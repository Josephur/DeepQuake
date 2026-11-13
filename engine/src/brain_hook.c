/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * The brain hook (#31): the machinery that lets one nominated monster be
 * driven by a policy instead of its QuakeC decisions. Shared by the
 * simulator (libdqsim, which wraps it with session/error handling) and
 * the play client (M5, which nominates from its own code and feeds the
 * policy from ONNX inference, see brain_onnx.c).
 *
 * sv_phys.c is compiled with PR_ExecuteProgram renamed to
 * dqsim_hook_execute_program (see engine/CMakeLists.txt), so every QuakeC
 * execution the physics initiates arrives here first. Only one shape is
 * intercepted: the nominated monster's scheduled think, when the think
 * function is one of its stand/walk/run state functions — the points
 * where QuakeC makes a movement or attack decision. Attack, pain and
 * death state functions are not decisions of that kind and execute under
 * QuakeC as usual, so their animation and damage stay stock.
 */
#include "quakedef.h"

#include <stdlib.h>
#include <string.h>

#include "brain_hook.h"
#include "dq_internal.h"
#include "dqsim/dqsim.h"

/* One policy-driven monster (#31, many at once from M6): the nomination's
 * callback and caller state, the QuakeC names it drives, its handicap and
 * its own animation clock. Slots fill in order from dq_brain_hook_nominate
 * and empty on release, clear, and every level start. A fixed table, so
 * nominating never allocates and never fails for want of memory. */
typedef struct dq_hook_slot {
    int entity_index; /* entity number, 0 = free slot */
    dqsim_hook_callback callback;
    void *user;
    char prefix[32]; /* QuakeC state prefix, e.g. "army_" */
    func_t fire_think; /* the attack-sequence think that fires, 0 = none */
    func_t attack_think; /* the function that starts an attack, 0 = none */
    float stand_frame, run_frame;
    int run_frames;
    unsigned anim_tick;
    dq_handicap_state handicap;
    int skill;
} dq_hook_slot;

static dq_hook_slot hook_slots[DQ_BRAIN_MAX_MONSTERS];
/* Overrides are per process, not per monster: the simulator drives one
 * monster and its tests set the schedule for it. */
static int hook_override_ticks = -1;   /* -1: use the schedule */
static float hook_override_aim = -1.0f;
static int progs_is_stock;
static char hook_level_name[MAX_QPATH];
/* The edict block of the level episode the nominations belong to: a death
 * restart reloads the same map name, but SV_SpawnServer reallocates the
 * edicts, which is the signal snapshots already use for "new episode". */
static void *hook_edicts;

static int read_whole_file(const char *path, unsigned char **out, size_t *length);
static int read_pak_progs(const char *path, unsigned char **out, size_t *length);
static int read_whole_file(const char *path, unsigned char **out, size_t *length)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        return 0;
    }
    *out = malloc((size_t)size);
    if (!*out || fread(*out, 1, (size_t)size, f) != (size_t)size) {
        free(*out);
        fclose(f);
        return 0;
    }
    fclose(f);
    *length = (size_t)size;
    return 1;
}

/* Extract "progs.dat" from a PAK file into out (malloc'd). */
static int read_pak_progs(const char *path, unsigned char **out, size_t *length)
{
    unsigned char *pak_bytes;
    size_t pak_length;
    if (!read_whole_file(path, &pak_bytes, &pak_length) || pak_length < 12)
        return 0;
    int32_t dir_offset, dir_length;
    memcpy(&dir_offset, pak_bytes + 4, 4);
    memcpy(&dir_length, pak_bytes + 8, 4);
    int found = 0;
    if (memcmp(pak_bytes, "PACK", 4) == 0 && dir_offset >= 0 && dir_length >= 0 &&
        (size_t)(dir_offset + dir_length) <= pak_length) {
        for (int32_t at = dir_offset; at + 64 <= dir_offset + dir_length && !found; at += 64) {
            char name[57];
            memcpy(name, pak_bytes + at, 56);
            name[56] = '\0';
            if (strcmp(name, "progs.dat") != 0)
                continue;
            int32_t offset, size;
            memcpy(&offset, pak_bytes + at + 56, 4);
            memcpy(&size, pak_bytes + at + 60, 4);
            if (offset >= 0 && size > 0 && (size_t)(offset + size) <= pak_length) {
                *out = malloc((size_t)size);
                if (*out) {
                    memcpy(*out, pak_bytes + offset, (size_t)size);
                    *length = (size_t)size;
                    found = 1;
                }
            }
        }
    }
    free(pak_bytes);
    return found;
}

/* Identity of the stock id1 progs.dat, recorded by the embedding program
 * (dqsim_init reads id1 for the simulator; the client passes its basedir
 * to dq_brain_stock_identity_from_base). */
static uint64_t stock_progs_hash;
static size_t stock_progs_length;
static int stock_identity_valid;

/* The stock identity: progs.dat as the engine finds it under id1 - the
 * pak files in descending order (the engine searches later paks first),
 * then a loose id1/progs.dat. */
int dq_brain_stock_identity_from_base(const char *base)
{
    char path[1088];
    unsigned char *bytes = NULL;
    size_t length = 0;
    int highest = -1;
    for (int n = 0; n < 16; n++) {
        q_snprintf(path, sizeof(path), "%s/id1/pak%d.pak", base, n);
        FILE *f = fopen(path, "rb");
        if (!f)
            break;
        fclose(f);
        highest = n;
    }
    for (int n = highest; n >= 0 && !bytes; n--) {
        q_snprintf(path, sizeof(path), "%s/id1/pak%d.pak", base, n);
        read_pak_progs(path, &bytes, &length);
    }
    if (!bytes) {
        q_snprintf(path, sizeof(path), "%s/id1/progs.dat", base);
        read_whole_file(path, &bytes, &length);
    }
    if (!bytes)
        return 0;
    dq_brain_set_stock_identity(dq_fnv1a(DQ_FNV1A_SEED, bytes, length), length);
    stock_identity_valid = 1;
    free(bytes);
    return 1;
}

uint64_t dq_fnv1a(uint64_t hash, const void *data, size_t length)
{
    const unsigned char *bytes = data;

    for (size_t i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

unsigned char dq_classify_ray(vec3_t eye, vec3_t forward, float max_distance,
                              const trace_t *trace, float *out_frac)
{
    const float step = 8.0f; /* liquid and sky entries land within one step */
    vec3_t probe;

    if (trace->startsolid) {
        *out_frac = 0.0f;
        switch (SV_PointContents(eye)) {
        case CONTENTS_WATER: return DQSIM_SURFACE_WATER;
        case CONTENTS_SLIME: return DQSIM_SURFACE_SLIME;
        case CONTENTS_LAVA: return DQSIM_SURFACE_LAVA;
        case CONTENTS_SKY: return DQSIM_SURFACE_SKY;
        default: return DQSIM_SURFACE_WORLD;
        }
    }

    float reach = trace->fraction * max_distance;
    for (float d = step; d < reach; d += step) {
        VectorMA(eye, d, forward, probe);
        switch (SV_PointContents(probe)) {
        case CONTENTS_WATER: *out_frac = d / max_distance; return DQSIM_SURFACE_WATER;
        case CONTENTS_SLIME: *out_frac = d / max_distance; return DQSIM_SURFACE_SLIME;
        case CONTENTS_LAVA: *out_frac = d / max_distance; return DQSIM_SURFACE_LAVA;
        case CONTENTS_SKY: *out_frac = d / max_distance; return DQSIM_SURFACE_SKY;
        }
    }

    if (trace->fraction >= 1.0f) {
        *out_frac = 1.0f;
        return DQSIM_SURFACE_NOTHING;
    }
    *out_frac = trace->fraction;
    if (trace->ent && trace->ent != sv.edicts)
        return DQSIM_SURFACE_BRUSH; /* a door, a platform, a train */
    return DQSIM_SURFACE_WORLD;
}

static void effective_handicap(const dq_hook_slot *slot, int *reaction_ticks,
                               float *aim_error_deg)
{
    dq_handicap_schedule(slot->skill, reaction_ticks, aim_error_deg);
    if (hook_override_ticks >= 0)
        *reaction_ticks = hook_override_ticks;
    if (hook_override_aim >= 0.0f)
        *aim_error_deg = hook_override_aim;
}

int dq_brain_handicap_for_skill(int skill, dqsim_handicap *out)
{
    if (!out || skill < 0 || skill > 3)
        return -1;
    dq_handicap_schedule(skill, &out->reaction_ticks, &out->aim_error_deg);
    return 0;
}

void dq_brain_set_handicap_override(int32_t reaction_ticks, float aim_error_deg)
{
    hook_override_ticks = reaction_ticks;
    hook_override_aim = aim_error_deg;
}

void dq_brain_set_stock_identity(uint64_t hash, size_t length)
{
    stock_progs_hash = hash;
    stock_progs_length = length;
    stock_identity_valid = 1;
}

static void slot_reset(dq_hook_slot *slot)
{
    memset(slot, 0, sizeof(*slot));
    dq_handicap_reset(&slot->handicap);
}

size_t dq_brain_hook_handicaps(dq_handicap_state *out, size_t max)
{
    size_t count = 0;

    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++) {
        if (!hook_slots[i].entity_index)
            continue;
        if (count < max && out)
            out[count] = hook_slots[i].handicap;
        count++;
    }
    return count;
}

int dq_brain_set_handicap_states(const dq_handicap_state *in, size_t count)
{
    size_t occupied = 0;

    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        if (hook_slots[i].entity_index)
            occupied++;
    if (occupied != count)
        return -1;
    size_t at = 0;
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        if (hook_slots[i].entity_index)
            hook_slots[i].handicap = in[at++];
    return (int)count;
}

void dq_brain_hook_on_level_start(void)
{
    if (!stock_identity_valid)
        return;
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        slot_reset(&hook_slots[i]);
    hook_override_ticks = -1;
    hook_override_aim = -1.0f;
    /* Identity of the progs this level actually loaded: the raw file
     * bytes through the engine's search path (a game directory shadows
     * id1), hashed with its length. Reading the in-memory progs instead
     * would hash a byte-swapped copy under a stale com_filesize. */
    byte *raw = COM_LoadMallocFile("progs.dat", NULL);
    progs_is_stock = 0;
    if (raw) {
        size_t length = (size_t)com_filesize;
        progs_is_stock = length == stock_progs_length &&
                         dq_fnv1a(DQ_FNV1A_SEED, raw, length) == stock_progs_hash;
        free(raw);
    }
}

int dq_brain_progs_is_stock(void)
{
    return progs_is_stock;
}

static func_t find_function(const char *name)
{
    for (int i = 1; i < progs->numfunctions; i++)
        if (strcmp(PR_GetString(pr_functions[i].s_name), name) == 0)
            return i;
    return 0;
}

/* The attack-sequence think that actually fires: the "<prefix>atkN" state
 * function whose statements call "<prefix>fire" (#34). 0 when there is
 * none (a melee monster), in which case aim error has nothing to act on. */
static func_t find_fire_think(const char *prefix)
{
    char name[40];
    q_snprintf(name, sizeof(name), "%sfire", prefix);
    func_t fire = find_function(name);
    if (!fire)
        return 0;
    for (int n = 1; n <= 24; n++) {
        q_snprintf(name, sizeof(name), "%satk%d", prefix, n);
        func_t candidate = find_function(name);
        if (!candidate)
            continue;
        dfunction_t *function = &pr_functions[candidate];
        for (int s = 0; s < 512; s++) {
            dstatement_t *statement = &pr_statements[function->first_statement + s];
            if (statement->op == OP_DONE || statement->op == OP_RETURN)
                break;
            if (statement->op >= OP_CALL0 && statement->op <= OP_CALL8) {
                func_t called;
                memcpy(&called, &pr_globals[statement->a], sizeof(called));
                if (called == fire)
                    return candidate;
            }
        }
    }
    return 0;
}

/* Model frame a QuakeC state function selects, from its OP_STATE. */
static float state_frame(func_t fnum)
{
    if (!fnum)
        return -1;
    dstatement_t *statement = &pr_statements[pr_functions[fnum].first_statement];
    if (statement->op != OP_STATE)
        return -1;
    return pr_globals[statement->a];
}

static int is_decision_think(func_t fnum, const char *prefix)
{
    const char *name = PR_GetString(pr_functions[fnum].s_name);
    size_t length_prefix = strlen(prefix);
    if (strncmp(name, prefix, length_prefix) != 0)
        return 0;
    const char *rest = name + length_prefix;
    if (strncmp(rest, "stand", 5) != 0 && strncmp(rest, "walk", 4) != 0 &&
        strncmp(rest, "run", 3) != 0)
        return 0;
    size_t length = strlen(name);
    return length > 0 && name[length - 1] >= '0' && name[length - 1] <= '9';
}

static void hook_observe(int entity_index, edict_t *ent, dqsim_hook_observation *obs)
{
    static const float pitches[DQSIM_HOOK_RAY_PITCHES] = {0.0f, 30.0f};
    memset(obs, 0, sizeof(*obs));
    obs->time = sv.time;
    obs->entity_index = entity_index;
    VectorCopy(ent->v.origin, obs->origin);
    VectorCopy(ent->v.velocity, obs->velocity);
    VectorCopy(ent->v.angles, obs->angles);
    obs->health = ent->v.health;
    obs->on_ground = ((int)ent->v.flags & FL_ONGROUND) != 0;
    obs->waterlevel = (int)ent->v.waterlevel;
    eval_t *cooldown = GetEdictFieldValue(ent, "attack_finished");
    obs->attack_cooldown = cooldown && cooldown->_float > (float)sv.time
                               ? cooldown->_float - (float)sv.time
                               : 0.0f;

    vec3_t eye;
    VectorAdd(ent->v.origin, ent->v.view_ofs, eye);
    for (int p = 0; p < DQSIM_HOOK_RAY_PITCHES; p++) {
        for (int y = 0; y < DQSIM_HOOK_RAY_YAWS; y++) {
            vec3_t angles, forward, right, up, end;
            angles[PITCH] = pitches[p];
            angles[YAW] = ent->v.angles[YAW] + 360.0f * (float)y / DQSIM_HOOK_RAY_YAWS;
            angles[ROLL] = 0;
            AngleVectors(angles, forward, right, up);
            VectorMA(eye, DQSIM_HOOK_RAY_DISTANCE, forward, end);
            trace_t trace = SV_Move(eye, vec3_origin, vec3_origin, end, MOVE_NOMONSTERS, ent);
            float fraction;
            int slot = p * DQSIM_HOOK_RAY_YAWS + y;
            obs->ray_kinds[slot] = dq_classify_ray(eye, forward, DQSIM_HOOK_RAY_DISTANCE,
                                                   &trace, &fraction);
            obs->rays[slot] = fraction;
        }
    }

    for (int i = 1; i < sv.num_edicts; i++) {
        edict_t *other = EDICT_NUM(i);
        if (other == ent || other->free)
            continue;
        if (!((int)other->v.flags & (FL_CLIENT | FL_MONSTER | FL_ITEM)))
            continue;
        vec3_t target, delta;
        VectorAdd(other->v.origin, other->v.view_ofs, target);
        VectorSubtract(target, eye, delta);
        float distance = sqrtf(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
        if (distance > DQSIM_HOOK_SIGHT_RANGE)
            continue;
        trace_t trace = SV_Move(eye, vec3_origin, vec3_origin, target, MOVE_NOMONSTERS, ent);
        if (trace.fraction < 1.0f && trace.ent != other)
            continue; /* no line of sight */
        dqsim_hook_visible entry;
        memset(&entry, 0, sizeof(entry));
        entry.index = i;
        entry.flags = (int)other->v.flags;
        VectorSubtract(other->v.origin, ent->v.origin, entry.relative);
        VectorCopy(other->v.velocity, entry.velocity);
        entry.health = other->v.health;
        entry.distance = distance;
        /* Keep the nearest DQSIM_HOOK_MAX_VISIBLE, nearest first. */
        int at = obs->visible_count;
        if (at == DQSIM_HOOK_MAX_VISIBLE) {
            if (distance >= obs->visible[at - 1].distance)
                continue;
            at--;
        } else {
            obs->visible_count++;
        }
        while (at > 0 && obs->visible[at - 1].distance > distance) {
            obs->visible[at] = obs->visible[at - 1];
            at--;
        }
        obs->visible[at] = entry;
    }
}

static void hook_apply(edict_t *ent, func_t fnum, dq_hook_slot *slot,
                       const dqsim_hook_observation *obs,
                       const dqsim_hook_action *action)
{
    ent->v.angles[YAW] = action->desired_yaw;
    ent->v.v_angle[YAW] = action->desired_yaw;

    vec3_t angles = {0, action->desired_yaw, 0}, forward, right, up, wish;
    AngleVectors(angles, forward, right, up);
    wish[0] = forward[0] * action->forward + right[0] * action->side;
    wish[1] = forward[1] * action->forward + right[1] * action->side;
    wish[2] = 0;
    float speed = sqrtf(wish[0] * wish[0] + wish[1] * wish[1]);
    if (speed > 0.01f) {
        if ((int)ent->v.flags & FL_ONGROUND) {
            vec3_t move;
            VectorScale(wish, (float)host_frametime, move);
            SV_movestep(ent, move, true);
        } else {
            ent->v.velocity[0] = wish[0];
            ent->v.velocity[1] = wish[1];
        }
    }

    int attacked = 0;
    if (action->attack) {
        edict_t *enemy = NULL;
        for (int i = 0; i < obs->visible_count; i++) {
            if (obs->visible[i].flags & FL_CLIENT) {
                enemy = EDICT_NUM(obs->visible[i].index);
                break;
            }
        }
        eval_t *cooldown = enemy ? GetEdictFieldValue(ent, "attack_finished") : NULL;
        func_t attack = enemy ? slot->attack_think : 0;
        if (attack && (!cooldown || sv.time >= cooldown->_float)) {
            int reaction_ticks;
            float aim_error_deg;
            effective_handicap(slot, &reaction_ticks, &aim_error_deg);
            (void)reaction_ticks;
            dq_handicap_roll_aim(&slot->handicap, aim_error_deg);
            ent->v.enemy = EDICT_TO_PROG(enemy);
            pr_global_struct->self = EDICT_TO_PROG(ent);
            pr_global_struct->other = EDICT_TO_PROG(enemy);
            pr_global_struct->time = sv.time;
            PR_ExecuteProgram(attack);
            attacked = 1;
        }
    }

    if (!attacked) {
        /* Animate from the state functions' own frames: run while moving,
         * stand while still, advancing at the QuakeC think cadence. */
        slot->anim_tick++;
        if (speed > 0.01f && slot->run_frame >= 0)
            ent->v.frame = slot->run_frame + (float)((slot->anim_tick / 7) % slot->run_frames);
        else if (slot->stand_frame >= 0)
            ent->v.frame = slot->stand_frame + (float)((slot->anim_tick / 7) % 8);
        ent->v.think = fnum;
        ent->v.nextthink = (float)(sv.time + host_frametime);
    }
}

/* Client extension point (M5): called while no monster is nominated, so
 * an embedding program (the play client's ONNX inference) can nominate
 * one. NULL in the simulator. */
static void (*hook_tick)(void);

void dq_brain_hook_set_tick(void (*fn)(void))
{
    hook_tick = fn;
}

static dq_hook_slot *slot_for_edict(const edict_t *ent)
{
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++) {
        dq_hook_slot *slot = &hook_slots[i];
        if (slot->entity_index && EDICT_NUM(slot->entity_index) == ent)
            return slot;
    }
    return NULL;
}

void dqsim_hook_execute_program(func_t fnum)
{
    /* A new level, or the same map reloaded after a death: both clear
     * every nomination, because edict indices belong to one episode. The
     * simulator's load_map calls dq_brain_hook_on_level_start itself; the
     * client loads maps through the console and is covered by this test.
     * The client's tick runs first: it records the stock identity on its
     * first call, which the level start needs. */
    int level_changed = strcmp(sv.name, hook_level_name) != 0 || sv.edicts != hook_edicts;
    if (level_changed) {
        q_strlcpy(hook_level_name, sv.name, sizeof(hook_level_name));
        hook_edicts = sv.edicts;
    }
    if (hook_tick)
        hook_tick();
    if (level_changed)
        dq_brain_hook_on_level_start();
    dq_hook_slot *slot = NULL;
    if (progs_is_stock && fnum > 0 && pr_global_struct->self) {
        edict_t *self = PROG_TO_EDICT(pr_global_struct->self);
        if (self)
            slot = slot_for_edict(self);
    }
    if (slot) {
        edict_t *ent = EDICT_NUM(slot->entity_index);
        if (!ent->free && pr_global_struct->self == EDICT_TO_PROG(ent)) {
            /* The pending aim error acts where the shot is actually aimed:
             * the fire think reads the enemy's origin, so offset that
             * origin tangentially for this one execution and restore it. */
            if (slot->handicap.aim_pending && fnum == slot->fire_think && ent->v.enemy) {
                edict_t *enemy = PROG_TO_EDICT(ent->v.enemy);
                vec3_t saved, delta;
                VectorCopy(enemy->v.origin, saved);
                VectorSubtract(enemy->v.origin, ent->v.origin, delta);
                float distance = sqrtf(delta[0] * delta[0] + delta[1] * delta[1]);
                if (distance > 1.0f) {
                    float along = distance * tanf(slot->handicap.aim_yaw_deg *
                                                  3.14159265f / 180.0f);
                    enemy->v.origin[0] += -delta[1] / distance * along;
                    enemy->v.origin[1] += delta[0] / distance * along;
                    enemy->v.origin[2] += distance * tanf(slot->handicap.aim_pitch_deg *
                                                          3.14159265f / 180.0f);
                }
                slot->handicap.aim_pending = 0;
                PR_ExecuteProgram(fnum);
                VectorCopy(saved, enemy->v.origin);
                return;
            }
            if (ent->v.health > 0 && (func_t)ent->v.think == fnum && ent->v.nextthink == 0 &&
                is_decision_think(fnum, slot->prefix)) {
                dqsim_hook_observation obs;
                hook_observe(slot->entity_index, ent, &obs);
                int visible_enemy = -1;
                for (int i = 0; i < obs.visible_count; i++)
                    if (obs.visible[i].flags & FL_CLIENT) {
                        visible_enemy = obs.visible[i].index;
                        break;
                    }
                int reaction_ticks;
                float aim_error_deg;
                effective_handicap(slot, &reaction_ticks, &aim_error_deg);
                (void)aim_error_deg;
                int respond = dq_handicap_may_respond(&slot->handicap, visible_enemy, obs.time,
                                                      reaction_ticks, host_frametime);
                dqsim_hook_action action;
                memset(&action, 0, sizeof(action));
                action.desired_yaw = ent->v.angles[YAW];
                slot->callback(&obs, &action, slot->user);
                if (!ent->free) {
                    if (!respond) {
                        /* Inside the reaction delay: the callback ran,
                         * but nothing is applied (no turn, move or attack). */
                        memset(&action, 0, sizeof(action));
                        action.desired_yaw = ent->v.angles[YAW];
                    }
                    hook_apply(ent, fnum, slot, &obs, &action);
                }
                return;
            }
        }
    }
    PR_ExecuteProgram(fnum);
}

/* The function that starts this class's attack (the hook has no aim-error
 * target for a monster without one). The Grunt chains "<prefix>atk1..."
 * and fires from "<prefix>fire"; a melee monster such as the Dog bites
 * from "<prefix>atta1..." instead, so try the known shapes in order. */
static func_t find_attack_think(const char *prefix)
{
    static const char *const names[] = {"atk1", "atta1", "attack", "melee1"};
    char name[40];

    for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++) {
        q_snprintf(name, sizeof(name), "%s%s", prefix, names[i]);
        func_t found = find_function(name);
        if (found)
            return found;
    }
    return 0;
}

const char *dq_brain_hook_nominate(int entity_index, dqsim_hook_callback callback, void *user)
{
    if (!callback)
        return "callback is NULL";
    if (entity_index <= 0)
        return "entity_index must be positive";
    if (!sv.active)
        return "no map is loaded";
    if (!progs_is_stock)
        return "the loaded progs.dat is not the stock one";
    if (entity_index >= sv.num_edicts)
        return "no such entity";
    edict_t *ent = EDICT_NUM(entity_index);
    const char *classname = ent->free ? "" : PR_GetString(ent->v.classname);
    if (ent->free || strncmp(classname, "monster_", 8) != 0)
        return "the entity is not a monster";
    if (dq_brain_hook_is_nominated(entity_index))
        return "the entity is already policy-driven";

    dq_hook_slot *slot = NULL;
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS && !slot; i++)
        if (!hook_slots[i].entity_index)
            slot = &hook_slots[i];
    if (!slot)
        return "no free monster slot (DQ_BRAIN_MAX_MONSTERS reached)";

    slot_reset(slot);
    slot->entity_index = entity_index;
    slot->callback = callback;
    slot->user = user;
    slot->skill = current_skill < 0 ? 0 : current_skill > 3 ? 3 : current_skill;
    hook_override_ticks = -1;
    hook_override_aim = -1.0f;
    q_snprintf(slot->prefix, sizeof(slot->prefix), "%s_", classname + 8);
    slot->fire_think = find_fire_think(slot->prefix);
    slot->attack_think = find_attack_think(slot->prefix);
    char name[40];
    q_snprintf(name, sizeof(name), "%sstand1", slot->prefix);
    slot->stand_frame = state_frame(find_function(name));
    q_snprintf(name, sizeof(name), "%srun1", slot->prefix);
    slot->run_frame = state_frame(find_function(name));
    slot->run_frames = 0;
    for (int i = 1; i <= 24; i++) {
        q_snprintf(name, sizeof(name), "%srun%d", slot->prefix, i);
        if (!find_function(name))
            break;
        slot->run_frames = i;
    }
    if (slot->run_frames == 0)
        slot->run_frames = 1;
    return NULL;
}

void dq_brain_hook_release(int entity_index)
{
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        if (hook_slots[i].entity_index == entity_index)
            slot_reset(&hook_slots[i]);
}

void dq_brain_hook_clear(void)
{
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        slot_reset(&hook_slots[i]);
}

int dq_brain_hook_active(void)
{
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        if (hook_slots[i].entity_index)
            return hook_slots[i].entity_index;
    return 0;
}

int dq_brain_hook_count(void)
{
    int count = 0;

    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        if (hook_slots[i].entity_index)
            count++;
    return count;
}

int dq_brain_hook_is_nominated(int entity_index)
{
    if (entity_index <= 0)
        return 0;
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        if (hook_slots[i].entity_index == entity_index)
            return 1;
    return 0;
}

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
#define DQ_MAX_FIRE_THINKS 4
/* The longest stock stand/run chain is well under this (the Shambler stands
 * in 17 states, the Dog and Shalrath run in 12); the walk caps here. */
#define DQ_MAX_ANIM_FRAMES 32
/* Turning is rate-limited like the stock ai_turn/ChangeYaw instead of snapping
 * to the policy's requested facing (#61). Snapping made the monster dart across
 * visibility-leaf boundaries and flicker. deg/s; the policy's turn grid is at
 * most 30 deg per 6-tick decision = 360 deg/s, so this keeps up but smooths the
 * step into a continuous turn. */
#define DQ_MONSTER_TURN_RATE 360.0f
/* How far past the next tick a driven monster's think is scheduled (#61).
 * The hook thinks every tick. Scheduling the think for exactly the next tick
 * made the server tell the client "finish interpolating in 0 seconds"
 * (sv_main.c sends Q_rint((nextthink - sv.time) * 255) after sv.time has
 * already advanced), and the client then blended the monster's position and
 * pose with 0 / 0 (r_alias.c: (cl.time - lerpstart) / (lerpfinish -
 * lerpstart), all three equal on a local server). The NaN position or pose
 * draws nothing, so the monster blinked out for a tick every time it moved,
 * turned or changed frame - and stopped blinking in an attack, whose stock
 * states think 0.1 s apart. The margin keeps that interval at two 1/255 s
 * steps, clear of zero with room for float rounding late in a long level; it
 * is capped at half a tick so the think still runs on the very next tick
 * (SV_RunThink runs a think that is due before the end of the tick). */
#define DQ_THINK_MARGIN 0.006
/* A driven monster has two facings (#61). The policy's facing is the frame
 * its observation is taken in and its forward/side moves are measured from;
 * the policy strafes and backs up freely, as a player does. A Quake monster
 * model has only run-forward frames, so showing the policy's facing made the
 * monster run with its back to where it was going, or slide sideways. The
 * facing shown to clients therefore follows what the body actually does: the
 * direction it really moved this tick, or the policy's facing when it did not
 * move. Both live in the edict, so snapshots and the state hash carry them
 * with no extra state: angles[YAW] is the shown facing (what clients draw and
 * the stock attack states turn), v_angle[YAW] the policy's facing and
 * v_angle[PITCH] the shown facing as the hook last wrote it. Stock QuakeC
 * never reads a monster's v_angle. When angles[YAW] is no longer what the hook
 * wrote, QuakeC turned the monster (an attack's ai_face) and the policy's
 * facing adopts the result. deg/s; twice the policy's own turn rate, so the
 * body visibly leads into a change of direction rather than snapping. */
#define DQ_SHOWN_TURN_RATE 720.0f
/* The monster counts as moving when it covered at least this fraction of the
 * step it was asked for; a step into a wall covers none of it. */
#define DQ_MOVED_FRACTION 0.25f
/* A melee attack - any attack that is not a projectile or a beam - starts
 * only when the enemy is within reach (#63). Asked to attack an enemy it can
 * see but not reach, the monster charges: it turns to the enemy and runs at
 * it, and a class with a leap leaps when the stock AI would. Without this a
 * Dog, Fiend or Knight played its whole bite or swing where it stood, again
 * and again, at any distance.
 *
 * Reach is flat distance between origins: the monster's half width, the
 * player's (16), and 32 more for the step the stock attack states take toward
 * the enemy before the blow lands. That is 64 for a Grunt-sized monster and 80
 * for an Ogre-sized one, whose corner-to-corner contact distance is already
 * 68. The height limit keeps a monster from swinging at a player on a ledge
 * above it. The leap window is the stock one: CheckDogJump leaps between 80
 * and 150 units, CheckDemonJump between 100 and 200. The charge runs at the
 * class's own run speed (see DQ_STATE_INTERVAL); DQ_CHARGE_SPEED, in units per
 * second, stands in for a class whose run chain names no distance. */
#define DQ_MELEE_REACH_MARGIN 48.0f
#define DQ_MELEE_REACH_HEIGHT 64.0f
#define DQ_LEAP_MIN 100.0f
#define DQ_LEAP_MAX 200.0f
#define DQ_CHARGE_SPEED 320.0f
/* A driven monster moves at its own body's speed (#61). The policy commands
 * the player's 400 units a second; a stock Grunt runs at about 110 and its run
 * frames are drawn for that, so at 400 it skated across the floor with its
 * legs barely moving. Each class's speeds are read from its own states: the
 * distances its walk and run chains pass to ai_walk and ai_run, one state
 * every 0.1 s. Whatever the policy asks for is capped there. It walks, with
 * its walk frames, until it has an enemy - it saw the player or was hurt, the
 * stock rule for leaving the walk states - and runs from then on. */
#define DQ_STATE_INTERVAL 0.1f
typedef struct dq_hook_slot {
    int entity_index; /* entity number, 0 = free slot */
    dqsim_hook_callback callback;
    void *user;
    char prefix[32]; /* QuakeC state prefix, e.g. "army_" */
    /* The attack-sequence frames that fire the shot (the Grunt's "army_atk5",
     * the Enforcer's "enf_atk6" and "enf_atk10"): the aim error acts there.
     * Empty for a melee monster, which has nothing to aim off. */
    func_t fire_thinks[DQ_MAX_FIRE_THINKS];
    int fire_count;
    func_t attack_think; /* the function that starts an attack, 0 = none */
    /* The two attack entries the stock AI picks between (#53): th_melee when
     * the enemy is in melee range (QuakeC's RANGE_MELEE is 64), the ranged
     * entry otherwise. The ranged entry is the spawn's th_missile when its
     * chain fires a projectile (the Ogre's grenade throw, unreachable from a
     * name-shaped probe), else the shape probe's entry - the Dog's th_missile
     * is a pounce, and one entry execution per decision cannot charge with
     * it. 0 = the class sets none. */
    func_t melee_think, missile_think;
    /* Does each entry hit from a distance - a projectile, a beam (#63)? One
     * that does not is a melee attack, and a melee attack out of reach is
     * never started: the monster charges instead. */
    int melee_ranged, missile_ranged;
    /* The class's leap, if it has one: a spawn th_missile that fires nothing
     * (the Dog's "dog_leap1", the Fiend's "demon1_jump1"). The charge uses it
     * the way the stock AI does, to cross the last stretch. 0 = none. */
    func_t leap_think;
    /* The stand and run animations as the stock QuakeC actually plays them:
     * the frame of each state in the chain, in order (#61). Cycling these
     * sequences - rather than offsetting from the first frame - keeps every
     * frame the hook sets one the class's own states set, so a chain that is
     * shorter than eight states, a single-state stand (the Shalrath), or a
     * run that loops back onto an earlier frame (the Shalrath's run12) can
     * never name a frame the model does not have. */
    float stand_seq[DQ_MAX_ANIM_FRAMES];
    int stand_len;
    float run_seq[DQ_MAX_ANIM_FRAMES];
    int run_len;
    float walk_seq[DQ_MAX_ANIM_FRAMES];
    int walk_len;
    /* The body's own speeds in units per second (see DQ_STATE_INTERVAL);
     * 0 = the class's chain names no distance, and nothing is capped. */
    float run_speed, walk_speed;
    /* Everything the hook changes per monster while it runs (RETRAIN_SPEC
     * part 0B): the animation clock, the investigation phase, the heard
     * noise and the alert pace. Carried by snapshots and the state hash
     * (dq_brain_hook_slot_states), which also closes the gap that the
     * animation clock was never carried. */
    dq_slot_state state;
    /* The monster's health when nominated: the observation's health input
     * is a fraction of it (part 0D uses it; the field is kept from here). */
    float start_health;
    dq_handicap_state handicap;
    int skill;
} dq_hook_slot;

static dq_hook_slot hook_slots[DQ_BRAIN_MAX_MONSTERS];

/* Resolve a class's QuakeC functions from its prefix: the stand and run
 * frames, the attack entry and the shot. Defined below, next to the
 * lookups it uses. */
static int resolve_prefix(dq_hook_slot *slot, const char *prefix);
static void find_enemy_globals(void);

/* The class prefix of a decision think's name ("enf_stand1" -> "enf_"). */
static int decision_think_prefix(const char *name, char *out, size_t size);
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

/* Hearing (RETRAIN_SPEC part 0B). Process-wide settings that survive map
 * loads; the skill table gives the defaults that `skill` chooses between
 * and that every entry of can be overridden. A monster's own range and
 * run chance come from its slot's skill when the settings name no value. */
static dqsim_hearing hearing = {1, 0.0f, -1.0f, 0};
static const struct {
    float range;
    float run_chance;
} hearing_table[4] = {
    {500.0f, 0.25f}, /* 0 easy */
    {1000.0f, 0.5f}, /* 1 normal */
    {1000.0f, 0.5f}, /* 2 hard */
    {1500.0f, 0.75f}, /* 3 nightmare */
};
/* The explosion temp-entity write the hearing watches for: svc_temp_entity
 * (23), then TE_EXPLOSION (3) or TE_TAREXPLOSION (4), then three
 * coordinates, written to sv.datagram by QuakeC (PF_WriteByte and
 * PF_WriteCoord, renamed into the wrappers below). */
#define DQ_HEAR_TE_NONE 0
#define DQ_HEAR_TE_MARKER 1
#define DQ_HEAR_TE_KIND 2
static int hear_te_state;
static int hear_te_coords;
static vec3_t hear_te_origin;

void dq_brain_set_hearing(const dqsim_hearing *settings)
{
    if (settings)
        hearing = *settings;
    else
        hearing = (dqsim_hearing){1, 0.0f, -1.0f, 0};
}

int dq_brain_hearing_for_skill(int skill, dqsim_hearing *out)
{
    if (skill < 0 || skill > 3 || !out)
        return -1;
    out->enabled = 1;
    out->range = hearing_table[skill].range;
    out->run_chance = hearing_table[skill].run_chance;
    out->player_noises = 0;
    return 0;
}

/* The hearing this slot uses: its own skill's table row where the settings
 * name no value. */
static void hearing_for_slot(const dq_hook_slot *slot, dqsim_hearing *out)
{
    *out = hearing;
    if (out->range <= 0.0f)
        out->range = hearing_table[slot->skill].range;
    if (out->run_chance < 0.0f)
        out->run_chance = hearing_table[slot->skill].run_chance;
}

/* Whether the executing QuakeC function is the emitter's own pain handler:
 * a cry is the emitter's death sound (health gone) or its th_pain. */
static int hear_is_cry(edict_t *entity)
{
    if (entity->v.health <= 0.0f)
        return 1;
    eval_t *th_pain = GetEdictFieldValue(entity, "th_pain");
    if (th_pain && th_pain->function && pr_xfunction &&
        (func_t)(pr_xfunction - pr_functions) == th_pain->function)
        return 1;
    return 0;
}

/* Hand one noise to every nominated, live monster other than the emitter
 * that is within its hearing range. Each monster keeps the most recent
 * noise; two in the same tick, the nearer wins. A monster without an enemy
 * that hears while remembering nothing starts an alert, whose pace is
 * drawn once from the seeded stream. */
static void hear_deliver(int kind, const vec3_t origin, const edict_t *emitter)
{
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++) {
        dq_hook_slot *slot = &hook_slots[i];
        if (!slot->entity_index)
            continue;
        edict_t *ent = EDICT_NUM(slot->entity_index);
        if (ent->free || ent->v.health <= 0.0f || ent == emitter)
            continue;
        dqsim_hearing own;
        hearing_for_slot(slot, &own);
        float dx = origin[0] - ent->v.origin[0];
        float dy = origin[1] - ent->v.origin[1];
        float dz = origin[2] - ent->v.origin[2];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);
        if (getenv("DQ_HEAR_DEBUG"))
            fprintf(stderr, "DBGDEL kind=%d slot=%d dist=%.0f t=%.1f origin=%.0f/%.0f/%.0f\n",
                    kind, slot->entity_index, distance, sv.time, origin[0], origin[1],
                    origin[2]);
        if (distance > own.range)
            continue;
        if (slot->state.heard_valid) {
            if (slot->state.heard_time > sv.time)
                continue; /* a future noise from another tick's clock: keep it */
            if (slot->state.heard_time == sv.time) {
                float ox = ent->v.origin[0] - slot->state.heard_origin[0];
                float oy = ent->v.origin[1] - slot->state.heard_origin[1];
                float oz = ent->v.origin[2] - slot->state.heard_origin[2];
                if (distance * distance >= ox * ox + oy * oy + oz * oz)
                    continue; /* two noises in one tick: the nearer wins */
            }
        }
        int fresh_alert = !slot->state.heard_valid;
        slot->state.heard_valid = 1;
        VectorCopy(origin, slot->state.heard_origin);
        slot->state.heard_time = sv.time;
        slot->state.heard_kind = kind;
        if (fresh_alert && !ent->v.enemy) {
            /* The alert's pace, once per alert: run or investigate, from the
             * hearing stream (separate from the game's, per the maintainer's
             * decision on the issue; deterministic for a load seed). */
            double draw = dq_hear_rng() / (double)RAND_MAX;
            slot->state.pace = draw < own.run_chance ? DQSIM_PACE_RUN
                                                     : DQSIM_PACE_INVESTIGATE;
        }
    }
}

/* The SV_StartSound every QuakeC sound() arrives through in the simulator
 * and the client (pr_cmds.c, and sv_phys.c, are compiled with the name
 * renamed to this): classify, deliver, then make the real sound. With
 * hearing off or no nominated monster this changes nothing. */
void dq_hear_start_sound(edict_t *entity, int channel, const char *sample, int volume,
                         float attenuation)
{
    int kind = 0;
    int flags = entity ? (int)entity->v.flags : 0;
    if (hearing.enabled && entity) {
        if ((channel & 7) == 1 && (flags & (FL_CLIENT | FL_MONSTER)))
            kind = DQSIM_NOISE_GUNFIRE;
        else if ((channel & 7) == 2 && (flags & FL_MONSTER) && hear_is_cry(entity))
            kind = DQSIM_NOISE_CRY;
        else if (hearing.player_noises && (flags & FL_CLIENT))
            kind = DQSIM_NOISE_PLAYER;
    }
    if (kind)
        hear_deliver(kind, entity->v.origin, entity);
    SV_StartSound(entity, channel, sample, volume, attenuation);
}

void dq_hear_write_byte(sizebuf_t *sb, int c)
{
    if (sb == &sv.datagram) {
        if (hear_te_state == DQ_HEAR_TE_NONE) {
            if (c == 23) /* svc_temp_entity */
                hear_te_state = DQ_HEAR_TE_MARKER;
        } else if (hear_te_state == DQ_HEAR_TE_MARKER) {
            if (c == 3 || c == 4) { /* TE_EXPLOSION, TE_TAREXPLOSION */
                hear_te_state = DQ_HEAR_TE_KIND;
            } else {
                hear_te_state = DQ_HEAR_TE_NONE;
            }
            hear_te_coords = 0;
        } else {
            hear_te_state = DQ_HEAR_TE_NONE;
        }
    }
    MSG_WriteByte(sb, c);
}

void dq_hear_write_coord(sizebuf_t *sb, float f, unsigned int flags)
{
    if (sb == &sv.datagram) {
        if (hear_te_state == DQ_HEAR_TE_KIND) {
            hear_te_origin[hear_te_coords++] = f;
            if (hear_te_coords == 3) {
                hear_deliver(DQSIM_NOISE_EXPLOSION, hear_te_origin, NULL);
                hear_te_state = DQ_HEAR_TE_NONE;
            }
        } else if (hear_te_state != DQ_HEAR_TE_NONE) {
            hear_te_state = DQ_HEAR_TE_NONE;
        }
    }
    MSG_WriteCoord(sb, f, flags);
}

/* Forget a heard noise past its memory and end the alert. Runs per
 * nominated monster per applied tick, and before each report. */
static void hear_tick(dq_hook_slot *slot, const edict_t *ent)
{
    if (slot->state.heard_valid &&
        sv.time - slot->state.heard_time >= DQSIM_HOOK_HEARD_SECONDS) {
        slot->state.heard_valid = 0;
        slot->state.pace = DQSIM_PACE_WALK;
    }
    (void)ent;
}

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

size_t dq_brain_hook_slot_states(dq_slot_state *out, size_t max)
{
    size_t count = 0;

    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++) {
        if (!hook_slots[i].entity_index)
            continue;
        if (count < max && out)
            out[count] = hook_slots[i].state;
        count++;
    }
    return count;
}

int dq_brain_set_slot_states(const dq_slot_state *in, size_t count)
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
            hook_slots[i].state = in[at++];
    return (int)count;
}

void dq_brain_hook_on_level_start(void)
{
    if (progs_is_stock)
        find_enemy_globals();
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

/* The attack-sequence frames that actually fire: the "<prefix>atkN" state
 * functions whose statements call the class's shot function. A class can
 * fire more than once per attack (the Enforcer's chain fires at "enf_atk6"
 * and at "enf_atk10"), and every shot of the attack has to be handicapped,
 * so all of them are kept. None for a melee monster, which has nothing to
 * aim off.
 *
 * The shot is not always named after the class: the Grunt's is "army_fire"
 * and the Enforcer's "enforcer_fire", while the Enforcer's prefix is "enf_".
 * So the shot is found by what the attack chain calls, not by its name: a
 * function whose name contains "fire" or "shot" fires one. The Ogre's
 * grenade launch is "OgreFireGrenade", which contains "Fire" without ending
 * in it (#53); "SUB_CheckRefire" is not a shot, and the engine's own helpers
 * that merely share the syllable (the doors' "door_fire") are not reached
 * from a monster's attack chain. The Shambler's beam is "CastLightning" (#63):
 * without it the beam chain counted as firing nothing, the Shambler's ranged
 * entry was dropped, and it only ever clawed. */
static int is_shot_function(const char *name)
{
    static const char *const shots[] = {"fire", "shot", "missile", "lightning"};
    if (strncmp(name, "SUB_", 4) == 0)
        return 0;
    for (const char *c = name; *c; c++)
        for (size_t i = 0; i < sizeof(shots) / sizeof(*shots); i++) {
            size_t length = strlen(shots[i]);
            if (strncasecmp(c, shots[i], length) == 0)
                return 1;
        }
    return 0;
}

/* The stock AI sets these three globals from the monster's own sight before
 * any attack (ai_stand/ai_run's FindTargetDistance); the attack chains read
 * them - the Ogre's grenade throw picks its lead from "enemy_range" and
 * gates on "enemy_vis" (#53). The hook replaces the think that would have
 * set them, so it sets the same three by the same rule: RANGE_MELEE 1 below
 * 120 units, RANGE_NEAR 2 below 500, RANGE_MID 3 below 1000, RANGE_FAR 4.
 * Offsets into pr_globals, found once per progs load; 0 = the global does
 * not exist in this progs. */
static int range_global_ofs, vis_global_ofs, infront_global_ofs;
static int enemy_globals_found;

static void find_enemy_globals(void)
{
    ddef_t *defs = (ddef_t *)((byte *)progs + progs->ofs_globaldefs);
    for (int i = 0; i < progs->numglobaldefs; i++) {
        const char *name = PR_GetString(defs[i].s_name);
        if (strcmp(name, "enemy_range") == 0)
            range_global_ofs = defs[i].ofs;
        else if (strcmp(name, "enemy_vis") == 0)
            vis_global_ofs = defs[i].ofs;
        else if (strcmp(name, "enemy_infront") == 0)
            infront_global_ofs = defs[i].ofs;
    }
    enemy_globals_found = 1;
}

static int chain_has_shot_call(func_t fnum)
{
    dfunction_t *function = &pr_functions[fnum];
    for (int s = 0; s < 512; s++) {
        dstatement_t *statement = &pr_statements[function->first_statement + s];
        if (statement->op == OP_DONE || statement->op == OP_RETURN)
            break;
        if (statement->op >= OP_CALL0 && statement->op <= OP_CALL8) {
            func_t called;
            memcpy(&called, &pr_globals[statement->a], sizeof(called));
            if (called > 0 && called < progs->numfunctions &&
                is_shot_function(PR_GetString(pr_functions[called].s_name)))
                return 1;
        }
    }
    return 0;
}

/* The state function this one chains to: an attack chain is a run of frame
 * functions, each ending in OP_STATE (frame, next think). */
static func_t chain_next(func_t fnum)
{
    dstatement_t *statement = &pr_statements[pr_functions[fnum].first_statement];
    for (int s = 0; s < 512; s++, statement++) {
        if (statement->op == OP_DONE || statement->op == OP_RETURN)
            return 0;
        if (statement->op == OP_STATE) {
            func_t next;
            memcpy(&next, &pr_globals[statement->b], sizeof(next));
            return next > 0 && next < progs->numfunctions ? next : 0;
        }
    }
    return 0;
}

/* Does this attack chain reach a shot anywhere? Walked with a visited set:
 * the chains end back in the run cycle, which loops. */
static int chain_reaches_shot(func_t entry)
{
    func_t visited[96];
    int visited_count = 0;
    for (func_t f = entry; f; f = chain_next(f)) {
        int seen = 0;
        for (int i = 0; i < visited_count; i++)
            if (visited[i] == f) {
                seen = 1;
                break;
            }
        if (seen)
            return 0;
        if (visited_count < (int)(sizeof(visited) / sizeof(*visited)))
            visited[visited_count++] = f;
        if (chain_has_shot_call(f))
            return 1;
    }
    return 0;
}

/* Does this attack entry hit from a distance? Its own state chain reaches a
 * shot, or it is a dispatcher - a plain function that picks among chains,
 * like the Scrag's "Wiz_Missile" or the Ogre's "ogre_melee" - and a chain it
 * can start does. */
static int entry_is_ranged(func_t entry, int depth)
{
    if (!entry)
        return 0;
    if (chain_reaches_shot(entry))
        return 1;
    dfunction_t *function = &pr_functions[entry];
    if (function->first_statement <= 0 || depth >= 2)
        return 0;
    for (int s = 0; s < 512; s++) {
        dstatement_t *statement = &pr_statements[function->first_statement + s];
        if (statement->op == OP_DONE || statement->op == OP_RETURN)
            break;
        if (statement->op == OP_STATE)
            return 0; /* a state function: its chain was walked above */
        if (statement->op >= OP_CALL0 && statement->op <= OP_CALL8) {
            func_t called;
            memcpy(&called, &pr_globals[statement->a], sizeof(called));
            if (called > 0 && called < progs->numfunctions &&
                pr_functions[called].first_statement > 0 && entry_is_ranged(called, depth + 1))
                return 1;
        }
    }
    return 0;
}

/* The attack chain's own frames that fire the shot: walk the state chain
 * from each attack entry - the spawn's th_melee and th_missile, else the
 * shape probe - and record every frame that calls a shot function. The
 * Enforcer's chain reaches "enforcer_fire" twice (atk6 and atk10) and the
 * Ogre's throw chain calls "OgreFireGrenade" from its "ogre_nail" frames,
 * which no "<prefix>atk<n>" name scan could find (#53). */
static int find_fire_thinks(const dq_hook_slot *slot, func_t *out, int max)
{
    const func_t entries[3] = {slot->melee_think, slot->missile_think, slot->attack_think};
    func_t visited[96];
    int visited_count = 0;
    int count = 0;

    for (int e = 0; e < 3; e++) {
        if (!entries[e])
            continue;
        for (func_t f = entries[e]; f; f = chain_next(f)) {
            int seen = 0;
            for (int i = 0; i < visited_count; i++)
                if (visited[i] == f) {
                    seen = 1;
                    break;
                }
            if (seen)
                break;
            if (visited_count < (int)(sizeof(visited) / sizeof(*visited)))
                visited[visited_count++] = f;
            if (count < max && chain_has_shot_call(f))
                out[count++] = f;
        }
    }
    return count;
}

/* Is this think one of the frames that fires the shot? */
static int is_fire_think(const dq_hook_slot *slot, func_t fnum)
{
    for (int i = 0; i < slot->fire_count; i++)
        if (slot->fire_thinks[i] == fnum)
            return 1;
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

/* The distance a state moves the monster: the constant it passes to `mover`
 * ("ai_run", "ai_walk"), 0 when it does not call it. QuakeC compiles the call
 * as a store of the constant into the first parameter (a vector store, whatever
 * the type), then the call. */
static float state_move_distance(func_t fnum, const char *mover)
{
    dfunction_t *function = &pr_functions[fnum];
    float parm = 0.0f;
    for (int s = 0; s < 512; s++) {
        dstatement_t *statement = &pr_statements[function->first_statement + s];
        if (statement->op == OP_DONE || statement->op == OP_RETURN)
            break;
        if (statement->op >= OP_STORE_F && statement->op <= OP_STORE_FNC &&
            statement->b == OFS_PARM0)
            parm = pr_globals[statement->a];
        if (statement->op == OP_CALL1) {
            func_t called;
            memcpy(&called, &pr_globals[statement->a], sizeof(called));
            if (called > 0 && called < progs->numfunctions &&
                strcmp(PR_GetString(pr_functions[called].s_name), mover) == 0)
                return parm;
        }
    }
    return 0.0f;
}

/* Collect the frames of a stand/run chain in play order (#61): follow each
 * state's next think, recording its frame, until the chain loops back on a
 * state already seen, leaves the animation states, or fills the table. The
 * result is exactly the cycle the stock QuakeC animates - including a run
 * whose last state reuses an earlier frame - so every entry is a frame the
 * class's model has. Returns the number of frames collected. With `mover`,
 * `speed` receives the chain's average speed in units per second. */
static int walk_chain_frames(func_t start, float *out, int max, const char *mover, float *speed)
{
    func_t visited[DQ_MAX_ANIM_FRAMES];
    int n = 0;
    float travelled = 0.0f;
    for (func_t f = start; f && n < max; f = chain_next(f)) {
        int seen = 0;
        for (int i = 0; i < n; i++)
            if (visited[i] == f) {
                seen = 1;
                break;
            }
        if (seen)
            break;
        float frame = state_frame(f);
        if (frame < 0.0f)
            break;
        visited[n] = f;
        out[n++] = frame;
        if (mover)
            travelled += state_move_distance(f, mover);
    }
    if (speed)
        *speed = n > 0 ? travelled / ((float)n * DQ_STATE_INTERVAL) : 0.0f;
    return n;
}

/* Is this think one of the monster's own decision states? The class's
 * prefix is not always its classname (#49): the Enforcer's states are
 * "enf_stand1" and friends, not "monster_enforcer_stand1". The first
 * decision think of a class teaches the hook its prefix; the ones after it
 * must match, so a think from anywhere else is left to QuakeC. */
static int is_decision_think(func_t fnum, dq_hook_slot *slot)
{
    char prefix[32];

    if (!decision_think_prefix(PR_GetString(pr_functions[fnum].s_name), prefix, sizeof(prefix)))
        return 0;
    if (strcmp(prefix, slot->prefix) != 0) {
        /* The class names its states differently: resolve them again. */
        if (!resolve_prefix(slot, prefix))
            return 0;
    }
    return 1;
}

/* Turn `from` toward `to` by at most `max_turn` degrees, the short way. */
static float turn_toward(float from, float to, float max_turn)
{
    float diff = to - from;
    while (diff > 180.0f)
        diff -= 360.0f;
    while (diff < -180.0f)
        diff += 360.0f;
    if (diff > max_turn)
        diff = max_turn;
    else if (diff < -max_turn)
        diff = -max_turn;
    return from + diff;
}

/* The policy's facing (see DQ_SHOWN_TURN_RATE): its own, unless QuakeC has
 * turned the monster since the hook last wrote the shown facing. */
static float policy_yaw(edict_t *ent)
{
    if (ent->v.angles[YAW] != ent->v.v_angle[PITCH])
        ent->v.v_angle[YAW] = ent->v.angles[YAW];
    return ent->v.v_angle[YAW];
}

static void hook_observe(const dq_hook_slot *slot, edict_t *ent, dqsim_hook_observation *obs)
{
    static const float pitches[DQSIM_HOOK_RAY_PITCHES] = {0.0f, 30.0f};
    memset(obs, 0, sizeof(*obs));
    obs->time = sv.time;
    obs->entity_index = slot->entity_index;
    VectorCopy(ent->v.origin, obs->origin);
    VectorCopy(ent->v.velocity, obs->velocity);
    VectorCopy(ent->v.angles, obs->angles);
    obs->angles[YAW] = policy_yaw(ent);
    obs->health = ent->v.health;
    obs->on_ground = ((int)ent->v.flags & FL_ONGROUND) != 0;
    obs->waterlevel = (int)ent->v.waterlevel;
    eval_t *cooldown = GetEdictFieldValue(ent, "attack_finished");
    obs->attack_cooldown = cooldown && cooldown->_float > (float)sv.time
                               ? cooldown->_float - (float)sv.time
                               : 0.0f;
    /* Hearing (RETRAIN_SPEC part 0B): the remembered noise and the pace the
     * hook moves the monster at now — with an enemy it runs, whatever the
     * alert rolled. hear_tick already forgot an expired noise. */
    obs->start_health = slot->start_health;
    if (slot->state.heard_valid) {
        obs->heard = 1;
        obs->heard_kind = slot->state.heard_kind;
        VectorSubtract(slot->state.heard_origin, ent->v.origin, obs->heard_relative);
        obs->heard_age = (float)(sv.time - slot->state.heard_time);
    }
    obs->pace = ent->v.enemy ? DQSIM_PACE_RUN : slot->state.pace;
    if (obs->pace == DQSIM_PACE_RUN && ent->v.enemy) {
        edict_t *e = PROG_TO_EDICT(ent->v.enemy);
        int eidx = (int)(e - sv.edicts);
        fprintf(stderr, "DBGPACE ent=%d enemy=%d (classname %s) state_pace=%d t=%.1f\n",
                slot->entity_index, eidx, PR_GetString(e->v.classname), slot->state.pace,
                sv.time);
    }

    vec3_t eye;
    VectorAdd(ent->v.origin, ent->v.view_ofs, eye);
    for (int p = 0; p < DQSIM_HOOK_RAY_PITCHES; p++) {
        for (int y = 0; y < DQSIM_HOOK_RAY_YAWS; y++) {
            vec3_t angles, forward, right, up, end;
            angles[PITCH] = pitches[p];
            angles[YAW] = obs->angles[YAW] + 360.0f * (float)y / DQSIM_HOOK_RAY_YAWS;
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

/* The attack entry the monster may start on this enemy now, 0 when its
 * attack from here would be a melee attack out of reach. */
static func_t attack_for(const edict_t *ent, const dq_hook_slot *slot, const edict_t *enemy)
{
    float dx = enemy->v.origin[0] - ent->v.origin[0];
    float dy = enemy->v.origin[1] - ent->v.origin[1];
    float distance = sqrtf(dx * dx + dy * dy);
    int in_reach = distance <= ent->v.maxs[0] + DQ_MELEE_REACH_MARGIN &&
                   fabsf(enemy->v.origin[2] - ent->v.origin[2]) <= DQ_MELEE_REACH_HEIGHT;
    if (in_reach && slot->melee_think)
        return slot->melee_think;
    if (slot->missile_think)
        return slot->missile_ranged || in_reach ? slot->missile_think : 0;
    if (slot->melee_think)
        return slot->melee_ranged || in_reach ? slot->melee_think : 0;
    return slot->attack_think;
}

static void hook_apply(edict_t *ent, func_t fnum, dq_hook_slot *slot,
                       const dqsim_hook_observation *obs,
                       const dqsim_hook_action *action)
{
    /* The enemy an attack would be at: the nearest client in sight. */
    edict_t *enemy = NULL;
    if (action->attack)
        for (int i = 0; i < obs->visible_count; i++)
            if (obs->visible[i].flags & FL_CLIENT) {
                enemy = EDICT_NUM(obs->visible[i].index);
                break;
            }
    /* Seeing a live player gives the monster its enemy, as the stock sight
     * check does; being hurt already did (T_Damage). With an enemy it runs,
     * without one it walks (see DQ_STATE_INTERVAL). */
    if (!ent->v.enemy)
        for (int i = 0; i < obs->visible_count; i++)
            if ((obs->visible[i].flags & FL_CLIENT) && obs->visible[i].health > 0) {
                ent->v.enemy = EDICT_TO_PROG(EDICT_NUM(obs->visible[i].index));
                if (getenv("DQ_HEAR_DEBUG"))
                    fprintf(stderr, "DBGENEMY ent=%d enemy=%d t=%.1f\n", slot->entity_index,
                            obs->visible[i].index, sv.time);
                break;
            }
    /* The pace the monster moves at now (RETRAIN_SPEC part 0B): with an
     * enemy it runs, as today; an alert (a heard noise) chooses between
     * running and investigating, at half the run speed, until the noise is
     * forgotten. A class with no walk chain runs at every pace. */
    hear_tick(slot, ent);
    int pace = ent->v.enemy ? DQSIM_PACE_RUN : slot->state.pace;
    int running;
    float body_speed;
    if (ent->v.enemy || slot->walk_len == 0 || slot->walk_speed <= 0.0f) {
        running = 1;
        body_speed = slot->run_speed;
    } else if (pace == DQSIM_PACE_INVESTIGATE) {
        running = 0;
        body_speed = 0.5f * slot->run_speed;
    } else if (pace == DQSIM_PACE_RUN) {
        running = 1;
        body_speed = slot->run_speed;
    } else {
        running = 0;
        body_speed = slot->walk_speed;
    }
    float charge_speed = slot->run_speed > 0.0f ? slot->run_speed : DQ_CHARGE_SPEED;
    eval_t *cooldown = enemy ? GetEdictFieldValue(ent, "attack_finished") : NULL;
    int ready = !cooldown || sv.time >= cooldown->_float;

    /* A melee attack asked for out of reach becomes a charge (see
     * DQ_MELEE_REACH_MARGIN): run at the enemy, or leap the last stretch. */
    float want_yaw = action->desired_yaw, want_forward = action->forward,
          want_side = action->side;
    func_t leap = 0;
    int charging = enemy && !attack_for(ent, slot, enemy);
    if (charging) {
        float dx = enemy->v.origin[0] - ent->v.origin[0];
        float dy = enemy->v.origin[1] - ent->v.origin[1];
        float distance = sqrtf(dx * dx + dy * dy);
        want_yaw = atan2f(dy, dx) * 180.0f / 3.14159265f;
        want_forward = charge_speed;
        want_side = 0;
        if (slot->leap_think && ready && ((int)ent->v.flags & FL_ONGROUND) &&
            distance >= DQ_LEAP_MIN && distance <= DQ_LEAP_MAX &&
            fabsf(enemy->v.origin[2] - ent->v.origin[2]) <= DQ_MELEE_REACH_HEIGHT)
            leap = slot->leap_think;
    }

    /* Turn the policy's facing toward the one it asked for at a bounded rate
     * instead of snapping (#61): the stock ai_turn/ChangeYaw turns gradually.
     * Moves are measured from the turning facing. A leap goes where the
     * monster faces, so it faces the enemy outright and does not step. */
    float new_yaw = leap ? want_yaw
                         : turn_toward(policy_yaw(ent), want_yaw,
                                       DQ_MONSTER_TURN_RATE * (float)host_frametime);
    /* On the ground the charge is the engine's own chase step, the one every
     * stock monster runs at its enemy with: it turns the body, steps, and
     * picks a way round what is in the way, where a straight line stops dead
     * at the first crate. It turns by yaw_speed per call and the stock states
     * call it ten times a second, so for this per-tick call the turn is
     * scaled to the hook's own turn rate. The body's facing is the policy's
     * for as long as the charge lasts. */
    int chased = 0, chase_moved = 0;
    float moved_dist = 0.0f; /* what the body covered this tick, for the pace's frames */
    if (charging && !leap && ((int)ent->v.flags & FL_ONGROUND)) {
        vec3_t before;
        float yaw_speed = ent->v.yaw_speed;
        VectorCopy(ent->v.origin, before);
        ent->v.angles[YAW] = policy_yaw(ent);
        ent->v.enemy = EDICT_TO_PROG(enemy);
        ent->v.goalentity = EDICT_TO_PROG(enemy);
        ent->v.yaw_speed = DQ_MONSTER_TURN_RATE * (float)host_frametime;
        pr_global_struct->self = EDICT_TO_PROG(ent);
        G_FLOAT(OFS_PARM0) = charge_speed * (float)host_frametime;
        SV_MoveToGoal();
        ent->v.yaw_speed = yaw_speed;
        float dx = ent->v.origin[0] - before[0], dy = ent->v.origin[1] - before[1];
        float asked = charge_speed * (float)host_frametime * DQ_MOVED_FRACTION;
        chase_moved = dx * dx + dy * dy >= asked * asked;
        if (chase_moved)
            moved_dist = sqrtf(dx * dx + dy * dy);
        chased = 1;
        new_yaw = ent->v.angles[YAW];
    }
    ent->v.v_angle[YAW] = new_yaw;

    vec3_t angles = {0, new_yaw, 0}, forward, right, up, wish;
    AngleVectors(angles, forward, right, up);
    wish[0] = forward[0] * want_forward + right[0] * want_side;
    wish[1] = forward[1] * want_forward + right[1] * want_side;
    wish[2] = 0;
    if (leap || chased)
        wish[0] = wish[1] = 0;
    float speed = sqrtf(wish[0] * wish[0] + wish[1] * wish[1]);
    if (body_speed > 0.0f && speed > body_speed) {
        wish[0] *= body_speed / speed;
        wish[1] *= body_speed / speed;
        speed = body_speed;
    }
    /* What the body actually did this tick: the direction it travelled, or
     * nothing when the step was refused (a wall, a ledge, another monster). */
    int moved = chase_moved;
    float travel_yaw = new_yaw;
    if (speed > 0.01f) {
        if ((int)ent->v.flags & FL_ONGROUND) {
            vec3_t move, before;
            VectorScale(wish, (float)host_frametime, move);
            VectorCopy(ent->v.origin, before);
            SV_movestep(ent, move, true);
            float dx = ent->v.origin[0] - before[0], dy = ent->v.origin[1] - before[1];
            float asked = speed * (float)host_frametime * DQ_MOVED_FRACTION;
            if (dx * dx + dy * dy >= asked * asked) {
                moved = 1;
                moved_dist = sqrtf(dx * dx + dy * dy);
                travel_yaw = atan2f(dy, dx) * 180.0f / 3.14159265f;
            }
        } else {
            ent->v.velocity[0] = wish[0];
            ent->v.velocity[1] = wish[1];
            moved = 1;
            moved_dist = speed * (float)host_frametime;
            travel_yaw = atan2f(wish[1], wish[0]) * 180.0f / 3.14159265f;
        }
    }
    /* The shown facing follows the travel, or settles on the policy's facing
     * when the monster is not going anywhere. */
    if (!chased)
        ent->v.angles[YAW] = turn_toward(ent->v.angles[YAW], moved ? travel_yaw : new_yaw,
                                         DQ_SHOWN_TURN_RATE * (float)host_frametime);
    ent->v.v_angle[PITCH] = ent->v.angles[YAW];

    int attacked = 0;
    if (enemy) {
        /* The stock AI runs the spawn's own th_melee when the enemy is in
         * melee reach and th_missile otherwise; a class without one entry
         * falls back to the other, and a class with neither uses the shape
         * probe (#53). An entry that is a melee attack is not started out of
         * reach (#63): the monster charged above instead, or leaps now. */
        func_t attack = leap ? leap : attack_for(ent, slot, enemy);
        if (attack && ready) {
            int reaction_ticks;
            float aim_error_deg;
            effective_handicap(slot, &reaction_ticks, &aim_error_deg);
            (void)reaction_ticks;
            dq_handicap_roll_aim(&slot->handicap, aim_error_deg);
            /* The stock attack states turn and aim from the monster's facing:
             * they start from the policy's, as they always have. */
            ent->v.angles[YAW] = new_yaw;
            ent->v.enemy = EDICT_TO_PROG(enemy);
            pr_global_struct->self = EDICT_TO_PROG(ent);
            pr_global_struct->other = EDICT_TO_PROG(enemy);
            pr_global_struct->time = sv.time;
            /* The think the stock AI would have run set the sight globals
             * first; the chains read them (see find_enemy_globals). */
            if (enemy_globals_found) {
                float dx = enemy->v.origin[0] - ent->v.origin[0];
                float dy = enemy->v.origin[1] - ent->v.origin[1];
                float distance = sqrtf(dx * dx + dy * dy);
                if (range_global_ofs)
                    pr_globals[range_global_ofs] =
                        distance < 120.0f ? 1.0f
                                          : distance < 500.0f ? 2.0f
                                                              : distance < 1000.0f ? 3.0f : 4.0f;
                if (vis_global_ofs)
                    pr_globals[vis_global_ofs] = 1.0f;
                if (infront_global_ofs)
                    pr_globals[infront_global_ofs] = 1.0f;
            }
            PR_ExecuteProgram(attack);
            attacked = 1;
        }
    }

    if (!attacked) {
        /* Animate from the state functions' own frames: run while really
         * moving, stand while still or held by a wall (running on the spot is
         * what a refused step used to look like), advancing at the QuakeC
         * think cadence. The investigation pace plays the walk frames
         * advanced by distance instead (RETRAIN_SPEC part 0B): a stock walk
         * covers walk_speed * DQ_STATE_INTERVAL per frame and the
         * investigation pace covers the same, so the feet keep pace with
         * the ground. */
        slot->state.anim_tick++;
        unsigned step = slot->state.anim_tick / 7;
        if (pace == DQSIM_PACE_INVESTIGATE) {
            if (moved) {
                slot->state.investigate_phase +=
                    moved_dist / (slot->walk_speed * DQ_STATE_INTERVAL);
                ent->v.frame =
                    slot->walk_seq[(unsigned)slot->state.investigate_phase %
                                   (unsigned)slot->walk_len];
            } else if (slot->stand_len > 0)
                ent->v.frame = slot->stand_seq[step % (unsigned)slot->stand_len];
        } else if (moved && !running)
            ent->v.frame = slot->walk_seq[step % (unsigned)slot->walk_len];
        else if (moved && slot->run_len > 0)
            ent->v.frame = slot->run_seq[step % (unsigned)slot->run_len];
        else if (slot->stand_len > 0)
            ent->v.frame = slot->stand_seq[step % (unsigned)slot->stand_len];
        ent->v.think = fnum;
        double margin = 0.5 * host_frametime;
        if (margin > DQ_THINK_MARGIN)
            margin = DQ_THINK_MARGIN;
        ent->v.nextthink = (float)(sv.time + host_frametime + margin);
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
             * origin tangentially for this one execution and restore it.
             * It stays pending for the rest of the attack: a class whose
             * chain fires more than once (the Enforcer fires twice) would
             * otherwise keep an exact second shot (#49). */
            if (slot->handicap.aim_pending && is_fire_think(slot, fnum) && ent->v.enemy) {
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
                /* Shots that aim from the monster's own facing rather than
                 * from the enemy's origin - the Ogre's grenade throws with
                 * makevectors(self.angles) and only takes its lead from the
                 * enemy - cannot be bent by displacing the enemy, so the
                 * monster's facing is rolled by the same error too and
                 * restored after the frame (#53). */
                float saved_yaw = ent->v.angles[YAW];
                ent->v.angles[YAW] += slot->handicap.aim_yaw_deg;
                PR_ExecuteProgram(fnum);
                ent->v.angles[YAW] = saved_yaw;
                VectorCopy(saved, enemy->v.origin);
                return;
            }
            if (ent->v.health > 0 && (func_t)ent->v.think == fnum && ent->v.nextthink == 0 &&
                is_decision_think(fnum, slot)) {
                /* The monster is deciding again: the attack it was in is
                 * over, so the aim error of that attack does not carry into
                 * the next one. */
                slot->handicap.aim_pending = 0;
                dqsim_hook_observation obs;
                hear_tick(slot, ent);
                hook_observe(slot, ent, &obs);
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
                action.desired_yaw = obs.angles[YAW];
                slot->callback(&obs, &action, slot->user);
                if (!ent->free) {
                    if (!respond) {
                        /* Inside the reaction delay: the callback ran,
                         * but nothing is applied (no turn, move or attack). */
                        memset(&action, 0, sizeof(action));
                        action.desired_yaw = obs.angles[YAW];
                    }
                    hook_apply(ent, fnum, slot, &obs, &action);
                }
                return;
            }
        }
    }
    PR_ExecuteProgram(fnum);
}

/* The fallback attack entry, by name shape: the entries the hook actually
 * runs come from the spawn's own th_melee/th_missile fields (#53), and this
 * probe covers a class whose spawn sets neither - the Grunt's "army_atk1",
 * which chains to "army_fire", and the Dog's "dog_atta1", a lunge whose
 * chain bites from "dog_bite" (#47). "<class>CheckAttack" is *not* an attack
 * entry: it is the AI's range classifier, it reads the global "enemy_range"
 * that only the stock AI sets, and it starts nothing. The Ogre's "ogre_melee"
 * and the Shalrath's "shal_attack1" are the shapes that forced this list
 * (#49). */
static func_t find_attack_think(const char *prefix)
{
    static const char *const names[] = {"atk1", "atta1", "attack", "attack1", "melee", "melee1"};
    char name[40];

    for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++) {
        q_snprintf(name, sizeof(name), "%s%s", prefix, names[i]);
        func_t found = find_function(name);
        if (found)
            return found;
    }
    return 0;
}

/* The QuakeC prefix of a class's own state functions: the Grunt's is
 * "army_", the Dog's "dog_", the Enforcer's "enf_", the Shambler's "sham_".
 * It is not always the classname, so the hook reads it off the monster's own
 * decision think ("<prefix>_stand<n>", "<prefix>_walk<n>", "<prefix>_run<n>")
 * instead of assuming one. Writes the prefix, with its trailing
 * underscore, and returns 1 when the name is such a think. */
static int decision_think_prefix(const char *name, char *out, size_t size)
{
    /* The Shambler-like single-state stand has no frame digits (#56): the
     * Shalrath's stand think is "shal_stand", while its walk and run states
     * are shal_walk1..12 and shal_run1..12. Digits are therefore optional;
     * the underscore before the kind is not, so a bare "run" still cannot
     * pretend to be a class's state. */
    static const char *const kinds[] = {"stand", "walk", "run"};
    const char *underscore = strrchr(name, '_');

    if (!underscore)
        return 0;
    for (size_t i = 0; i < sizeof(kinds) / sizeof(*kinds); i++) {
        size_t length = strlen(kinds[i]);
        if (strncmp(underscore + 1, kinds[i], length) != 0)
            continue;
        const char *digits = underscore + 1 + length;
        for (const char *digit = digits; *digit; digit++)
            if (*digit < '0' || *digit > '9')
                return 0;
        size_t prefix_length = (size_t)(underscore - name) + 1;
        if (prefix_length >= size)
            return 0;
        memcpy(out, name, prefix_length);
        out[prefix_length] = '\0';
        return 1;
    }
    return 0;
}

/* Resolve everything the hook needs from a class's QuakeC prefix: the stand
 * and run frame sequences it animates with, and the function that starts an
 * attack (the aim error acts on the attack-sequence frames, found separately).
 * The sequences are walked from the stock states themselves (#61), so they
 * are correct for any chain length or shape. The prefix is only kept when the
 * class's frames are found, so a think that is not a class's own state cannot
 * change it. */
static int resolve_prefix(dq_hook_slot *slot, const char *prefix)
{
    char name[40];
    func_t stand_start, run_start;

    q_snprintf(name, sizeof(name), "%sstand1", prefix);
    stand_start = find_function(name);
    if (!stand_start) {
        /* The digitless single-state stand: the Shalrath's "shal_stand" (#56). */
        q_snprintf(name, sizeof(name), "%sstand", prefix);
        stand_start = find_function(name);
    }
    q_snprintf(name, sizeof(name), "%srun1", prefix);
    run_start = find_function(name);

    q_snprintf(name, sizeof(name), "%swalk1", prefix);
    func_t walk_start = find_function(name);

    slot->stand_len =
        walk_chain_frames(stand_start, slot->stand_seq, DQ_MAX_ANIM_FRAMES, NULL, NULL);
    slot->run_len = walk_chain_frames(run_start, slot->run_seq, DQ_MAX_ANIM_FRAMES, "ai_run",
                                      &slot->run_speed);
    slot->walk_len = walk_chain_frames(walk_start, slot->walk_seq, DQ_MAX_ANIM_FRAMES,
                                       "ai_walk", &slot->walk_speed);
    if (slot->stand_len == 0 && slot->run_len == 0)
        return 0; /* not a class with the usual frames: leave the slot alone */

    q_strlcpy(slot->prefix, prefix, sizeof(slot->prefix));
    slot->attack_think = find_attack_think(slot->prefix);
    return 1;
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
    slot->start_health = ent->v.health;
    hook_override_ticks = -1;
    hook_override_aim = -1.0f;
    /* The class's own prefix is not always its classname (the Enforcer's
     * states are "enf_..."), so the classname stem is only the first guess:
     * the monster's own decision think settles it (#49). */
    char class_prefix[32];
    q_snprintf(class_prefix, sizeof(class_prefix), "%s_", classname + 8);
    resolve_prefix(slot, class_prefix);
    /* The spawn's own attack entries: the stock AI runs th_melee in melee
     * range and th_missile otherwise. The Ogre's throw lives only in
     * th_missile, so a name-shaped probe finds only its melee entry (#53).
     * The fields are QuakeC-level, beyond entvars_t, hence the field lookup. */
    eval_t *melee_field = GetEdictFieldValue(ent, "th_melee");
    eval_t *missile_field = GetEdictFieldValue(ent, "th_missile");
    slot->melee_think = melee_field ? melee_field->function : 0;
    slot->missile_think = missile_field ? missile_field->function : 0;
    /* The ranged entry is the spawn's th_missile only when its chain actually
     * fires a projectile: the Dog's th_missile is a one-shot pounce, and the
     * hook runs an entry once per decision, so a pounce cannot close distance
     * the way the lunge the shape probe finds does (#53). */
    func_t probe = slot->attack_think;
    func_t melee = slot->melee_think;
    func_t missile = slot->missile_think;
    slot->melee_think = melee ? melee : probe;
    slot->missile_think = missile && chain_reaches_shot(missile) ? missile : probe;
    if (!melee && !missile)
        slot->melee_think = probe; /* no spawn entries: the probe's guess everywhere */
    slot->melee_ranged = entry_is_ranged(slot->melee_think, 0);
    slot->missile_ranged = entry_is_ranged(slot->missile_think, 0);
    /* A leap is a state chain of the class's own; a dispatcher the ranged
     * rule could not follow (the Scrag's) is not one. */
    slot->leap_think =
        missile && state_frame(missile) >= 0.0f && !entry_is_ranged(missile, 0) ? missile : 0;
    slot->fire_count = find_fire_thinks(slot, slot->fire_thinks, DQ_MAX_FIRE_THINKS);
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

/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Public API of libdqsim, implemented on the vendored QuakeSpasm server.
 *
 * The engine is started the way upstream starts a dedicated server with one
 * client slot. The player is a client connected through net_fake.c: this
 * file queues the same bytes a real Quake client sends, and the unmodified
 * server code does the rest.
 */
#include "quakedef.h"

#include <stddef.h>

#include "brain_hook.h"
#include "dq_internal.h"
#include "dqsim/dqsim.h"

/* Defined in upstream host.c. */
extern jmp_buf host_abortserver;
extern int host_framecount;
/* Defined in upstream cmd.c: the pending console command text. */
extern sizebuf_t cmd_text;

enum { STATE_NEW, STATE_READY, STATE_DEAD, STATE_SHUTDOWN };

static int state = STATE_NEW;
static double tick_seconds;
static quakeparms_t parms;
static char basedir[1024];
static char game_dir[256];
static char *engine_argv[7];
static int level_count;
/* Bumped whenever a new level episode starts, so old snapshots are refused. */
static uint32_t level_serial;

/*
 * Every API function that enters the engine starts with DQ_ENTER. If the
 * engine raises a fatal error it lands back here, the simulator is marked
 * dead, and the function returns DQSIM_ERROR.
 */
#define DQ_ENTER()                                                             \
    do {                                                                       \
        if (state != STATE_READY)                                              \
            return fail_not_ready();                                           \
        if (setjmp(dq_fatal_jmp)) {                                            \
            state = STATE_DEAD;                                                \
            return DQSIM_ERROR;                                                \
        }                                                                      \
        if (setjmp(host_abortserver)) {                                        \
            state = STATE_DEAD;                                                \
            return DQSIM_ERROR;                                                \
        }                                                                      \
    } while (0)

static int fail(const char *message)
{
    q_strlcpy(dq_error, message, sizeof(dq_error));
    return DQSIM_ERROR;
}

static int fail_not_ready(void)
{
    switch (state) {
    case STATE_NEW:
        return fail("dqsim_init has not been called");
    case STATE_SHUTDOWN:
        return fail("the simulator was shut down");
    default:
        /* keep the original fatal message in dq_error */
        return DQSIM_ERROR;
    }
}

const char *dqsim_version(void)
{
    return DQSIM_VERSION_STRING;
}

const char *dqsim_last_error(void)
{
    return dq_error;
}

/* The server portion of upstream _Host_Frame, with a fixed frame time.
 *
 * host_frametime is rounded through a float first because that is what a
 * stock server does: the frame time comes from the host_framerate cvar, and
 * cvar_t stores a float. Feeding the raw double makes products such as
 * sv_gravity * host_frametime land on the other side of a float rounding
 * boundary, drifting one ULP per tick from stock Quake, and the physics
 * amplifies that at contact boundaries (see the fidelity test).
 */
static void run_frame(void)
{
    host_frametime = (double)(float)tick_seconds;
    realtime += tick_seconds;
    Cbuf_Execute();
    if (sv.active)
        Host_ServerFrame();
    host_framecount++;
}

static void queue_command(const char *command)
{
    byte data[128];
    sizebuf_t buf = {.data = data, .maxsize = sizeof(data)};
    MSG_WriteChar(&buf, clc_stringcmd);
    MSG_WriteString(&buf, command);
    dq_net_queue(buf.data, buf.cursize);
}

/*
 * Run the sign-on sequence a client performs after connecting or after the
 * server starts a level: prespawn, spawn, begin, one frame each.
 */
static void sign_on(void)
{
    queue_command("prespawn");
    run_frame();
    queue_command("spawn");
    run_frame();
    queue_command("begin");
    run_frame();
    if (!sv.active || !svs.clients[0].active || !svs.clients[0].spawned)
        Sys_Error("the player failed to spawn on '%s'", sv.name);
}

/* A game directory name is a single component under the basedir (#37).
 * Upstream would raise a fatal error on bad names; reject them here so
 * the caller gets a dqsim_ error and the one-shot init is not spent. */
static int game_name_valid(const char *game)
{
    if (!game || !game[0])
        return 1; /* unset: id1 only */
    size_t length = strlen(game);
    if (length >= sizeof(game_dir) || game[0] == '-')
        return 0;
    if (strcmp(game, ".") == 0 || strstr(game, "..") != NULL)
        return 0;
    for (const char *p = game; *p; p++)
        if (*p == '/' || *p == '\\' || *p == ':' || (unsigned char)*p < 32 || *p == 127)
            return 0;
    return 1;
}


int dqsim_init(const dqsim_config *config)
{
    char pak[sizeof(basedir) + 32];
    FILE *f;

    if (state != STATE_NEW)
        return fail("dqsim_init may only be called once per process");
    if (!config || !config->basedir || !config->basedir[0])
        return fail("dqsim_init: config.basedir is required");
    if (config->heap_mb < 0 || config->heap_mb > 2047 || config->tick_seconds < 0 ||
        config->tick_seconds > 0.1)
        return fail("dqsim_init: heap_mb must be 0-2047 and tick_seconds 0-0.1");
    if (!game_name_valid(config->game))
        return fail("dqsim_init: game must be a single directory name under basedir");

    /* The engine itself starts without game data and fails later, obscurely. */
    q_snprintf(pak, sizeof(pak), "%s/id1/pak0.pak", config->basedir);
    f = fopen(pak, "rb");
    if (!f) {
        q_snprintf(dq_error, sizeof(dq_error), "dqsim_init: cannot open %s", pak);
        return DQSIM_ERROR;
    }
    fclose(f);
    if (!dq_brain_stock_identity_from_base(config->basedir))
        return fail("dqsim_init: cannot find progs.dat under id1");

    dq_verbose = config->verbose;
    tick_seconds = config->tick_seconds > 0 ? config->tick_seconds : DQSIM_DEFAULT_TICK_SECONDS;
    q_strlcpy(basedir, config->basedir, sizeof(basedir));
    game_dir[0] = '\0';
    if (config->game && config->game[0])
        q_strlcpy(game_dir, config->game, sizeof(game_dir));

    engine_argv[0] = "dqsim";
    engine_argv[1] = "-dedicated"; /* no client subsystems */
    engine_argv[2] = "1";          /* one client slot: single-player rules */
    engine_argv[3] = "-basedir";
    engine_argv[4] = basedir;
    int engine_argc = 5;
    if (game_dir[0]) {
        engine_argv[5] = "-game";
        engine_argv[6] = game_dir;
        engine_argc = 7;
    }

    memset(&parms, 0, sizeof(parms));
    parms.basedir = basedir;
    parms.userdir = basedir;
    parms.argc = engine_argc;
    parms.argv = engine_argv;
    parms.memsize = (config->heap_mb ? config->heap_mb : 128) * 1024 * 1024;
    parms.membase = malloc(parms.memsize);
    parms.numcpus = 1;
    if (!parms.membase)
        return fail("dqsim_init: out of memory allocating the engine heap");
    host_parms = &parms;

    if (setjmp(dq_fatal_jmp)) {
        state = STATE_DEAD;
        return DQSIM_ERROR;
    }
    if (setjmp(host_abortserver)) {
        state = STATE_DEAD;
        return DQSIM_ERROR;
    }
    Host_Init();
    /* Upstream's dedicated start-up queues "map start"; discard it. */
    SZ_Clear(&cmd_text);
    state = STATE_READY;
    return DQSIM_OK;
}

int dqsim_load_map(const char *map, int skill, uint32_t seed)
{
    char command[128];

    if (!map || !map[0] || strlen(map) > 60 || strpbrk(map, " \t\n\r;\"'"))
        return fail("dqsim_load_map: invalid map name");
    if (skill < 0 || skill > 3)
        return fail("dqsim_load_map: skill must be 0-3");
    DQ_ENTER();

    dq_rand_seed(seed);
    realtime = 0;
    host_framecount = 0;

    q_snprintf(command, sizeof(command), "skill %d\nmap %s\n", skill, map);
    Cbuf_AddText(command);
    Cbuf_Execute();
    if (!sv.active || q_strcasecmp(sv.name, map) != 0) {
        q_snprintf(dq_error, sizeof(dq_error), "could not load map '%s'", map);
        return DQSIM_ERROR;
    }

    /* Connect the player; the server accepts it during the next frame. */
    dq_net_request_connect();
    run_frame();
    sign_on();
    level_count = 0;
    level_serial++;
    dq_brain_hook_on_level_start();
    return DQSIM_OK;
}

int dqsim_step(const dqsim_action *action)
{
    byte data[64];
    sizebuf_t buf = {.data = data, .maxsize = sizeof(data)};

    if (!action)
        return fail("dqsim_step: action is NULL");
    DQ_ENTER();
    if (!sv.active || !svs.clients[0].spawned)
        return fail("dqsim_step: no map is loaded");

    /* The clc_move message, as read by SV_ReadClientMove. */
    const float angles[3] = {action->pitch, action->yaw, 0};
    MSG_WriteChar(&buf, clc_move);
    MSG_WriteFloat(&buf, sv.time);
    for (int i = 0; i < 3; i++) {
        if (sv.protocol == PROTOCOL_NETQUAKE)
            MSG_WriteAngle(&buf, angles[i], sv.protocolflags);
        else
            MSG_WriteAngle16(&buf, angles[i], sv.protocolflags);
    }
    MSG_WriteShort(&buf, (int)action->forward);
    MSG_WriteShort(&buf, (int)action->side);
    MSG_WriteShort(&buf, (int)action->up);
    MSG_WriteByte(&buf, action->buttons & 3);
    MSG_WriteByte(&buf, action->impulse & 255);
    dq_net_queue(buf.data, buf.cursize);

    run_frame();

    /*
     * The game restarts the level when a dead player presses a button, and
     * loads the next one when the player reaches an exit. Either way the
     * server waits for the client to sign on again.
     */
    if (sv.active && svs.clients[0].active && !svs.clients[0].spawned) {
        sign_on();
        level_count++;
        level_serial++;
        dq_brain_hook_on_level_start();
    }
    return DQSIM_OK;
}

int dqsim_step_repeat(const dqsim_action *action, int count)
{
    for (int i = 0; i < count; i++) {
        if (dqsim_step(action) != DQSIM_OK)
            return DQSIM_ERROR;
    }
    return DQSIM_OK;
}

int dqsim_client_command(const char *command)
{
    if (!command || !command[0] || strlen(command) > 100 || strpbrk(command, "\n\r"))
        return fail("dqsim_client_command: invalid command");
    DQ_ENTER();
    if (!sv.active || !svs.clients[0].spawned)
        return fail("dqsim_client_command: no map is loaded");
    queue_command(command);
    return DQSIM_OK;
}

int dqsim_teleport_player(const float origin[3], float yaw)
{
    vec3_t target;

    if (!origin)
        return fail("dqsim_teleport_player: origin is NULL");
    DQ_ENTER();
    if (!sv.active || !svs.clients[0].spawned)
        return fail("dqsim_teleport_player: no map is loaded");

    edict_t *ent = svs.clients[0].edict;
    VectorCopy(origin, target);
    trace_t trace = SV_Move(target, ent->v.mins, ent->v.maxs, target, MOVE_NORMAL, ent);
    if (trace.startsolid || trace.allsolid)
        return fail("dqsim_teleport_player: the position is blocked");

    VectorCopy(target, ent->v.origin);
    VectorCopy(target, ent->v.oldorigin);
    VectorCopy(vec3_origin, ent->v.velocity);
    VectorCopy(vec3_origin, ent->v.angles);
    ent->v.angles[YAW] = yaw;
    VectorCopy(ent->v.angles, ent->v.v_angle);
    ent->v.fixangle = 1;
    ent->v.flags = (int)ent->v.flags & ~FL_ONGROUND;
    SV_LinkEdict(ent, false);
    return DQSIM_OK;
}

/*
 * What one ray ran into (#26). A solid trace alone cannot tell lava from
 * floor, and the engine's hull trace passes through sky exactly as it
 * passes through water (world.c: SV_RecursiveHullCheck treats CONTENTS_SKY
 * as water), so the ray also steps along its own path and reports a liquid
 * or sky volume at the first step inside it. That ordering is what an eye
 * would see: the first non-empty thing along the line. The one thing the
 * march never sees is sky directly behind a wall within one step of it,
 * where the wall has already won.
 * Sets *out_frac to the fraction of max_distance where the reported thing
 * was met.
 */
int dqsim_ray_fan(const float *pitches, int n_pitch, int n_yaw, float max_distance, float *out,
                  unsigned char *out_kind)
{
    if (!pitches || !out || n_pitch < 1 || n_yaw < 1 || max_distance <= 0)
        return fail("dqsim_ray_fan: invalid arguments");
    DQ_ENTER();
    if (!sv.active || !svs.clients[0].spawned)
        return fail("dqsim_ray_fan: no map is loaded");

    edict_t *ent = svs.clients[0].edict;
    vec3_t eye, end, angles, forward, right, up;
    VectorAdd(ent->v.origin, ent->v.view_ofs, eye);

    for (int p = 0; p < n_pitch; p++) {
        for (int y = 0; y < n_yaw; y++) {
            angles[PITCH] = pitches[p];
            angles[YAW] = ent->v.v_angle[YAW] + 360.0f * (float)y / (float)n_yaw;
            angles[ROLL] = 0;
            AngleVectors(angles, forward, right, up);
            VectorMA(eye, max_distance, forward, end);
            trace_t trace = SV_Move(eye, vec3_origin, vec3_origin, end, MOVE_NOMONSTERS, ent);
            float fraction;
            unsigned char kind = dq_classify_ray(eye, forward, max_distance, &trace, &fraction);
            out[p * n_yaw + y] = fraction;
            if (out_kind)
                out_kind[p * n_yaw + y] = kind;
        }
    }
    return DQSIM_OK;
}

/*
 * The brain hook itself (#31) lives in brain_hook.c, shared with the play
 * client (M5). The public API below wraps it with this file's session and
 * error handling.
 */

int dqsim_handicap_for_skill(int skill, dqsim_handicap *out)
{
    if (!out || skill < 0 || skill > 3)
        return fail("dqsim_handicap_for_skill: skill must be 0-3 and out non-NULL");
    if (dq_brain_handicap_for_skill(skill, out))
        return DQSIM_ERROR;
    return DQSIM_OK;
}

int dqsim_set_handicap_override(int32_t reaction_ticks, float aim_error_deg)
{
    DQ_ENTER();
    if (dq_brain_hook_active() <= 0)
        return fail("dqsim_set_handicap_override: no monster is nominated");
    dq_brain_set_handicap_override(reaction_ticks, aim_error_deg);
    return DQSIM_OK;
}

int dqsim_progs_is_stock(void)
{
    DQ_ENTER();
    if (!sv.active)
        return fail("dqsim_progs_is_stock: no map is loaded");
    return dq_brain_progs_is_stock();
}

int dqsim_set_policy_hook(int entity_index, dqsim_hook_callback callback, void *user)
{
    if (!callback)
        return fail("dqsim_set_policy_hook: callback is NULL (use dqsim_clear_policy_hook)");
    DQ_ENTER();
    const char *error = dq_brain_hook_nominate(entity_index, callback, user);
    if (error)
        return fail(error);
    return DQSIM_OK;
}

int dqsim_clear_policy_hook(void)
{
    DQ_ENTER();
    dq_brain_hook_clear();
    return DQSIM_OK;
}

/* Separate from dqsim_get_entities so no local lives across its setjmp. */
static int copy_entities(dqsim_entity *out, int max)
{
    int count = 0;

    for (int i = 0; i < sv.num_edicts; i++) {
        const edict_t *ent = EDICT_NUM(i);
        if (ent->free)
            continue;
        if (count < max) {
            dqsim_entity *e = &out[count];
            memset(e, 0, sizeof(*e));
            e->index = i;
            q_strlcpy(e->classname, PR_GetString(ent->v.classname), sizeof(e->classname));
            for (int k = 0; k < 3; k++) {
                e->origin[k] = ent->v.origin[k];
                e->absmin[k] = ent->v.absmin[k];
                e->absmax[k] = ent->v.absmax[k];
            }
            e->health = ent->v.health;
            e->solid = (int)ent->v.solid;
            e->flags = (int)ent->v.flags;
        }
        count++;
    }
    return count;
}

int dqsim_get_entities(dqsim_entity *out, int max)
{
    if ((!out && max > 0) || max < 0)
        return fail("dqsim_get_entities: invalid arguments");
    DQ_ENTER();
    if (!sv.active)
        return fail("dqsim_get_entities: no map is loaded");
    return copy_entities(out, max);
}

const char *dqsim_map_name(void)
{
    if (state != STATE_READY || !sv.active)
        return "";
    return sv.name;
}

int dqsim_level_count(void)
{
    return level_count;
}

int dqsim_get_player(dqsim_player_state *out)
{
    if (!out)
        return fail("dqsim_get_player: out is NULL");
    DQ_ENTER();
    if (!sv.active || !svs.clients[0].spawned)
        return fail("dqsim_get_player: no map is loaded");

    const entvars_t *v = &svs.clients[0].edict->v;
    for (int i = 0; i < 3; i++) {
        out->origin[i] = v->origin[i];
        out->velocity[i] = v->velocity[i];
        out->view_angles[i] = v->v_angle[i];
    }
    out->health = v->health;
    out->on_ground = ((int)v->flags & FL_ONGROUND) != 0;
    out->waterlevel = (int)v->waterlevel;
    return DQSIM_OK;
}

double dqsim_time(void)
{
    if (state != STATE_READY || !sv.active)
        return -1.0;
    return sv.time;
}

uint64_t dqsim_state_hash(void)
{
    uint64_t hash = 0xcbf29ce484222325ULL;
    if (state != STATE_READY || !sv.active)
        return 0;

    const size_t fields_size = (size_t)progs->entityfields * 4;
    hash = dq_fnv1a(hash, &sv.time, sizeof(sv.time));
    hash = dq_fnv1a(hash, &sv.num_edicts, sizeof(sv.num_edicts));
    for (int i = 0; i < sv.num_edicts; i++) {
        const edict_t *ent = EDICT_NUM(i);
        hash = dq_fnv1a(hash, &ent->free, sizeof(ent->free));
        if (!ent->free)
            hash = dq_fnv1a(hash, &ent->v, fields_size);
    }
    /* Every policy-driven monster's handicap, in slot order (M6: a level
     * can have dozens). This is the hook-owned state, so the deterministic
     * hash covers what the engine itself decides about handicaps. */
    dq_handicap_state handicaps[DQ_BRAIN_MAX_MONSTERS];
    int hook_count = (int)dq_brain_hook_handicaps(handicaps, DQ_BRAIN_MAX_MONSTERS);
    hash = dq_fnv1a(hash, &hook_count, sizeof(hook_count));
    for (int i = 0; i < hook_count; i++)
        hash = dq_fnv1a(hash, &handicaps[i], sizeof(handicaps[i]));
    return hash;
}

/*
 * Snapshots
 * =========
 *
 * A snapshot is the mutable part of the server state, copied into one block
 * of memory. It is in-memory only and valid only in this process and this
 * level episode: a new map, a level change or a death restart runs
 * SV_SpawnServer(), which frees and reallocates sv.edicts, so level_serial
 * distinguishes the episode and stale snapshots are refused.
 *
 * The map hunk is deliberately not captured: models, progs.dat and the
 * QuakeC string table never change during play. Models load only at map load
 * (PF_precache_model refuses outside ss_loading; PF_setmodel only indexes
 * precached models) and strings are appended, never rewritten. Restore
 * therefore never rewinds the hunk, which keeps mod_known and
 * pr_knownstrings consistent.
 *
 * The world's spatial index (sv_areanodes, static in world.c) is not
 * captured either. It is rebuilt from the restored edicts' area links, which
 * encode each list's exact order.
 */
#define DQ_SNAPSHOT_MAGIC 0x4451534eU /* "DQSN" */

typedef struct dq_snapshot_header {
    uint32_t magic;
    uint32_t level_serial;
    char map[64];
    uint64_t rng_state;
    int hook_count; /* policy-driven monsters in slot order (M6) */
    dq_handicap_state handicaps[DQ_BRAIN_MAX_MONSTERS]; /* #34 handicap state */
    double realtime;
    int host_framecount;
    int level_count;
    int num_edicts;
    int pr_edict_size;
    int maxclientslimit;
    int numglobals;
    size_t clients_offset;
    size_t edicts_offset;
    size_t globals_offset;
    size_t net_offset;
    size_t cmd_struct_offset;
    size_t cmd_offset;
    size_t cmd_size;
    size_t total_size;
} dq_snapshot_header;

struct dqsim_snapshot {
    dq_snapshot_header header;
    unsigned char payload[];
};

/* Is this link one of the edicts' own area links rather than a list head? */
static qboolean is_edict_link(const link_t *link)
{
    const unsigned char *base = (const unsigned char *)sv.edicts;
    const unsigned char *p = (const unsigned char *)link;
    const size_t area_offset = offsetof(edict_t, area);

    if (p < base + area_offset)
        return false;
    size_t diff = (size_t)(p - (base + area_offset));
    if (diff >= (size_t)sv.num_edicts * (size_t)pr_edict_size)
        return false;
    return diff % (size_t)pr_edict_size == 0;
}

/*
 * Recreate sv_areanodes in the exact order the snapshot had. SV_ClearWorld()
 * builds the nodes with empty, self-linked lists; the restored edicts still
 * carry the saved links, in which the first edict of each list points back at
 * its head and the last points forward at it. Setting those two head links is
 * enough to rebuild every list, order included.
 */
static void rebuild_world_links(void)
{
    SV_ClearWorld();

    for (int i = 0; i < sv.num_edicts; i++) {
        edict_t *ent = EDICT_NUM(i);
        if (!ent->area.prev)
            continue; /* not linked */
        if (!is_edict_link(ent->area.prev))
            ent->area.prev->next = &ent->area; /* first in the list */
        if (ent->area.next && !is_edict_link(ent->area.next))
            ent->area.next->prev = &ent->area; /* last in the list */
    }
}

int dqsim_snapshot_save(dqsim_snapshot **out)
{
    if (!out)
        return fail("dqsim_snapshot_save: out is NULL");
    DQ_ENTER();
    if (!sv.active || !svs.clients[0].spawned)
        return fail("dqsim_snapshot_save: no map is loaded");

    const size_t client_bytes = (size_t)svs.maxclientslimit * sizeof(client_t);
    const size_t edict_bytes = (size_t)sv.num_edicts * (size_t)pr_edict_size;
    const size_t globals_bytes = (size_t)progs->numglobals * sizeof(float);
    const size_t cmd_bytes = cmd_text.data ? (size_t)cmd_text.cursize : 0;

    const size_t clients_offset = sizeof(server_t) + sizeof(server_static_t);
    const size_t edicts_offset = clients_offset + client_bytes;
    const size_t globals_offset = edicts_offset + edict_bytes;
    const size_t net_offset = globals_offset + globals_bytes;
    const size_t cmd_struct_offset = net_offset + sizeof(dq_net_state);
    const size_t cmd_offset = cmd_struct_offset + sizeof(sizebuf_t);
    const size_t payload_size = cmd_offset + cmd_bytes;
    const size_t total_size = sizeof(dq_snapshot_header) + payload_size;

    dqsim_snapshot *snap = malloc(sizeof(*snap) + payload_size);
    if (!snap)
        return fail("dqsim_snapshot_save: out of memory");

    dq_snapshot_header *h = &snap->header;
    memset(h, 0, sizeof(*h));
    h->magic = DQ_SNAPSHOT_MAGIC;
    h->level_serial = level_serial;
    q_strlcpy(h->map, sv.name, sizeof(h->map));
    h->rng_state = dq_rand_state();
    h->hook_count = (int)dq_brain_hook_handicaps(h->handicaps, DQ_BRAIN_MAX_MONSTERS);
    h->realtime = realtime;
    h->host_framecount = host_framecount;
    h->level_count = level_count;
    h->num_edicts = sv.num_edicts;
    h->pr_edict_size = pr_edict_size;
    h->maxclientslimit = svs.maxclientslimit;
    h->numglobals = (int)progs->numglobals;
    h->clients_offset = clients_offset;
    h->edicts_offset = edicts_offset;
    h->globals_offset = globals_offset;
    h->net_offset = net_offset;
    h->cmd_struct_offset = cmd_struct_offset;
    h->cmd_offset = cmd_offset;
    h->cmd_size = cmd_bytes;
    h->total_size = total_size;

    unsigned char *p = snap->payload;
    memcpy(p, &sv, sizeof(sv));
    memcpy(p + sizeof(server_t), &svs, sizeof(svs));
    memcpy(p + clients_offset, svs.clients, client_bytes);
    memcpy(p + edicts_offset, sv.edicts, edict_bytes);
    memcpy(p + globals_offset, pr_global_struct, globals_bytes);
    dq_net_get_state((dq_net_state *)(p + net_offset));
    memcpy(p + cmd_struct_offset, &cmd_text, sizeof(cmd_text));
    if (cmd_bytes)
        memcpy(p + cmd_offset, cmd_text.data, cmd_bytes);

    *out = snap;
    return DQSIM_OK;
}

/* Every reason a snapshot may not apply here, before anything is changed. */
static int check_snapshot(const dqsim_snapshot *snap)
{
    if (snap->header.magic != DQ_SNAPSHOT_MAGIC)
        return fail("dqsim_snapshot_restore: not a snapshot");
    if (snap->header.level_serial != level_serial ||
        q_strcasecmp(snap->header.map, sv.name) != 0)
        return fail("dqsim_snapshot_restore: the snapshot is from a different level episode");
    if (snap->header.pr_edict_size != pr_edict_size ||
        snap->header.maxclientslimit != svs.maxclientslimit ||
        snap->header.numglobals != (int)progs->numglobals)
        return fail("dqsim_snapshot_restore: the snapshot is from a different engine configuration");
    if (snap->header.num_edicts < 1 || snap->header.num_edicts > sv.max_edicts)
        return fail("dqsim_snapshot_restore: the snapshot has an invalid edict count");
    /* The nominations themselves are process state, not snapshot state; the
     * handicap states are stored in slot order, so only a snapshot taken
     * with the same nominations can be applied. */
    if (snap->header.hook_count != dq_brain_hook_count())
        return fail("dqsim_snapshot_restore: the snapshot was taken with a different set of "
                    "policy-driven monsters");

    const unsigned char *p = snap->payload;
    const server_t *saved_sv = (const server_t *)p;
    const server_static_t *saved_svs = (const server_static_t *)(p + sizeof(server_t));
    if (saved_sv->edicts != sv.edicts || saved_svs->clients != svs.clients)
        return fail("dqsim_snapshot_restore: the snapshot is from a different level episode");
    return DQSIM_OK;
}

int dqsim_snapshot_restore(const dqsim_snapshot *snapshot)
{
    if (!snapshot)
        return fail("dqsim_snapshot_restore: snapshot is NULL");
    DQ_ENTER();
    if (!sv.active || !svs.clients[0].spawned)
        return fail("dqsim_snapshot_restore: no map is loaded");
    if (check_snapshot(snapshot) != DQSIM_OK)
        return DQSIM_ERROR;

    const dq_snapshot_header *h = &snapshot->header;
    const unsigned char *p = snapshot->payload;

    memcpy(&sv, p, sizeof(sv));
    memcpy(&svs, p + sizeof(server_t), sizeof(svs));
    memcpy(svs.clients, p + h->clients_offset,
           (size_t)h->maxclientslimit * sizeof(client_t));
    memcpy(sv.edicts, p + h->edicts_offset, (size_t)h->num_edicts * (size_t)h->pr_edict_size);
    memcpy(pr_global_struct, p + h->globals_offset, (size_t)h->numglobals * sizeof(float));
    dq_net_set_state((const dq_net_state *)(p + h->net_offset));
    dq_rand_set_state(h->rng_state);
    dq_brain_set_handicap_states(h->handicaps, (size_t)h->hook_count);
    memcpy(&cmd_text, p + h->cmd_struct_offset, sizeof(cmd_text));
    if (h->cmd_size)
        memcpy(cmd_text.data, p + h->cmd_offset, h->cmd_size);

    host_framecount = h->host_framecount;
    realtime = h->realtime;
    level_count = h->level_count;

    rebuild_world_links();
    return DQSIM_OK;
}

size_t dqsim_snapshot_size(const dqsim_snapshot *snapshot)
{
    return snapshot ? snapshot->header.total_size : 0;
}

void dqsim_snapshot_free(dqsim_snapshot *snapshot)
{
    free(snapshot);
}

void dqsim_shutdown(void)
{
    if (state == STATE_NEW || state == STATE_SHUTDOWN)
        return;
    if (state == STATE_READY) {
        if (setjmp(dq_fatal_jmp) == 0)
            Host_Shutdown();
    }
    free(parms.membase);
    parms.membase = NULL;
    state = STATE_SHUTDOWN;
}

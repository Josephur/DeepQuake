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
/* Identity of the stock id1 progs.dat, read from disk at dqsim_init (#37):
 * what the loaded progs of every level is compared against. */
static uint64_t stock_progs_hash;
static size_t stock_progs_length;
static int level_count;
/* Bumped whenever a new level episode starts, so old snapshots are refused. */
static uint32_t level_serial;

/* Brain hook state and level-start housekeeping; see the hook code below
 * dqsim_ray_fan. */
static void hook_on_level_start(void);

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

static uint64_t fnv1a(uint64_t hash, const void *data, size_t length)
{
    const unsigned char *p = data;
    for (size_t i = 0; i < length; i++) {
        hash ^= p[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

#define FNV1A_SEED 14695981039346656037ULL

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

/* The stock identity: progs.dat as the engine finds it under id1 — the
 * pak files in descending order (the engine searches later paks first),
 * then a loose id1/progs.dat. */
static int compute_stock_identity(const char *base)
{
    char path[sizeof(basedir) + 64];
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
    stock_progs_hash = fnv1a(FNV1A_SEED, bytes, length);
    stock_progs_length = length;
    free(bytes);
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
    if (!compute_stock_identity(config->basedir))
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
    hook_on_level_start();
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
        hook_on_level_start();
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
static unsigned char classify_ray(vec3_t eye, vec3_t forward, float max_distance,
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
            unsigned char kind = classify_ray(eye, forward, max_distance, &trace, &fraction);
            out[p * n_yaw + y] = fraction;
            if (out_kind)
                out_kind[p * n_yaw + y] = kind;
        }
    }
    return DQSIM_OK;
}

/*
 * Brain hook (#31). sv_phys.c is compiled with PR_ExecuteProgram renamed
 * to dqsim_hook_execute_program (see engine/CMakeLists.txt), so every
 * QuakeC execution the physics initiates arrives here first. Only one
 * shape is intercepted: the nominated monster's scheduled think, when the
 * think function is one of its stand/walk/run state functions — the
 * points where QuakeC makes a movement or attack decision. Attack, pain
 * and death state functions are not decisions of that kind and execute
 * under QuakeC as usual, so their animation and damage stay stock.
 */
static dqsim_hook_callback hook_callback;
static void *hook_user;
static int hook_index; /* entity number, 0 = no nomination */
static char hook_prefix[32]; /* QuakeC state prefix, e.g. "army_" */
static float hook_stand_frame, hook_run_frame;
static int hook_run_frames;
static unsigned hook_anim_tick;
static int progs_is_stock;

static void hook_on_level_start(void)
{
    hook_callback = NULL;
    hook_user = NULL;
    hook_index = 0;
    /* Identity of the progs this level actually loaded: the raw file
     * bytes through the engine's search path (a game directory shadows
     * id1), hashed with its length. Reading the in-memory progs instead
     * would hash a byte-swapped copy under a stale com_filesize. */
    byte *raw = COM_LoadMallocFile("progs.dat", NULL);
    progs_is_stock = 0;
    if (raw) {
        size_t length = (size_t)com_filesize;
        progs_is_stock = length == stock_progs_length &&
                         fnv1a(FNV1A_SEED, raw, length) == stock_progs_hash;
        free(raw);
    }
}

int dqsim_progs_is_stock(void)
{
    DQ_ENTER();
    if (!sv.active)
        return fail("dqsim_progs_is_stock: no map is loaded");
    return progs_is_stock ? 1 : 0;
}

static func_t find_function(const char *name)
{
    for (int i = 1; i < progs->numfunctions; i++)
        if (strcmp(PR_GetString(pr_functions[i].s_name), name) == 0)
            return i;
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

static int is_decision_think(func_t fnum)
{
    const char *name = PR_GetString(pr_functions[fnum].s_name);
    size_t prefix = strlen(hook_prefix);
    if (strncmp(name, hook_prefix, prefix) != 0)
        return 0;
    const char *rest = name + prefix;
    if (strncmp(rest, "stand", 5) != 0 && strncmp(rest, "walk", 4) != 0 &&
        strncmp(rest, "run", 3) != 0)
        return 0;
    size_t length = strlen(name);
    return length > 0 && name[length - 1] >= '0' && name[length - 1] <= '9';
}

static void hook_observe(edict_t *ent, dqsim_hook_observation *obs)
{
    static const float pitches[DQSIM_HOOK_RAY_PITCHES] = {0.0f, 30.0f};
    memset(obs, 0, sizeof(*obs));
    obs->time = sv.time;
    obs->entity_index = hook_index;
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
            obs->ray_kinds[slot] = classify_ray(eye, forward, DQSIM_HOOK_RAY_DISTANCE, &trace,
                                                &fraction);
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

static void hook_apply(edict_t *ent, func_t fnum, const dqsim_hook_observation *obs,
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
        char attack_name[40];
        q_snprintf(attack_name, sizeof(attack_name), "%satk1", hook_prefix);
        func_t attack = enemy ? find_function(attack_name) : 0;
        if (attack && (!cooldown || sv.time >= cooldown->_float)) {
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
        hook_anim_tick++;
        if (speed > 0.01f && hook_run_frame >= 0)
            ent->v.frame = hook_run_frame + (float)((hook_anim_tick / 7) % hook_run_frames);
        else if (hook_stand_frame >= 0)
            ent->v.frame = hook_stand_frame + (float)((hook_anim_tick / 7) % 8);
        ent->v.think = fnum;
        ent->v.nextthink = (float)(sv.time + host_frametime);
    }
}

void dqsim_hook_execute_program(func_t fnum)
{
    if (hook_callback && hook_index > 0 && fnum > 0 && progs_is_stock) {
        edict_t *ent = EDICT_NUM(hook_index);
        if (!ent->free && ent->v.health > 0 && pr_global_struct->self == EDICT_TO_PROG(ent) &&
            (func_t)ent->v.think == fnum && ent->v.nextthink == 0 && is_decision_think(fnum)) {
            dqsim_hook_observation obs;
            hook_observe(ent, &obs);
            dqsim_hook_action action;
            memset(&action, 0, sizeof(action));
            action.desired_yaw = ent->v.angles[YAW];
            hook_callback(&obs, &action, hook_user);
            if (!ent->free)
                hook_apply(ent, fnum, &obs, &action);
            return;
        }
    }
    PR_ExecuteProgram(fnum);
}

int dqsim_set_policy_hook(int entity_index, dqsim_hook_callback callback, void *user)
{
    if (!callback)
        return fail("dqsim_set_policy_hook: callback is NULL (use dqsim_clear_policy_hook)");
    if (entity_index <= 0)
        return fail("dqsim_set_policy_hook: entity_index must be positive");
    DQ_ENTER();
    if (!sv.active)
        return fail("dqsim_set_policy_hook: no map is loaded");
    if (!progs_is_stock)
        return fail("dqsim_set_policy_hook: the loaded progs.dat is not the stock one");
    if (entity_index >= sv.num_edicts)
        return fail("dqsim_set_policy_hook: no such entity");
    edict_t *ent = EDICT_NUM(entity_index);
    const char *classname = ent->free ? "" : PR_GetString(ent->v.classname);
    if (ent->free || strncmp(classname, "monster_", 8) != 0)
        return fail("dqsim_set_policy_hook: the entity is not a monster");

    hook_callback = callback;
    hook_user = user;
    hook_index = entity_index;
    hook_anim_tick = 0;
    q_snprintf(hook_prefix, sizeof(hook_prefix), "%s_", classname + 8);
    char name[40];
    q_snprintf(name, sizeof(name), "%sstand1", hook_prefix);
    hook_stand_frame = state_frame(find_function(name));
    q_snprintf(name, sizeof(name), "%srun1", hook_prefix);
    hook_run_frame = state_frame(find_function(name));
    hook_run_frames = 0;
    for (int i = 1; i <= 24; i++) {
        q_snprintf(name, sizeof(name), "%srun%d", hook_prefix, i);
        if (!find_function(name))
            break;
        hook_run_frames = i;
    }
    if (hook_run_frames == 0)
        hook_run_frames = 1;
    return DQSIM_OK;
}

int dqsim_clear_policy_hook(void)
{
    DQ_ENTER();
    hook_callback = NULL;
    hook_user = NULL;
    hook_index = 0;
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
    hash = fnv1a(hash, &sv.time, sizeof(sv.time));
    hash = fnv1a(hash, &sv.num_edicts, sizeof(sv.num_edicts));
    for (int i = 0; i < sv.num_edicts; i++) {
        const edict_t *ent = EDICT_NUM(i);
        hash = fnv1a(hash, &ent->free, sizeof(ent->free));
        if (!ent->free)
            hash = fnv1a(hash, &ent->v, fields_size);
    }
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

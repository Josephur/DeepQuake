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
static char *engine_argv[5];
static int level_count;

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

/* The server portion of upstream _Host_Frame, with a fixed frame time. */
static void run_frame(void)
{
    host_frametime = tick_seconds;
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

    /* The engine itself starts without game data and fails later, obscurely. */
    q_snprintf(pak, sizeof(pak), "%s/id1/pak0.pak", config->basedir);
    f = fopen(pak, "rb");
    if (!f) {
        q_snprintf(dq_error, sizeof(dq_error), "dqsim_init: cannot open %s", pak);
        return DQSIM_ERROR;
    }
    fclose(f);

    dq_verbose = config->verbose;
    tick_seconds = config->tick_seconds > 0 ? config->tick_seconds : DQSIM_DEFAULT_TICK_SECONDS;
    q_strlcpy(basedir, config->basedir, sizeof(basedir));

    engine_argv[0] = "dqsim";
    engine_argv[1] = "-dedicated"; /* no client subsystems */
    engine_argv[2] = "1";          /* one client slot: single-player rules */
    engine_argv[3] = "-basedir";
    engine_argv[4] = basedir;

    memset(&parms, 0, sizeof(parms));
    parms.basedir = basedir;
    parms.userdir = basedir;
    parms.argc = 5;
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

int dqsim_ray_fan(const float *pitches, int n_pitch, int n_yaw, float max_distance, float *out)
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
            out[p * n_yaw + y] = trace.startsolid ? 0.0f : trace.fraction;
        }
    }
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
    return DQSIM_OK;
}

double dqsim_time(void)
{
    if (state != STATE_READY || !sv.active)
        return -1.0;
    return sv.time;
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

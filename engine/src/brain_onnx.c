/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * ONNX inference for the play client's brain hook (#43, M5).
 *
 * A model produced by deepquake.export_onnx (GRU actor-critic: inputs
 * observation + recurrent state, outputs per-head probabilities, value,
 * new state) runs here through the system ONNX Runtime; the greedy
 * action per head maps to the same discrete decisions the training
 * environment used (deepquake/monster_env.py): forward/back-stop-ahead
 * and strafe at the monster command speed, turn steps accumulated into
 * the facing, and the attack bit that starts the real QuakeC sequence.
 *
 * The observation the model consumes is deepquake.monster's (#33): the
 * dq_brain_build_observation function below constructs exactly that
 * layout, and tests/test_brain_onnx.c verifies the C pipeline against
 * the Python policy on synthetic hook reports.
 *
 * Without a model (no dq_model cvar, or a load failure) the client plays
 * the stock game (#44's fallback): the hook is simply never nominated.
 */
#include "quakedef.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brain_hook.h"
#include "dq_internal.h"

#include <onnxruntime_c_api.h>

/* deepquake.monster.MONSTER_OBS_DIM and MEMORY_SECONDS. */
#define DQ_MONSTER_OBS_DIM 119
#define DQ_MONSTER_MEMORY_SECONDS 5.0f
/* deepquake.monster_env.MONSTER_SPEED: commanded forward/side speed the
 * policy was trained with (the engine's monster physics cap what moves). */
#define DQ_MONSTER_SPEED 400.0f
#define DQ_MONSTER_MAX_HIDDEN 256

/* The action layout of the model (deepquake.monster_env.ACTION_NVEC) and
 * the turn grid (combat.TURN_DEGREES), overridable from a layout sidecar
 * next to the model so a differently-shaped model needs no recompile. */
static int dq_nvec[8] = {3, 3, 7, 2};
static int dq_nvec_count = 4;
static float dq_turn_degrees[8] = {-30.0f, -15.0f, -5.0f, 0.0f, 5.0f, 15.0f, 30.0f, 0.0f};

/* Obtained with Cvar_Get at registration, so "+set" on the command line
 * and these are the same instances. */
static cvar_t dq_model_var = {"dq_model", "", CVAR_ARCHIVE};
static cvar_t dq_style_var = {"dq_style", "0.5 0.5 0.5", CVAR_ARCHIVE};
static cvar_t dq_seed_var = {"dq_seed", "7", CVAR_ARCHIVE};
static cvar_t dq_cpu_var = {"dq_onnx_cpu", "0", CVAR_ARCHIVE};
static cvar_t *cvar_model, *cvar_style, *cvar_seed, *cvar_cpu;

static OrtEnv *ort_env;
static OrtSession *ort_session;
static OrtMemoryInfo *ort_memory;
static char *name_in_obs, *name_in_hidden;
static char *name_out_probs, *name_out_value, *name_out_hidden;
static float hidden_state[DQ_MONSTER_MAX_HIDDEN];
static size_t hidden_dim;
static int obs_dim;

/* Monster observation state (#33): the last-seen enemy memory. */
static float last_seen[3];
static double last_seen_time;
static int last_seen_valid;
static float monster_yaw;
static int dq_cvars_registered;
static int seeded;

static const OrtApi *ort(void)
{
    static const OrtApi *api;
    if (!api)
        api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return api;
}

static float parse_float(const char **cursor)
{
    while (**cursor == ' ' || **cursor == '\t')
        (*cursor)++;
    char *end;
    float value = (float)strtod(*cursor, &end);
    *cursor = end;
    return value;
}

/* Read "<model>.nvec": a line of ACTION_NVEC integers, optionally a line
 * of turn degrees. Without the sidecar the Grunt layout is assumed. */
static void read_layout(const char *model_path)
{
    char layout[1040];
    q_snprintf(layout, sizeof(layout), "%s.nvec", model_path);
    FILE *f = fopen(layout, "r");
    if (!f)
        return;
    int count = 0;
    while (count < 8 && fscanf(f, " %d", &dq_nvec[count]) == 1) count++, dq_nvec_count = count;
    char line[256];
    if (fgets(line, sizeof(line), f) && fgets(line, sizeof(line), f)) {
        const char *cursor = line;
        int got = 0;
        while (got < 8) {
            float value = parse_float(&cursor);
            if (cursor == line)
                break;
            dq_turn_degrees[got++] = value;
            line[0] = '\0';
        }
    }
    fclose(f);
}

static int load_model(const char *path)
{
    const OrtApi *api = ort();
    if (api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "deepquake-client", &ort_env))
        return -1;

    /* CUDA when present, CPU as the fallback (ARCHITECTURE.md M5). */
    for (int attempt = 0; attempt < 2 && !ort_session; attempt++) {
        OrtSessionOptions *options;
        if (api->CreateSessionOptions(&options))
            return -1;
        api->SetIntraOpNumThreads(options, 1);
        if (attempt == 0 && !(cvar_cpu && cvar_cpu->value != 0.0f))
            api->SessionOptionsAppendExecutionProvider(options, "CUDA", NULL, NULL, 0);
        else if (attempt == 0) {
            /* dq_onnx_cpu 1: skip the CUDA attempt (latency measurement). */
            api->ReleaseSessionOptions(options);
            continue;
        }
        OrtStatus *status = api->CreateSession(ort_env, path, options, &ort_session);
        api->ReleaseSessionOptions(options);
        if (status) {
            Con_Printf("deepquake: %s execution provider unavailable: %s\n",
                       attempt == 0 ? "CUDA" : "CPU", api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            ort_session = NULL;
        }
    }
    if (!ort_session)
        return -1;

    OrtAllocator *allocator;
    api->GetAllocatorWithDefaultOptions(&allocator);
    api->SessionGetInputName(ort_session, 0, allocator, &name_in_obs);
    api->SessionGetInputName(ort_session, 1, allocator, &name_in_hidden);
    api->SessionGetOutputName(ort_session, 0, allocator, &name_out_probs);
    api->SessionGetOutputName(ort_session, 1, allocator, &name_out_value);
    api->SessionGetOutputName(ort_session, 2, allocator, &name_out_hidden);
    api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_memory);

    OrtTypeInfo *type;
    const OrtTensorTypeAndShapeInfo *tensor;
    size_t dims;
    int64_t shape[4];
    api->SessionGetInputTypeInfo(ort_session, 1, &type);
    api->CastTypeInfoToTensorInfo(type, &tensor);
    api->GetDimensionsCount(tensor, &dims);
    api->GetDimensions(tensor, shape, dims);
    api->ReleaseTypeInfo(type);
    hidden_dim = (size_t)shape[1];
    if (hidden_dim > DQ_MONSTER_MAX_HIDDEN)
        hidden_dim = DQ_MONSTER_MAX_HIDDEN;
    api->SessionGetInputTypeInfo(ort_session, 0, &type);
    api->CastTypeInfoToTensorInfo(type, &tensor);
    api->GetDimensionsCount(tensor, &dims);
    api->GetDimensions(tensor, shape, dims);
    api->ReleaseTypeInfo(type);
    obs_dim = (int)shape[1];
    return 0;
}

void dq_brain_build_observation(const dqsim_hook_observation *hook, int skill,
                                const float style[3], float *out)
{
    const int rays = DQSIM_HOOK_RAY_PITCHES * DQSIM_HOOK_RAY_YAWS;
    memset(out, 0, sizeof(float) * DQ_MONSTER_OBS_DIM);
    for (int i = 0; i < rays; i++) {
        out[i] = hook->rays[i];
        out[rays + i] = hook->ray_kinds[i] / 6.0f;
    }

    int at = 2 * rays;
    float yaw_rad = hook->angles[1] * (float)M_PI / 180.0f;
    float cos_yaw = cosf(yaw_rad), sin_yaw = sinf(yaw_rad);
    out[at + 0] = (hook->velocity[0] * cos_yaw + hook->velocity[1] * sin_yaw) / 400.0f;
    out[at + 1] = (-hook->velocity[0] * sin_yaw + hook->velocity[1] * cos_yaw) / 400.0f;
    out[at + 2] = hook->velocity[2] / 400.0f;
    out[at + 3] = hook->on_ground ? 1.0f : 0.0f;
    out[at + 4] = hook->health < 0.0f ? 0.0f : (hook->health > 100.0f ? 1.0f : hook->health / 100.0f);
    out[at + 5] = hook->attack_cooldown < 0.0f ? 0.0f
                  : (hook->attack_cooldown > 1.0f ? 1.0f : hook->attack_cooldown);
    out[at + 6] = hook->waterlevel / 3.0f;

    const dqsim_hook_visible *enemy = NULL;
    for (int i = 0; i < hook->visible_count; i++) {
        if (hook->visible[i].flags & FL_CLIENT) {
            enemy = &hook->visible[i];
            break;
        }
    }
    if (enemy) {
        for (int k = 0; k < 3; k++)
            last_seen[k] = hook->origin[k] + enemy->relative[k];
        last_seen_time = hook->time;
        last_seen_valid = 1;
        float dx = enemy->relative[0], dy = enemy->relative[1];
        float distance = sqrtf(dx * dx + dy * dy);
        float scale = 1.0f / (distance > 1e-6f ? distance : 1e-6f);
        out[at + 7] = (dx * cos_yaw + dy * sin_yaw) * scale;
        out[at + 8] = (-dx * sin_yaw + dy * cos_yaw) * scale;
        float dz = enemy->relative[2];
        out[at + 9] = dz / 256.0f;
        if (out[at + 9] < -1.0f) out[at + 9] = -1.0f;
        if (out[at + 9] > 1.0f) out[at + 9] = 1.0f;
        out[at + 10] = distance / 1024.0f;
        if (out[at + 10] > 2.0f) out[at + 10] = 2.0f;
        out[at + 11] = 1.0f;
        out[at + 12] = enemy->health < 0.0f ? 0.0f
                       : (enemy->health > 100.0f ? 1.0f : enemy->health / 100.0f);
    }
    if (last_seen_valid) {
        float age = (float)(hook->time - last_seen_time);
        if (age < 0.0f)
            age = 0.0f;
        if (age > DQ_MONSTER_MEMORY_SECONDS) {
            last_seen_valid = 0;
        } else {
            float rx = last_seen[0] - hook->origin[0];
            float ry = last_seen[1] - hook->origin[1];
            float rz = last_seen[2] - hook->origin[2];
            float distance = sqrtf(rx * rx + ry * ry);
            float scale = 1.0f / (distance > 1e-6f ? distance : 1e-6f);
            out[at + 13] = (rx * cos_yaw + ry * sin_yaw) * scale;
            out[at + 14] = (-rx * sin_yaw + ry * cos_yaw) * scale;
            out[at + 15] = rz / 256.0f;
            if (out[at + 15] < -1.0f) out[at + 15] = -1.0f;
            if (out[at + 15] > 1.0f) out[at + 15] = 1.0f;
            out[at + 16] = distance / 1024.0f;
            if (out[at + 16] > 2.0f) out[at + 16] = 2.0f;
            out[at + 17] = age / DQ_MONSTER_MEMORY_SECONDS;
            out[at + 18] = 1.0f;
        }
    }

    out[at + 19] = skill / 3.0f;
    out[at + 20] = style[0];
    out[at + 21] = style[1];
    out[at + 22] = style[2];
}

/* The client's policy: the brain-hook callback running the model. */
static void onnx_policy(const dqsim_hook_observation *hook, dqsim_hook_action *action,
                        void *user)
{
    const OrtApi *api = ort();
    (void)user;

    /* The fixed style the cross-check test uses (0.25, 0.5, 0.75). */
    float style[3] = {0.25f, 0.5f, 0.75f};

    float obs[DQ_MONSTER_OBS_DIM];
    dq_brain_build_observation(hook, dq_brain_hook_skill(), style, obs);

    int64_t shape[2] = {1, (int64_t)obs_dim};
    int64_t hshape[2] = {1, (int64_t)hidden_dim};
    OrtValue *input_obs = NULL, *input_hidden = NULL;
    api->CreateTensorWithDataAsOrtValue(ort_memory, obs, sizeof(float) * obs_dim, shape, 2,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_obs);
    api->CreateTensorWithDataAsOrtValue(ort_memory, hidden_state,
                                        sizeof(float) * hidden_dim, hshape, 2,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_hidden);
    const OrtValue *inputs[2] = {input_obs, input_hidden};
    const char *output_names[3] = {name_out_probs, name_out_value, name_out_hidden};
    OrtValue *output_values[3] = {NULL, NULL, NULL};
    const char *input_names[2] = {name_in_obs, name_in_hidden};
    OrtStatus *status = api->Run(ort_session, NULL, input_names, inputs, 2,
                                 output_names, 3, output_values);
    api->ReleaseValue(input_obs);
    api->ReleaseValue(input_hidden);
    if (status) {
        Con_Printf("deepquake: inference failed: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return; /* hold the facing, do nothing this think */
    }

    float *probs;
    float *new_hidden;
    api->GetTensorMutableData(output_values[0], (void **)&probs);
    api->GetTensorMutableData(output_values[2], (void **)&new_hidden);
    memcpy(hidden_state, new_hidden, sizeof(float) * hidden_dim);

    int actions[8];
    int offset = 0;
    for (int head = 0; head < dq_nvec_count; head++) {
        int best = 0;
        for (int i = 1; i < dq_nvec[head]; i++)
            if (probs[offset + i] > probs[offset + best])
                best = i;
        actions[head] = best;
        offset += dq_nvec[head];
    }
    api->ReleaseValue(output_values[0]);
    api->ReleaseValue(output_values[1]);
    api->ReleaseValue(output_values[2]);

    monster_yaw = fmodf(monster_yaw + dq_turn_degrees[actions[2]], 360.0f);
    if (monster_yaw < 0.0f)
        monster_yaw += 360.0f;
    action->desired_yaw = monster_yaw;
    action->forward = (actions[0] - 1) * DQ_MONSTER_SPEED;
    action->side = (1 - actions[1]) * DQ_MONSTER_SPEED;
    action->attack = actions[3] != 0;
}
static void register_cvars(void);
static void dq_selftest_run(const char *model, const char *report);

/* Nominate the first stock Grunt once the model is ready. Runs from the
 * brain hook's client extension point while nothing is nominated, so it
 * also retries after a level change (the nomination clears per level). */
static void try_nominate(void)
{
    register_cvars();
    if (!cvar_model->string[0])
        return;
    if (!ort_session && load_model(cvar_model->string)) {
        Con_Printf("deepquake: playing stock AI (model %s did not load)\n", cvar_model->string);
        Cvar_Set("dq_model", ""); /* do not retry every tick */
        return;
    }
    if (!sv.active || !dq_brain_progs_is_stock())
        return;
    for (int i = 1; i < sv.num_edicts; i++) {
        edict_t *ent = EDICT_NUM(i);
        if (ent->free)
            continue;
        const char *classname = PR_GetString(ent->v.classname);
        if (strncmp(classname, "monster_army", 12) != 0)
            continue;
        if (!seeded) {
            seeded = 1;
            dq_rand_seed((uint32_t)(int)cvar_seed->value);
        }
        /* The recurrent state starts at zero for every episode, like
         * training; the facing continues from the monster's own. */
        memset(hidden_state, 0, sizeof(hidden_state));
        last_seen_valid = 0;
        monster_yaw = ent->v.angles[YAW];
        const char *error = dq_brain_hook_nominate(i, onnx_policy, NULL);
        if (error)
            Con_Printf("deepquake: Grunt %d stays stock: %s\n", i, error);
        else
            Con_Printf("deepquake: Grunt %d is driven by %s\n", i, cvar_model->string);
        return;
    }
}

/* Installed as the brain hook's client tick (see brain_hook.c). */
static void client_tick(void)
{
    {
        static int probes;
        if (probes < 3)
            Con_Printf("client probe %d\n", ++probes);
    }
    /* The stock progs identity, once. The client's basedir handling is
     * upstream's own (Sys_GetBasedir), so the identity is read through
     * the engine's search path: for the v1 client (no -game) that is the
     * id1 progs the levels load. The simulator keeps the strict id1
     * reader (dqsim_init) for its mod gate. */
    static int identity_done;
    if (!identity_done) {
        identity_done = 1;
        byte *raw = COM_LoadMallocFile("progs.dat", NULL);
        if (raw) {
            dq_brain_set_stock_identity(dq_fnv1a(DQ_FNV1A_SEED, raw, (size_t)com_filesize),
                                        (size_t)com_filesize);
            free(raw);
        } else {
            Con_Printf("deepquake: no progs.dat on the search path - stock AI only\n");
        }
    }
    try_nominate();
#ifdef DQ_CLIENT_SELFTEST
    /* Verification build: once a level is live, run the synthetic-report
     * selftest and quit - a headless acceptance run. */
    static int selftest_done;
    if (!selftest_done && sv.active && dq_brain_progs_is_stock()) {
        selftest_done = 1;
        dq_selftest_run(DQ_CLIENT_SELFTEST_MODEL, DQ_CLIENT_SELFTEST_REPORT);
        Cbuf_AddText("quit\n");
    }
#endif
}

/* Runs before main: storing a function pointer needs no engine state, so
 * the client needs no vendored-file change for start-up order. */
__attribute__((constructor)) static void install_tick(void)
{
    dq_brain_hook_set_tick(client_tick);
}

/* ---- selftest (#43): synthetic hook reports through the real pipeline ----
 *
 * Run from the client console: dq_selftest <model> <report_path>. Writes,
 * per case, the observation vector the C builder produced, the model's
 * probabilities and the greedy actions. tests/test_brain_onnx.py builds
 * the same reports through deepquake.monster and onnxruntime and compares.
 */
#include <ctype.h>
#include <time.h>

static void dq_selftest_make_hook(double time, int visible, dqsim_hook_observation *hook)
{
    static float rays[DQSIM_HOOK_RAY_PITCHES * DQSIM_HOOK_RAY_YAWS];
    static unsigned char kinds[DQSIM_HOOK_RAY_PITCHES * DQSIM_HOOK_RAY_YAWS];
    memset(hook, 0, sizeof(*hook));
    hook->time = time;
    hook->entity_index = 9;
    hook->origin[0] = time > 0.25 ? 10.0f : 0.0f;
    hook->velocity[0] = 100.0f;
    hook->angles[1] = 90.0f;
    hook->health = 80.0f;
    hook->on_ground = true;
    for (int i = 0; i < DQSIM_HOOK_RAY_PITCHES * DQSIM_HOOK_RAY_YAWS; i++) {
        rays[i] = (float)(i % 7) * 0.1f + 0.05f;
        kinds[i] = (unsigned char)(i % 7);
    }
    memcpy(hook->rays, rays, sizeof(rays));
    memcpy(hook->ray_kinds, kinds, sizeof(kinds));
    hook->attack_cooldown = 0.3f;
    if (visible) {
        static dqsim_hook_visible entry;
        entry.index = 1;
        entry.flags = FL_CLIENT;
        entry.relative[0] = 100.0f;
        entry.relative[1] = 0.0f;
        entry.relative[2] = 24.0f;
        entry.health = 60.0f;
        entry.distance = 100.0f;
        memcpy(hook->visible, &entry, sizeof(entry));
        hook->visible_count = 1;
    }
}

static void dq_selftest_run(const char *model, const char *report)
{
    if (!ort_session && load_model(model)) {
        Con_Printf("dq_selftest: could not load %s\n", model);
        return;
    }
    /* Fresh policy state, exactly as at a nomination. */
    memset(hidden_state, 0, sizeof(hidden_state));
    last_seen_valid = 0;
    monster_yaw = 90.0f;
    /* The fixed style the cross-check test uses (0.25, 0.5, 0.75). */
    float style[3] = {0.25f, 0.5f, 0.75f};

    FILE *out = fopen(report, "w");
    if (!out) {
        Con_Printf("dq_selftest: cannot write %s\n", report);
        return;
    }
    const OrtApi *api = ort();
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    int inferences = 0;
    for (int case_index = 0; case_index < 3; case_index++) {
        dqsim_hook_observation hook;
        dq_selftest_make_hook(case_index == 0 ? 0.0 : (case_index == 1 ? 0.5 : 6.0),
                              case_index < 2, &hook);
        float obs[DQ_MONSTER_OBS_DIM];
        dq_brain_build_observation(&hook, 2, style, obs);

        int64_t shape[2] = {1, obs_dim};
        int64_t hshape[2] = {1, (int64_t)hidden_dim};
        OrtValue *in_obs = NULL, *in_hidden = NULL;
        api->CreateTensorWithDataAsOrtValue(ort_memory, obs, sizeof(float) * obs_dim, shape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_obs);
        api->CreateTensorWithDataAsOrtValue(ort_memory, hidden_state,
                                            sizeof(float) * hidden_dim, hshape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_hidden);
        const char *input_names[2] = {name_in_obs, name_in_hidden};
        const char *output_names[3] = {name_out_probs, name_out_value, name_out_hidden};
        OrtValue *output_values[3] = {NULL, NULL, NULL};
        const OrtValue *inputs[2] = {in_obs, in_hidden};
        OrtStatus *status = api->Run(ort_session, NULL, input_names, inputs, 2, output_names, 3,
                                     output_values);
        api->ReleaseValue(in_obs);
        api->ReleaseValue(in_hidden);
        if (status) {
            Con_Printf("dq_selftest: inference failed: %s\n", api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            fclose(out);
            return;
        }
        float *probs;
        float *new_hidden;
        api->GetTensorMutableData(output_values[0], (void **)&probs);
        api->GetTensorMutableData(output_values[2], (void **)&new_hidden);
        memcpy(hidden_state, new_hidden, sizeof(float) * hidden_dim);

        int actions[8];
        int offset = 0;
        for (int head = 0; head < dq_nvec_count; head++) {
            int best = 0;
            for (int i = 1; i < dq_nvec[head]; i++)
                if (probs[offset + i] > probs[offset + best])
                    best = i;
            actions[head] = best;
            offset += dq_nvec[head];
        }
        api->ReleaseValue(output_values[0]);
        api->ReleaseValue(output_values[1]);
        api->ReleaseValue(output_values[2]);

        fprintf(out, "case %d\nobs", case_index);
        for (int i = 0; i < DQ_MONSTER_OBS_DIM; i++)
            fprintf(out, " %.6f", obs[i]);
        fprintf(out, "\nprobs");
        for (int i = 0; i < offset; i++)
            fprintf(out, " %.7f", probs[i]);
        fprintf(out, "\nactions");
        for (int head = 0; head < dq_nvec_count; head++)
            fprintf(out, " %d", actions[head]);
        fprintf(out, "\n");
        fflush(out);
        inferences++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double ms = (t_end.tv_sec - t_start.tv_sec) * 1e3 + (t_end.tv_nsec - t_start.tv_nsec) / 1e6;
    Con_Printf("dq_selftest: %d inferences in %.2f ms (%.0f us/decision)\n",
               inferences, ms, ms * 1000.0 / inferences);
    fclose(out);
    Con_Printf("dq_selftest: wrote %s\n", report);
}


static void register_cvars(void)
{
    if (dq_cvars_registered)
        return;
    dq_cvars_registered = 1;
    /* Register ours, then adopt what "+set" asked for: the command line
     * was processed before this registration, and Cvar_Set refuses
     * unknown cvars, so the values live only in host_parms.argv. */
    if (!(cvar_model = Cvar_FindVar("dq_model"))) {
        cvar_model = &dq_model_var;
        Cvar_RegisterVariable(cvar_model);
    }
    if (!(cvar_style = Cvar_FindVar("dq_style"))) {
        cvar_style = &dq_style_var;
        Cvar_RegisterVariable(cvar_style);
    }
    if (!(cvar_seed = Cvar_FindVar("dq_seed"))) {
        cvar_seed = &dq_seed_var;
        Cvar_RegisterVariable(cvar_seed);
    if (!(cvar_cpu = Cvar_FindVar("dq_onnx_cpu"))) {
        cvar_cpu = &dq_cpu_var;
        Cvar_RegisterVariable(cvar_cpu);
    }
    }
    for (int i = 1; i < host_parms->argc - 2; i++) {
        if (strcmp(host_parms->argv[i], "+set") != 0)
            continue;
        const char *name = host_parms->argv[i + 1];
        cvar_t *var = strcmp(name, "dq_model") == 0   ? cvar_model
                      : strcmp(name, "dq_style") == 0 ? cvar_style
                      : strcmp(name, "dq_seed") == 0  ? cvar_seed
                                                      : NULL;
        if (var)
            Cvar_Set(name, host_parms->argv[i + 2]);
    }
}

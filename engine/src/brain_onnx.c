/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * ONNX inference for the play client's brain hook (#43, M5; every
 * monster of a class from #46, M6).
 *
 * A model produced by deepquake.export_onnx (GRU actor-critic: inputs
 * observation + recurrent state, outputs per-head probabilities, value,
 * new state) runs here through the system ONNX Runtime; the greedy
 * action per head maps to the same discrete decisions the training
 * environment used (deepquake/monster_env.py): forward/back-stop-ahead
 * and strafe at the monster command speed, turn steps accumulated into
 * the facing, and the attack bit that starts the real QuakeC sequence.
 *
 * Every monster the model covers is nominated in the hook's slot table,
 * each with its own recurrent state, last-seen memory and facing: a level
 * has up to 35 Grunts, and one shared hidden state would smear their
 * episodes together. The number is capped by dq_max_monsters (0 = all).
 *
 * A model is either a v1 single-type one (119 observation floats, the
 * Grunt only) or a shared one (#48) whose input ends with a one-hot of
 * the monster type; "<path>.monsters" names the classes and their type
 * index, and each nominated monster carries its own index into the
 * observation, so one session drives a mixed level.
 *
 * The observation the model consumes is deepquake.monster's (#33): the
 * dq_build_observation function below constructs exactly that layout.
 *
 * Without a model (no dq_model cvar, or a load failure) the client plays
 * the stock game (#44's fallback): the hook is simply never nominated.
 */
#include "quakedef.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "brain_hook.h"
#include "dq_internal.h"

#include <onnxruntime_c_api.h>

/* deepquake.monster.MONSTER_OBS_BASE_DIM, MEMORY_SECONDS and the type
 * vocabulary width (#48). A v1 model (models-v1) is 119 wide and has no
 * type input; a shared model appends one one-hot entry per type. */
#define DQ_MONSTER_OBS_BASE_DIM 119
#define DQ_MONSTER_MAX_TYPES 8
#define DQ_MONSTER_OBS_MAX (DQ_MONSTER_OBS_BASE_DIM + DQ_MONSTER_MAX_TYPES)
#define DQ_MONSTER_MEMORY_SECONDS 5.0f
/* deepquake.monster_env.MONSTER_SPEED: commanded forward/side speed the
 * policy was trained with (the engine's monster physics cap what moves). */
#define DQ_MONSTER_SPEED 400.0f
#define DQ_MONSTER_MAX_HIDDEN 256
/* dq_model is a space-separated list; one session per model, and each
 * model names the monster classes it drives in "<path>.monsters". */
#define DQ_MAX_MODELS 4
#define DQ_MAX_CLASSES 8
#define DQ_MAX_PATH 512

typedef struct dq_model_class {
    char classname[32];
    int type_index;
} dq_model_class;

typedef struct dq_model {
    char path[DQ_MAX_PATH];
    OrtSession *session;
    char *name_in_obs, *name_in_hidden;
    char *name_out_probs, *name_out_value, *name_out_hidden;
    size_t hidden_dim;
    int obs_dim;
    /* How many monster-type entries the observation ends with (#48): 0 for
     * a v1 model, n_types for a shared one. */
    int type_dim;
    /* The action layout (deepquake.monster_env.ACTION_NVEC) and the turn
     * grid (combat.TURN_DEGREES), from the ".nvec" sidecar; the Grunt's
     * layout is the default so a model without one still runs. */
    int nvec[8];
    int nvec_count;
    float turn[8];
    dq_model_class classes[DQ_MAX_CLASSES];
    int class_count;
} dq_model;

/* One policy-driven monster's inference state (#46): the recurrent state,
 * the observation's last-seen memory and the facing the turn actions
 * accumulate into. All per monster, never shared. */
typedef struct dq_monster_state {
    int entity_index; /* 0 = free */
    dq_model *model;
    float hidden[DQ_MONSTER_MAX_HIDDEN];
    float last_seen[3];
    double last_seen_time;
    int last_seen_valid;
    float yaw;
    int skill;
    /* Which entry of the model's type one-hot this monster is (#48). */
    int type_index;
} dq_monster_state;

static dq_monster_state monster_states[DQ_BRAIN_MAX_MONSTERS];
static dq_model models[DQ_MAX_MODELS];
static int model_count;
static int models_tried;
static char last_level[MAX_QPATH];
/* The edict block this client's per-monster states belong to: a death
 * restart reloads the same map name but allocates new edicts, so the
 * hidden states must not carry across it (training zeroes them per
 * episode). */
static void *last_edicts;
static int reported_driven = -1;
/* Inference cost, reported once a frame at most (never part of the sim). */
static long long inference_ns;
static int inference_count;

/* Obtained with Cvar_Get at registration, so "+set" on the command line
 * and these are the same instances. */
static cvar_t dq_model_var = {"dq_model", "", CVAR_ARCHIVE};
static cvar_t dq_style_var = {"dq_style", "0.5 0.5 0.5", CVAR_ARCHIVE};
static cvar_t dq_seed_var = {"dq_seed", "7", CVAR_ARCHIVE};
static cvar_t dq_cpu_var = {"dq_onnx_cpu", "0", CVAR_ARCHIVE};
static cvar_t dq_max_var = {"dq_max_monsters", "0", CVAR_ARCHIVE};
static cvar_t *cvar_model, *cvar_style, *cvar_seed, *cvar_cpu, *cvar_max;

static OrtEnv *ort_env;
static OrtMemoryInfo *ort_memory;
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
static void read_layout(dq_model *model)
{
    char layout[DQ_MAX_PATH + 8];
    q_snprintf(layout, sizeof(layout), "%s.nvec", model->path);
    FILE *f = fopen(layout, "r");
    if (!f)
        return;
    int count = 0;
    while (count < 8 && fscanf(f, " %d", &model->nvec[count]) == 1)
        count++, model->nvec_count = count;
    char line[256];
    if (fgets(line, sizeof(line), f) && fgets(line, sizeof(line), f)) {
        const char *cursor = line;
        int got = 0;
        while (got < 8) {
            float value = parse_float(&cursor);
            if (cursor == line)
                break;
            model->turn[got++] = value;
            line[0] = '\0';
        }
    }
    fclose(f);
}

/* Read "<model>.monsters" (written by deepquake.export_onnx): one
 * "<classname> <type index>" per line. Without it the model is the Grunt
 * policy, which is what models-v1 shipped. */
static void read_classes(dq_model *model)
{
    char list[DQ_MAX_PATH + 16];
    q_snprintf(list, sizeof(list), "%s.monsters", model->path);
    FILE *f = fopen(list, "r");
    if (!f)
        return;
    char classname[32];
    int type_index;
    while (model->class_count < DQ_MAX_CLASSES &&
           fscanf(f, " %31s %d", classname, &type_index) == 2) {
        dq_model_class *entry = &model->classes[model->class_count++];
        q_strlcpy(entry->classname, classname, sizeof(entry->classname));
        entry->type_index = type_index;
    }
    fclose(f);
}

static int load_model(dq_model *model)
{
    const OrtApi *api = ort();
    if (!ort_env && api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "deepquake-client", &ort_env))
        return -1;

    /* CUDA when present, CPU as the fallback (ARCHITECTURE.md M5). */
    for (int attempt = 0; attempt < 2 && !model->session; attempt++) {
        OrtSessionOptions *options;
        if (api->CreateSessionOptions(&options))
            return -1;
        api->SetIntraOpNumThreads(options, 1);
        if (attempt == 0 && !(cvar_cpu && cvar_cpu->value != 0.0f)) {
            api->SessionOptionsAppendExecutionProvider(options, "CUDA", NULL, NULL, 0);
        } else if (attempt == 0) {
            /* dq_onnx_cpu 1: skip the CUDA attempt (latency measurement). */
            api->ReleaseSessionOptions(options);
            continue;
        }
        OrtStatus *status = api->CreateSession(ort_env, model->path, options, &model->session);
        api->ReleaseSessionOptions(options);
        if (status) {
            Con_Printf("deepquake: %s execution provider unavailable for %s: %s\n",
                       attempt == 0 ? "CUDA" : "CPU", model->path, api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            model->session = NULL;
        }
    }
    if (!model->session)
        return -1;

    OrtAllocator *allocator;
    api->GetAllocatorWithDefaultOptions(&allocator);
    api->SessionGetInputName(model->session, 0, allocator, &model->name_in_obs);
    api->SessionGetInputName(model->session, 1, allocator, &model->name_in_hidden);
    api->SessionGetOutputName(model->session, 0, allocator, &model->name_out_probs);
    api->SessionGetOutputName(model->session, 1, allocator, &model->name_out_value);
    api->SessionGetOutputName(model->session, 2, allocator, &model->name_out_hidden);
    if (!ort_memory)
        api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_memory);

    OrtTypeInfo *type;
    const OrtTensorTypeAndShapeInfo *tensor;
    size_t dims;
    int64_t shape[4];
    api->SessionGetInputTypeInfo(model->session, 1, &type);
    api->CastTypeInfoToTensorInfo(type, &tensor);
    api->GetDimensionsCount(tensor, &dims);
    api->GetDimensions(tensor, shape, dims);
    api->ReleaseTypeInfo(type);
    model->hidden_dim = (size_t)shape[1];
    if (model->hidden_dim > DQ_MONSTER_MAX_HIDDEN)
        model->hidden_dim = DQ_MONSTER_MAX_HIDDEN;
    api->SessionGetInputTypeInfo(model->session, 0, &type);
    api->CastTypeInfoToTensorInfo(type, &tensor);
    api->GetDimensionsCount(tensor, &dims);
    api->GetDimensions(tensor, shape, dims);
    api->ReleaseTypeInfo(type);
    model->obs_dim = (int)shape[1];
    /* The client drives monster policies only (#35): a model with another
     * observation width is the wrong architecture - refuse it for stock
     * play with a clear log line instead of feeding it misaligned input.
     * Two widths are legal: 119 (a v1 single-type model) and
     * 119 + n_types (a shared model, #48). */
    if (model->obs_dim == DQ_MONSTER_OBS_BASE_DIM) {
        model->type_dim = 0;
    } else if (model->obs_dim > DQ_MONSTER_OBS_BASE_DIM &&
               model->obs_dim <= DQ_MONSTER_OBS_MAX) {
        model->type_dim = model->obs_dim - DQ_MONSTER_OBS_BASE_DIM;
    } else {
        Con_Printf("deepquake: refusing %s - observation width %d, the client "
                   "expects %d (a single-type model) or %d-%d (a shared model)\n",
                   model->path, model->obs_dim, DQ_MONSTER_OBS_BASE_DIM,
                   DQ_MONSTER_OBS_BASE_DIM + 1, DQ_MONSTER_OBS_MAX);
        return -1;
    }
    return 0;
}

/* A model only drives classes its observation can name: a v1 model (no
 * type input) covers the single type 0, a shared model every index it has
 * an entry for (#48). */
static int classes_fit(dq_model *model)
{
    for (int c = 0; c < model->class_count; c++) {
        int index = model->classes[c].type_index;
        int ok = model->type_dim == 0 ? index == 0
                                      : index >= 0 && index < model->type_dim;
        if (!ok) {
            Con_Printf("deepquake: refusing %s - class %s has type index %d "
                       "but the model has %d type input(s)\n",
                       model->path, model->classes[c].classname, index,
                       model->type_dim);
            return 0;
        }
    }
    return 1;
}

static void discard_model(dq_model *model)
{
    if (model->session)
        ort()->ReleaseSession(model->session);
    memset(model, 0, sizeof(*model));
}

/* Load every path in dq_model once. A failure is logged once and leaves
 * the classes that model would drive on the stock AI (#44). */
static void load_models(const char *list)
{
    char buffer[1024];
    q_strlcpy(buffer, list, sizeof(buffer));
    char *save = NULL;
    for (char *token = strtok_r(buffer, " \t", &save);
         token && model_count < DQ_MAX_MODELS; token = strtok_r(NULL, " \t", &save)) {
        dq_model *model = &models[model_count];
        memset(model, 0, sizeof(*model));
        q_strlcpy(model->path, token, sizeof(model->path));
        model->nvec[0] = 3;
        model->nvec[1] = 3;
        model->nvec[2] = 7;
        model->nvec[3] = 2;
        model->nvec_count = 4;
        static const float grunt_turn[8] = {-30.0f, -15.0f, -5.0f,
                                            0.0f,   5.0f,   15.0f, 30.0f, 0.0f};
        memcpy(model->turn, grunt_turn, sizeof(grunt_turn));
        read_layout(model);
        read_classes(model);
        if (!model->class_count) {
            q_strlcpy(model->classes[0].classname, "monster_army",
                      sizeof(model->classes[0].classname));
            model->classes[0].type_index = 0;
            model->class_count = 1;
        }
        if (load_model(model) || !classes_fit(model)) {
            Con_Printf("deepquake: %s did not load; its monsters stay stock\n", model->path);
            discard_model(model);
            continue;
        }
        Con_Printf("deepquake: loaded %s\n", model->path);
        model_count++;
    }
}

/* The model driving a class, and which type entry of its observation that
 * class is (unused by a v1 model). */
static dq_model *model_for_class(const char *classname, int *type_index)
{
    for (int m = 0; m < model_count; m++)
        for (int c = 0; c < models[m].class_count; c++)
            if (strcmp(models[m].classes[c].classname, classname) == 0) {
                *type_index = models[m].classes[c].type_index;
                return &models[m];
            }
    return NULL;
}

/* The observation deepquake.monster builds (#33), from a live hook report
 * and this monster's memory. Kept in C so the client needs no Python. */
static void dq_build_observation(dq_monster_state *state,
                                 const dqsim_hook_observation *hook, float *out)
{
    const int rays = DQSIM_HOOK_RAY_PITCHES * DQSIM_HOOK_RAY_YAWS;
    memset(out, 0, sizeof(float) * DQ_MONSTER_OBS_MAX);
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
    out[at + 4] = hook->health < 0.0f ? 0.0f
                  : (hook->health > 100.0f ? 1.0f : hook->health / 100.0f);
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
            state->last_seen[k] = hook->origin[k] + enemy->relative[k];
        state->last_seen_time = hook->time;
        state->last_seen_valid = 1;
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
    if (state->last_seen_valid) {
        float age = (float)(hook->time - state->last_seen_time);
        if (age < 0.0f)
            age = 0.0f;
        if (age > DQ_MONSTER_MEMORY_SECONDS) {
            state->last_seen_valid = 0;
        } else {
            float rx = state->last_seen[0] - hook->origin[0];
            float ry = state->last_seen[1] - hook->origin[1];
            float rz = state->last_seen[2] - hook->origin[2];
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

    out[at + 19] = state->skill / 3.0f;
    /* The fixed style the cross-check test uses (0.25, 0.5, 0.75); exposing
     * dq_style to the policy is the M6 cvar item. */
    out[at + 20] = 0.25f;
    out[at + 21] = 0.5f;
    out[at + 22] = 0.75f;
    /* The body the policy is wearing (#48): a shared model ends its input
     * with one one-hot entry per type, and its head is chosen from it. A
     * v1 model (type_dim 0) has no such slice. */
    if (state->model->type_dim > 0)
        out[at + 23 + state->type_index] = 1.0f;
}

/* The client's policy: the brain-hook callback running this monster's
 * model on this monster's state. */
static void onnx_policy(const dqsim_hook_observation *hook, dqsim_hook_action *action,
                        void *user)
{
    dq_monster_state *state = user;
    dq_model *model = state->model;
    const OrtApi *api = ort();

    float obs[DQ_MONSTER_OBS_MAX];
    dq_build_observation(state, hook, obs);

    int64_t shape[2] = {1, (int64_t)model->obs_dim};
    int64_t hshape[2] = {1, (int64_t)model->hidden_dim};
    OrtValue *input_obs = NULL, *input_hidden = NULL;
    api->CreateTensorWithDataAsOrtValue(ort_memory, obs, sizeof(float) * model->obs_dim, shape,
                                        2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_obs);
    api->CreateTensorWithDataAsOrtValue(ort_memory, state->hidden,
                                        sizeof(float) * model->hidden_dim, hshape, 2,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_hidden);
    const OrtValue *inputs[2] = {input_obs, input_hidden};
    const char *output_names[3] = {model->name_out_probs, model->name_out_value,
                                   model->name_out_hidden};
    OrtValue *output_values[3] = {NULL, NULL, NULL};
    const char *input_names[2] = {model->name_in_obs, model->name_in_hidden};
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    OrtStatus *status = api->Run(model->session, NULL, input_names, inputs, 2, output_names, 3,
                                 output_values);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    inference_ns += (t_end.tv_sec - t_start.tv_sec) * 1000000000LL +
                    (t_end.tv_nsec - t_start.tv_nsec);
    inference_count++;
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
    memcpy(state->hidden, new_hidden, sizeof(float) * model->hidden_dim);

    int actions[8];
    int offset = 0;
    for (int head = 0; head < model->nvec_count; head++) {
        int best = 0;
        for (int i = 1; i < model->nvec[head]; i++)
            if (probs[offset + i] > probs[offset + best])
                best = i;
        actions[head] = best;
        offset += model->nvec[head];
    }
    api->ReleaseValue(output_values[0]);
    api->ReleaseValue(output_values[1]);
    api->ReleaseValue(output_values[2]);

    state->yaw = fmodf(state->yaw + model->turn[actions[2]], 360.0f);
    if (state->yaw < 0.0f)
        state->yaw += 360.0f;
    action->desired_yaw = state->yaw;
    action->forward = (actions[0] - 1) * DQ_MONSTER_SPEED;
    action->side = (1 - actions[1]) * DQ_MONSTER_SPEED;
    action->attack = actions[3] != 0;
}

static void register_cvars(void);
static void dq_selftest_run(const char *model, const char *report);

static dq_monster_state *free_state(void)
{
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++)
        if (!monster_states[i].entity_index)
            return &monster_states[i];
    return NULL;
}

/* Forget monsters that died or that the hook no longer drives (a level
 * change clears the hook's slots, so this is also the re-nomination path). */
static void release_dead_states(void)
{
    for (int i = 0; i < DQ_BRAIN_MAX_MONSTERS; i++) {
        dq_monster_state *state = &monster_states[i];
        if (!state->entity_index)
            continue;
        if (state->entity_index >= sv.num_edicts) {
            memset(state, 0, sizeof(*state));
            continue;
        }
        edict_t *ent = EDICT_NUM(state->entity_index);
        if (ent->free || ent->v.health <= 0.0f || !dq_brain_hook_is_nominated(state->entity_index))
            memset(state, 0, sizeof(*state));
    }
}

/* Nominate every live monster whose class a loaded model covers, each
 * with its own inference state, up to dq_max_monsters. Runs once per
 * engine tick from the client tick; a level change resets the indices. */
static void nominate_pass(void)
{
    if (strcmp(sv.name, last_level) != 0 || sv.edicts != last_edicts) {
        q_strlcpy(last_level, sv.name, sizeof(last_level));
        last_edicts = sv.edicts;
        memset(monster_states, 0, sizeof(monster_states));
        reported_driven = -1;
    }
    release_dead_states();
    if (!cvar_model->string[0])
        return;
    if (!models_tried) {
        models_tried = 1;
        load_models(cvar_model->string);
    }
    if (!model_count || !sv.active || !dq_brain_progs_is_stock())
        return;
    if (!cvar_model || !cvar_max)
        return;

    int cap = cvar_max->value > 0.0f ? (int)cvar_max->value : DQ_BRAIN_MAX_MONSTERS;
    if (cap > DQ_BRAIN_MAX_MONSTERS)
        cap = DQ_BRAIN_MAX_MONSTERS;
    int driven = dq_brain_hook_count();
    if (!seeded) {
        seeded = 1;
        dq_rand_seed((uint32_t)(int)cvar_seed->value);
    }
    for (int i = 1; i < sv.num_edicts && driven < cap; i++) {
        edict_t *ent = EDICT_NUM(i);
        if (ent->free)
            continue;
        int type_index = 0;
        dq_model *model = model_for_class(PR_GetString(ent->v.classname), &type_index);
        if (!model)
            continue;
        if (dq_brain_hook_is_nominated(i))
            continue;
        dq_monster_state *state = free_state();
        if (!state)
            break;
        state->entity_index = i;
        state->model = model;
        state->type_index = type_index;
        state->skill = current_skill < 0 ? 0 : current_skill > 3 ? 3 : current_skill;
        /* The recurrent state starts at zero for every episode, like
         * training; the facing continues from the monster's own. */
        state->yaw = ent->v.angles[YAW];
        const char *error = dq_brain_hook_nominate(i, onnx_policy, state);
        if (error) {
            memset(state, 0, sizeof(*state));
            continue;
        }
        driven++;
    }
    if (driven != reported_driven) {
        reported_driven = driven;
        /* Say which classes are driven and at which type index (#48): one
         * session can serve a mixed level only if each class got its own
         * observation slot. */
        char classes[512];
        int used = 0;
        classes[0] = '\0';
        for (int m = 0; m < model_count; m++)
            for (int c = 0; c < models[m].class_count; c++) {
                int count = 0;
                for (int s = 0; s < DQ_BRAIN_MAX_MONSTERS; s++)
                    if (monster_states[s].entity_index &&
                        monster_states[s].model == &models[m] &&
                        monster_states[s].type_index == models[m].classes[c].type_index)
                        count++;
                if (!count)
                    continue;
                used += q_snprintf(classes + used, sizeof(classes) - used, "%s%s %d (type %d)",
                                   used ? ", " : "", models[m].classes[c].classname, count,
                                   models[m].classes[c].type_index);
            }
        Con_Printf("deepquake: %d monster(s) driven by the model%s%s\n", driven,
                   driven ? (classes[0] ? ": " : "") : " (stock AI)", classes);
    }
}

/* Installed as the brain hook's client tick (see brain_hook.c). */
static void client_tick(void)
{
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
    register_cvars();

    /* Once per engine tick, not once per QuakeC execution: nomination
     * scans the edict list, and the timing line is a whole-tick average. */
    static double last_pass = -1.0;
    if (sv.time == last_pass)
        return;
    last_pass = sv.time;
    nominate_pass();
    static int seconds;
    if (inference_count && (int)sv.time != seconds) {
        seconds = (int)sv.time;
        if (seconds % 10 == 0)
            Con_Printf("deepquake: %d decisions, %.0f us/decision, %d monster(s)\n",
                       inference_count, (double)inference_ns / 1000.0 / inference_count,
                       dq_brain_hook_count());
    }
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
 * probabilities and the greedy actions. A shared model reports the type
 * index each case wore (#48). The Python side (export parity tests) builds
 * the same reports through deepquake.monster and onnxruntime and compares.
 */
static void dq_selftest_make_hook(double time, int visible, dqsim_hook_observation *hook)
{
    memset(hook, 0, sizeof(*hook));
    hook->time = time;
    hook->entity_index = 9;
    hook->origin[0] = time > 0.25 ? 10.0f : 0.0f;
    hook->velocity[0] = 100.0f;
    hook->angles[1] = 90.0f;
    hook->health = 80.0f;
    hook->on_ground = true;
    for (int i = 0; i < DQSIM_HOOK_RAY_PITCHES * DQSIM_HOOK_RAY_YAWS; i++) {
        hook->rays[i] = (float)(i % 7) * 0.1f + 0.05f;
        hook->ray_kinds[i] = (unsigned char)(i % 7);
    }
    hook->attack_cooldown = 0.3f;
    if (visible) {
        hook->visible[0].index = 1;
        hook->visible[0].flags = FL_CLIENT;
        hook->visible[0].relative[0] = 100.0f;
        hook->visible[0].relative[1] = 0.0f;
        hook->visible[0].relative[2] = 24.0f;
        hook->visible[0].health = 60.0f;
        hook->visible[0].distance = 100.0f;
        hook->visible_count = 1;
    }
}

static void dq_selftest_run(const char *model_path, const char *report)
{
    if (!model_count) {
        memset(&models[0], 0, sizeof(models[0]));
        q_strlcpy(models[0].path, model_path, sizeof(models[0].path));
        models[0].nvec[0] = 3;
        models[0].nvec[1] = 3;
        models[0].nvec[2] = 7;
        models[0].nvec[3] = 2;
        models[0].nvec_count = 4;
        read_layout(&models[0]);
        if (load_model(&models[0])) {
            Con_Printf("dq_selftest: could not load %s\n", model_path);
            return;
        }
        model_count = 1;
    }
    dq_monster_state state;
    memset(&state, 0, sizeof(state));
    state.model = &models[0];
    state.skill = 2;
    state.yaw = 90.0f;

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
        /* A shared model sees one body per case (Grunt, Dog, Grunt), so the
         * report covers the type one-hot both ways (#48). */
        if (state.model->type_dim > 0)
            state.type_index = case_index % state.model->type_dim;
        dq_selftest_make_hook(case_index == 0 ? 0.0 : (case_index == 1 ? 0.5 : 6.0),
                              case_index < 2, &hook);
        float obs[DQ_MONSTER_OBS_MAX];
        dq_build_observation(&state, &hook, obs);

        int64_t shape[2] = {1, state.model->obs_dim};
        int64_t hshape[2] = {1, (int64_t)state.model->hidden_dim};
        OrtValue *in_obs = NULL, *in_hidden = NULL;
        api->CreateTensorWithDataAsOrtValue(ort_memory, obs, sizeof(float) * state.model->obs_dim,
                                            shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                            &in_obs);
        api->CreateTensorWithDataAsOrtValue(ort_memory, state.hidden,
                                            sizeof(float) * state.model->hidden_dim, hshape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_hidden);
        const char *input_names[2] = {state.model->name_in_obs, state.model->name_in_hidden};
        const char *output_names[3] = {state.model->name_out_probs, state.model->name_out_value,
                                       state.model->name_out_hidden};
        OrtValue *output_values[3] = {NULL, NULL, NULL};
        const OrtValue *inputs[2] = {in_obs, in_hidden};
        OrtStatus *status = api->Run(state.model->session, NULL, input_names, inputs, 2,
                                    output_names, 3, output_values);
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
        memcpy(state.hidden, new_hidden, sizeof(float) * state.model->hidden_dim);

        int actions[8];
        int offset = 0;
        for (int head = 0; head < state.model->nvec_count; head++) {
            int best = 0;
            for (int i = 1; i < state.model->nvec[head]; i++)
                if (probs[offset + i] > probs[offset + best])
                    best = i;
            actions[head] = best;
            offset += state.model->nvec[head];
        }
        fprintf(out, "case %d type %d\nobs", case_index, state.type_index);
        for (int i = 0; i < state.model->obs_dim; i++)
            fprintf(out, " %.6f", obs[i]);
        fprintf(out, "\nprobs");
        for (int i = 0; i < offset; i++)
            fprintf(out, " %.7f", probs[i]);
        fprintf(out, "\nactions");
        for (int head = 0; head < state.model->nvec_count; head++)
            fprintf(out, " %d", actions[head]);
        fprintf(out, "\n");
        fflush(out);
        inferences++;
        api->ReleaseValue(output_values[0]);
        api->ReleaseValue(output_values[1]);
        api->ReleaseValue(output_values[2]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double ms = (t_end.tv_sec - t_start.tv_sec) * 1e3 + (t_end.tv_nsec - t_start.tv_nsec) / 1e6;
    Con_Printf("dq_selftest: %d inferences in %.2f ms (%.0f us/decision)\n",
               inferences, ms, ms * 1000.0 / inferences);
    fclose(out);
    Con_Printf("dq_selftest: wrote %s\n", report);
}

static cvar_t *adopt_cvar(cvar_t *ours, const char *name)
{
    cvar_t *found = Cvar_FindVar(name);
    if (found)
        return found;
    Cvar_RegisterVariable(ours);
    return ours;
}

static void register_cvars(void)
{
    if (dq_cvars_registered)
        return;
    dq_cvars_registered = 1;
    /* Register ours, then adopt what "+set" asked for: the command line
     * was processed before this registration, and Cvar_Set refuses
     * unknown cvars, so the values live only in host_parms.argv. */
    cvar_model = adopt_cvar(&dq_model_var, "dq_model");
    cvar_style = adopt_cvar(&dq_style_var, "dq_style");
    cvar_seed = adopt_cvar(&dq_seed_var, "dq_seed");
    cvar_cpu = adopt_cvar(&dq_cpu_var, "dq_onnx_cpu");
    cvar_max = adopt_cvar(&dq_max_var, "dq_max_monsters");
    struct {
        const char *name;
        cvar_t *var;
    } const adopted[] = {
        {"dq_model", cvar_model},
        {"dq_style", cvar_style},
        {"dq_seed", cvar_seed},
        {"dq_onnx_cpu", cvar_cpu},
        {"dq_max_monsters", cvar_max},
    };
    for (int i = 1; i < host_parms->argc - 2; i++) {
        if (strcmp(host_parms->argv[i], "+set") != 0)
            continue;
        const char *name = host_parms->argv[i + 1];
        for (size_t k = 0; k < sizeof(adopted) / sizeof(*adopted); k++)
            if (strcmp(name, adopted[k].name) == 0)
                Cvar_Set(name, host_parms->argv[i + 2]);
    }
}

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
 * A model is a shared one (#48) whose input ends with a one-hot of the
 * monster type; "<path>.monsters" names the classes and their type
 * index, and each nominated monster carries its own index into the
 * observation, so one session drives a mixed level.
 *
 * The observation the model consumes is deepquake.monster's (#33), in the
 * layout the retrain spec's part 0D settled: the dq_build_observation
 * function below constructs it, exactly as the Python MonsterObserver does.
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
 * vocabulary width (#48; ten types since the retrain spec's part 0D added
 * the Scrag). A model's observation is the base layout followed by one
 * one-hot entry per type it knows. */
#define DQ_MONSTER_OBS_BASE_DIM 132
#define DQ_MONSTER_MAX_TYPES 10
#define DQ_MONSTER_OBS_MAX (DQ_MONSTER_OBS_BASE_DIM + DQ_MONSTER_MAX_TYPES)
#define DQ_MONSTER_MEMORY_SECONDS 5.0f
/* deepquake.monster_env.MONSTER_SPEED: commanded forward/side speed the
 * policy was trained with (the engine's monster physics cap what moves). */
#define DQ_MONSTER_SPEED 400.0f
/* deepquake.nav.FRAME_SKIP: ticks per decision the policy was trained with
 * (12 decisions/s). The turn action is a delta applied ONCE per decision and
 * the whole action is held for the block; deciding every engine tick instead
 * accumulates the turn ~6x too fast and the monster spins/darts (#61). */
#define DQ_FRAME_SKIP 6
#define DQ_MONSTER_MAX_HIDDEN 256
/* dq_model is a space-separated list; one session per model, and each
 * model names the monster classes it drives in "<path>.monsters". */
#define DQ_MAX_MODELS 4
#define DQ_MAX_CLASSES 10
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
    /* How many monster-type entries the observation ends with (#48): one per
     * type the model knows, 1 to DQ_MONSTER_MAX_TYPES. */
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
    int skill;
    /* The style the policy is asked to play in (aggression, caution,
     * stealth), from dq_style at nomination; training uses its default. */
    float style[3];
    /* Which entry of the model's type one-hot this monster is (#48). */
    int type_index;
    /* Decision cadence (#61): the action is held for DQ_FRAME_SKIP ticks like
     * training. skip counts the holds left before the next inference; the held
     * forward/side/attack are replayed on the holds. held_turn is the turn of
     * the decision in degrees, spread over the block: every tick (decision and
     * hold alike) asks for the facing the hook just reported plus a sixth of
     * it, so a turn QuakeC's attack made is kept (RETRAIN_SPEC part 0A, #68).
     * The split is safe: the hook turns at most DQ_MONSTER_TURN_RATE
     * (360 degrees/s) times the tick - 5 degrees at 72 Hz - and the largest
     * turn, 30 degrees over 6 ticks, is 5 per tick. */
    int skip;
    float held_turn;
    float held_forward;
    float held_side;
    int held_attack;
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
/* The hearing settings of retrain spec part 0B, as cvars. A range of 0 and
 * a run chance below 0 mean the skill table's. */
static cvar_t dq_hearing_var = {"dq_hearing", "1", CVAR_ARCHIVE};
static cvar_t dq_hear_range_var = {"dq_hear_range", "0", CVAR_ARCHIVE};
static cvar_t dq_hear_run_var = {"dq_hear_run", "-1", CVAR_ARCHIVE};
static cvar_t dq_hear_player_var = {"dq_hear_player", "0", CVAR_ARCHIVE};
static cvar_t *cvar_model, *cvar_style, *cvar_seed, *cvar_cpu, *cvar_max;
static cvar_t *cvar_hearing, *cvar_hear_range, *cvar_hear_run, *cvar_hear_player;

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
 * "<classname> <type index>" per line. Without it the model drives the
 * Grunt alone, as the first released models did. */
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
     * The width is the base layout plus one one-hot entry per type the
     * model knows, 1 to DQ_MONSTER_MAX_TYPES of them (#48). The 119-wide
     * single-type models and the 128-wide nine-type ones predate the
     * retrain spec's part 0D and no longer fit. */
    if (model->obs_dim > DQ_MONSTER_OBS_BASE_DIM && model->obs_dim <= DQ_MONSTER_OBS_MAX) {
        model->type_dim = model->obs_dim - DQ_MONSTER_OBS_BASE_DIM;
    } else {
        Con_Printf("deepquake: refusing %s - observation width %d, the client "
                   "expects %d-%d (the layout plus one entry per monster type)\n",
                   model->path, model->obs_dim, DQ_MONSTER_OBS_BASE_DIM + 1, DQ_MONSTER_OBS_MAX);
        return -1;
    }
    return 0;
}

/* A model only drives classes its observation can name: every class needs a
 * type entry the model has (#48). */
static int classes_fit(dq_model *model)
{
    for (int c = 0; c < model->class_count; c++) {
        int index = model->classes[c].type_index;
        int ok = index >= 0 && index < model->type_dim;
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
 * class is. */
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

/* Four channels for a point relative to the monster - a seen enemy, the
 * place it was last seen, a heard noise: the unit vector to it in the
 * monster's frame (forward, left), its height difference / 256 clipped to
 * [-1, 1], and its flat distance / 1024 clipped to 2. */
static void dq_target_channels(float *out, const float *relative, float cos_yaw, float sin_yaw)
{
    float dx = relative[0], dy = relative[1];
    float distance = sqrtf(dx * dx + dy * dy);
    float scale = 1.0f / (distance > 1e-6f ? distance : 1e-6f);
    out[0] = (dx * cos_yaw + dy * sin_yaw) * scale;
    out[1] = (-dx * sin_yaw + dy * cos_yaw) * scale;
    out[2] = relative[2] / 256.0f;
    if (out[2] < -1.0f)
        out[2] = -1.0f;
    if (out[2] > 1.0f)
        out[2] = 1.0f;
    out[3] = distance / 1024.0f;
    if (out[3] > 2.0f)
        out[3] = 2.0f;
}

/* The observation deepquake.monster builds (#33), in the layout the retrain
 * spec's part 0D settled, from a live hook report and this monster's memory.
 * Kept in C so the client needs no Python; python/deepquake/monster.py's
 * docstring holds the table, and the two builders must agree. */
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
    /* Health as a fraction of the monster's own health when nominated. A
     * report without a start health divides by 100, as the Python one does. */
    float start = hook->start_health > 0.0f ? hook->start_health : 100.0f;
    float health = hook->health < 0.0f ? 0.0f : hook->health / start;
    out[at + 4] = health > 1.0f ? 1.0f : health;
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
        dq_target_channels(out + at + 7, enemy->relative, cos_yaw, sin_yaw);
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
            float relative[3];
            for (int k = 0; k < 3; k++)
                relative[k] = state->last_seen[k] - hook->origin[k];
            dq_target_channels(out + at + 13, relative, cos_yaw, sin_yaw);
            out[at + 17] = age / DQ_MONSTER_MEMORY_SECONDS;
            out[at + 18] = 1.0f;
        }
    }

    out[at + 19] = state->skill / 3.0f;
    for (int k = 0; k < 3; k++)
        out[at + 20 + k] = state->style[k];

    /* The last heard noise has channels of its own and never becomes the
     * last-seen place; what kind it was is not an input (part 0D). */
    if (hook->heard) {
        dq_target_channels(out + at + 23, hook->heard_relative, cos_yaw, sin_yaw);
        float age = hook->heard_age < 0.0f ? 0.0f : hook->heard_age / DQSIM_HOOK_HEARD_SECONDS;
        out[at + 27] = age > 1.0f ? 1.0f : age;
        out[at + 28] = 1.0f;
    }
    /* The pace the hook moves the monster at, one-hot: walk, investigate,
     * run. One outside the three lights nothing. */
    if (hook->pace >= DQSIM_PACE_WALK && hook->pace <= DQSIM_PACE_RUN)
        out[at + 29 + hook->pace] = 1.0f;

    /* The nearest visible live monster other than itself (the first such in
     * the hook's nearest-first list), and how many there are. */
    const dqsim_hook_visible *ally = NULL;
    int allies = 0;
    for (int i = 0; i < hook->visible_count; i++) {
        const dqsim_hook_visible *v = &hook->visible[i];
        if (!(v->flags & FL_MONSTER) || v->health <= 0.0f || v->index == hook->entity_index)
            continue;
        if (!ally)
            ally = v;
        allies++;
    }
    if (ally) {
        float bearing[4];
        dq_target_channels(bearing, ally->relative, cos_yaw, sin_yaw);
        /* Its bearing and distance; its height is not an input. */
        out[at + 32] = bearing[0];
        out[at + 33] = bearing[1];
        out[at + 34] = bearing[3];
    }
    float crowd = allies / 4.0f;
    out[at + 35] = crowd > 1.0f ? 1.0f : crowd;

    /* The body the policy is wearing (#48): the observation ends with one
     * one-hot entry per type, and the head is chosen from it. */
    out[at + 36 + state->type_index] = 1.0f;
}

/* The client's policy: the brain-hook callback running this monster's
 * model on this monster's state. */
static void onnx_policy(const dqsim_hook_observation *hook, dqsim_hook_action *action,
                        void *user)
{
    dq_monster_state *state = user;
    /* Hold the last decision for the rest of its block (#61): the policy was
     * trained deciding every DQ_FRAME_SKIP ticks, so re-deciding every engine
     * tick would accumulate the turn delta ~6x too fast and the monster would
     * spin/dart. Replay the held action without inferring. */
    if (state->skip > 0) {
        state->skip--;
        action->desired_yaw = hook->angles[1] + state->held_turn / DQ_FRAME_SKIP;
        action->forward = state->held_forward;
        action->side = state->held_side;
        action->attack = state->held_attack;
        return;
    }
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

    state->held_turn = model->turn[actions[2]];
    action->desired_yaw = hook->angles[1] + state->held_turn / DQ_FRAME_SKIP;
    action->forward = (actions[0] - 1) * DQ_MONSTER_SPEED;
    action->side = (1 - actions[1]) * DQ_MONSTER_SPEED;
    action->attack = actions[3] != 0;
    /* Hold this decision for the block, as in training (#61). */
    state->held_forward = action->forward;
    state->held_side = action->side;
    state->held_attack = action->attack;
    state->skip = DQ_FRAME_SKIP - 1;
}

static void register_cvars(void);
static void dq_selftest_run(const char *model, const char *report);

/* The style a nominated monster plays with, from dq_style: three numbers
 * (aggression, caution, stealth), each clipped to [0, 1]. One the cvar does
 * not give, or does not give as a number, stays at 0.5, the value training
 * uses. */
static void parse_style(const char *text, float style[3])
{
    const char *cursor = text;
    for (int k = 0; k < 3; k++) {
        style[k] = 0.5f;
        const char *before = cursor;
        float value = parse_float(&cursor);
        if (cursor == before)
            return;
        if (value >= 0.0f && value <= 1.0f)
            style[k] = value;
        else if (value < 0.0f)
            style[k] = 0.0f;
        else if (value > 1.0f)
            style[k] = 1.0f;
    }
}

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
        /* The client loads maps through the console, so no load seed reaches
         * it: the hearing stream seeds from the map name (part 0B), keeping
         * the paces deterministic per map. */
        dq_hear_rng_seed(dq_fnv1a(DQ_FNV1A_SEED, sv.name, strlen(sv.name)));
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
        parse_style(cvar_style->string, state->style);
        /* The recurrent state starts at zero for every episode, like training.
         * No facing is carried: every turn is measured from the facing the
         * hook reports (RETRAIN_SPEC part 0A, #68). */
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
    /* The hearing settings, every tick: what a cvar leaves at its default
     * (a range of 0, a run chance below 0) is the skill table's. */
    dqsim_hearing hearing = {
        cvar_hearing->value != 0.0f,
        cvar_hear_range->value > 0.0f ? cvar_hear_range->value : 0.0f,
        cvar_hear_run->value < 0.0f   ? -1.0f
        : cvar_hear_run->value > 1.0f ? 1.0f
                                      : cvar_hear_run->value,
        cvar_hear_player->value != 0.0f,
    };
    dq_brain_set_hearing(&hearing);
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
    hook->start_health = 100.0f;
    hook->on_ground = true;
    hook->pace = DQSIM_PACE_WALK;
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
    if (time == 0.5) {
        /* The second case also sees a live monster besides the player, hears
         * a noise, and is investigating: the part 0D channels. */
        hook->visible[1].index = 7;
        hook->visible[1].flags = FL_MONSTER;
        hook->visible[1].relative[0] = -60.0f;
        hook->visible[1].relative[1] = 90.0f;
        hook->visible[1].relative[2] = -12.0f;
        hook->visible[1].health = 30.0f;
        hook->visible[1].distance = 110.0f;
        hook->visible_count = 2;
        hook->heard = 1;
        hook->heard_kind = DQSIM_NOISE_EXPLOSION;
        hook->heard_relative[0] = -300.0f;
        hook->heard_relative[1] = 200.0f;
        hook->heard_relative[2] = 48.0f;
        hook->heard_age = 2.5f;
        hook->pace = DQSIM_PACE_INVESTIGATE;
    } else if (time > 0.5) {
        hook->pace = DQSIM_PACE_RUN;
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
    /* The style the Python side of the cross-check uses; dq_style is not
     * read here. */
    state.style[0] = 0.25f;
    state.style[1] = 0.5f;
    state.style[2] = 0.75f;

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
        /* A shared model sees one body per case (Grunt, Dog, Enforcer), so
         * the report covers the type one-hot (#48). */
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
    cvar_hearing = adopt_cvar(&dq_hearing_var, "dq_hearing");
    cvar_hear_range = adopt_cvar(&dq_hear_range_var, "dq_hear_range");
    cvar_hear_run = adopt_cvar(&dq_hear_run_var, "dq_hear_run");
    cvar_hear_player = adopt_cvar(&dq_hear_player_var, "dq_hear_player");
    struct {
        const char *name;
        cvar_t *var;
    } const adopted[] = {
        {"dq_model", cvar_model},
        {"dq_style", cvar_style},
        {"dq_seed", cvar_seed},
        {"dq_onnx_cpu", cvar_cpu},
        {"dq_max_monsters", cvar_max},
        {"dq_hearing", cvar_hearing},
        {"dq_hear_range", cvar_hear_range},
        {"dq_hear_run", cvar_hear_run},
        {"dq_hear_player", cvar_hear_player},
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

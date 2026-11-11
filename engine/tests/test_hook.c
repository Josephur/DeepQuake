/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * The brain hook (#31) on a real map: a scripted policy drives one Grunt
 * (move, then attack the player it can see), deterministically. Needs game
 * data: pass the directory containing id1/ as the first argument. Exits
 * with 77 (skipped) when that directory has no id1/pak0.pak.
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dqsim/dqsim.h"

#define CHECK(condition, ...)                                                  \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int calls;
static int saw_player;
static int mode; /* 0 locate, 1 run forward, 2 attack */
static float start[3];
static float monster_yaw;

static void policy(const dqsim_hook_observation *obs, dqsim_hook_action *action, void *user)
{
    (void)user;
    if (calls == 0)
        memcpy(start, obs->origin, sizeof(start));
    calls++;
    monster_yaw = obs->angles[1];
    for (int i = 0; i < obs->visible_count; i++)
        if (obs->visible[i].flags & 8) /* FL_CLIENT */
            saw_player = 1;

    if (mode == 1) {
        action->forward = 200.0f;
    } else if (mode == 2) {
        for (int i = 0; i < obs->visible_count; i++) {
            if (obs->visible[i].flags & 8) {
                action->desired_yaw =
                    atan2f(obs->visible[i].relative[1], obs->visible[i].relative[0]) *
                    180.0f / 3.14159265f;
                action->attack = 1;
            }
        }
    }
}

static int find_grunt(dqsim_entity *entities, int count, const float *near)
{
    int best = -1;
    float best_distance = 1e30f;
    for (int i = 0; i < count; i++) {
        if (strcmp(entities[i].classname, "monster_army") != 0)
            continue;
        float d = 0;
        for (int k = 0; k < 3; k++) {
            float delta = entities[i].origin[k] - near[k];
            d += delta * delta;
        }
        if (d < best_distance) {
            best_distance = d;
            best = i;
        }
    }
    return best;
}

/* One full scenario; returns the end state hash, or 0 on a failed CHECK. */
static uint64_t scenario(void)
{
    dqsim_player_state player;
    static dqsim_entity entities[1024];
    const dqsim_action idle = {0};
    /* The CHECK macro returns from this function with 0; the caller
     * distinguishes that from a hash because a hash of 0 is not used. */
    CHECK(dqsim_load_map("e1m1", 0, 7) == DQSIM_OK, "load e1m1");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get player");
    int count = dqsim_get_entities(entities, 1024);
    CHECK(count > 0, "get entities");
    int grunt = find_grunt(entities, count, player.origin);
    CHECK(grunt >= 0, "no monster_army on e1m1");

    calls = 0;
    saw_player = 0;
    mode = 0;
    CHECK(dqsim_set_policy_hook(entities[grunt].index, policy, NULL) == DQSIM_OK,
          "nominate grunt: %s", dqsim_last_error());
    /* Nomination refuses the player and a missing entity. */
    CHECK(dqsim_set_policy_hook(1, policy, NULL) == DQSIM_ERROR, "player nomination accepted");
    CHECK(dqsim_set_policy_hook(10000, policy, NULL) == DQSIM_ERROR, "bogus nomination accepted");

    mode = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 3) == DQSIM_OK, "run steps");
    CHECK(calls > 100, "policy called %d times in 3 s", calls);
    count = dqsim_get_entities(entities, 1024);
    float moved = 0;
    for (int i = 0; i < count; i++)
        if (entities[i].index == entities[grunt].index) {
            for (int k = 0; k < 2; k++) {
                float delta = entities[i].origin[k] - start[k];
                moved += delta * delta;
            }
        }
    moved = sqrtf(moved);
    CHECK(moved > 40.0f, "grunt moved %.1f units in 3 s", moved);

    /* Stand the player in front of the grunt and let the policy shoot. */
    float rad = monster_yaw * 3.14159265f / 180.0f;
    float spot[3] = {start[0] + cosf(rad) * 140.0f, start[1] + sinf(rad) * 140.0f,
                     start[2] + 24.0f};
    CHECK(dqsim_get_entities(entities, 1024) > 0, "entities again");
    CHECK(dqsim_teleport_player(spot, monster_yaw + 180.0f) == DQSIM_OK,
          "teleport player in front: %s", dqsim_last_error());
    mode = 2;
    calls = 0;
    int hurt = 0;
    for (int second = 0; second < 15 && !hurt; second++) {
        CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "attack steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player after attack");
        hurt = player.health < 100.0f;
    }
    CHECK(saw_player, "the grunt never saw the player");
    CHECK(hurt, "player health still %.0f after 15 s of attacks", player.health);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return dqsim_state_hash();
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: test_hook <gamedata-dir>\n");
        return 77;
    }
    char pak[1024];
    snprintf(pak, sizeof(pak), "%s/id1/pak0.pak", argv[1]);
    FILE *f = fopen(pak, "rb");
    if (!f)
        return 77;
    fclose(f);

    const dqsim_config config = {.basedir = argv[1]};
    if (dqsim_init(&config) != DQSIM_OK) {
        fprintf(stderr, "dqsim_init: %s\n", dqsim_last_error());
        return 1;
    }

    uint64_t first = scenario();
    if (!first)
        return 1;
    uint64_t second = scenario();
    if (!second)
        return 1;
    CHECK(first == second, "same seed and inputs gave different hashes");

    printf("hook scenario hash %" PRIu64 "\n", first);
    dqsim_shutdown();
    return 0;
}

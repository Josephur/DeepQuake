/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * End-to-end test of the simulator on a real map. Needs game data: pass the
 * directory containing id1/ as the first argument. Exits with 77 (skipped)
 * when that directory has no id1/pak0.pak.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dqsim/dqsim.h"

#define TICKS 1000

#define CHECK(condition, ...)                                                  \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n  last error: %s\n", dqsim_last_error());       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* A fixed, varied input sequence: run, turn, jump, shoot. */
static dqsim_action action_for_tick(int tick)
{
    dqsim_action action = {0};
    action.forward = 400;
    action.side = (tick / 97) % 2 ? 200 : -200;
    action.yaw = 90.0f + (float)(tick % 360);
    action.pitch = (float)((tick % 40) - 20);
    if (tick % 50 == 0)
        action.buttons |= DQSIM_BUTTON_JUMP;
    if (tick % 7 == 0)
        action.buttons |= DQSIM_BUTTON_ATTACK;
    return action;
}

static void run_episode(const char *map, uint32_t seed, uint64_t *hashes)
{
    CHECK(dqsim_load_map(map, 1, seed) == DQSIM_OK, "load_map(%s)", map);
    for (int tick = 0; tick < TICKS; tick++) {
        dqsim_action action = action_for_tick(tick);
        CHECK(dqsim_step(&action) == DQSIM_OK, "step %d", tick);
        hashes[tick] = dqsim_state_hash();
    }
}

int main(int argc, char **argv)
{
    static uint64_t first[TICKS], second[TICKS], other_seed[TICKS];
    char pak[2048];
    dqsim_player_state player;
    dqsim_action idle = {0};

    if (argc != 2) {
        fprintf(stderr, "usage: %s <gamedata-dir>\n", argv[0]);
        return 2;
    }
    snprintf(pak, sizeof(pak), "%s/id1/pak0.pak", argv[1]);
    FILE *f = fopen(pak, "rb");
    if (!f) {
        fprintf(stderr, "SKIP: no game data at %s\n", pak);
        return 77;
    }
    fclose(f);

    /* Calls before init fail cleanly. */
    CHECK(dqsim_step(&idle) == DQSIM_ERROR, "step before init must fail");
    CHECK(dqsim_time() < 0, "time before init must be negative");

    dqsim_config config = {.basedir = argv[1]};
    CHECK(dqsim_init(&config) == DQSIM_OK, "init");
    CHECK(dqsim_init(&config) == DQSIM_ERROR, "second init must fail");
    CHECK(dqsim_step(&idle) == DQSIM_ERROR, "step before load_map must fail");

    /* A missing map is an ordinary error, not a fatal one. */
    CHECK(dqsim_load_map("no_such_map", 1, 0) == DQSIM_ERROR, "missing map must fail");
    CHECK(dqsim_load_map("bad name", 1, 0) == DQSIM_ERROR, "invalid name must fail");

    /* The player spawns at e1m1's info_player_start with full health. */
    CHECK(dqsim_load_map("e1m1", 1, 1) == DQSIM_OK, "load e1m1");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get_player");
    CHECK(fabsf(player.origin[0] - 480) < 1 && fabsf(player.origin[1] + 352) < 1,
          "spawn at (%.1f %.1f %.1f), expected (480 -352 ~88)", player.origin[0],
          player.origin[1], player.origin[2]);
    CHECK(player.health == 100, "health %.0f", player.health);

    /* Time advances by exactly one tick per step. */
    double before = dqsim_time();
    CHECK(dqsim_step(&idle) == DQSIM_OK, "step");
    CHECK(fabs(dqsim_time() - before - DQSIM_DEFAULT_TICK_SECONDS) < 1e-6, "tick length");

    /* Running forward moves the player. */
    dqsim_action forward = {.forward = 400, .yaw = 90};
    for (int i = 0; i < 72; i++)
        CHECK(dqsim_step(&forward) == DQSIM_OK, "step");
    dqsim_player_state moved;
    CHECK(dqsim_get_player(&moved) == DQSIM_OK, "get_player");
    CHECK(moved.origin[1] - player.origin[1] > 200, "moved only %.1f units in one second",
          moved.origin[1] - player.origin[1]);

    /*
     * Dying and pressing attack makes the game restart the level. Stepping
     * must carry on, and report it. On e1m3 a player who runs in circles
     * firing is killed well within the tick budget.
     */
    CHECK(dqsim_load_map("e1m3", 3, 5) == DQSIM_OK, "load e1m3");
    CHECK(strcmp(dqsim_map_name(), "e1m3") == 0, "map name '%s'", dqsim_map_name());
    CHECK(dqsim_level_count() == 0, "level count starts at 0");
    int died = 0;
    for (int tick = 0; tick < 40000 && dqsim_level_count() == 0; tick++) {
        dqsim_action action = {.forward = 400, .yaw = (float)(tick % 360)};
        if (tick % 5 == 0)
            action.buttons = DQSIM_BUTTON_ATTACK;
        CHECK(dqsim_step(&action) == DQSIM_OK, "step %d", tick);
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "get_player");
        if (player.health <= 0)
            died = 1;
    }
    CHECK(died, "the player never died on e1m3");
    CHECK(dqsim_level_count() == 1, "level did not restart after death");
    CHECK(dqsim_get_player(&player) == DQSIM_OK && player.health == 100,
          "health after restart is %.0f", player.health);

    /* Determinism: same seed and inputs give the same state at every tick. */
    run_episode("e1m1", 1234, first);
    run_episode("e1m1", 1234, second);
    for (int tick = 0; tick < TICKS; tick++)
        CHECK(first[tick] == second[tick], "state diverged at tick %d", tick);

    /* The seed matters: a different seed changes the trajectory. */
    run_episode("e1m1", 99, other_seed);
    CHECK(memcmp(first, other_seed, sizeof(first)) != 0, "seed had no effect");

    /* The final hash is printed so a second process can be compared. */
    printf("final-hash %016llx\n", (unsigned long long)first[TICKS - 1]);

    dqsim_shutdown();
    CHECK(dqsim_step(&idle) == DQSIM_ERROR, "step after shutdown must fail");
    return 0;
}

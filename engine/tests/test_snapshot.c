/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Snapshot and restore. Restoring a snapshot and replaying the same actions
 * must reproduce the same state hashes as the original run, at every tick.
 *
 * Needs game data: pass the directory containing id1/ as the first argument.
 * Exits with 77 (skipped) when that directory has no id1/pak0.pak.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dqsim/dqsim.h"

/* Ticks before the mid-episode snapshot, and ticks replayed after it. */
#define BEFORE 200
#define AFTER 400

#define CHECK(condition, ...)                                                  \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n  last error: %s\n", dqsim_last_error());       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* The same fixed, varied sequence test_sim.c replays: run, strafe, turn,
 * look up and down, jump now and then, and fire every seventh tick. */
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

/* Step `count` ticks starting at absolute tick `first`, recording hashes. */
static void run_ticks(int first, int count, uint64_t *hashes)
{
    for (int i = 0; i < count; i++) {
        dqsim_action action = action_for_tick(first + i);
        CHECK(dqsim_step(&action) == DQSIM_OK, "step %d", first + i);
        hashes[i] = dqsim_state_hash();
    }
}

static void compare_hashes(const uint64_t *expected, const uint64_t *actual, int count,
                           const char *what)
{
    for (int i = 0; i < count; i++)
        CHECK(expected[i] == actual[i], "%s diverged at replay tick %d", what, i);
}

int main(int argc, char **argv)
{
    static uint64_t original[AFTER], replay[AFTER];

    if (argc != 2) {
        fprintf(stderr, "usage: %s <gamedata-dir>\n", argv[0]);
        return 2;
    }
    char pak[2048];
    snprintf(pak, sizeof(pak), "%s/id1/pak0.pak", argv[1]);
    FILE *f = fopen(pak, "rb");
    if (!f) {
        fprintf(stderr, "SKIP: no game data at %s\n", pak);
        return 77;
    }
    fclose(f);

    /* No snapshot can be taken before the engine runs. */
    dqsim_snapshot *early = NULL;
    CHECK(dqsim_snapshot_save(&early) == DQSIM_ERROR, "save before init must fail");
    CHECK(early == NULL, "a failed save must not return a snapshot");
    CHECK(dqsim_snapshot_size(NULL) == 0, "size of a NULL snapshot must be 0");

    dqsim_config config = {.basedir = argv[1]};
    CHECK(dqsim_init(&config) == DQSIM_OK, "init");
    CHECK(dqsim_snapshot_save(NULL) == DQSIM_ERROR, "save with NULL out must fail");
    CHECK(dqsim_snapshot_restore(NULL) == DQSIM_ERROR, "restore of NULL must fail");
    CHECK(dqsim_snapshot_save(&early) == DQSIM_ERROR, "save before load_map must fail");

    /* --- Episode start: the snapshot a fast reset would keep. --- */
    CHECK(dqsim_load_map("e1m1", 1, 1234) == DQSIM_OK, "load e1m1");
    dqsim_snapshot *start = NULL;
    CHECK(dqsim_snapshot_save(&start) == DQSIM_OK, "save at episode start");
    CHECK(start != NULL && dqsim_snapshot_size(start) > 0, "empty snapshot");
    uint64_t start_hash = dqsim_state_hash();
    double start_time = dqsim_time();

    run_ticks(0, AFTER, original);

    CHECK(dqsim_snapshot_restore(start) == DQSIM_OK, "restore episode start");
    CHECK(dqsim_state_hash() == start_hash, "hash after restore differs from the snapshot");
    CHECK(dqsim_time() == start_time, "time after restore is %.9f, expected %.9f", dqsim_time(),
          start_time);

    run_ticks(0, AFTER, replay);
    compare_hashes(original, replay, AFTER, "episode-start replay");

    /* Restoring the same snapshot again must give the same state again. */
    CHECK(dqsim_snapshot_restore(start) == DQSIM_OK, "second restore of the same snapshot");
    CHECK(dqsim_state_hash() == start_hash, "second restore gave a different hash");
    dqsim_snapshot_free(start);

    /* --- Mid-episode, after combat has started: the harder case. --- */
    CHECK(dqsim_load_map("e1m1", 1, 1234) == DQSIM_OK, "reload e1m1");
    run_ticks(0, BEFORE, replay);
    dqsim_snapshot *mid = NULL;
    CHECK(dqsim_snapshot_save(&mid) == DQSIM_OK, "save mid-episode");
    uint64_t mid_hash = dqsim_state_hash();
    double mid_time = dqsim_time();

    run_ticks(BEFORE, AFTER, original);

    CHECK(dqsim_snapshot_restore(mid) == DQSIM_OK, "restore mid-episode");
    CHECK(dqsim_state_hash() == mid_hash, "mid-episode hash after restore differs");
    CHECK(dqsim_time() == mid_time, "mid-episode time after restore is %.9f, expected %.9f",
          dqsim_time(), mid_time);

    run_ticks(BEFORE, AFTER, replay);
    compare_hashes(original, replay, AFTER, "mid-episode replay");

    /* A snapshot taken after a restore restores too, and two snapshots of the
     * same state are identical in size. */
    dqsim_snapshot *again = NULL, *same = NULL;
    CHECK(dqsim_snapshot_save(&again) == DQSIM_OK, "save after the replay");
    CHECK(dqsim_snapshot_save(&same) == DQSIM_OK, "save the same state again");
    CHECK(dqsim_snapshot_size(again) == dqsim_snapshot_size(same),
          "two snapshots of one state differ in size: %zu vs %zu", dqsim_snapshot_size(again),
          dqsim_snapshot_size(same));
    CHECK(dqsim_snapshot_restore(same) == DQSIM_OK, "restore the second snapshot");
    CHECK(dqsim_state_hash() == original[AFTER - 1], "second snapshot restored the wrong state");
    CHECK(dqsim_snapshot_restore(again) == DQSIM_OK, "restore the first snapshot again");
    CHECK(dqsim_state_hash() == original[AFTER - 1], "first snapshot restored the wrong state");

    /* --- A snapshot from another episode must be refused, not applied. --- */
    CHECK(dqsim_load_map("e1m1", 1, 99) == DQSIM_OK, "start another episode");
    uint64_t untouched = dqsim_state_hash();
    CHECK(dqsim_snapshot_restore(mid) == DQSIM_ERROR, "restore across episodes must fail");
    CHECK(strstr(dqsim_last_error(), "different") != NULL,
          "unclear cross-episode error: %s", dqsim_last_error());
    CHECK(dqsim_state_hash() == untouched, "a refused restore changed the state");

    /* --- A death restart is a new episode too, and bumps the serial by itself. --- */
    CHECK(dqsim_load_map("e1m1", 1, 1234) == DQSIM_OK, "load for the death test");
    dqsim_snapshot *before_death = NULL;
    CHECK(dqsim_snapshot_save(&before_death) == DQSIM_OK, "save before death");
    CHECK(dqsim_client_command("kill") == DQSIM_OK, "kill the player");
    dqsim_action attack = {0};
    attack.buttons = DQSIM_BUTTON_ATTACK;
    int restarted = 0;
    for (int i = 0; i < 60 && !restarted; i++) {
        CHECK(dqsim_step(&attack) == DQSIM_OK, "step after death %d", i);
        restarted = dqsim_level_count() > 0;
    }
    CHECK(restarted, "the level did not restart after the player died");
    CHECK(dqsim_snapshot_restore(before_death) == DQSIM_ERROR,
          "restore after a death restart must fail");
    dqsim_snapshot_free(before_death);

    /* --- Snapshots are plain memory: they outlive the simulator. --- */
    dqsim_snapshot *last = NULL;
    CHECK(dqsim_snapshot_save(&last) == DQSIM_OK, "save before shutdown");
    dqsim_shutdown();
    CHECK(dqsim_snapshot_restore(last) == DQSIM_ERROR, "restore after shutdown must fail");
    CHECK(dqsim_snapshot_restore(mid) == DQSIM_ERROR, "restore of an older snapshot after shutdown");
    dqsim_snapshot_free(last);
    dqsim_snapshot_free(same);
    dqsim_snapshot_free(again);
    dqsim_snapshot_free(mid);
    return 0;
}

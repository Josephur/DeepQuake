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

    /* Entities: the level contains the player, monsters and items. */
    static dqsim_entity entities[1024];
    int n_entities = dqsim_get_entities(entities, 1024);
    CHECK(n_entities > 50 && n_entities <= 1024, "entity count %d", n_entities);
    CHECK(strcmp(entities[0].classname, "worldspawn") == 0, "entity 0 is '%s'",
          entities[0].classname);
    CHECK(strcmp(entities[1].classname, "player") == 0, "entity 1 is '%s'",
          entities[1].classname);
    int monsters = 0;
    const dqsim_entity *deathmatch_spot = NULL;
    for (int i = 0; i < n_entities; i++) {
        if (entities[i].flags & 32)
            monsters++;
        if (strcmp(entities[i].classname, "info_player_deathmatch") == 0)
            deathmatch_spot = &entities[i];
    }
    CHECK(monsters > 5, "only %d monsters on e1m1", monsters);
    CHECK(deathmatch_spot != NULL, "no deathmatch spawn point on e1m1");

    /* Teleport: allowed to a spawn point, refused inside a wall. */
    float inside_wall[3] = {100000, 100000, 100000};
    CHECK(dqsim_teleport_player(inside_wall, 0) == DQSIM_ERROR, "teleport into void must fail");
    CHECK(dqsim_teleport_player(deathmatch_spot->origin, 45) == DQSIM_OK, "teleport");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get_player");
    CHECK(fabsf(player.origin[0] - deathmatch_spot->origin[0]) < 0.01f, "teleport position");

    /* Rays: fractions in [0, 1]; in a level some hit walls and some do not.
     * Kinds are one of the seven surface values; a blocked ray is never
     * NOTHING, an unblocked one never reports a surface. */
    const float pitches[2] = {0, 30};
    float rays[2 * 16];
    unsigned char kinds[2 * 16];
    CHECK(dqsim_ray_fan(pitches, 2, 16, 512, rays, kinds) == DQSIM_OK, "ray_fan");
    int blocked = 0;
    for (int i = 0; i < 32; i++) {
        CHECK(rays[i] >= 0 && rays[i] <= 1, "ray %d fraction %f", i, rays[i]);
        CHECK(kinds[i] <= DQSIM_SURFACE_SKY, "ray %d kind %d", i, kinds[i]);
        if (rays[i] < 1) {
            blocked++;
            CHECK(kinds[i] != DQSIM_SURFACE_NOTHING, "blocked ray %d hit nothing", i);
        } else {
            CHECK(kinds[i] == DQSIM_SURFACE_NOTHING, "clear ray %d kind %d", i, kinds[i]);
        }
    }
    CHECK(blocked >= 16, "only %d of 32 rays hit anything", blocked);
    /* Every downward ray must hit the floor well before 512 units. */
    for (int i = 16; i < 32; i++)
        CHECK(rays[i] < 0.9f, "downward ray %d did not hit the floor", i);

    /*
     * Ray kinds at fixed spots on fixed maps (#26): a single ray along the
     * view yaw from a teleported position, measured once on this build and
     * hard-coded; the engine treats sky like water (it passes through the
     * volume), so all three hazard kinds come from the contents march.
     */
    static const struct {
        const char *map;
        float at[3];
        float pitch;
        int kind;
    } kind_spots[] = {
        /* e1m1's opening corridor walls; its nether slime pool. */
        {"e1m1", {480, -352, 89}, 0, DQSIM_SURFACE_WORLD},
        {"e1m1", {932, 2236, 24}, 70, DQSIM_SURFACE_SLIME},
        /* A shut door on e1m1 (a moving brush, not world geometry). */
        {"e1m1", {52, 564, 96}, 0, DQSIM_SURFACE_BRUSH},
        /* e1m2's moat, entered at the surface (met before the floor). */
        {"e1m2", {800, 368, 339}, 30, DQSIM_SURFACE_WATER},
        /* e1m5's courtyard sky, reached from below. */
        {"e1m5", {296, 2500, 480}, -80, DQSIM_SURFACE_SKY},
        /* e1m7's lake, from the platform above it. */
        {"e1m7", {304, 64, -48}, 30, DQSIM_SURFACE_LAVA},
    };
    for (size_t s = 0; s < sizeof(kind_spots) / sizeof(*kind_spots); s++) {
        CHECK(dqsim_load_map(kind_spots[s].map, 0, 1) == DQSIM_OK,
              "load %s", kind_spots[s].map);
        CHECK(dqsim_teleport_player(kind_spots[s].at, 0) == DQSIM_OK,
              "teleport to %s spot", kind_spots[s].map);
        float one_ray;
        unsigned char kind = 0xff;
        CHECK(dqsim_ray_fan(&kind_spots[s].pitch, 1, 1, 2048, &one_ray, &kind) == DQSIM_OK,
              "kind ray on %s", kind_spots[s].map);
        CHECK(kind == kind_spots[s].kind, "%s ray kind %d, wanted %d (fraction %f)",
              kind_spots[s].map, kind, kind_spots[s].kind, one_ray);
        CHECK(one_ray > 0 && one_ray < 1, "%s ray fraction %f", kind_spots[s].map, one_ray);
    }

    /* Client commands: god mode sets the flag on the player. */
    CHECK(dqsim_client_command("god") == DQSIM_OK, "client_command");
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "step_repeat");
    CHECK(dqsim_get_entities(entities, 2) > 1 && (entities[1].flags & 64), "god mode not set");

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

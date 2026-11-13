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

static int find_monster(dqsim_entity *entities, int count, const char *classname,
                        const float *near)
{
    int best = -1;
    float best_distance = 1e30f;
    for (int i = 0; i < count; i++) {
        if (strcmp(entities[i].classname, classname) != 0)
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
    int grunt = find_monster(entities, count, "monster_army", player.origin);
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

static int seen_multi[3];
static int multi_ids[3] = {0, 1, 2};
static float multi_start[3][3];
static int multi_started[3];
static int multi_forward;

static void multi_policy(const dqsim_hook_observation *obs, dqsim_hook_action *action,
                         void *user)
{
    int id = *(int *)user;
    seen_multi[id]++;
    if (!multi_started[id]) {
        multi_started[id] = 1;
        memcpy(multi_start[id], obs->origin, sizeof(multi_start[id]));
    }
    if (multi_forward)
        action->forward = 200.0f;
}

/* Many monsters at once (M6): three Grunts nominated together, each
 * driven through its own callback and user pointer, the state hash
 * covering all of them, and a snapshot carrying each one's handicap. */
static uint64_t multi_scenario(void)
{
    static dqsim_entity entities[1024];
    const dqsim_action idle = {0};
    CHECK(dqsim_load_map("e1m1", 0, 11) == DQSIM_OK, "load e1m1");
    int count = dqsim_get_entities(entities, 1024);
    int chosen[3], chosen_count = 0;
    for (int i = 0; i < count && chosen_count < 3; i++)
        if (strcmp(entities[i].classname, "monster_army") == 0)
            chosen[chosen_count++] = entities[i].index;
    CHECK(chosen_count == 3, "only %d Grunts on e1m1", chosen_count);

    memset(seen_multi, 0, sizeof(seen_multi));
    memset(multi_started, 0, sizeof(multi_started));
    for (int k = 0; k < 3; k++)
        CHECK(dqsim_set_policy_hook(chosen[k], multi_policy, &multi_ids[k]) == DQSIM_OK,
              "nominate %d: %s", chosen[k], dqsim_last_error());
    CHECK(dqsim_set_policy_hook(chosen[1], multi_policy, &multi_ids[1]) == DQSIM_ERROR,
          "the same entity was nominated twice");

    multi_forward = 0;
    CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "warm-up");
    for (int k = 0; k < 3; k++)
        CHECK(seen_multi[k] > 0, "monster %d was never asked for a decision", chosen[k]);

    /* Each monster's own callback moves it. */
    multi_forward = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "drive all three");
    count = dqsim_get_entities(entities, 1024);
    for (int k = 0; k < 3; k++) {
        int found = 0;
        float moved = 0;
        for (int i = 0; i < count; i++)
            if (entities[i].index == chosen[k]) {
                found = 1;
                for (int axis = 0; axis < 2; axis++) {
                    float delta = entities[i].origin[axis] - multi_start[k][axis];
                    moved += delta * delta;
                }
            }
        CHECK(found, "monster %d vanished", chosen[k]);
        CHECK(sqrtf(moved) > 20.0f, "monster %d moved %.1f units", chosen[k], sqrtf(moved));
    }

    /* The snapshot carries every nominated monster's handicap state. */
    uint64_t before = dqsim_state_hash();
    dqsim_snapshot *snap = NULL;
    CHECK(dqsim_snapshot_save(&snap) == DQSIM_OK, "save: %s", dqsim_last_error());
    CHECK(dqsim_step_repeat(&idle, 36) == DQSIM_OK, "step after save");
    CHECK(dqsim_state_hash() != before, "the state did not change");
    CHECK(dqsim_snapshot_restore(snap) == DQSIM_OK, "restore: %s", dqsim_last_error());
    CHECK(dqsim_state_hash() == before, "restore did not reproduce the hash");
    dqsim_snapshot_free(snap);

    /* ... and it is refused once the set of nominations differs. */
    snap = NULL;
    CHECK(dqsim_snapshot_save(&snap) == DQSIM_OK, "save again: %s", dqsim_last_error());
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    CHECK(dqsim_snapshot_restore(snap) == DQSIM_ERROR,
          "a snapshot with a different nomination set was accepted");
    dqsim_snapshot_free(snap);
    return dqsim_state_hash();
}

/* The Dog (#47): a melee monster. Like the Grunt's "army_atk1", its attack
 * entry is the function the spawn built-in puts in "th_melee" - "dog_atta1",
 * the first frame of a lunge that bites from dog_bite in dog_atta4 and ends
 * in dog_run1. The bite, not the hook, checks the range, so the dog may
 * close distance with it but must not damage a player it is not next to.
 *
 * Each scenario loads the map itself: a dog left in the middle of its bite
 * chain is not asked for decisions, so its observation is stale until the
 * chain returns to a stand state. */
static int dog_mode; /* 0 idle, 1 forward, 2 turn, 3 attack the player */
static int dog_calls;
static int dog_saw_player;
static float dog_yaw;
static float dog_origin[3];
static float dog_turn_to;

static void dog_policy(const dqsim_hook_observation *obs, dqsim_hook_action *action, void *user)
{
    (void)user;
    dog_calls++;
    memcpy(dog_origin, obs->origin, sizeof(dog_origin));
    dog_yaw = obs->angles[1];
    dog_saw_player = 0;
    for (int i = 0; i < obs->visible_count; i++)
        if (obs->visible[i].flags & 8) /* FL_CLIENT */
            dog_saw_player = 1;
    action->desired_yaw = obs->angles[1];
    if (dog_mode == 1) {
        action->forward = 200.0f;
    } else if (dog_mode == 2) {
        action->desired_yaw = dog_turn_to;
    } else if (dog_mode == 3) {
        for (int i = 0; i < obs->visible_count; i++) {
            if (obs->visible[i].flags & 8) { /* FL_CLIENT */
                action->desired_yaw =
                    atan2f(obs->visible[i].relative[1], obs->visible[i].relative[0]) * 180.0f /
                    3.14159265f;
                action->attack = 1;
            }
        }
    }
}

/* Does nothing: a nominated monster gets its decisions from here, so its
 * stock AI never runs and it neither moves nor attacks. */
static void frozen_policy(const dqsim_hook_observation *obs, dqsim_hook_action *action,
                          void *user)
{
    (void)obs;
    (void)action;
    (void)user;
}

/* Nominate every monster but `keep` with frozen_policy: the other monsters
 * of a live map would otherwise shoot the standing player and be counted as
 * the dog's damage. */
static int freeze_monsters_except(int keep)
{
    static dqsim_entity entities[2048];
    int count = dqsim_get_entities(entities, 2048);
    int frozen = 0;
    for (int i = 0; i < count; i++) {
        if (!(entities[i].flags & 32) || entities[i].index == keep) /* FL_MONSTER */
            continue;
        CHECK(dqsim_set_policy_hook(entities[i].index, frozen_policy, NULL) == DQSIM_OK,
              "freeze entity %d: %s", entities[i].index, dqsim_last_error());
        frozen++;
    }
    return frozen;
}

/* The dog with the least clutter around it: the observation reports only the
 * six nearest entities (deepquake.monster), so a dog standing among items
 * would not see a player standing near them, and the hook would find no
 * target for the bite. */
static int choose_dog(dqsim_entity *entities, int count, const float *player)
{
    int best = -1;
    int best_clutter = 1 << 30;
    float best_distance = 1e30f;
    for (int i = 0; i < count; i++) {
        if (strcmp(entities[i].classname, "monster_dog") != 0)
            continue;
        int clutter = 0;
        for (int j = 0; j < count; j++) {
            if (j == i || !(entities[j].flags & (32 | 256 | 8))) /* monster, item, client */
                continue;
            float dx = entities[j].origin[0] - entities[i].origin[0];
            float dy = entities[j].origin[1] - entities[i].origin[1];
            if (dx * dx + dy * dy < 120.0f * 120.0f)
                clutter++;
        }
        float dx = entities[i].origin[0] - player[0];
        float dy = entities[i].origin[1] - player[1];
        float distance = dx * dx + dy * dy;
        if (clutter < best_clutter || (clutter == best_clutter && distance < best_distance)) {
            best = i;
            best_clutter = clutter;
            best_distance = distance;
        }
    }
    return best;
}

/* Put the player `distance` units ahead of `origin` along `yaw`, facing it. */
static int place_player_ahead(const float origin[3], float yaw, float distance)
{
    float rad = yaw * 3.14159265f / 180.0f;
    float spot[3] = {origin[0] + cosf(rad) * distance, origin[1] + sinf(rad) * distance,
                     origin[2] + 24.0f};
    return dqsim_teleport_player(spot, yaw + 180.0f) == DQSIM_OK;
}

/* Put the player `distance` units from the dog in the first direction that is
 * both free and in the dog's sight (the check-attack function only starts an
 * attack it can see the target for). Leaves the dog in attack mode. */
static int place_player_visible(float distance)
{
    const dqsim_action idle = {0};
    for (int k = 0; k < 24; k++) {
        if (!place_player_ahead(dog_origin, dog_yaw + (float)(k * 15), distance))
            continue;
        dog_mode = 3;
        for (int tick = 0; tick < 24; tick++) {
            dog_saw_player = 0;
            if (dqsim_step_repeat(&idle, 1) != DQSIM_OK)
                return 0;
            if (dog_saw_player)
                return 1;
        }
    }
    return 0;
}

static float flat_distance(const float a[3], const float b[3])
{
    float dx = a[0] - b[0], dy = a[1] - b[1];
    return sqrtf(dx * dx + dy * dy);
}

/* Load, find the clearest dog, nominate it, freeze every other monster, and
 * let one second pass so the dog's own facing and origin are known. */
static int dog_start(int *dog_index)
{
    dqsim_player_state player;
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};

    CHECK(dqsim_load_map("e1m1", 2, 23) == DQSIM_OK, "load e1m1");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get player");
    int count = dqsim_get_entities(entities, 2048);
    CHECK(count > 0, "get entities");
    int dog = choose_dog(entities, count, player.origin);
    CHECK(dog >= 0, "no monster_dog on e1m1");
    *dog_index = entities[dog].index;
    dog_calls = 0;
    dog_mode = 0;
    CHECK(dqsim_set_policy_hook(*dog_index, dog_policy, NULL) == DQSIM_OK,
          "nominate dog: %s", dqsim_last_error());
    CHECK(freeze_monsters_except(*dog_index) >= 8, "fewer than 8 other monsters to freeze");
    /* dog_stand1 / dog_run1 / dog_walk1 are decision thinks, so the policy is
     * asked for decisions like the Grunt's. */
    CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "dog settle");
    CHECK(dog_calls > 10, "the dog was asked for %d decisions in 1 s", dog_calls);
    return 0;
}

/* The dog runs forward when the policy says so, and turns to the facing the
 * policy asks for. */
static int dog_drive(void)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int dog_index;
    if (dog_start(&dog_index))
        return 1;

    float start[3];
    memcpy(start, dog_origin, sizeof(start));
    dog_mode = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "dog forward");
    int count = dqsim_get_entities(entities, 2048);
    int found = 0;
    float moved = 0;
    for (int i = 0; i < count; i++)
        if (entities[i].index == dog_index) {
            found = 1;
            for (int k = 0; k < 2; k++) {
                float delta = entities[i].origin[k] - start[k];
                moved += delta * delta;
            }
        }
    CHECK(found, "the dog vanished");
    CHECK(sqrtf(moved) > 20.0f, "the dog moved %.1f units in 2 s", sqrtf(moved));

    float before_turn = dog_yaw;
    dog_turn_to = dog_yaw + 90.0f;
    dog_mode = 2;
    CHECK(dqsim_step_repeat(&idle, 36) == DQSIM_OK, "dog turn");
    float turned = fabsf(remainderf(dog_yaw - before_turn, 360.0f));
    CHECK(turned > 45.0f, "the dog turned %.1f of the 90 degrees asked", turned);
    printf("dog: moved %.0f units in 2 s, turned %.0f degrees\n", sqrtf(moved), turned);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* At bite range the bite lands, and the QuakeC chain gates the rate: a hit
 * per tick, or one hit of more than a bite's 7-21 damage, would mean the hook
 * bypassed the chain. */
static int dog_bite(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int dog_index;
    if (dog_start(&dog_index))
        return 1;
    CHECK(place_player_visible(50.0f), "no free spot in the dog's sight at bite range");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at bite range");

    float health = player.health;
    float last = health;
    float biggest = 0;
    int bites = 0, first_tick = -1;
    dog_mode = 3;
    for (int tick = 0; tick < 72 * 3 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "bite steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the bites");
        float drop = last - player.health;
        if (drop > 0.5f) {
            bites++;
            if (first_tick < 0)
                first_tick = tick;
            if (drop > biggest)
                biggest = drop;
        }
        last = player.health;
    }
    printf("dog: %d bites in 3 s at 50 units, first after %.2f s, largest %.0f, "
           "health %.0f -> %.0f\n",
           bites, first_tick < 0 ? -1.0f : first_tick / 72.0f, biggest, health, player.health);
    CHECK(bites > 0, "the dog bit nothing in 3 s standing 50 units in front of it");
    CHECK(bites <= 7, "the dog hit the player %d times in 3 s: the chain is not gating", bites);
    CHECK(biggest <= 25.0f, "one hit took %.0f health, more than a bite's 7-21", biggest);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The bite is a lunge - dog_atta1 starts dog_atta2..8, every frame moving
 * at the player - so from 300 units the dog closes the distance and then
 * bites. What it must not do is damage anybody it is not actually next to:
 * the first damage has to land at contact range. */
static int dog_range(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int dog_index;
    if (dog_start(&dog_index))
        return 1;
    CHECK(place_player_visible(300.0f), "no free spot in the dog's sight 300 units away");
    float start[3];
    memcpy(start, dog_origin, sizeof(start));
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player far away");
    float far[3];
    memcpy(far, player.origin, sizeof(far));
    float health = player.health;

    float first_damage_distance = -1.0f;
    float closest = -1.0f;
    dog_mode = 3;
    for (int tick = 0; tick < 72 * 4 && first_damage_distance < 0.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "charge steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the charge");
        float now = flat_distance(dog_origin, player.origin);
        if (closest < 0.0f || now < closest)
            closest = now;
        if (player.health < health - 0.5f)
            first_damage_distance = now;
    }
    printf("dog: from %.0f units it charged to %.0f, first damage at %.0f, health %.0f -> %.0f\n",
           flat_distance(start, far), closest, first_damage_distance, health, player.health);
    CHECK(first_damage_distance >= 0.0f,
          "the dog never damaged the player: its attack chain does not reach dog_bite");
    CHECK(first_damage_distance < 100.0f,
          "the first damage landed %.0f units away: the bite is not contact-range",
          first_damage_distance);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
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

    uint64_t multi_first = multi_scenario();
    if (!multi_first)
        return 1;
    uint64_t multi_second = multi_scenario();
    if (!multi_second)
        return 1;
    CHECK(multi_first == multi_second, "many monsters: same seed and inputs gave different hashes");

    if (dog_drive() || dog_bite() || dog_range())
        return 1;

    printf("hook scenario hash %" PRIu64 ", many-monster hash %" PRIu64 "\n", first,
           multi_first);
    dqsim_shutdown();
    return 0;
}

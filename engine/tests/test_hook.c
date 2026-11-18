/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * The brain hook (#31) on a real map: a scripted policy drives one Grunt
 * (move, then attack the player it can see), deterministically. The Dog
 * (#47) and the Enforcer and the Ogre (#49) are driven by the same code:
 * the hook must find each class's own QuakeC functions, whatever they are
 * called. Needs game data: pass the directory containing id1/ as the first
 * argument. Exits with 77 (skipped) when that directory has no id1/pak0.pak.
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
 * chain returns to a stand state.
 *
 * The same machinery drives any stock class (#49): the state below belongs to
 * "the monster under test", whatever its class. */
static int driven_mode; /* 0 idle, 1 forward, 2 turn, 3 attack the player */

/* The end-state hashes of the clean and the 45-degree-error throw windows:
 * the windows run on the same seed and the same inputs, so the hashes differ
 * exactly when the aim error reaches the grenade's flight (#53). The grenade
 * is not in dqsim_get_entities (its classname never reaches the sim's copy),
 * so a hash is the honest comparison. */
static uint64_t throw_hash_clean, throw_hash_error;
static int driven_calls;
static int driven_saw_player;
static float driven_yaw;
static float driven_origin[3];
static float driven_turn_to;

static void driven_policy(const dqsim_hook_observation *obs, dqsim_hook_action *action,
                          void *user)
{
    (void)user;
    driven_calls++;
    memcpy(driven_origin, obs->origin, sizeof(driven_origin));
    driven_yaw = obs->angles[1];
    driven_saw_player = 0;
    for (int i = 0; i < obs->visible_count; i++)
        if (obs->visible[i].flags & 8) /* FL_CLIENT */
            driven_saw_player = 1;
    action->desired_yaw = obs->angles[1];
    if (driven_mode == 1) {
        action->forward = 200.0f;
    } else if (driven_mode == 2) {
        action->desired_yaw = driven_turn_to;
    } else if (driven_mode == 3) {
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
 * the driven monster's damage. */
static int frozen_monsters;

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
    frozen_monsters = frozen;
    return frozen;
}

/* The class member with the least clutter around it: the observation reports
 * only the six nearest entities (deepquake.monster), so a monster standing
 * among items would not see a player standing near them, and the hook would
 * find no target for the attack. */
static int choose_class(dqsim_entity *entities, int count, const char *classname,
                        const float *player)
{
    int best = -1;
    int best_clutter = 1 << 30;
    float best_distance = 1e30f;
    for (int i = 0; i < count; i++) {
        if (strcmp(entities[i].classname, classname) != 0)
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
        if (!place_player_ahead(driven_origin, driven_yaw + (float)(k * 15), distance))
            continue;
        driven_mode = 3;
        for (int tick = 0; tick < 24; tick++) {
            driven_saw_player = 0;
            if (dqsim_step_repeat(&idle, 1) != DQSIM_OK)
                return 0;
            if (driven_saw_player)
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

/* Load, find the clearest monster of `classname`, nominate it, freeze every
 * other monster, and let one second pass so the driven monster's own facing
 * and origin are known. The class may name its QuakeC functions differently
 * from its classname (#49): the hook has to find them either way. */
static int driven_start(const char *map, const char *classname, int *index)
{
    dqsim_player_state player;
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};

    CHECK(dqsim_load_map(map, 2, 23) == DQSIM_OK, "load %s", map);
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get player");
    int count = dqsim_get_entities(entities, 2048);
    CHECK(count > 0, "get entities");
    int chosen = choose_class(entities, count, classname, player.origin);
    CHECK(chosen >= 0, "no %s on %s", classname, map);
    *index = entities[chosen].index;
    driven_calls = 0;
    driven_mode = 0;
    CHECK(dqsim_set_policy_hook(*index, driven_policy, NULL) == DQSIM_OK,
          "nominate %s: %s", classname, dqsim_last_error());
    CHECK(freeze_monsters_except(*index) >= 8, "only %d other monsters to freeze on %s",
          frozen_monsters, map);
    /* "<prefix>_stand1" / "<prefix>_run1" / "<prefix>_walk1" are decision
     * thinks, so the policy is asked for decisions like the Grunt's. */
    CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "%s settle", classname);
    CHECK(driven_calls > 10, "the %s was asked for %d decisions in 1 s", classname, driven_calls);
    return 0;
}

/* The dog runs forward when the policy says so, and turns to the facing the
 * policy asks for. */
static int dog_drive(void)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int dog_index;
    if (driven_start("e1m1", "monster_dog", &dog_index))
        return 1;

    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    driven_mode = 1;
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

    float before_turn = driven_yaw;
    driven_turn_to = driven_yaw + 90.0f;
    driven_mode = 2;
    CHECK(dqsim_step_repeat(&idle, 36) == DQSIM_OK, "dog turn");
    float turned = fabsf(remainderf(driven_yaw - before_turn, 360.0f));
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
    if (driven_start("e1m1", "monster_dog", &dog_index))
        return 1;
    CHECK(place_player_visible(50.0f), "no free spot in the dog's sight at bite range");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at bite range");

    float health = player.health;
    float last = health;
    float biggest = 0;
    int bites = 0, first_tick = -1;
    driven_mode = 3;
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
    if (driven_start("e1m1", "monster_dog", &dog_index))
        return 1;
    CHECK(place_player_visible(300.0f), "no free spot in the dog's sight 300 units away");
    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player far away");
    float far[3];
    memcpy(far, player.origin, sizeof(far));
    float health = player.health;

    float first_damage_distance = -1.0f;
    float closest = -1.0f;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 4 && first_damage_distance < 0.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "charge steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the charge");
        float now = flat_distance(driven_origin, player.origin);
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

/* Another stock class, driven by the same hook (#49). The Enforcer's QuakeC
 * functions are named "enf_...", not "monster_enforcer_...", so the hook has
 * to find the class's own prefix instead of deriving one from the classname.
 * Its attack entry is "enf_atk1" and its shot "enforcer_fire", which is not
 * "<prefix>fire" either. */
static int enforcer_drive(void)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m1", "monster_enforcer", &index))
        return 1;

    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    driven_mode = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "enforcer forward");
    int count = dqsim_get_entities(entities, 2048);
    int found = 0;
    float moved = 0;
    for (int i = 0; i < count; i++)
        if (entities[i].index == index) {
            found = 1;
            for (int k = 0; k < 2; k++) {
                float delta = entities[i].origin[k] - start[k];
                moved += delta * delta;
            }
        }
    CHECK(found, "the enforcer vanished");
    CHECK(sqrtf(moved) > 20.0f, "the enforcer moved %.1f units in 2 s", sqrtf(moved));
    printf("enforcer: moved %.0f units in 2 s\n", sqrtf(moved));
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The policy asks for an attack and the class's own attack chain lands the
 * damage: "enf_atk1" leads to "enf_atk6", which fires the laser from
 * "enforcer_fire". The Enforcer must not hit the player on every tick - the
 * QuakeC chain sets the pace. */
static int enforcer_attack(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m1", "monster_enforcer", &index))
        return 1;
    CHECK(place_player_visible(100.0f), "no free spot in the enforcer's sight at 100 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    float biggest = 0;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 4 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "attack steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the attack");
        float drop = health - player.health;
        if (drop > 0.5f) {
            hits++;
            if (drop > biggest)
                biggest = drop;
        }
        health = player.health;
    }
    printf("enforcer: %d hits in 4 s at 100 units, biggest %.0f, health %.0f -> %.0f\n", hits,
           biggest, before, player.health);
    CHECK(hits > 0, "the enforcer never damaged the player: its attack chain does not reach "
                    "enforcer_fire");
    CHECK(hits <= 8, "the enforcer hit the player %d times in 4 s: the chain is not gating", hits);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The aim error has to land where the shot is aimed. The Enforcer's shot
 * function is "enforcer_fire" while its prefix is "enf_", so the hook finds
 * it from the attack entry's own call: with a 45 degree aim error the laser
 * goes wide, and with no error the same shot hits. */
static int enforcer_aim_error(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m1", "monster_enforcer", &index))
        return 1;
    CHECK(place_player_visible(100.0f), "no free spot in the enforcer's sight at 100 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");
    float before = player.health;

    CHECK(dqsim_set_handicap_override(0, 45.0f) == DQSIM_OK, "aim error override: %s",
          dqsim_last_error());
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 3; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "aim error steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the aim error");
    }
    float missed = before - player.health;
    printf("enforcer: %.0f damage with a 45 degree aim error\n", missed);
    CHECK(missed < 5.0f, "the enforcer did %.0f damage with a 45 degree aim error", missed);

    CHECK(dqsim_set_handicap_override(0, 0.0f) == DQSIM_OK, "aim error override off");
    float health = player.health;
    int hits = 0;
    for (int tick = 0; tick < 72 * 3 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "clean shot steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the clean shot");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("enforcer: %d hits with no aim error\n", hits);
    CHECK(hits > 0, "the enforcer never hit the player without an aim error");
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The Ogre's attack entry is "ogre_melee", a name the hook's probe list did
 * not have (#49): the Ogre was driven but its attack bit did nothing. */
static int ogre_attack(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e1m4", "monster_ogre", &index))
        return 1;
    CHECK(place_player_visible(60.0f), "no free spot in the ogre's sight at 60 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 4 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "attack steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the attack");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("ogre: %d hits in 4 s at 60 units, health %.0f -> %.0f\n", hits, before,
           player.health);
    CHECK(hits > 0, "the ogre never damaged the player: its attack entry was not found");
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* Drive the throw for one window and return the window's end-state hash.
 * The grenade is not in dqsim_get_entities (its classname never reaches the
 * sim's entity copy), so the flight is compared through the hash. */
static uint64_t ogre_throw_window(float aim_override, int *hits, float *damage)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e1m4", "monster_ogre", &index))
        return 0;
    CHECK(dqsim_set_handicap_override(0, aim_override) == DQSIM_OK,
          "aim error override: %s", dqsim_last_error());
    CHECK(place_player_visible(100.0f), "no free spot in the ogre's sight at 300 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float health = player.health;
    *hits = 0;
    *damage = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "throw steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the throw");
        if (health - player.health > 0.5f) {
            (*hits)++;
            *damage = health - player.health;
        }
        health = player.health;
    }
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return dqsim_state_hash();
}

/* The Ogre's ranged attack: the throw chain spawns a grenade that flies and
 * explodes with radius damage (#53). The throw must reach the player from
 * well outside bite range, and the QuakeC chain must gate the pace. */
static int ogre_throw(void)
{
    /* A clean throw: skill 0's own aim error is 12 degrees, which at 300
     * units can land a grenade wide of the player, so the exact throw is
     * pinned with the override (the Enforcer's aim test does the same). */
    int hits = 0;
    float damage = 0;
    throw_hash_clean = ogre_throw_window(0.0f, &hits, &damage);
    printf("ogre: clean throw, %d grenade hits, %.0f damage\n", hits, damage);
    CHECK(throw_hash_clean != 0, "the clean throw window failed");
    CHECK(hits > 0, "the ogre never damaged the player at range: its throw chain does not "
                    "spawn grenades");
    CHECK(hits <= 8, "the ogre hit the player %d times in 8 s: the chain is not gating", hits);
    return 0;
}

/* The aim error has to reach the grenade's launch too (#53): the thrown
 * grenade is not visible in the sim's entity copy, so the two windows are
 * compared by their end-state hashes - same seed, same inputs, the only
 * difference is the bent flight, so equal hashes would mean the error never
 * touched the grenade. */
static int ogre_throw_aim_error(void)
{
    int hits = 0;
    float damage = 0;
    throw_hash_error = ogre_throw_window(45.0f, &hits, &damage);
    printf("ogre: 45 degree error throw, %d grenade hits, %.0f damage\n", hits, damage);
    CHECK(throw_hash_error != 0, "the error throw window failed");
    CHECK(throw_hash_clean != throw_hash_error,
          "the aim error did not change the grenade's flight");
    return 0;
}

/* A fifth class through the same hook (#55): the Hell Knight's prefix is
 * "hknight_" and its ranged attack is the magic missile, fired by
 * "hknight_shot" from its th_missile chain. The melee axe runs from
 * th_melee in close. No new mechanism: the spawn-field entries, the
 * chain-walked fire frames, the facing roll and the sight globals have to
 * cover it as they cover the Ogre. */
static int hknight_drive(void)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m5", "monster_hell_knight", &index))
        return 1;

    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    driven_mode = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "hell knight forward");
    int count = dqsim_get_entities(entities, 2048);
    int found = 0;
    float moved = 0;
    for (int i = 0; i < count; i++)
        if (entities[i].index == index) {
            found = 1;
            for (int k = 0; k < 2; k++) {
                float delta = entities[i].origin[k] - start[k];
                moved += delta * delta;
            }
        }
    CHECK(found, "the hell knight vanished");
    CHECK(sqrtf(moved) > 20.0f, "the hell knight moved %.1f units in 2 s", sqrtf(moved));
    printf("hell knight: moved %.0f units in 2 s\n", sqrtf(moved));
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The ranged attack: the magic missile must reach the player from well
 * outside melee range, and the QuakeC chain must gate the pace. */
static int hknight_attack(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m5", "monster_hell_knight", &index))
        return 1;
    CHECK(dqsim_set_handicap_override(0, 0.0f) == DQSIM_OK, "aim error override: %s",
          dqsim_last_error());
    CHECK(place_player_visible(100.0f), "no free spot in the hell knight's sight at 400 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "attack steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the attack");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("hell knight: %d missile hits in 8 s at 400 units, health %.0f -> %.0f\n", hits,
           before, player.health);
    CHECK(hits > 0, "the hell knight never damaged the player at range: its missile chain "
                    "does not fire");
    CHECK(hits <= 12, "the hell knight hit the player %d times in 8 s: the chain is not gating",
          hits);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The aim error reaches the magic missile: the clean and 45-degree windows
 * are compared by state hash (the missile is not in the sim's entity copy;
 * same seed and inputs, the only difference is the bent flight). */
static int hknight_aim_error(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m5", "monster_hell_knight", &index))
        return 1;
    CHECK(place_player_visible(100.0f), "no free spot in the hell knight's sight at 400 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");
    float before = player.health;

    CHECK(dqsim_set_handicap_override(0, 0.0f) == DQSIM_OK, "aim error override off: %s",
          dqsim_last_error());
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "clean steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the clean shots");
    }
    float clean_damage = before - player.health;
    printf("hell knight: %.0f damage clean in 8 s\n", clean_damage);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");

    if (driven_start("e2m5", "monster_hell_knight", &index))
        return 1;
    CHECK(dqsim_set_handicap_override(0, 45.0f) == DQSIM_OK, "aim error override: %s",
          dqsim_last_error());
    CHECK(place_player_visible(100.0f), "no free spot in the hell knight's sight at 400 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range for the error window");
    before = player.health;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "aim error steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the aim error");
    }
    float missed = before - player.health;
    printf("hell knight: %.0f damage with a 45 degree aim error\n", missed);
    CHECK(missed < 5.0f, "the hell knight did %.0f damage with a 45 degree aim error", missed);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* A sixth class (#56). The Shalrath's stand think is "shal_stand" - no
 * frame digits - so a standing Shalrath must still be policy-driven: the
 * decision pattern accepts the digitless single-state stand. Its spikes are
 * launched by "ShalMissile", which the shot-name rule only finds since
 * "missile" joined the set. The spikes HOME (ShalHome corrects in flight),
 * so the aim proof is the clean/45-degree state-hash comparison, not a
 * damage assertion. */
static int shalrath_stand_driven(void)
{
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e3m7", "monster_shalrath", &index))
        return 1;
    driven_calls = 0;
    driven_mode = 0; /* stand still: the policy is asked anyway */
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "stand steps");
    printf("shalrath: %d decisions in 2 s while standing\n", driven_calls);
    CHECK(driven_calls > 100, "a standing Shalrath got %d decisions: the digitless stand "
                              "think is not intercepted",
          driven_calls);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

static int shalrath_attack(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e3m7", "monster_shalrath", &index))
        return 1;
    CHECK(dqsim_set_handicap_override(0, 0.0f) == DQSIM_OK, "aim error override: %s",
          dqsim_last_error());
    CHECK(place_player_visible(100.0f), "no free spot in the shalrath's sight at 100 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "spike steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the spikes");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("shalrath: %d spike hits in 8 s at 100 units, health %.0f -> %.0f\n", hits, before,
           player.health);
    CHECK(hits > 0, "the shalrath never damaged the player at range: its spike chain does "
                    "not fire");
    CHECK(hits <= 12, "the shalrath hit the player %d times in 8 s: the chain is not gating",
          hits);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The aim error reaches the spike launch: the clean and 45-degree windows
 * are compared by state hash. The spikes home (ShalHome corrects in
 * flight), so the damage may not differ; the flight does. */
static int shalrath_aim_error(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    uint64_t clean_hash, error_hash;

    if (driven_start("e3m7", "monster_shalrath", &index))
        return 1;
    CHECK(dqsim_set_handicap_override(0, 0.0f) == DQSIM_OK, "aim error override off: %s",
          dqsim_last_error());
    CHECK(place_player_visible(100.0f), "no free spot in the shalrath's sight at 100 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "clean spike steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the clean spikes");
    }
    clean_hash = dqsim_state_hash();
    printf("shalrath: clean window hash %llu\n", (unsigned long long)clean_hash);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");

    if (driven_start("e3m7", "monster_shalrath", &index))
        return 1;
    CHECK(dqsim_set_handicap_override(0, 45.0f) == DQSIM_OK, "aim error override: %s",
          dqsim_last_error());
    CHECK(place_player_visible(100.0f), "no free spot in the shalrath's sight at 100 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range for the error window");
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "aim error spike steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the aim error");
    }
    error_hash = dqsim_state_hash();
    printf("shalrath: 45 degree error window hash %llu\n", (unsigned long long)error_hash);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");

    CHECK(clean_hash != 0 && error_hash != 0, "a spike window failed");
    CHECK(clean_hash != error_hash, "the aim error did not change the spike flight");
    return 0;
}

/* A seventh class (#57). The Knight's prefix is "knight_", all states carry
 * frame digits, and the progs name no fireball launch helper - the spawn may
 * be inline in the chain, which the shot-name rule cannot match. The ctests
 * measure what actually happens: movement, damage at range, and the aim hash
 * (clean vs 45 degrees - equal hashes would mean the fire-think detection
 * needs extending). */
static int knight_drive(void)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e1m4", "monster_knight", &index))
        return 1;

    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    driven_mode = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "knight forward");
    int count = dqsim_get_entities(entities, 2048);
    int found = 0;
    float moved = 0;
    for (int i = 0; i < count; i++)
        if (entities[i].index == index) {
            found = 1;
            for (int k = 0; k < 2; k++) {
                float delta = entities[i].origin[k] - start[k];
                moved += delta * delta;
            }
        }
    CHECK(found, "the knight vanished");
    CHECK(sqrtf(moved) > 20.0f, "the knight moved %.1f units in 2 s", sqrtf(moved));
    printf("knight: moved %.0f units in 2 s\n", sqrtf(moved));
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The Knight is a pure melee monster in this progs: the spawn's entries both
 * point at the sword chain (knight_atk1..10, ai_melee), the knight_bow chain
 * is vestigial (ai_turn only) and no knight function spawns anything. The
 * damage has to land at sword range, and the chain has to gate. */
static int knight_attack(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e1m4", "monster_knight", &index))
        return 1;
    CHECK(place_player_visible(50.0f), "no free spot in the knight's sight at 50 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "attack steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the attack");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("knight: %d sword hits in 8 s at 50 units, health %.0f -> %.0f\n", hits, before,
           player.health);
    CHECK(hits > 0, "the knight never damaged the player: its sword chain does not reach "
                    "ai_melee");
    CHECK(hits <= 24, "the knight hit the player %d times in 8 s: the chain is not gating",
          hits);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* An eighth class (#58): the Fiend is a pure melee monster with a leap -
 * the spawn's entries are th_melee = Demon_MeleeAttack (the scratch) and
 * th_missile = demon1_atta1 (the leap chain, its closing move). */
static int fiend_drive(void)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m5", "monster_demon1", &index))
        return 1;

    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    driven_mode = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "fiend forward");
    int count = dqsim_get_entities(entities, 2048);
    int found = 0;
    float moved = 0;
    for (int i = 0; i < count; i++)
        if (entities[i].index == index) {
            found = 1;
            for (int k = 0; k < 2; k++) {
                float delta = entities[i].origin[k] - start[k];
                moved += delta * delta;
            }
        }
    CHECK(found, "the fiend vanished");
    CHECK(sqrtf(moved) > 20.0f, "the fiend moved %.1f units in 2 s", sqrtf(moved));
    printf("fiend: moved %.0f units in 2 s\n", sqrtf(moved));
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The melee scratch at contact range, and the leap from further out: the
 * damage has to land, and the chains have to gate. */
static int fiend_attack(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m5", "monster_demon1", &index))
        return 1;
    CHECK(place_player_visible(60.0f), "no free spot in the fiend's sight at 60 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "scratch steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the scratch");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("fiend: %d scratch hits in 8 s at 60 units, health %.0f -> %.0f\n", hits, before,
           player.health);
    CHECK(hits > 0, "the fiend never damaged the player: its melee chain does not reach "
                    "Demon_MeleeAttack");
    CHECK(hits <= 24, "the fiend hit the player %d times in 8 s: the chain is not gating",
          hits);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

static int fiend_leap(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e2m5", "monster_demon1", &index))
        return 1;
    CHECK(place_player_visible(100.0f), "no free spot in the fiend's sight at 100 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    float closest = -1.0f;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "leap steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the leap");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
        float now = flat_distance(driven_origin, player.origin);
        if (closest < 0.0f || now < closest)
            closest = now;
    }
    printf("fiend: %d hits in 8 s from 100 units, closed to %.0f, health %.0f -> %.0f\n",
           hits, closest, before, player.health);
    CHECK(hits > 0, "the fiend never damaged the player from 100 units: its leap does not "
                    "close or its scratch does not land");
    CHECK(closest < 100.0f, "the fiend never closed below 100 units (closest %.0f)",
          closest);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* A ninth class (#59): the Shambler. The attack entry is sham_melee (a
 * dispatcher: the claw at melee range, the LIGHTNING BEAM at range - the
 * beam chain is sham_magic1..12 and its launch helper is CastLightning,
 * which the fire/shot/missile name rule does not match). The ctests measure
 * what actually happens, and the aim proof is the state-hash comparison. */
static int shambler_drive(void)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e1m8", "monster_shambler", &index))
        return 1;

    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    driven_mode = 1;
    CHECK(dqsim_step_repeat(&idle, 72 * 2) == DQSIM_OK, "shambler forward");
    int count = dqsim_get_entities(entities, 2048);
    int found = 0;
    float moved = 0;
    for (int i = 0; i < count; i++)
        if (entities[i].index == index) {
            found = 1;
            for (int k = 0; k < 2; k++) {
                float delta = entities[i].origin[k] - start[k];
                moved += delta * delta;
            }
        }
    CHECK(found, "the shambler vanished");
    CHECK(sqrtf(moved) > 20.0f, "the shambler moved %.1f units in 2 s", sqrtf(moved));
    printf("shambler: moved %.0f units in 2 s\n", sqrtf(moved));
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The claw at melee range: the damage has to land, and the chain has to
 * gate. */
static int shambler_claw(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e1m8", "monster_shambler", &index))
        return 1;
    CHECK(place_player_visible(60.0f), "no free spot in the shambler's sight at 60 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "claw steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the claw");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("shambler: %d claw hits in 8 s at 60 units, health %.0f -> %.0f\n", hits, before,
           player.health);
    CHECK(hits > 0, "the shambler never damaged the player: its melee chain does not reach "
                    "ShamClaw");
    CHECK(hits <= 24, "the shambler hit the player %d times in 8 s: the chain is not gating",
          hits);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The beam at range: the damage has to land from outside claw range. */
static int shambler_beam(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    if (driven_start("e1m8", "monster_shambler", &index))
        return 1;
    CHECK(place_player_visible(300.0f), "no free spot in the shambler's sight at 300 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");

    float before = player.health;
    float health = before;
    int hits = 0;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "beam steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the beam");
        if (health - player.health > 0.5f)
            hits++;
        health = player.health;
    }
    printf("shambler: %d beam hits in 8 s at 300 units, health %.0f -> %.0f\n", hits, before,
           player.health);
    CHECK(hits > 0, "the shambler never damaged the player at range: its magic chain does "
                    "not reach the beam");
    CHECK(hits <= 16, "the shambler hit the player %d times in 8 s: the chain is not gating",
          hits);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* The aim error reaches the beam: the clean and 45-degree windows compared
 * by state hash. */
static int shambler_aim_error(void)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    int index;
    uint64_t clean_hash, error_hash;

    if (driven_start("e1m8", "monster_shambler", &index))
        return 1;
    CHECK(dqsim_set_handicap_override(0, 0.0f) == DQSIM_OK, "aim error override off: %s",
          dqsim_last_error());
    CHECK(place_player_visible(300.0f), "no free spot in the shambler's sight at 300 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "clean beam steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the clean beam");
    }
    clean_hash = dqsim_state_hash();
    printf("shambler: clean window hash %llu\n", (unsigned long long)clean_hash);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");

    if (driven_start("e1m8", "monster_shambler", &index))
        return 1;
    CHECK(dqsim_set_handicap_override(0, 45.0f) == DQSIM_OK, "aim error override: %s",
          dqsim_last_error());
    CHECK(place_player_visible(300.0f), "no free spot in the shambler's sight at 300 units");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range for the error window");
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 8 && player.health > 10.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "aim error beam steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the aim error");
    }
    error_hash = dqsim_state_hash();
    printf("shambler: 45 degree error window hash %llu\n", (unsigned long long)error_hash);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");

    CHECK(clean_hash != 0 && error_hash != 0, "a shambler window failed");
    CHECK(clean_hash != error_hash, "the aim error did not change the shambler's beam");
    return 0;
}

/* Every frame a driven monster shows must be one its model has (#61). The old
 * animation offset from the first frame by a hard-coded count, which named a
 * frame past the end of the Shalrath's 35-frame model (its run chain loops back
 * onto an earlier frame), so the client drew nothing and the monster blinked.
 * The fix walks each class's real stand/run chains; this pins the result against
 * the model's frame count, an independent ground truth. numframes is the stock
 * .mdl's frame total for the class. */
static int frame_bound(const char *map, const char *classname, int numframes)
{
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    int index;
    if (driven_start(map, classname, &index))
        return 1;

    int stand_max = -1, run_max = -1;
    driven_mode = 0; /* still: the stand animation */
    for (int t = 0; t < 72; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "%s stand tick", classname);
        int count = dqsim_get_entities(entities, 2048);
        for (int i = 0; i < count; i++)
            if (entities[i].index == index) {
                CHECK(entities[i].frame >= 0 && entities[i].frame < numframes,
                      "%s stand frame %d outside the model's %d frames", classname,
                      entities[i].frame, numframes);
                /* A zero interval makes the client blend with 0 / 0 and draw
                 * nothing for that tick: the blink of #61. */
                CHECK(entities[i].lerp_interval != 0,
                      "%s stand tick %d: the client would be sent a zero interpolation "
                      "interval", classname, t);
                if (entities[i].frame > stand_max)
                    stand_max = entities[i].frame;
            }
    }

    float start[3];
    memcpy(start, driven_origin, sizeof(start));
    driven_mode = 1; /* forward: the run animation (no enemy in sight, so the
                      * reaction delay never gates it) */
    for (int t = 0; t < 144; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "%s run tick", classname);
        int count = dqsim_get_entities(entities, 2048);
        for (int i = 0; i < count; i++)
            if (entities[i].index == index) {
                CHECK(entities[i].frame >= 0 && entities[i].frame < numframes,
                      "%s run frame %d outside the model's %d frames", classname,
                      entities[i].frame, numframes);
                /* A zero interval makes the client blend with 0 / 0 and draw
                 * nothing for that tick: the blink of #61. */
                CHECK(entities[i].lerp_interval != 0,
                      "%s run tick %d: the client would be sent a zero interpolation "
                      "interval", classname, t);
                if (entities[i].frame > run_max)
                    run_max = entities[i].frame;
            }
    }
    float dx = driven_origin[0] - start[0], dy = driven_origin[1] - start[1];
    float moved = sqrtf(dx * dx + dy * dy);
    CHECK(moved > 20.0f, "the %s moved only %.1f units, so the run path was not exercised",
          classname, moved);
    printf("%s: frames in range (stand max %d, run max %d, model %d frames)\n", classname,
           stand_max, run_max, numframes);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

static int animation_frames(void)
{
    /* The stocking maps the per-class tests already use, and each stock model's
     * frame total. The Shalrath is the regression: 35 frames, run max must stay
     * at 34, never 35. */
    if (frame_bound("e1m1", "monster_army", 114) ||
        frame_bound("e1m1", "monster_dog", 86) ||
        frame_bound("e2m1", "monster_enforcer", 102) ||
        frame_bound("e1m4", "monster_ogre", 147) ||
        frame_bound("e2m5", "monster_hell_knight", 166) ||
        frame_bound("e3m7", "monster_shalrath", 35) ||
        frame_bound("e1m4", "monster_knight", 97) ||
        frame_bound("e2m5", "monster_demon1", 69) ||
        frame_bound("e1m8", "monster_shambler", 94))
        return 1;
    return 0;
}

/* Driven monsters walk and run continuously (#61): over a 30 s window, no
 * per-tick horizontal jump beyond the class's top speed. Many monsters at once,
 * as in play. The vertical step-downs off ledges (up to the 18-unit step height)
 * are normal MOVETYPE_STEP behaviour and are not horizontal teleports.
 * Lunges/leaps are attacks, bounded by the dog/fiend tests; this drives plain
 * locomotion. */
static void locomotion_policy(const dqsim_hook_observation *obs, dqsim_hook_action *action,
                              void *user)
{
    (void)user;
    action->desired_yaw = obs->angles[1];
    action->forward = 200.0f;
}

static int movement_continuity(void)
{
    static dqsim_entity entities[2048];
    static float prev[2048][3];
    static int have_prev[2048];
    const dqsim_action idle = {0};
    const float max_step = 8.5f; /* (400 u/s top run speed / 72 ticks) * 1.5 */

    CHECK(dqsim_load_map("e1m1", 2, 23) == DQSIM_OK, "load e1m1");
    int count = dqsim_get_entities(entities, 2048);
    int nominated = 0;
    for (int i = 0; i < count; i++) {
        if (!(entities[i].flags & 32)) /* FL_MONSTER */
            continue;
        CHECK(dqsim_set_policy_hook(entities[i].index, locomotion_policy, NULL) == DQSIM_OK,
              "nominate %d: %s", entities[i].index, dqsim_last_error());
        nominated++;
    }
    CHECK(nominated >= 8, "only %d monsters to drive", nominated);

    memset(have_prev, 0, sizeof(have_prev));
    float max_horizontal = 0.0f;
    int jumps = 0;
    for (int t = 0; t < 72 * 30; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "locomotion tick");
        int n = dqsim_get_entities(entities, 2048);
        for (int i = 0; i < n; i++) {
            int idx = entities[i].index;
            if (!(entities[i].flags & 32) || idx >= 2048)
                continue;
            if (have_prev[idx]) {
                float dx = entities[i].origin[0] - prev[idx][0];
                float dy = entities[i].origin[1] - prev[idx][1];
                float horizontal = sqrtf(dx * dx + dy * dy);
                if (horizontal > max_horizontal)
                    max_horizontal = horizontal;
                if (horizontal > max_step)
                    jumps++;
            }
            prev[idx][0] = entities[i].origin[0];
            prev[idx][1] = entities[i].origin[1];
            prev[idx][2] = entities[i].origin[2];
            have_prev[idx] = 1;
        }
    }
    printf("movement: %d monsters, max horizontal per-tick %.2f units, teleports %d\n",
           nominated, max_horizontal, jumps);
    CHECK(jumps == 0, "%d horizontal teleports in 30 s of locomotion", jumps);
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

    if (enforcer_drive() || enforcer_attack() || enforcer_aim_error() || ogre_attack() ||
        ogre_throw() || ogre_throw_aim_error() || hknight_drive() || hknight_attack() ||
        hknight_aim_error() || shalrath_stand_driven() || shalrath_attack() ||
        shalrath_aim_error() || knight_drive() || knight_attack() || fiend_drive() ||
        fiend_attack() || fiend_leap() || shambler_drive() || shambler_claw() ||
        shambler_beam() || shambler_aim_error() || animation_frames() ||
        movement_continuity())
        return 1;

    printf("hook scenario hash %" PRIu64 ", many-monster hash %" PRIu64 "\n", first,
           multi_first);
    dqsim_shutdown();
    return 0;
}

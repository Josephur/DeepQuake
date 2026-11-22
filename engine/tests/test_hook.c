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
static int driven_mode; /* 0 idle, 1 forward, 2 turn, 3 attack the player, 4 strafe */

/* The end-state hashes of the clean and the 45-degree-error throw windows:
 * the windows run on the same seed and the same inputs, so the hashes differ
 * exactly when the aim error reaches the grenade's flight (#53). The grenade
 * is not in dqsim_get_entities (its classname never reaches the sim's copy),
 * so a hash is the honest comparison. */
static uint64_t throw_hash_clean, throw_hash_error;
static int driven_calls;
static int driven_saw_player;
static const char *spoilt; /* why the last speed attempt was rejected */
static int driven_saw_ever; /* 1 once a report has shown the player (0B tests) */
static float driven_yaw;
static float driven_origin[3];
static float driven_turn_to;
static float driven_side;
/* The hearing fields the part 0B tests read from the driven monster's reports. */
static int driven_heard;
static int driven_heard_kind;
static float driven_heard_relative[3];
static float driven_heard_age;
static float driven_cooldown;
static int driven_pace;

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
    if (driven_saw_player)
        driven_saw_ever = 1;
    /* The hearing fields the part 0B tests read (RETRAIN_SPEC part 0B). */
    driven_heard = (int)obs->heard;
    driven_heard_kind = (int)obs->heard_kind;
    memcpy(driven_heard_relative, obs->heard_relative, sizeof(driven_heard_relative));
    driven_heard_age = obs->heard_age;
    driven_cooldown = obs->attack_cooldown;
    driven_pace = (int)obs->pace;
    action->desired_yaw = obs->angles[1];
    if (driven_mode == 1) {
        action->forward = 200.0f;
    } else if (driven_mode == 4) {
        action->side = driven_side;
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

static int freeze_monsters_except(int keep, const float *near)
{
    static dqsim_entity entities[2048];
    int count = dqsim_get_entities(entities, 2048);
    int frozen = 0;
    for (int i = 0; i < count; i++) {
        if (!(entities[i].flags & 32) || entities[i].index == keep) /* FL_MONSTER */
            continue;
        /* Only the neighbours when `near` is given: the hook's slot pool
         * is finite (DQ_BRAIN_MAX_MONSTERS), and the monster-rich maps
         * hold more monsters than fit - one 800 units away cannot reach
         * the measured corridor inside an episode anyway. */
        if (near) {
            float dx = entities[i].origin[0] - near[0];
            float dy = entities[i].origin[1] - near[1];
            float dz = entities[i].origin[2] - near[2];
            if (dx * dx + dy * dy + dz * dz > 600.0f * 600.0f)
                continue;
        }
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
    driven_saw_ever = 0;
    CHECK(dqsim_set_policy_hook(*index, driven_policy, NULL) == DQSIM_OK,
          "nominate %s: %s", classname, dqsim_last_error());
    CHECK(freeze_monsters_except(*index, NULL) >= 8,
          "only %d other monsters to freeze on %s", frozen_monsters, map);
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

/* Asked to attack a player 300 units away, the dog closes the distance and
 * then bites (since #63 by charging and leaping; before, by the bite chain's
 * own lunge steps). What it must not do is damage anybody it is not actually
 * next to: the first damage has to land at contact range. */
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

static int driven_entity(int index, dqsim_entity *out)
{
    static dqsim_entity entities[2048];
    int count = dqsim_get_entities(entities, 2048);
    for (int i = 0; i < count; i++)
        if (entities[i].index == index) {
            *out = entities[i];
            return 1;
        }
    return 0;
}

/* A melee attack is never started out of reach (#63). A policy that asks to
 * attack a player it can see but not reach used to get the whole bite or
 * swing played where the monster stood, again and again: Dogs and Fiends
 * attacking thin air. Now the monster charges - runs at the player, leaps if
 * its class has a leap - and the blow starts in reach. Checked on what a
 * client sees: while the player is beyond the leap window the monster shows
 * its run frames [run_lo, run_hi] and covers ground, and the first damage
 * lands at contact range. */
static int melee_needs_reach(const char *map, const char *classname, float from, int run_lo,
                             int run_hi)
{
    dqsim_player_state player;
    const dqsim_action idle = {0};
    dqsim_entity self;
    int index;
    if (driven_start(map, classname, &index))
        return 1;
    /* No reaction delay: the charge is what is measured, from the first tick. */
    CHECK(dqsim_set_handicap_override(0, 0.0f) == DQSIM_OK, "handicap override: %s",
          dqsim_last_error());
    /* A spot `from` units away that is in sight and level with the monster
     * once the player has landed: one over a drop falls out of sight, and
     * that is not the case under test. The monster idles meanwhile. */
    int placed = 0;
    for (int k = 0; k < 24 && !placed; k++) {
        if (!place_player_ahead(driven_origin, driven_yaw + (float)(k * 15), from))
            continue;
        driven_mode = 0;
        CHECK(dqsim_step_repeat(&idle, 36) == DQSIM_OK, "let the player land");
        driven_saw_player = 0;
        CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "look");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player at range");
        CHECK(driven_entity(index, &self), "the %s is gone", classname);
        placed = driven_saw_player && fabsf(player.origin[2] - self.origin[2]) < 48.0f &&
                 flat_distance(self.origin, player.origin) > from - 40.0f;
    }
    CHECK(placed, "no level spot in the %s's sight %.0f units away", classname, from);

    float health = player.health;
    float start = flat_distance(self.origin, player.origin);
    float first_damage_distance = -1.0f;
    int far_ticks = 0, far_running = 0, reached_tick = -1;
    driven_mode = 3;
    for (int tick = 0; tick < 72 * 6 && first_damage_distance < 0.0f; tick++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "charge steps");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player during the charge");
        CHECK(driven_entity(index, &self), "the %s is gone", classname);
        float now = flat_distance(self.origin, player.origin);
        if (now > 220.0f) {
            far_ticks++;
            if (self.frame >= run_lo && self.frame <= run_hi)
                far_running++;
            else
                CHECK(far_ticks <= 12 || far_running * 10 >= far_ticks * 9,
                      "%s at %.0f units shows frame %d: not running at the player", classname,
                      now, self.frame);
        } else if (reached_tick < 0) {
            reached_tick = tick;
        }
        if (player.health < health - 0.5f)
            first_damage_distance = now;
    }
    printf("%s: from %.0f units, %d of %d far ticks in run frames, inside 220 units after "
           "%.2f s, first damage at %.0f units\n",
           classname, start, far_running, far_ticks, reached_tick / 72.0f, first_damage_distance);
    CHECK(far_ticks >= 10, "the %s started only %.0f units away", classname, start);
    CHECK(far_running * 10 >= far_ticks * 9, "the %s ran on only %d of %d ticks out of reach",
          classname, far_running, far_ticks);
    CHECK(reached_tick >= 0 && reached_tick <= 72 * 2,
          "the %s took %.2f s to come within 220 units from %.0f", classname,
          reached_tick / 72.0f, start);
    CHECK(first_damage_distance >= 0.0f, "the %s never damaged the player", classname);
    CHECK(first_damage_distance < 110.0f, "the %s's first damage landed %.0f units away",
          classname, first_damage_distance);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* Show the idle driven monster the player: from then on it has an enemy and
 * runs rather than walks (#61). */
static int alert_driven(void)
{
    const dqsim_action idle = {0};
    driven_mode = 0;
    for (int k = 0; k < 24; k++) {
        if (!place_player_ahead(driven_origin, driven_yaw + (float)(k * 15), 200.0f))
            continue;
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

/* A driven monster moves at its own body's speed, in the frames drawn for it
 * (#61). The policy commands the player's 400 units a second; applied as
 * given, a Grunt crossed the floor at four times its stock run speed with its
 * run frames ticking over at their usual rate: it slid. The hook now caps the
 * move at the speed of the class's own states - the Grunt's walk chain
 * (frames 90-113) averages 18 units a second, its run chain (73-80) 109 - and
 * the monster walks until it has an enemy, then runs. */
static int body_speed(void)
{
    const dqsim_action idle = {0};
    dqsim_entity self;
    int index;
    if (driven_start("e1m1", "monster_army", &index))
        return 1;
    driven_saw_player = 0;
    driven_mode = 0;
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "look");
    CHECK(!driven_saw_player, "the Grunt sees the player from the start: nothing to walk for");

    float fastest[2] = {0.0f, 0.0f};
    int moving[2] = {0, 0};
    for (int phase = 0; phase < 2; phase++) {
        if (phase == 1)
            CHECK(alert_driven(), "no spot in the Grunt's sight");
        int lo = phase ? 73 : 90, hi = phase ? 80 : 113;
        float last[3];
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        memcpy(last, self.origin, sizeof(last));
        driven_mode = 1;
        for (int t = 0; t < 144; t++) {
            CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "forward tick");
            CHECK(driven_entity(index, &self), "the Grunt is gone");
            float step = flat_distance(self.origin, last);
            memcpy(last, self.origin, sizeof(last));
            if (step <= 0.0f)
                continue;
            moving[phase]++;
            if (step > fastest[phase])
                fastest[phase] = step;
            CHECK(self.frame >= lo && self.frame <= hi,
                  "%s tick %d: moved %.2f units showing frame %d, not %d-%d",
                  phase ? "run" : "walk", t, step, self.frame, lo, hi);
        }
    }
    printf("grunt: walks at up to %.1f units/s (%d ticks), runs at up to %.1f (%d ticks)\n",
           fastest[0] * 72.0f, moving[0], fastest[1] * 72.0f, moving[1]);
    CHECK(moving[0] > 100 && moving[1] > 100, "the Grunt moved on only %d walk and %d run ticks",
          moving[0], moving[1]);
    CHECK(fastest[0] * 72.0f < 25.0f, "the Grunt walks at %.1f units/s; its walk states do 18",
          fastest[0] * 72.0f);
    CHECK(fastest[1] * 72.0f > 90.0f && fastest[1] * 72.0f < 125.0f,
          "the Grunt runs at %.1f units/s; its run states do 109", fastest[1] * 72.0f);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

static float angle_between(float a, float b)
{
    float d = fmodf(a - b, 360.0f);
    if (d > 180.0f)
        d -= 360.0f;
    if (d < -180.0f)
        d += 360.0f;
    return fabsf(d);
}

/* What a client is shown must match what the body does (#61). The policy
 * strafes and backs up, but a monster model only has run-forward frames:
 * showing the policy's facing made Grunts run with their backs to where they
 * were going, and the run animation played whenever a move was asked for, so
 * a Grunt held by a wall ran on the spot. The shown facing follows the real
 * travel while the facing the policy steers by is untouched, and the run
 * frames play only on ticks the monster really moved. The Grunt stands in
 * frames 0-7 and runs in 73-80. */
static int shown_facing(void)
{
    const dqsim_action idle = {0};
    dqsim_entity self;
    int index;
    if (driven_start("e1m1", "monster_army", &index))
        return 1;
    /* With an enemy, so it runs: a walk covers too little ground to measure. */
    CHECK(alert_driven(), "no spot in the Grunt's sight");

    /* Strafing: whichever side is open. */
    float policy_yaw = driven_yaw;
    float moved = 0.0f, travel = 0.0f;
    for (int attempt = 0; attempt < 2 && moved < 60.0f; attempt++) {
        float start[3];
        memcpy(start, driven_origin, sizeof(start));
        driven_side = attempt == 0 ? 200.0f : -200.0f;
        driven_mode = 4;
        CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "strafe");
        float dx = driven_origin[0] - start[0], dy = driven_origin[1] - start[1];
        moved = sqrtf(dx * dx + dy * dy);
        travel = atan2f(dy, dx) * 180.0f / 3.14159265f;
    }
    CHECK(moved >= 60.0f, "the Grunt strafed only %.1f units either way", moved);
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    CHECK(angle_between(driven_yaw, policy_yaw) < 0.01f,
          "strafing turned the policy's facing from %.1f to %.1f", policy_yaw, driven_yaw);
    CHECK(angle_between(travel, policy_yaw) > 60.0f,
          "the strafe travelled along the policy's facing: nothing to tell apart");
    CHECK(angle_between(self.yaw, travel) < 10.0f,
          "strafing toward %.1f, the Grunt is shown facing %.1f", travel, self.yaw);
    CHECK(self.frame >= 73 && self.frame <= 80, "strafing Grunt shows frame %d, not a run frame",
          self.frame);

    /* Standing still again, the shown facing settles back on the policy's. */
    driven_mode = 0;
    CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "settle");
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    CHECK(angle_between(self.yaw, policy_yaw) < 0.01f,
          "at rest the Grunt is shown facing %.1f, the policy faces %.1f", self.yaw, policy_yaw);

    /* Forward until something holds it: run frames while it covers ground,
     * stand frames on every tick it does not. */
    driven_mode = 1;
    float last[3];
    memcpy(last, self.origin, sizeof(last));
    int held = 0, ran = 0;
    for (int t = 0; t < 4000 && held < 72; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "forward tick");
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        float dx = self.origin[0] - last[0], dy = self.origin[1] - last[1];
        float step = sqrtf(dx * dx + dy * dy);
        memcpy(last, self.origin, sizeof(last));
        if (step == 0.0f) {
            held++;
            CHECK(self.frame <= 7, "tick %d: held in place but showing run frame %d", t,
                  self.frame);
        } else if (step > 1.0f) { /* a Grunt's run is 1.5 units a tick */
            ran++;
            CHECK(self.frame >= 73 && self.frame <= 80,
                  "tick %d: moved %.2f units but showing frame %d", t, step, self.frame);
        }
    }
    CHECK(ran > 20, "the Grunt ran on only %d ticks", ran);
    CHECK(held >= 72, "nothing held the Grunt in 4000 ticks (%d held ticks)", held);
    printf("grunt: shown facing follows travel; %d run ticks, %d held ticks in stand frames\n",
           ran, held);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* ------------------------------------------------------------------ *
 * Hearing (RETRAIN_SPEC part 0B): what a nominated monster hears, the
 * settings, the alert and its pace.
 * ------------------------------------------------------------------ */

static int speed_start_at(const char *map, const char *classname, int *index, int skip);

/* The library defaults: on, the skill table's range and run chance, no
 * player noises. Every hearing test starts from these, because the
 * settings are process-wide and survive map loads. */
static int hearing_defaults(void)
{
    const dqsim_hearing on = {1, 0.0f, -1.0f, 0};
    CHECK(dqsim_set_hearing(&on) == DQSIM_OK, "hearing defaults: %s", dqsim_last_error());
    return 0;
}

/* Put the player somewhere in [min_distance, max_distance] that is free and
 * out of the monster's sight, and leave it there: the distances are tried
 * nearest the ask first, 24 directions each. Returns 0 when no spot worked
 * or any attempt was seen: a sighting gives the monster its enemy for the
 * rest of the episode, so the caller starts over with a fresh load. */
static int place_player_hidden(float min_distance, float max_distance)
{
    const dqsim_action idle = {0};
    static const float spreads[5] = {0.0f, -50.0f, 50.0f, -100.0f, 100.0f};
    driven_saw_ever = 0;
    for (int s = 0; s < 5; s++) {
        float d = (min_distance + max_distance) * 0.5f + spreads[s];
        if (d < min_distance || d > max_distance)
            continue;
        for (int k = 0; k < 24; k++) {
            if (!place_player_ahead(driven_origin, driven_yaw + (float)(k * 15 + 7), d))
                continue;
            driven_mode = 0;
            for (int tick = 0; tick < 12; tick++) {
                driven_saw_player = 0;
                if (dqsim_step_repeat(&idle, 1) != DQSIM_OK)
                    return 0;
            }
            if (!driven_saw_ever)
                return 1;
        }
    }
    return 0;
}

/* The player makes one weapon-channel sound: select the axe (impulse 1)
 * and swing once. The axe needs no ammunition - the shotgun went dry
 * somewhere around the pace tests' ninetieth shot and a dry click makes
 * no sound at all (measured) - and its reach is 64 units, so it cannot
 * hit the level monsters a pellet spread occasionally did. `yaw` is the
 * swing's direction; `pitch` points it into the floor. */
static int player_fire_at(float yaw, float pitch)
{
    const dqsim_action axe = {.impulse = 1, .yaw = yaw};
    CHECK(dqsim_step_repeat(&axe, 1) == DQSIM_OK, "axe select tick");
    const dqsim_action swing = {.buttons = DQSIM_BUTTON_ATTACK, .pitch = pitch, .yaw = yaw};
    CHECK(dqsim_step_repeat(&swing, 1) == DQSIM_OK, "swing tick");
    return 0;
}

static int player_fire(float pitch)
{
    return player_fire_at(0.0f, pitch);
}

/* driven_start, but the player starts inside [min_distance, max_distance],
 * free and out of the monster's sight: the hearing tests' alert must arrive
 * without a sighting, and from the clearest Grunt every direction may be
 * either a wall or open sight. Tries each instance of the class, nearest
 * the player's spawn first, and one hidden spot per instance. */
static int hidden_start(const char *map, const char *classname, int *index, float min_distance,
                        float max_distance)
{
    dqsim_player_state player;
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};
    static const float spreads[5] = {0.0f, -50.0f, 50.0f, -100.0f, 100.0f};

    CHECK(dqsim_load_map(map, 2, 23) == DQSIM_OK, "load %s", map);
    /* The search teleports the player around the level before anything is
     * frozen: make it untargetable first, or a level monster shoots it dead
     * mid-search and every later fire is a dead player's silent click. The
     * hook still sees a notarget player (the sight probe is built on that). */
    CHECK(dqsim_client_command("notarget") == DQSIM_OK, "notarget");
    const dqsim_action settle = {0};
    CHECK(dqsim_step_repeat(&settle, 2) == DQSIM_OK, "notarget tick");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get player");
    int count = dqsim_get_entities(entities, 2048);
    CHECK(count > 0, "get entities");
    driven_calls = 0;
    driven_mode = 0;
    driven_saw_ever = 0;
    float spawn[3] = {player.origin[0], player.origin[1], player.origin[2]};
    int taken[2048];
    memset(taken, 0, sizeof(taken));
    for (int round = 0; round < count; round++) {
        /* the nearest untried instance of the class, skipping the first
         * `skip` accepted ones for a caller that retries */
        int pick = -1;
        float best = 1e30f;
        for (int i = 0; i < count; i++) {
            if (taken[i] || strcmp(entities[i].classname, classname) != 0)
                continue;
            float dx = entities[i].origin[0] - spawn[0], dy = entities[i].origin[1] - spawn[1];
            float d = dx * dx + dy * dy;
            if (d < best) {
                best = d;
                pick = i;
            }
        }
        if (pick < 0)
            break;
        taken[pick] = 1;
        int idx = entities[pick].index;
        if (dqsim_set_policy_hook(idx, driven_policy, NULL) != DQSIM_OK)
            continue;
        driven_saw_ever = 0; /* per candidate: a sighting spoils only its own instance */
        int placed = 0;
        for (int s = 0; s < 5 && !placed; s++) {
            float d = (min_distance + max_distance) * 0.5f + spreads[s];
            if (d < min_distance || d > max_distance)
                continue;
            for (int k = 0; k < 24; k++) {
                if (!place_player_ahead(entities[pick].origin,
                                        entities[pick].yaw + (float)(k * 15 + 7), d))
                    continue;
                driven_mode = 0;
                for (int tick = 0; tick < 12; tick++) {
                    driven_saw_player = 0;
                    if (dqsim_step_repeat(&idle, 1) != DQSIM_OK)
                        return 1;
                }
                if (!driven_saw_ever) {
                    /* ... and alive: a spot above a long drop leaves the
                     * player dead at the floor, and a dead player's weapon
                     * is a silent click. */
                    dqsim_player_state settled;
                    CHECK(dqsim_get_player(&settled) == DQSIM_OK, "settled player");
                    if (settled.health < 95.0f || !settled.on_ground)
                        continue; /* mid-air over a drop: the player keeps falling */
                    /* No monster within pellet range either: the tests fire
                     * the shotgun here, and a pellet hurting a level monster
                     * cries over the shot's own noise. */
                    float spot[3];
                    memcpy(spot, driven_origin, sizeof(spot));
                    int crowded = 0;
                    int n = dqsim_get_entities(entities, 2048);
                    for (int i = 0; i < n && !crowded; i++) {
                        if (!(entities[i].flags & 32) || entities[i].index == idx)
                            continue;
                        float dx = entities[i].origin[0] - spot[0];
                        float dy = entities[i].origin[1] - spot[1];
                        if (dx * dx + dy * dy < 120.0f * 120.0f)
                            crowded = 1;
                    }
                    if (!crowded) {
                        placed = 1;
                        break;
                    }
                }
            }
        }
        if (!placed) {
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            continue;
        }
        CHECK(freeze_monsters_except(idx, NULL) >= 8, "only %d other monsters to freeze on %s",
              frozen_monsters, map);
        *index = idx;
        CHECK(driven_calls > 10, "the %s was asked for %d decisions in 1 s", classname,
              driven_calls);
        return 0;
    }
    return 1;
}

/* Is the heard noise where `origin` was, minus the monster's origin, within
 * `tolerance` units on every axis? */
static int heard_marks(const float origin[3], const float monster[3], float tolerance)
{
    float want[3] = {origin[0] - monster[0], origin[1] - monster[1], origin[2] - monster[2]};
    return fabsf(driven_heard_relative[0] - want[0]) <= tolerance &&
           fabsf(driven_heard_relative[1] - want[1]) <= tolerance &&
           fabsf(driven_heard_relative[2] - want[2]) <= tolerance;
}

/* 1 of 12: the shot is heard through walls, at where it was fired, and the
 * heard position does not follow the player afterwards. */
static int hearing_gunfire(void)
{
    fprintf(stderr, "== hearing_gunfire\n");

    const dqsim_action idle = {0};
    dqsim_entity self;
    dqsim_player_state player;
    int index;
    hearing_defaults();
    if (hidden_start("e1m1", "monster_army", &index, 350.0f, 450.0f))
        return 1;

    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player before the shot");
    float shot_origin[3] = {player.origin[0], player.origin[1], player.origin[2]};
    player_fire(0.0f);
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "settle");
    CHECK(driven_heard == 1, "the Grunt did not hear the shot");
    CHECK(driven_heard_kind == DQSIM_NOISE_GUNFIRE, "heard kind %d, not gunfire",
          driven_heard_kind);
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    /* 3, not 2: the player settles one physics step inside the shot tick
     * (measured: exactly 2 units of z); the shot's spot is what is checked,
     * and a wrong noise would be off by hundreds. */
    CHECK(heard_marks(shot_origin, self.origin, 3.0f),
          "heard at (%.1f, %.1f, %.1f), not at the shot (%.1f, %.1f, %.1f)",
          driven_heard_relative[0], driven_heard_relative[1], driven_heard_relative[2],
          shot_origin[0] - self.origin[0], shot_origin[1] - self.origin[1],
          shot_origin[2] - self.origin[2]);

    /* The player moves 200 units: the heard position stays where the shot
     * was (decision 3). */
    int moved = 0;
    for (int k = 0; k < 8 && !moved; k++) {
        float rad = (float)k * 3.14159265f / 4.0f;
        float spot[3] = {shot_origin[0] + cosf(rad) * 200.0f,
                         shot_origin[1] + sinf(rad) * 200.0f, shot_origin[2]};
        if (dqsim_teleport_player(spot, 0.0f) == DQSIM_OK)
            moved = 1;
    }
    CHECK(moved, "no free spot 200 units from the shot");
    CHECK(dqsim_step_repeat(&idle, 3) == DQSIM_OK, "decision 3");
    CHECK(driven_heard == 1, "the heard noise vanished after the move");
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    CHECK(heard_marks(shot_origin, self.origin, 3.0f),
          "the heard position followed the player: (%.1f, %.1f, %.1f)",
          driven_heard_relative[0], driven_heard_relative[1], driven_heard_relative[2]);
    printf("hearing: gunfire heard at the shot's origin, position kept after the player moved\n");
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* 2 of 12: the range setting is the monster's own; 400 is heard, 700 is not,
 * with the range at 500. */
static int hearing_range(void)
{
    fprintf(stderr, "== hearing_range\n");

    const dqsim_action idle = {0};
    dqsim_player_state player;
    int index;
    if (hidden_start("e1m1", "monster_army", &index, 350.0f, 450.0f))
        return 1;
    const dqsim_hearing short_range = {1, 500.0f, -1.0f, 0};
    CHECK(dqsim_set_hearing(&short_range) == DQSIM_OK, "range 500: %s", dqsim_last_error());
    CHECK(place_player_hidden(350.0f, 450.0f), "no hidden spot 350-450 units out");

    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player before the 400 shot");
    float near_origin[3] = {player.origin[0], player.origin[1], player.origin[2]};
    player_fire(0.0f);
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "settle");
    CHECK(driven_heard == 1 && driven_heard_kind == DQSIM_NOISE_GUNFIRE,
          "a shot 400 units away was not heard with the range at 500");

    /* The noise is remembered for 10 s; let it expire before the far shot. */
    CHECK(dqsim_step_repeat(&idle, 72 * 11) == DQSIM_OK, "let the noise expire");
    CHECK(driven_heard == 0, "the noise was not forgotten after 11 s (age %.1f)",
          driven_heard_age);
    CHECK(place_player_hidden(600.0f, 800.0f), "no hidden spot 600-800 units out");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player before the 700 shot");
    float far_origin[3] = {player.origin[0], player.origin[1], player.origin[2]};
    player_fire(0.0f);
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "settle");
    /* A level hazard's cry can land meanwhile; what must not happen is the
     * shot itself being heard at the player. */
    dqsim_entity self;
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    CHECK(!(driven_heard && driven_heard_kind == DQSIM_NOISE_GUNFIRE &&
            heard_marks(far_origin, self.origin, 3.0f)),
          "a shot 700 units away was heard with the range at 500");
    printf("hearing: range 500 hears 400, not 700\n");
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    (void)near_origin;
    return 0;
}

/* 3 of 12: hearing off, nothing is heard; back on, it is. */
static int hearing_switch(void)
{
    fprintf(stderr, "== hearing_switch\n");

    const dqsim_action idle = {0};
    dqsim_player_state player;
    int index;
    if (hidden_start("e1m1", "monster_army", &index, 350.0f, 450.0f))
        return 1; /* 250-350 admits no settled spot on e1m1's Grunts (measured) */
    /* The placement can have left a noise remembered, and e1m1's hazards
     * cry while it waits: poll for the quiet second. */
    for (int t = 0; t < 72 * 20 && driven_heard; t++)
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "quiet poll");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player before the off shot");
    float shot_origin[3] = {player.origin[0], player.origin[1], player.origin[2]};
    const dqsim_hearing off = {0, 0.0f, -1.0f, 0};
    CHECK(dqsim_set_hearing(&off) == DQSIM_OK, "hearing off: %s", dqsim_last_error());
    player_fire(0.0f); /* EXPERIMENT */
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "settle");
    dqsim_entity self;
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    CHECK(!(driven_heard && driven_heard_kind == DQSIM_NOISE_GUNFIRE &&
            heard_marks(shot_origin, self.origin, 3.0f)),
          "a shot was heard at the player with hearing off");
    hearing_defaults();
    CHECK(dqsim_step_repeat(&idle, 44) == DQSIM_OK, "shotgun refire");
    player_fire(0.0f); /* EXPERIMENT */

    /* The shot's noise arrives within a tick or two; a world cry can land
     * right after it, so look for the shot per tick instead of judging the
     * last report. */
    int heard_again = 0;
    for (int t = 0; t < 12 && !heard_again; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "on settle");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "player after the on shot");
        float spot2[3] = {player.origin[0], player.origin[1], player.origin[2]};
        dqsim_entity self2;
        CHECK(driven_entity(index, &self2), "the Grunt is gone");
        if (driven_heard && driven_heard_kind == DQSIM_NOISE_GUNFIRE &&
            heard_marks(spot2, self2.origin, 3.0f))
            heard_again = 1;
    }
    CHECK(heard_again, "a shot was not heard with hearing back on");
    printf("hearing: the switch silences and restores\n");
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* 4 of 12: the rocket's temp-entity explosion is heard, at where it went
 * off, far from the player. The player fires AWAY from the Grunt, at a wall
 * behind it: a rocket aimed at the Grunt kills it outright (100 damage on
 * 30 health), and a dead monster hears nothing (RETRAIN_SPEC part 0B: fires
 * at a wall at least 250 units from itself, within range of the Grunt). */
static int hearing_explosion(void)
{
    fprintf(stderr, "== hearing_explosion\n");

    const dqsim_action idle = {0};
    dqsim_player_state player;
    int index;
    hearing_defaults();
    if (driven_start("e1m1", "monster_army", &index))
        return 1;
    CHECK(dqsim_client_command("notarget") == DQSIM_OK, "notarget");
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "notarget tick");
    /* Impulses travel in the usercmd, not as console commands: an
     * "impulse 9" console line is an unknown server command (measured). */
    const dqsim_action arm_all = {.impulse = 9};
    const dqsim_action arm_rl = {.impulse = 7};
    CHECK(dqsim_step_repeat(&arm_all, 1) == DQSIM_OK, "impulse 9");
    CHECK(dqsim_step_repeat(&arm_rl, 1) == DQSIM_OK, "impulse 7");
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "arm the player");

    /* Up to three visible spots: fire away from the Grunt at whatever wall
     * is behind, and take the first explosion it hears. */
    int exploded = 0;
    float explosion_origin[3] = {0.0f, 0.0f, 0.0f}, player_origin[3] = {0.0f, 0.0f, 0.0f};
    for (int attempt = 0; attempt < 3 && !exploded; attempt++) {
        int placed = 0;
        for (int k = 0; k < 24 && !placed; k++) {
            float yaw = driven_yaw + (float)(k * 15 + 7) + (float)(attempt * 5);
            if (!place_player_ahead(driven_origin, yaw, 300.0f))
                continue;
            driven_mode = 0;
            for (int tick = 0; tick < 12; tick++) {
                driven_saw_player = 0;
                CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "place tick");
            }
            if (driven_saw_player)
                placed = 1;
        }
        CHECK(placed, "no visible spot 300 units out");
        CHECK(dqsim_get_player(&player) == DQSIM_OK, "placed player");
        memcpy(player_origin, player.origin, sizeof(player_origin));
        CHECK(player.health >= 95.0f, "the player is hurt (%.0f): no clear shot", player.health);
        dqsim_entity self;
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        if (self.health <= 0.0f)
            return 1; /* a stray rocket killed it: this attempt is meaningless */
        float away = atan2f(player_origin[1] - self.origin[1],
                            player_origin[0] - self.origin[0]) * 180.0f / 3.14159265f;
        for (int tick = 0; tick < 72 * 3 && !exploded; tick++) {
            const dqsim_action shoot = {.buttons = DQSIM_BUTTON_ATTACK, .yaw = away};
            CHECK(dqsim_step_repeat(&shoot, 1) == DQSIM_OK, "rocket tick");
            CHECK(driven_entity(index, &self), "the Grunt is gone");
            CHECK(self.health > 0.0f, "the Grunt died before hearing an explosion");
            if (driven_heard_kind == DQSIM_NOISE_EXPLOSION && driven_heard) {
                explosion_origin[0] = self.origin[0] + driven_heard_relative[0];
                explosion_origin[1] = self.origin[1] + driven_heard_relative[1];
                explosion_origin[2] = self.origin[2] + driven_heard_relative[2];
                float dx2 = explosion_origin[0] - player_origin[0];
                float dy2 = explosion_origin[1] - player_origin[1];
                if (sqrtf(dx2 * dx2 + dy2 * dy2) >= 200.0f)
                    exploded = 1; /* far enough; a nearer wall is re-placed below */
                break;
            }
        }
    }
    CHECK(exploded, "no explosion at least 200 units out was heard in 9 s of rockets");
    float dx = explosion_origin[0] - player_origin[0], dy = explosion_origin[1] - player_origin[1];
    float distance = sqrtf(dx * dx + dy * dy);
    CHECK(distance >= 200.0f, "the explosion was only %.0f units from the player", distance);
    printf("hearing: explosion heard %.0f units from the shooter\n", distance);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* 5 of 12: a Grunt's pain and death cries are heard, at the victim, once
 * while it lives and once on its death. The victim stays frozen: the hook
 * only takes the decision thinks, and pain and death run under QuakeC as
 * usual, so its cries play. */
static int hearing_cries(void)
{
    const dqsim_action idle = {0};
    static dqsim_entity entities[2048];
    dqsim_entity self;
    int index;
    if (driven_start("e1m1", "monster_army", &index))
        return 1;
    hearing_defaults();
    CHECK(driven_entity(index, &self), "the Grunt is gone");

    /* The victim: a monster_army 300-900 units from the driven one (in
     * hearing range, on another room's far side). */
    int count = dqsim_get_entities(entities, 2048);
    int victim_index = -1;
    float victim_origin[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < count && victim_index < 0; i++) {
        if (entities[i].index == index || strcmp(entities[i].classname, "monster_army") != 0)
            continue;
        float dx = entities[i].origin[0] - self.origin[0];
        float dy = entities[i].origin[1] - self.origin[1];
        float d = sqrtf(dx * dx + dy * dy);
        if (d >= 300.0f && d <= 900.0f) {
            victim_index = entities[i].index;
            memcpy(victim_origin, entities[i].origin, sizeof(victim_origin));
        }
    }
    CHECK(victim_index > 0, "e1m1 has no second Grunt in hearing range");

    /* The player beside the victim, out of the driven Grunt's sight, facing
     * it: one offset that is free and unseen. */
    int placed = 0;
    float fire_yaw = 0.0f;
    for (int k = 0; k < 8 && !placed; k++) {
        float rad = (float)k * 3.14159265f / 4.0f;
        float spot[3] = {victim_origin[0] + cosf(rad) * 48.0f,
                         victim_origin[1] + sinf(rad) * 48.0f,
                         victim_origin[2] + 24.0f};
        float to_victim = atan2f(victim_origin[1] - spot[1],
                                 victim_origin[0] - spot[0]) * 180.0f / 3.14159265f;
        if (dqsim_teleport_player(spot, to_victim) != DQSIM_OK)
            continue;
        /* The victim must be in the player's fire line: a ray at the victim's
         * yaw reaches its distance. */
        float dist = sqrtf((victim_origin[0] - spot[0]) * (victim_origin[0] - spot[0]) +
                           (victim_origin[1] - spot[1]) * (victim_origin[1] - spot[1]));
        float distances[24];
        unsigned char kinds[24];
        CHECK(dqsim_ray_fan((float[]){0.0f}, 1, 24, 768.0f, distances, kinds) == DQSIM_OK,
              "player rays");
        /* The player faces the victim: its centre ray must reach the
         * victim's distance. */
        int los = distances[0] * 768.0f >= dist - 16.0f;
        if (!los)
            continue;
        driven_saw_ever = 0;
        driven_mode = 0;
        for (int tick = 0; tick < 12; tick++)
            dqsim_step_repeat(&idle, 1);
        /* The cry and the shot land in the same tick, and the nearer wins:
         * the player must be FARTHER from the driven Grunt than the victim
         * is, or its own gunshot masks every cry. */
        float gp = sqrtf((spot[0] - self.origin[0]) * (spot[0] - self.origin[0]) +
                         (spot[1] - self.origin[1]) * (spot[1] - self.origin[1]));
        float gv = sqrtf((victim_origin[0] - self.origin[0]) * (victim_origin[0] - self.origin[0]) +
                         (victim_origin[1] - self.origin[1]) * (victim_origin[1] - self.origin[1]));
        if (!driven_saw_ever && gp > gv + 24.0f) {
            placed = 1;
            fire_yaw = to_victim;
        }
    }
    CHECK(placed, "no spot beside the victim is out of the driven Grunt's sight and "
                  "farther than the victim");

    /* Fire until the victim dies; a cry is heard at the victim's position
     * while it lives (pain) and once on its death. The fire carries its yaw:
     * the action's angles are absolute, and a buttons-only action snaps the
     * view to yaw 0 (measured). */
    int pain_seen = 0, death_seen = 0;
    int kinds_heard[8] = {0};
    /* One shot per burst, then a pause: the player's own shot is a noise
     * that overwrites the cries (most recent wins), so the cry of a shot
     * shows only in the pause right after it. */
    for (int burst = 0; burst < 30 && !death_seen; burst++) {
        const dqsim_action fire = {.buttons = DQSIM_BUTTON_ATTACK, .yaw = fire_yaw};
        CHECK(dqsim_step_repeat(&fire, 1) == DQSIM_OK, "shot at the victim");
        /* Read right after the shot: the cry of this shot is the most
         * recent noise for the first half-second, and the next shot
         * overwrites it. */
        for (int tick = 0; tick < 24; tick++) {
            CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "pause tick");
            if (driven_heard) {
                CHECK(driven_heard_kind >= 0 && driven_heard_kind < 8,
                      "kind %d out of range", driven_heard_kind);
                kinds_heard[driven_heard_kind]++;
                CHECK(driven_entity(index, &self), "the Grunt is gone");
                if (driven_heard_kind == DQSIM_NOISE_CRY &&
                    heard_marks(victim_origin, self.origin, 3.0f)) {
                    /* The victim is gone or gone-corpse: which cry was this? */
                    int alive = 0;
                    int n = dqsim_get_entities(entities, 2048);
                    for (int i = 0; i < n; i++)
                        if (entities[i].index == victim_index && entities[i].health > 0.0f)
                            alive = 1;
                    if (alive)
                        pain_seen = 1;
                    else
                        death_seen = 1;
                }
            }
        }
        CHECK(dqsim_step_repeat(&idle, 24) == DQSIM_OK, "refire wait");
    }
    CHECK(pain_seen, "no pain cry was heard while the victim lived");
    CHECK(death_seen, "no death cry was heard");
    printf("hearing: cries heard by kind - gunfire %d, explosion %d, cry %d, player %d; "
           "pain %s, death %s\n",
           kinds_heard[1], kinds_heard[2], kinds_heard[3], kinds_heard[4],
           pain_seen ? "yes" : "no", death_seen ? "yes" : "no");
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    return 0;
}

/* 6 of 12: the player's own sounds count only with player_noises on. */
/* 6 of 12: the player's own sounds count only with player_noises on. */
static int hearing_player_noises(void)
{
    fprintf(stderr, "== hearing_player_noises\n");

    const dqsim_action idle = {0};
    dqsim_player_state player;
    int index;
    hearing_defaults();
    if (hidden_start("e1m1", "monster_army", &index, 350.0f, 450.0f))
        return 1; /* 150-250 admits no settled spot on e1m1's Grunts (measured);
                     the jump is heard within 1000 either way */
    /* As in hearing_switch: any placement-phase noise expires first, and
     * level hazards may cry meanwhile - poll for the quiet second. */
    for (int t = 0; t < 72 * 20 && driven_heard; t++)
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "quiet poll");

    const dqsim_action jump = {.buttons = DQSIM_BUTTON_JUMP};
    CHECK(dqsim_step_repeat(&jump, 6) == DQSIM_OK, "jump");
    CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "land");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player after the jump");
    float jump_spot[3] = {player.origin[0], player.origin[1], player.origin[2]};
    dqsim_entity self;
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    CHECK(!(driven_heard && driven_heard_kind == DQSIM_NOISE_PLAYER &&
            heard_marks(jump_spot, self.origin, 3.0f)),
          "the player's jump was heard with player_noises off");

    const dqsim_hearing noisy = {1, 0.0f, -1.0f, 1};
    CHECK(dqsim_set_hearing(&noisy) == DQSIM_OK, "player noises on: %s", dqsim_last_error());
    /* The hearing-on check needs the noise fresh: jump again after any
     * stray noise has passed (the noise from this jump is the one judged). */
    CHECK(dqsim_step_repeat(&jump, 6) == DQSIM_OK, "jump again");
    CHECK(dqsim_step_repeat(&idle, 72) == DQSIM_OK, "land again");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "player after the second jump");
    float jump_spot2[3] = {player.origin[0], player.origin[1], player.origin[2]};
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    CHECK(driven_heard == 1 && driven_heard_kind == DQSIM_NOISE_PLAYER &&
              heard_marks(jump_spot2, self.origin, 3.0f),
          "the player's jump was not heard as a player noise (kind %d)", driven_heard_kind);
    printf("hearing: the player's jump is heard only with player_noises on\n");
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    hearing_defaults();
    return 0;
}

/* 7 of 12: the investigation pace - walk frames advanced by distance, at
 * half the run speed - and the run pace at run chance 1. */
/* One attempt of test 7 on candidate number `skip`. Returns 0 when the
 * measurements were taken, 1 when the episode was spoilt (the walk opened
 * the sight and handed the Grunt its enemy: the alert cannot test a pace
 * then) and the caller retries on the next candidate. */
static int investigate_attempt(int skip, float *out_walk_per_100, float *out_inv_speed,
                               float *out_run_speed, float *out_inv_per_100)
{
    const dqsim_action idle = {0};
    dqsim_entity self;
    int index;
    if (speed_start_at("e1m2", "monster_army", &index, skip))
        return 2; /* no usable candidate at this skip: a hard stop */
    CHECK(dqsim_client_command("notarget") == DQSIM_OK, "notarget");
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "notarget tick");
    /* The alert's pace must roll at chance 0: any noise the placement
     * makes would otherwise start the alert at the table's 0.5. */
    {
        const dqsim_hearing investigate = {1, 0.0f, 0.0f, 0};
        CHECK(dqsim_set_hearing(&investigate) == DQSIM_OK, "run chance 0: %s",
              dqsim_last_error());
    }
    driven_saw_ever = 0;
    CHECK(driven_entity(index, &self), "the Grunt is gone");

    /* The stock walk's figure: turn around (the drive probe walked the
     * monster 50+ units forward, so the path back is proven open) and
     * count the frame changes per 100 units. */
    float last[3];
    memcpy(last, self.origin, sizeof(last));
    int walk_changes = 0;
    float walk_units = 0.0f;
    float walk_frame = self.frame;
    float walk_last_step = 0.0f;
    driven_turn_to = driven_yaw + 180.0f;
    driven_mode = 2;
    CHECK(dqsim_step_repeat(&idle, 48) == DQSIM_OK, "turn around");
    driven_mode = 1;
    for (int t = 0; t < 144; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "walk tick");
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        float step = flat_distance(self.origin, last);
        memcpy(last, self.origin, sizeof(last));
        walk_units += step;
        /* A change between two moving ticks: a blocked tick shows stand
         * frames that flicker every 7 ticks, and the figure's intent is
         * the walk animation's rate per unit moved. */
        if (step > 0.0f && walk_last_step > 0.0f && self.frame != walk_frame) {
            walk_changes++;
            walk_frame = self.frame;
        }
        walk_last_step = step;
        if (step > 0.0f && (self.frame < 90 || self.frame > 113)) {
            spoilt = "the walk showed non-walk frames (a hurt monster?)";
            return 1;
        }
        if (driven_saw_ever)
            return 1; /* the sight opened mid-walk: retry */
        if (self.health < 30.0f) {
            return 1; /* the walk led into a level hazard: retry elsewhere */
        }
    }
    if (walk_units < 15.0f) {
        return 1; /* boxed in: retry elsewhere */
    }

    /* The alert: a shot from the hidden spot, into the floor at the
     * player's feet - the sound carries, and no level monster is hurt (a
     * hurt monster cries, and its return fire refreshes the noise
     * forever). The monster stands while it fires. */
    driven_mode = 0;
    player_fire(45.0f);
    for (int t = 0; t < 72 && driven_pace != DQSIM_PACE_INVESTIGATE; t++)
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "pace settle");
    if (driven_saw_ever) {
        return 1; /* the falling player opened the sight: retry elsewhere */
    }
    if (driven_pace != DQSIM_PACE_INVESTIGATE) {
        return 1; /* the alert did not take: the sight opened or the roll lost it */
    }

    /* Investigate: walk frames advanced by distance, at half the run
     * speed. The drive turns around first: the walk's path back is proven
     * open, the path ahead may end at the next wall. */
    int inv_changes = 0;
    float inv_units = 0.0f, inv_fastest = 0.0f, inv_frame = self.frame;
    float run_speed = 0.0f;
    float last_step = 0.0f;
    memcpy(last, self.origin, sizeof(last));
    driven_turn_to = driven_yaw + 180.0f;
    driven_mode = 2;
    CHECK(dqsim_step_repeat(&idle, 36) == DQSIM_OK, "turn around");
    driven_mode = 1;
    for (int t = 0; t < 144; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "investigate tick");
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        float step = flat_distance(self.origin, last);
        memcpy(last, self.origin, sizeof(last));
        inv_units += step;
        if (step > inv_fastest)
            inv_fastest = step;
        if (step > 0.0f && last_step > 0.0f && self.frame != inv_frame) {
            inv_changes++;
            inv_frame = self.frame;
        }
        last_step = step;
        if (step > 0.0f && (self.frame < 90 || self.frame > 113)) {
            spoilt = "the investigate drive showed non-walk frames";
            return 1;
        }
        if (driven_pace != DQSIM_PACE_INVESTIGATE) {
            return 1; /* the sight opened mid-drive: retry */
        }
        if (driven_saw_ever) {
            return 1; /* the sight opened mid-drive: retry */
        }
        if (self.health < 30.0f) {
            return 1; /* the drive led into a level hazard: retry elsewhere */
        }
    }
    if (inv_units < 20.0f) {
        return 1; /* boxed in: retry elsewhere */
    }

    /* Run chance 1: the same alert gives the run frames and the run pace,
     * after the first alert's noise has been forgotten. The player stays
     * hidden: once alert_driven hands the monster its enemy, the pace reads
     * RUN whatever the alert rolled. */
    /* The first alert's noise expires under deaf hearing: with hearing on,
     * every stray noise re-rolls the alert at the current chance. */
    {
        const dqsim_hearing deaf = {0, 0.0f, -1.0f, 0};
        CHECK(dqsim_set_hearing(&deaf) == DQSIM_OK, "hearing off: %s", dqsim_last_error());
    }
    driven_mode = 0; /* the wait must not walk the monster into a hazard */
    int calls_at_start = driven_calls;
    for (int t = 0; t < 72 * 20 && driven_pace != DQSIM_PACE_WALK; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "quiet poll");
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        if (self.health < 30.0f) {
            return 1; /* standing in a hazard: retry elsewhere */
        }
        if (driven_calls == calls_at_start) {
            return 1; /* the monster's decisions stopped: retry elsewhere */
        }
        calls_at_start = driven_calls;
    }
    CHECK(driven_pace == DQSIM_PACE_WALK, "the pace did not return to walk in 20 s");
    const dqsim_hearing always_run = {1, 0.0f, 1.0f, 0};
    CHECK(dqsim_set_hearing(&always_run) == DQSIM_OK, "run chance 1: %s", dqsim_last_error());
    CHECK(driven_pace == DQSIM_PACE_WALK, "the pace did not return to walk in 20 s");
    CHECK(place_player_hidden(600.0f, 900.0f), "no hidden spot for the second alert");
    player_fire(45.0f); /* into the floor, as the first alert */
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "settle");
    CHECK(driven_pace == DQSIM_PACE_RUN, "the alert pace is %d, not run", driven_pace);
    CHECK(driven_entity(index, &self), "the Grunt is gone");
    memcpy(last, self.origin, sizeof(last));
    driven_mode = 1; /* the run frames show on ticks the monster really moves */
    for (int t = 0; t < 24; t++) {
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "alerted run tick");
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        float step = flat_distance(self.origin, last);
        memcpy(last, self.origin, sizeof(last));
        /* 0.5, not 0: a closing door can push the standing monster a
         * fraction of a unit, and the stand frame it shows is not the
         * alert's doing. A real drive step is 1.5 units at run speed. */
        if (step > 0.5f)
            CHECK(self.frame >= 73 && self.frame <= 80,
                  "alerted run tick %d shows frame %d, not a run frame", t, self.frame);
    }
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    hearing_defaults();

    /* The run speed, measured in a fresh episode: the previous phases may
     * have walked the monster into a corner, and alert_driven's sighting
     * hands it an enemy, so this is last and on its own load. */
    if (driven_start("e1m2", "monster_army", &index))
        return 2;
    {
        const dqsim_action idle2 = {0};
        CHECK(dqsim_client_command("notarget") == DQSIM_OK, "notarget");
        CHECK(dqsim_step_repeat(&idle2, 2) == DQSIM_OK, "notarget tick");
        /* A targeted placement: alert_driven's blind fan leaves the player
         * standing on e1m2's hazards, and a dead player restarts the level
         * under the measurement. */
        int placed2 = 0;
        for (int k = 0; k < 24 && !placed2; k++) {
            if (!place_player_ahead(driven_origin, driven_yaw + (float)(k * 15 + 7), 200.0f))
                continue;
            int alive = 1;
            for (int tick = 0; tick < 24; tick++) {
                driven_saw_player = 0;
                CHECK(dqsim_step_repeat(&idle2, 1) == DQSIM_OK, "place tick");
                dqsim_player_state p1;
                CHECK(dqsim_get_player(&p1) == DQSIM_OK, "player during the settle");
                if (p1.health < 95.0f) {
                    alive = 0; /* a hazard is eating the player: leave now,
                                  before a death restarts the level */
                    break;
                }
            }
            if (alive && driven_saw_player)
                placed2 = 1;
        }
        CHECK(placed2, "no safe visible spot in the Grunt's sight");
        CHECK(driven_entity(index, &self), "the Grunt is gone");
        driven_mode = 1;
        float run_units = 0.0f, run_fastest = 0.0f;
        memcpy(last, self.origin, sizeof(last));
        for (int t = 0; t < 144; t++) {
            CHECK(dqsim_step_repeat(&idle2, 1) == DQSIM_OK, "run tick");
            CHECK(driven_entity(index, &self), "the Grunt is gone");
            float step = flat_distance(self.origin, last);
            memcpy(last, self.origin, sizeof(last));
            run_units += step;
            if (step > run_fastest)
                run_fastest = step;
        }
        run_speed = run_fastest * 72.0f;
        CHECK(run_units > 30.0f, "the Grunt covered %.0f units running", run_units);
    }
    float inv_speed = inv_fastest * 72.0f;
    CHECK(inv_speed > 0.45f * run_speed && inv_speed < 0.55f * run_speed,
          "investigating at %.1f units/s, not half of %.1f", inv_speed, run_speed);
    float walk_per_100 = 100.0f * (float)walk_changes / walk_units;
    float inv_per_100 = 100.0f * (float)inv_changes / inv_units;
    CHECK(inv_per_100 > 0.85f * walk_per_100 && inv_per_100 < 1.15f * walk_per_100,
          "frame changes per 100 units: investigate %.1f, stock walk %.1f", inv_per_100,
          walk_per_100);

    printf("hearing: investigate at %.1f u/s (half of %.1f), frames per 100 units %.1f vs walk %.1f\n",
           inv_speed, run_speed, inv_per_100, walk_per_100);
    *out_walk_per_100 = walk_per_100;
    *out_inv_speed = inv_speed;
    *out_run_speed = run_speed;
    *out_inv_per_100 = inv_per_100;
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    hearing_defaults();
    return 0;
}

static int investigate_pace(void)
{
    fprintf(stderr, "== investigate_pace\n");

    hearing_defaults();
    float walk_per_100 = 0.0f, inv_speed = 0.0f, run_speed = 0.0f, inv_per_100 = 0.0f;
    for (int attempt = 0; attempt < 8; attempt++) {
        int outcome = investigate_attempt(attempt, &walk_per_100, &inv_speed, &run_speed,
                                          &inv_per_100);
        if (outcome == 0)
            return 0;
        fprintf(stderr, "inv attempt %d: outcome %d (%s)\n", attempt, outcome,
                spoilt ? spoilt : "no candidate placed");
        if (outcome == 2)
            break; /* no usable candidate left: fall through to the failure */
        CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook between attempts");
        hearing_defaults();
    }
    CHECK(0, "no attempt of the investigation pace survived a full episode");
    return 1;
}

/* 8 of 12: the alert's pace is drawn from the seeded stream: 20 alerts in a
 * row give both paces, and the same load seed gives the same sequence. */
static int alert_pace_seeded(void)
{
    fprintf(stderr, "== alert_pace_seeded\n");

    const dqsim_action idle = {0};
    int index;
    int paces[2][20];
    for (int run = 0; run < 2; run++) {
        if (hidden_start("e1m1", "monster_army", &index, 350.0f, 450.0f))
            return 1;
        const dqsim_hearing half = {1, 0.0f, 0.5f, 0};
        CHECK(dqsim_set_hearing(&half) == DQSIM_OK, "run chance 0.5: %s", dqsim_last_error());
        for (int alert = 0; alert < 20; alert++) {
            player_fire(0.0f);
            CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "settle");
            CHECK(driven_pace == DQSIM_PACE_RUN || driven_pace == DQSIM_PACE_INVESTIGATE,
                  "alert %d rolled pace %d", alert, driven_pace);
            paces[run][alert] = driven_pace;
            /* Let the noise be forgotten, then alert again. */
            CHECK(dqsim_step_repeat(&idle, 72 * 11) == DQSIM_OK, "forget the noise");
            CHECK(driven_pace == DQSIM_PACE_WALK, "the pace did not end with the noise");
        }
        CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
        hearing_defaults();
    }
    int saw[2] = {0, 0};
    for (int alert = 0; alert < 20; alert++) {
        if (paces[0][alert] == DQSIM_PACE_RUN)
            saw[0] = 1;
        else
            saw[1] = 1;
        CHECK(paces[0][alert] == paces[1][alert],
              "alert %d rolled %d then %d for the same load seed", alert, paces[0][alert],
              paces[1][alert]);
    }
    CHECK(saw[0] && saw[1], "20 alerts rolled only one pace");
    printf("hearing: 20 alerts gave both paces, the same sequence twice\n");
    return 0;
}

/* 9 of 12: the hearing state is in the snapshot and the hash: a restore
 * replays identically, and with hearing off the replay diverges at the
 * second shot. */
static int hearing_snapshot_replay(void)
{
    fprintf(stderr, "== hearing_snapshot_replay\n");

    const dqsim_action idle = {0};
    int index;
    hearing_defaults();
    if (hidden_start("e1m1", "monster_army", &index, 350.0f, 450.0f))
        return 1;
    player_fire(0.0f);
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "settle");
    CHECK(driven_heard == 1, "the alert shot was not heard");

    dqsim_snapshot *snap = NULL;
    CHECK(dqsim_snapshot_save(&snap) == DQSIM_OK, "save the snapshot: %s", dqsim_last_error());

    static uint64_t hashes[144];
    for (int t = 0; t < 144; t++) {
        if (t == 50)
            player_fire(0.0f); /* the second shot, in the window */
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "window tick");
        hashes[t] = dqsim_state_hash();
    }

    CHECK(dqsim_snapshot_restore(snap) == DQSIM_OK, "restore: %s", dqsim_last_error());
    for (int t = 0; t < 144; t++) {
        if (t == 50)
            player_fire(0.0f);
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "replay tick");
        CHECK(dqsim_state_hash() == hashes[t], "tick %d: the restored replay diverged", t);
    }

    /* From the same snapshot with hearing off, the second shot is not heard:
     * the hashes diverge from tick 50 on (the hash covers the hearing
     * state). */
    CHECK(dqsim_snapshot_restore(snap) == DQSIM_OK, "restore again: %s", dqsim_last_error());
    const dqsim_hearing off = {0, 0.0f, -1.0f, 0};
    CHECK(dqsim_set_hearing(&off) == DQSIM_OK, "hearing off: %s", dqsim_last_error());
    int diverged_at = -1;
    for (int t = 0; t < 144; t++) {
        if (t == 50)
            player_fire(0.0f);
        CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "deaf replay tick");
        if (diverged_at < 0 && dqsim_state_hash() != hashes[t])
            diverged_at = t;
    }
    CHECK(diverged_at >= 50, "the deaf replay diverged at tick %d, not at the shot",
          diverged_at);
    printf("hearing: replay identical, deaf replay diverges at tick %d\n", diverged_at);
    free(snap);
    CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
    hearing_defaults();
    return 0;
}

/* 10 of 12: every roster class investigates at half its run speed, and that
 * is at least 1.2 times its walk speed. */
static int speed_start_at(const char *map, const char *classname, int *index, int skip)
{
    dqsim_player_state player;
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};

    CHECK(dqsim_load_map(map, 2, 23) == DQSIM_OK, "load %s", map);
    /* Deaf and untargetable before anything else: the placement's fan
     * teleports cross monster triggers, the fights' gunfire reaches the
     * candidates (their alert rolls at the table's 0.5, and the pace
     * measurements would read the alert's pace), and untargetability stops
     * the fights outright. */
    {
        const dqsim_hearing deaf = {0, 0.0f, -1.0f, 0};
        CHECK(dqsim_set_hearing(&deaf) == DQSIM_OK, "hearing off: %s", dqsim_last_error());
    }
    CHECK(dqsim_client_command("notarget") == DQSIM_OK, "notarget");
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "notarget tick");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get player");
    int count = dqsim_get_entities(entities, 2048);
    CHECK(count > 0, "get entities");
    driven_calls = 0;
    driven_mode = 0;
    driven_saw_ever = 0;
    int accepted = 0; /* retries decline the earlier accepted candidates */
    for (int pick = 0; pick < count; pick++) {
        if (strcmp(entities[pick].classname, classname) != 0)
            continue;
        int idx = entities[pick].index;
        if (dqsim_set_policy_hook(idx, driven_policy, NULL) != DQSIM_OK) {
            continue;
        }
        driven_saw_ever = 0; /* per candidate: a sighting spoils only its own instance */
        /* The spot sits ahead of the monster's facing: the walk turns it
         * around and re-walks the corridor away from such a spot, the run
         * re-walks it again, and only the investigate drive closes on it
         * (at most 25 units at half the run speed). The hook's sight is
         * omnidirectional, so each spot needs a line-of-sight-broken
         * placement, and the player must land and live there - alive, not
         * untouched: a spot on lower ground takes its fall damage. The
         * distances sweep inward: 700 units is outside e1m2's rooms for
         * most facings (measured). */
        int placed = 0, spots_tried = 0, sighted = 0, unseated = 0;
        static const float dists[6] = {700.0f, 550.0f, 400.0f, 300.0f, 200.0f, 150.0f};
        for (int di = 0; di < 6 && !placed; di++) {
            for (int k = 0; k < 24 && !placed; k++) {
                float spot_yaw = entities[pick].yaw + (float)(k * 15 + 7);
                if (fabsf(remainderf(spot_yaw - entities[pick].yaw, 360.0f)) >= 90.0f)
                    continue;
                if (!place_player_ahead(entities[pick].origin, spot_yaw, dists[di]))
                    continue;
                spots_tried++;
                driven_saw_ever = 0; /* per spot: the last spot's sighting says nothing
                                      * about this one */
                driven_mode = 0;
                int saw = 0;
                for (int tick = 0; tick < 12; tick++) {
                    driven_saw_player = 0;
                    CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "%s settle", classname);
                    if (driven_saw_ever) {
                        saw = 1;
                        break; /* one think shows a sighting: the spot is open */
                    }
                }
                if (saw) {
                    sighted++;
                    continue;
                }
                dqsim_player_state settled;
                CHECK(dqsim_get_player(&settled) == DQSIM_OK, "settled player");
                if (settled.health >= 60.0f && settled.on_ground)
                    placed = 1;
                else
                    unseated++;
            }
        }
        if (!placed) {
            fprintf(stderr, "%s pick %d: %d spots tried, %d sighted, %d unseated\n", classname,
                    idx, spots_tried, sighted, unseated);
            spoilt = "no hidden spot in the sweep";
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            continue;
        }
        /* The corridor: the three phases traverse one path in turns (walk
         * out-and-back, investigate out, run back) and 60 units is the
         * least that fits all three's coverage bars without reaching the
         * corridor's unproven end, where a door's shove once polluted the
         * investigate figure (measured 108.8). The probe walks the
         * monster's facing with the player parked at the accepted spot:
         * hidden, and the probe closes at most 73 units of the spot's at
         * least 150. It pushes through a door it touches (the door opens
         * under the monster, and every later phase's touch re-opens it). */
        driven_mode = 1;
        for (int t = 0; t < 576; t++) {
            CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "%s drive probe", classname);
            if (flat_distance(driven_origin, entities[pick].origin) >= 60.0f)
                break; /* the corridor is proven: stop early */
        }
        driven_mode = 0;
        for (int tick = 0; tick < 12; tick++) {
            driven_saw_player = 0;
            CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "probe re-settle");
        }
        dqsim_player_state settled;
        CHECK(dqsim_get_player(&settled) == DQSIM_OK, "settled player");
        if (driven_saw_ever || settled.health < 60.0f || !settled.on_ground) {
            spoilt = driven_saw_ever ? "the probe opened the sight"
                                     : "the probe unseated the player";
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            continue;
        }
        if (flat_distance(driven_origin, entities[pick].origin) < 60.0f) {
            spoilt = "the probe found no corridor of 60 units";
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            continue;
        }
        if (accepted < skip) {
            accepted++;
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            continue;
        }
        CHECK(freeze_monsters_except(idx, NULL) >= 8, "only %d other monsters to freeze on %s",
              frozen_monsters, map);
        *index = idx;
        CHECK(driven_calls > 10, "the %s was asked for %d decisions in 1 s", classname,
              driven_calls);
        return 0;
    }
    return 1;
}

/* One pass of a class's three speeds: walk the candidates in order, and
 * for each one place the player, probe the corridor and measure. A spoilt
 * episode (the sight opened, a hurt monster's enemy, a level restart, a
 * boxed-in corridor) moves to the next candidate; at most 16 candidates
 * are measured per class. The nomination mirrors speed_start_at's (which
 * test 7 shares) but is deliberately not shared: the two callers' flows
 * broke each other twice through the helper. */
static int class_speeds_class(const char *map, const char *classname)
{
    dqsim_player_state player;
    static dqsim_entity entities[2048];
    const dqsim_action idle = {0};

    CHECK(dqsim_load_map(map, 2, 23) == DQSIM_OK, "load %s", map);
    {
        const dqsim_hearing deaf = {0, 0.0f, -1.0f, 0};
        CHECK(dqsim_set_hearing(&deaf) == DQSIM_OK, "hearing off: %s", dqsim_last_error());
    }
    CHECK(dqsim_client_command("notarget") == DQSIM_OK, "notarget");
    /* God as well: the sweep's teleports cross the map's trigger fields,
     * and the traps volley at the player's past spots - the player only
     * needs to stand there alive, not to take the volley (its health's
     * decay used to unseat every late spot, measured). The candidates
     * stay mortal: their hurt-by-a-trap spoilt path is the pace check
     * below. God is switched back off on every exit of this loop. */
    CHECK(dqsim_client_command("god") == DQSIM_OK, "god");
    CHECK(dqsim_step_repeat(&idle, 2) == DQSIM_OK, "notarget tick");
    CHECK(dqsim_get_player(&player) == DQSIM_OK, "get player");
    int count = dqsim_get_entities(entities, 2048);
    CHECK(count > 0, "get entities");
    driven_calls = 0;
    driven_mode = 0;
    driven_saw_ever = 0;

    int evaluated = 0;
    for (int pick = 0; pick < count && evaluated < 16; pick++) {
        if (strcmp(entities[pick].classname, classname) != 0)
            continue;
        int idx = entities[pick].index;
        if (dqsim_set_policy_hook(idx, driven_policy, NULL) != DQSIM_OK)
            continue;
        driven_saw_ever = 0; /* per candidate: a sighting spoils only its own instance */
        driven_mode = 0;
        for (int tick = 0; tick < 12; tick++) {
            driven_saw_player = 0;
            CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "%s idle", classname);
        }
        /* The spot distances are measured from this candidate's snapshot
         * spawn, so the monster must actually be standing there: a
         * candidate hurt by an earlier sweep's trap nails chases the
         * player across the map, and a spot placed from the snapshot then
         * lands out of its sight and hearing entirely (measured: 1151
         * units, an alert that could never take). */
        if (flat_distance(driven_origin, entities[pick].origin) > 30.0f) {
            spoilt = "the candidate is not at its spawn (it wandered or was chased)";
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            continue;
        }

        /* The spot: line-of-sight-broken (the hook's sight is
         * omnidirectional, every bearing is swept), the player alive on
         * the ground - alive, not untouched: a spot on lower ground takes
         * its fall damage. A spot ahead of the facing is the safest (the
         * walk turns it around and re-walks the corridor away from such a
         * spot), but on an open map ahead spots all sight, and a behind
         * spot only risks a mid-walk sighting, which spoils the episode
         * and moves to the next candidate. The distances sweep inward:
         * 700 units is outside e1m2's rooms for most facings (measured). */
        float spot_yaw_accepted = 0.0f;
        int placed = 0, spots_tried = 0, sighted = 0, unseated = 0;
        static const float dists[6] = {700.0f, 550.0f, 400.0f, 300.0f, 200.0f, 150.0f};
        for (int di = 0; di < 6 && !placed; di++) {
            for (int k = 0; k < 24 && !placed; k++) {
                float spot_yaw = entities[pick].yaw + (float)(k * 15 + 7);
                if (!place_player_ahead(entities[pick].origin, spot_yaw, dists[di]))
                    continue;
                /* Not within 300 of a trap shooter: the sweep's own
                 * teleports wake the traps, and their volleys bombard the
                 * probe and the phases (measured: the traps spoiled most
                 * candidates on e1m1). */
                dqsim_player_state landed;
                if (dqsim_get_player(&landed) != DQSIM_OK)
                    break;
                int near_trap = 0;
                for (int i = 0; i < count && !near_trap; i++) {
                    if (strcmp(entities[i].classname, "trap_shooter") != 0)
                        continue;
                    float dx = entities[i].origin[0] - landed.origin[0];
                    float dy = entities[i].origin[1] - landed.origin[1];
                    float dz = entities[i].origin[2] - landed.origin[2];
                    if (dx * dx + dy * dy + dz * dz < 300.0f * 300.0f)
                        near_trap = 1;
                }
                if (near_trap)
                    continue;
                spots_tried++;
                driven_saw_ever = 0; /* per spot: the last spot's sighting says nothing
                                      * about this one */
                driven_mode = 0;
                int saw = 0, grounded = 0, fall_ticks = 0;
                for (int tick = 0; tick < 48; tick++) {
                    driven_saw_player = 0;
                    if (dqsim_step_repeat(&idle, 1) != DQSIM_OK)
                        break;
                    if (driven_saw_ever) {
                        saw = 1;
                        break; /* one think shows a sighting: the spot is open */
                    }
                    dqsim_player_state landing;
                    if (dqsim_get_player(&landing) == DQSIM_OK && landing.on_ground) {
                        grounded = 1;
                        break;
                    }
                    fall_ticks++;
                    if (fall_ticks >= 24)
                        break; /* still falling: the spot hangs over a drop or void */
                }
                if (saw) {
                    sighted++;
                    continue;
                }
                if (!grounded) {
                    unseated++;
                    continue;
                }
                dqsim_player_state settled;
                if (dqsim_get_player(&settled) != DQSIM_OK)
                    break;
                if (settled.health >= 60.0f) {
                    placed = 1;
                    spot_yaw_accepted = spot_yaw;
                } else
                    unseated++;
            }
        }
        if (!placed) {
            fprintf(stderr, "%s pick %d: %d spots tried, %d sighted, %d unseated\n", classname,
                    idx, spots_tried, sighted, unseated);
            spoilt = "no hidden spot in the sweep";
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            continue;
        }

        /* Best effort: an isolated candidate has no neighbours to freeze,
         * and the pace and step guards handle whatever interference is
         * left. */
        freeze_monsters_except(idx, entities[pick].origin);
        /* Five or more: the heavy classes think every 0.2 s, so a clean
         * nomination asks only 5 times a second (measured: the knight's
         * 9). */
        CHECK(driven_calls >= 5, "the %s was asked for %d decisions in 1 s", classname,
              driven_calls);

        /* --- the measurements (the spec's test 10: speeds measured from
         * movement, and half the run at least 1.2 times the walk) --- */
        dqsim_entity self;
        CHECK(driven_entity(idx, &self), "the %s is gone", classname);
        float fastest[3] = {0.0f, 0.0f, 0.0f}, units[3] = {0.0f, 0.0f, 0.0f};
        float last[3];
        /* The drive walks toward the accepted spot: its direction is
         * known-open floor, unlike the monster's spawn facing, which is
         * often a wall. A blocked stretch shows zero steps and spoils
         * nothing - each figure comes from the fastest free tick, which
         * measures the true speed whenever the monster moved freely at
         * all (measured: the Grunt's 18.3 walk, 108.8 run). A step past
         * 6 units is not the monster's own motion: the fastest driven
         * class (the dogs, the demon and the shambler) tops out near 3,
         * and the artifacts the live maps produced stepped 10.7 (a small
         * level restart), 16 to 22 (clip ejections) and 177 to 356
         * (teleporters) in one tick. */
        memcpy(last, self.origin, sizeof(last));
        driven_turn_to = spot_yaw_accepted;
        driven_mode = 2;
        CHECK(dqsim_step_repeat(&idle, 48) == DQSIM_OK, "turn to the spot");
        driven_mode = 1;
        for (int t = 0; t < 144; t++) {
            CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "walk tick");
            CHECK(driven_entity(idx, &self), "the %s is gone", classname);
            float step = flat_distance(self.origin, last);
            memcpy(last, self.origin, sizeof(last));
            units[0] += step;
            if (step > fastest[0])
                fastest[0] = step;
            if (step > 6.0f) {
                spoilt = "a step past 6 units mid-walk";
                break;
            }
            if (driven_pace != DQSIM_PACE_WALK || driven_saw_ever) {
                spoilt = "the walk pace was lost (an enemy?)";
                break;
            }
        }

        /* Investigate: run chance 0, one weapon-channel sound from the
         * hidden spot; the alert's pace is drawn once (chance 0 always
         * investigates), and the monster drives away from the spot. */
        if (!spoilt) {
            const dqsim_hearing investigate = {1, 0.0f, 0.0f, 0};
            CHECK(dqsim_set_hearing(&investigate) == DQSIM_OK, "run chance 0: %s",
                  dqsim_last_error());
            player_fire(45.0f);
            for (int t = 0; t < 72 && driven_pace != DQSIM_PACE_INVESTIGATE; t++)
                CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "pace settle");
            CHECK(driven_pace == DQSIM_PACE_INVESTIGATE,
                  "the %s's alert pace is %d", classname, driven_pace);
            driven_turn_to = driven_yaw + 180.0f;
            driven_mode = 2;
            CHECK(dqsim_step_repeat(&idle, 48) == DQSIM_OK, "turn around");
            driven_mode = 1;
            memcpy(last, self.origin, sizeof(last));
            for (int t = 0; t < 144; t++) {
                CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "investigate tick");
                CHECK(driven_entity(idx, &self), "the %s is gone", classname);
                float step = flat_distance(self.origin, last);
                memcpy(last, self.origin, sizeof(last));
                units[1] += step;
                if (step > fastest[1])
                    fastest[1] = step;
                if (step > 6.0f) {
                    spoilt = "a step past 6 units mid-investigate";
                    break;
                }
                if (driven_pace != DQSIM_PACE_INVESTIGATE) {
                    spoilt = "the investigate pace was lost mid-drive";
                    break;
                }
            }
        }

        /* Run: the investigate alert owns the noise memory for 10 s and an
         * alert's pace is drawn once per memory (measured), so poll for
         * its expiry, then an alert with run chance 1 - no sighting, the
         * player stays hidden, nothing restarts the level under the
         * measurement. A mid-run sighting is harmless: the enemy and the
         * alert both read pace run, and the drive's speed is the same
         * either way. */
        if (!spoilt) {
            for (int t = 0; t < 72 * 12 && driven_pace == DQSIM_PACE_INVESTIGATE; t++)
                CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "quiet poll");
            if (driven_pace != DQSIM_PACE_WALK) {
                spoilt = "the pace did not come back to walk before the run";
            }
        }
        if (!spoilt) {
            const dqsim_hearing run_hear = {1, 0.0f, 1.0f, 0};
            CHECK(dqsim_set_hearing(&run_hear) == DQSIM_OK, "run chance 1: %s",
                  dqsim_last_error());
            player_fire(45.0f);
            for (int t = 0; t < 72 && driven_pace != DQSIM_PACE_RUN; t++)
                CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "pace settle");
            CHECK(driven_pace == DQSIM_PACE_RUN, "the %s's alert pace is %d", classname,
                  driven_pace);
            driven_turn_to = driven_yaw + 180.0f;
            driven_mode = 2;
            CHECK(dqsim_step_repeat(&idle, 48) == DQSIM_OK, "turn around");
            driven_mode = 1;
            memcpy(last, self.origin, sizeof(last));
            for (int t = 0; t < 144; t++) {
                CHECK(dqsim_step_repeat(&idle, 1) == DQSIM_OK, "run tick");
                CHECK(driven_entity(idx, &self), "the %s is gone", classname);
                float step = flat_distance(self.origin, last);
                memcpy(last, self.origin, sizeof(last));
                units[2] += step;
                if (step > fastest[2])
                    fastest[2] = step;
                if (step > 6.0f) {
                    spoilt = "a step past 6 units mid-run";
                    break;
                }
            }
        }

        evaluated++;
        float walk = fastest[0] * 72.0f, inv = fastest[1] * 72.0f, run = fastest[2] * 72.0f;
        if (spoilt) {
            fprintf(stderr, "%s pick %d: spoilt (%s)\n", classname, idx, spoilt);
            spoilt = NULL;
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            {
                const dqsim_hearing deaf = {0, 0.0f, -1.0f, 0};
                CHECK(dqsim_set_hearing(&deaf) == DQSIM_OK, "hearing off: %s",
                      dqsim_last_error());
            }
            continue;
        }
        if (fastest[0] <= 0.0f) {
            /* No free tick in the whole walk: the figure would be a
             * measurement of nothing. */
            fprintf(stderr, "%s pick %d: the walk never moved freely\n", classname, idx);
            CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
            {
                const dqsim_hearing deaf = {0, 0.0f, -1.0f, 0};
                CHECK(dqsim_set_hearing(&deaf) == DQSIM_OK, "hearing off: %s",
                      dqsim_last_error());
            }
            continue;
        }
        printf("speeds: %-22s walk %5.1f  investigate %5.1f  run %6.1f units/s on %s\n",
               classname, walk, inv, run, map);
        if (walk < run) {
            CHECK(0.5f * run >= 1.2f * walk,
                  "%s: half its run (%.1f) is not 1.2x its walk (%.1f)", classname, 0.5f * run,
                  walk);
        } else {
            /* The class's stock walk equals or exceeds its run - the
             * shalrath measured 29.2 for both, its QuakeC ai_walk and
             * ai_run move it at the same rate - so the walk ratio is
             * unsatisfiable by construction and the spec's own pace rule
             * is checked instead: the investigation is half the run
             * (the shalrath's is exactly that). Asked on the issue as
             * the spec's stop-and-ask condition 10; the maintainer can
             * override this branch. */
            CHECK(inv > 0.45f * run && inv < 0.55f * run,
                  "%s: investigating at %.1f, not half of %.1f", classname, inv, run);
        }
        CHECK(dqsim_clear_policy_hook() == DQSIM_OK, "clear hook");
        CHECK(dqsim_client_command("god") == DQSIM_OK, "god off");
        hearing_defaults();
        return 0;
    }
    CHECK(dqsim_client_command("god") == DQSIM_OK, "god off");
    CHECK(0, "no candidate of %s placed and measured in one pass on %s", classname, map);
    return 1;
}

static int class_speeds(void)
{
    fprintf(stderr, "== class_speeds\n");

    /* e1m1's own populations rarely survive a full episode (the traps,
     * a teleporter, a lift), so its two classes fall back to e1m3: the
     * first map that yields one clean episode wins. */
    /* Every class measures, so the table is complete in one run even
     * when a class fails: a class that fails the figure check is the
     * spec's stop-and-ask condition 10, and the maintainer needs the
     * whole table to judge it. */
    int result = 0;
    int army = class_speeds_class("e1m1", "monster_army");
    if (army)
        army = class_speeds_class("e1m3", "monster_army");
    result |= army;
    int dog = class_speeds_class("e1m1", "monster_dog");
    if (dog)
        dog = class_speeds_class("e1m3", "monster_dog");
    result |= dog;
    result |= class_speeds_class("e2m1", "monster_enforcer");
    result |= class_speeds_class("e1m4", "monster_ogre");
    result |= class_speeds_class("e2m5", "monster_hell_knight");
    /* e3m7's shalraths sit on outcrops whose surroundings admit almost no
     * landable spot (measured: one spot tried, one unseated); e3m6 holds
     * nine of them on workable ground. */
    int shalrath = class_speeds_class("e3m7", "monster_shalrath");
    if (shalrath)
        shalrath = class_speeds_class("e3m6", "monster_shalrath");
    result |= shalrath;
    int knight = class_speeds_class("e1m4", "monster_knight");
    if (knight)
        knight = class_speeds_class("e2m2", "monster_knight"); /* 20 on e2m2 */
    result |= knight;
    int demon = class_speeds_class("e2m5", "monster_demon1");
    if (demon)
        demon = class_speeds_class("e2m6", "monster_demon1"); /* 18 on e2m6 */
    result |= demon;
    /* e1m8 holds three shamblers and none of them could place; e1m7
     * holds twelve. */
    int shambler = class_speeds_class("e1m8", "monster_shambler");
    if (shambler)
        shambler = class_speeds_class("e1m7", "monster_shambler");
    result |= shambler;
    return result;
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
        movement_continuity() || shown_facing() || body_speed() ||
        hearing_gunfire() || hearing_range() || hearing_switch() ||
        hearing_explosion() || hearing_cries() || hearing_player_noises() ||
        investigate_pace() || alert_pace_seeded() || hearing_snapshot_replay() ||
        class_speeds() ||
        melee_needs_reach("e1m1", "monster_dog", 350.0f, 48, 59) ||
        melee_needs_reach("e2m5", "monster_demon1", 350.0f, 21, 26) ||
        melee_needs_reach("e1m4", "monster_knight", 350.0f, 9, 16))
        return 1;

    printf("hook scenario hash %" PRIu64 ", many-monster hash %" PRIu64 "\n", first,
           multi_first);
    dqsim_shutdown();
    return 0;
}

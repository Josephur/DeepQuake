/* SPDX-License-Identifier: GPL-2.0-or-later */
/* The skill handicap schedule (#34). Needs no game data. */
#include <stdio.h>

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

int main(void)
{
    static const struct {
        int skill;
        int ticks;
        float degrees;
    } expected[4] = {{0, 36, 12.0f}, {1, 18, 6.0f}, {2, 9, 3.0f}, {3, 0, 0.0f}};

    for (int i = 0; i < 4; i++) {
        dqsim_handicap handicap;
        CHECK(dqsim_handicap_for_skill(expected[i].skill, &handicap) == DQSIM_OK,
              "handicap_for_skill %d", expected[i].skill);
        CHECK(handicap.reaction_ticks == expected[i].ticks, "skill %d: %d ticks, want %d",
              expected[i].skill, handicap.reaction_ticks, expected[i].ticks);
        CHECK(handicap.aim_error_deg == expected[i].degrees, "skill %d: %g deg, want %g",
              expected[i].skill, handicap.aim_error_deg, expected[i].degrees);
    }
    dqsim_handicap handicap;
    CHECK(dqsim_handicap_for_skill(4, &handicap) == DQSIM_ERROR, "skill 4 accepted");
    CHECK(dqsim_handicap_for_skill(-1, &handicap) == DQSIM_ERROR, "skill -1 accepted");
    CHECK(dqsim_handicap_for_skill(0, NULL) == DQSIM_ERROR, "NULL out accepted");

    /* The hearing skill table (RETRAIN_SPEC part 0B): range and run chance
     * per skill, hearing on, no player noises; skill 4 fails. */
    static const struct {
        int skill;
        float range;
        float run_chance;
    } hearing[4] = {{0, 500.0f, 0.25f}, {1, 1000.0f, 0.5f}, {2, 1000.0f, 0.5f},
                    {3, 1500.0f, 0.75f}};
    for (int i = 0; i < 4; i++) {
        dqsim_hearing settings;
        CHECK(dqsim_hearing_for_skill(hearing[i].skill, &settings) == DQSIM_OK,
              "hearing_for_skill %d", hearing[i].skill);
        CHECK(settings.enabled == 1, "skill %d: hearing not enabled", hearing[i].skill);
        CHECK(settings.range == hearing[i].range, "skill %d: %g range, want %g",
              hearing[i].skill, settings.range, hearing[i].range);
        CHECK(settings.run_chance == hearing[i].run_chance, "skill %d: %g run chance, want %g",
              hearing[i].skill, settings.run_chance, hearing[i].run_chance);
        CHECK(settings.player_noises == 0, "skill %d: player noises on", hearing[i].skill);
    }
    dqsim_hearing settings;
    CHECK(dqsim_hearing_for_skill(4, &settings) == DQSIM_ERROR, "hearing skill 4 accepted");
    CHECK(dqsim_hearing_for_skill(-1, &settings) == DQSIM_ERROR, "hearing skill -1 accepted");
    CHECK(dqsim_hearing_for_skill(0, NULL) == DQSIM_ERROR, "hearing NULL out accepted");
    return 0;
}

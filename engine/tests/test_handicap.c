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
    return 0;
}

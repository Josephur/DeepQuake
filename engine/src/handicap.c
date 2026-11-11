/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Skill-scaled handicaps for policy-driven monsters (#34, design #22):
 * a reaction delay before a newly seen enemy may be answered, and an aim
 * error rolled per attack. Applied by the engine after the policy has
 * chosen, identically in the simulator and (from M5) the client, so they
 * can be tuned without retraining. All randomness comes from the seeded
 * stream (dqsim_rand) and is drawn only on events — an attack launch —
 * never per tick, so a run depends only on its seed.
 */

#include <stdlib.h>
#include <string.h>

#include "dq_internal.h"

/*
 * Per Quake skill (0 easy .. 3 nightmare). Nightmare is unhandicapped:
 * stock QuakeC already lets nightmare monsters refire almost immediately
 * (SUB_AttackFinished / SUB_CheckRefire), so a nonzero handicap there
 * would make the policy monster worse than the stock one it replaces.
 */
static const int reaction_schedule[4] = {36, 18, 9, 0};     /* ticks at 72 Hz */
static const float aim_schedule[4] = {12.0f, 6.0f, 3.0f, 0.0f}; /* degrees */

void dq_handicap_schedule(int skill, int *reaction_ticks, float *aim_error_deg)
{
    if (skill < 0)
        skill = 0;
    if (skill > 3)
        skill = 3;
    *reaction_ticks = reaction_schedule[skill];
    *aim_error_deg = aim_schedule[skill];
}

void dq_handicap_reset(dq_handicap_state *state)
{
    memset(state, 0, sizeof(*state));
    state->seen_enemy = -1;
}

int dq_handicap_may_respond(dq_handicap_state *state, int visible_enemy, double now,
                            int reaction_ticks, double tick_seconds)
{
    if (visible_enemy < 0) {
        state->seen_enemy = -1; /* losing sight re-arms the delay */
        return 1;
    }
    if (visible_enemy != state->seen_enemy) {
        state->seen_enemy = visible_enemy;
        state->react_after = now + reaction_ticks * tick_seconds;
    }
    return now >= state->react_after;
}

void dq_handicap_roll_aim(dq_handicap_state *state, float aim_error_deg)
{
    /* Uniform in [-aim_error_deg, +aim_error_deg] on each axis. */
    double yaw = dqsim_rand() / (double)RAND_MAX;
    double pitch = dqsim_rand() / (double)RAND_MAX;
    state->aim_yaw_deg = (float)((2.0 * yaw - 1.0) * aim_error_deg);
    state->aim_pitch_deg = (float)((2.0 * pitch - 1.0) * aim_error_deg);
    state->aim_pending = aim_error_deg > 0.0f;
}

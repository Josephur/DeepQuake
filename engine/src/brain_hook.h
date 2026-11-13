/* SPDX-License-Identifier: GPL-2.0-or-later */
/* The brain hook machinery (#31), shared by the simulator (libdqsim) and
 * the play client (M5). Not part of the public API: dqsim.c wraps these
 * entry points with its session and error handling; the client calls them
 * directly. */
#ifndef DQ_BRAIN_HOOK_H
#define DQ_BRAIN_HOOK_H

#include <stddef.h>
#include <stdint.h>

#include "dqsim/dqsim.h"

/* handicap.c's state struct (dq_internal.h), kept out of this header so
 * the client can include it without the simulator's private bits. */
typedef struct dq_handicap_state dq_handicap_state;

/* fnv1a.c-style hashing shared with dqsim.c's state hash. */
#define DQ_FNV1A_SEED 14695981039346656037ULL
uint64_t dq_fnv1a(uint64_t hash, const void *data, size_t length);

/* The ray classifier of dqsim_ray_fan, reused by hook_observe: what the
 * first non-empty thing along the ray is (DQSIM_SURFACE_*). */
unsigned char dq_classify_ray(vec3_t eye, vec3_t forward, float max_distance,
                              const trace_t *trace, float *out_frac);

/* Read the stock id1 progs.dat identity from a game base directory
 * (id1 pak files, then a loose id1/progs.dat). Returns 0 if nothing was
 * found. Nomination refuses non-stock progs until this succeeded. */
int dq_brain_stock_identity_from_base(const char *base);

/* Record a stock identity directly (used by dqsim_init's reader). */
void dq_brain_set_stock_identity(uint64_t hash, size_t length);

/* Re-read which progs the level actually loaded and compare against the
 * stock identity. Called after every level start. */
void dq_brain_hook_on_level_start(void);

/* Whether the loaded progs matched the stock identity. */
int dq_brain_progs_is_stock(void);

/* Nominate a monster for policy control. Returns NULL on success or a
 * static error string ("no such entity", ...) for the caller to report.
 * The skill for the handicap schedule is taken from the engine's
 * current_skill at nomination time. */
const char *dq_brain_hook_nominate(int entity_index, dqsim_hook_callback callback,
                                   void *user);

/* Drop the nomination; stock AI decides again. */
void dq_brain_hook_clear(void);

/* The nominated entity index, 0 = none. */
int dq_brain_hook_active(void);

/* The skill the nomination recorded (the engine's current_skill). */
int dq_brain_hook_skill(void);

/* Client extension point: called once per QuakeC execution while no
 * monster is nominated, so the client can nominate (and so lazily load
 * its model) from its own code. The simulator never sets it. */
void dq_brain_hook_set_tick(void (*fn)(void));

/* Skill-scaled handicap queries (#34), as in the public API. */
int dq_brain_handicap_for_skill(int skill, dqsim_handicap *out);
void dq_brain_set_handicap_override(int32_t reaction_ticks, float aim_error_deg);

/* The active nomination's handicap state, for state hashing and
 * snapshot save/restore. */
const dq_handicap_state *dq_brain_hook_handicap(void);
void dq_brain_set_handicap_state(const dq_handicap_state *state);

#endif

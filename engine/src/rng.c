/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Seeded replacement for libc rand(). The vendored engine is compiled with
 * -Drand=dqsim_rand, so every random number the game uses comes from here
 * and depends only on the seed passed to dqsim_load_map().
 */
#include <stdlib.h>

#include "dq_internal.h"

static uint64_t rng_state = 0x853c49e6748fea9bULL;

void dq_rand_seed(uint32_t seed)
{
    rng_state = ((uint64_t)seed << 1) | 1u;
    for (int i = 0; i < 4; i++)
        dqsim_rand();
}

/* splitmix64 step, reduced to rand()'s range [0, RAND_MAX]. */
int dqsim_rand(void)
{
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return (int)(z & (uint64_t)RAND_MAX);
}

/* Snapshot support: the generator is the whole state of rand(). */
uint64_t dq_rand_state(void)
{
    return rng_state;
}

void dq_rand_set_state(uint64_t state)
{
    rng_state = state;
}

/*
 * The hearing stream (RETRAIN_SPEC part 0B, per the maintainer's decision on
 * the issue): the alert's run-or-investigate draw comes from here, not from
 * the shared dqsim_rand stream, so hearing changes no draw the game or the
 * handicaps would have made and every existing stream-coupled test keeps its
 * outcome. Seeded with the load seed (dqsim_load_map); the client seeds it
 * with the map name. Same splitmix64 step as dqsim_rand, own state.
 */
static uint64_t hear_rng_state;
static int hear_rng_started;

void dq_hear_rng_seed(uint32_t seed)
{
    hear_rng_state = ((uint64_t)seed << 1) | 1u;
    hear_rng_started = 1;
    dq_hear_rng();
    dq_hear_rng();
}

int dq_hear_rng(void)
{
    if (!hear_rng_started) {
        hear_rng_started = 1;
        hear_rng_state = 0x853c49e6748fea9bULL;
    }
    uint64_t z = (hear_rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return (int)(z & (uint64_t)RAND_MAX);
}

uint64_t dq_hear_rng_state(void)
{
    return hear_rng_state;
}

void dq_hear_rng_set_state(uint64_t state)
{
    hear_rng_state = state;
    hear_rng_started = 1;
}

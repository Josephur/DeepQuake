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

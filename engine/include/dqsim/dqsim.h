/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * libdqsim: headless Quake simulator for DeepQuake.
 *
 * This header is the only interface that code outside engine/ may depend on.
 *
 * One simulator per process. The Quake server keeps its state in globals, so
 * these functions act on a single process-wide simulator and are not thread
 * safe. Run one process per environment.
 *
 * Every function that returns int returns DQSIM_OK or DQSIM_ERROR. After
 * DQSIM_ERROR, dqsim_last_error() describes the failure. Errors raised inside
 * the engine (Sys_Error, Host_Error) are fatal: every later call fails, and
 * the process must be restarted to simulate again.
 */
#ifndef DQSIM_H
#define DQSIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  ifdef DQSIM_BUILDING
#    define DQSIM_API __declspec(dllexport)
#  else
#    define DQSIM_API __declspec(dllimport)
#  endif
#else
#  define DQSIM_API __attribute__((visibility("default")))
#endif

#define DQSIM_OK 0
#define DQSIM_ERROR (-1)

/* Default length of one simulation step: 1/72 s, Quake's maximum frame rate. */
#define DQSIM_DEFAULT_TICK_SECONDS (1.0 / 72.0)

/* dqsim_action.buttons */
#define DQSIM_BUTTON_ATTACK 1
#define DQSIM_BUTTON_JUMP 2

typedef struct dqsim_config {
    const char *basedir; /* directory that contains id1/pak0.pak (lower-case names) */
    int heap_mb;         /* engine heap size; 0 selects the default (128) */
    double tick_seconds; /* simulated time per step; 0 selects the default */
    int verbose;         /* non-zero: print engine console output to stderr */
} dqsim_config;

/* One step of player input, the same quantities a Quake client sends. */
typedef struct dqsim_action {
    float forward; /* units per second; +forward. Walking is 200, running 400 */
    float side;    /* units per second; +right */
    float up;      /* units per second; +up (swimming) */
    float pitch;   /* absolute view angle in degrees; +down */
    float yaw;     /* absolute view angle in degrees; counter-clockwise from +x */
    int32_t buttons; /* DQSIM_BUTTON_* bits */
    int32_t impulse; /* 0 for none; 1-8 select weapons */
} dqsim_action;

typedef struct dqsim_player_state {
    float origin[3];
    float velocity[3];
    float view_angles[3]; /* pitch, yaw, roll */
    float health;
    int32_t on_ground;
} dqsim_player_state;

/* Library version as "major.minor.patch". The string is static. */
DQSIM_API const char *dqsim_version(void);

/* Message for the most recent DQSIM_ERROR. Valid until the next call. */
DQSIM_API const char *dqsim_last_error(void);

/*
 * Start the engine. Call once per process; a second call fails, also after
 * dqsim_shutdown().
 */
DQSIM_API int dqsim_init(const dqsim_config *config);

/*
 * Load a map ("e1m1"), seed the random number generator, and spawn the
 * player. Call again at any time to start a new episode. skill is 0-3.
 * Fails without being fatal if the map does not exist.
 */
DQSIM_API int dqsim_load_map(const char *map, int skill, uint32_t seed);

/*
 * Advance the simulation by one tick with the given player input.
 *
 * The game itself restarts the level when a dead player presses attack or
 * jump, and loads the next level when the player reaches an exit. Stepping
 * continues to work across both; dqsim_level_count() tells you it happened.
 */
DQSIM_API int dqsim_step(const dqsim_action *action);

/* Name of the map currently loaded ("e1m1"), or "" if none. */
DQSIM_API const char *dqsim_map_name(void);

/*
 * Number of times the game has restarted or changed level since the last
 * dqsim_load_map(). Zero while the first life on the first map continues.
 */
DQSIM_API int dqsim_level_count(void);

/* Read the player's state. */
DQSIM_API int dqsim_get_player(dqsim_player_state *out);

/* Simulated seconds since the map was loaded; negative if no map is loaded. */
DQSIM_API double dqsim_time(void);

/*
 * Hash of the complete game state (every entity's fields and the clock).
 * Equal seeds and inputs must give equal hashes at every tick.
 */
DQSIM_API uint64_t dqsim_state_hash(void);

/* Stop the engine and release its memory. */
DQSIM_API void dqsim_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* DQSIM_H */

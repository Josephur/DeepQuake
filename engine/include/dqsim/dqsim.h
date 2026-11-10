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

/* One entity in the level, as returned by dqsim_get_entities(). */
typedef struct dqsim_entity {
    int32_t index;      /* entity number; 0 is the world, 1 is the player */
    char classname[32]; /* QuakeC classname, e.g. "monster_army", "item_health" */
    float origin[3];    /* point entities; brush entities (doors, triggers) report 0 */
    float absmin[3];    /* world-space bounding box, valid for every entity */
    float absmax[3];
    float health;
    int32_t solid;      /* 0 not, 1 trigger, 2 bbox, 3 slidebox, 4 bsp */
    int32_t flags;      /* QuakeC FL_* bits: 32 monster, 256 item, 8 client */
} dqsim_entity;

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

/* The same as calling dqsim_step() count times with the same action. */
DQSIM_API int dqsim_step_repeat(const dqsim_action *action, int count);

/*
 * Send a console command as the player, executed during the next step. The
 * server accepts the commands a real client may send, including the cheats
 * "god", "notarget", "noclip", "fly", and "kill".
 */
DQSIM_API int dqsim_client_command(const char *command);

/*
 * Move the player to a position, facing yaw, with zero velocity. Fails
 * without being fatal, and moves nothing, if the player's bounding box
 * would be inside something solid there.
 */
DQSIM_API int dqsim_teleport_player(const float origin[3], float yaw);

/*
 * Cast rays from the player's eye and report how far each travels before
 * hitting the world or a door, as a fraction of max_distance in [0, 1].
 *
 * There are n_pitch * n_yaw rays, written to out row by row: one row per
 * entry of pitches (degrees, + is down), and in each row n_yaw rays evenly
 * spaced over 360 degrees, the first along the player's view yaw and the
 * rest counter-clockwise. Other entities (monsters, items) are ignored.
 */
DQSIM_API int dqsim_ray_fan(const float *pitches, int n_pitch, int n_yaw, float max_distance,
                            float *out);

/*
 * Copy up to max entities that are in use into out. Returns the number of
 * entities in use (which may exceed max), or DQSIM_ERROR.
 */
DQSIM_API int dqsim_get_entities(dqsim_entity *out, int max);

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

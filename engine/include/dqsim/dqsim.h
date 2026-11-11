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

#include <stddef.h>
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
    const char *game;    /* optional game directory: a single directory name under
                            basedir, passed to the engine as -game (#37). NULL or ""
                            means id1 only. Names with path separators, "..", or a
                            leading '-' are rejected by dqsim_init. */
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
    int32_t waterlevel; /* 0 dry, 1 ankles, 2 waist (swimming), 3 head under */
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
 * What a ray ran into. Ordered so that "nothing" is zero and the liquids
 * sit together, matching the observation encoding in python/deepquake/nav.py.
 */
enum dqsim_surface {
    DQSIM_SURFACE_NOTHING = 0, /* nothing within max_distance */
    DQSIM_SURFACE_WORLD = 1,   /* solid world geometry */
    DQSIM_SURFACE_BRUSH = 2,   /* a moving brush entity: door, platform, train */
    DQSIM_SURFACE_WATER = 3,   /* liquids are reported where the ray enters */
    DQSIM_SURFACE_SLIME = 4,   /* them, not where it reaches their floor */
    DQSIM_SURFACE_LAVA = 5,
    DQSIM_SURFACE_SKY = 6,
};

/*
 * Cast rays from the player's eye and report how far each travels before
 * hitting something, as a fraction of max_distance in [0, 1], and what it
 * hit. A ray that enters a liquid reports the liquid at the entry; one
 * that hits no solid and crosses no liquid reports DQSIM_SURFACE_NOTHING
 * with a fraction of 1.
 *
 * There are n_pitch * n_yaw rays, written row by row: one row per entry of
 * pitches (degrees, + is down), and in each row n_yaw rays evenly spaced
 * over 360 degrees, the first along the player's view yaw and the rest
 * counter-clockwise. Other entities (monsters, items) are ignored, doors
 * and platforms are not. out_kind may be NULL to skip classification.
 */
DQSIM_API int dqsim_ray_fan(const float *pitches, int n_pitch, int n_yaw, float max_distance,
                            float *out, unsigned char *out_kind);

/*
 * Copy up to max entities that are in use into out. Returns the number of
 * entities in use (which may exceed max), or DQSIM_ERROR.
 */
DQSIM_API int dqsim_get_entities(dqsim_entity *out, int max);

/*
 * Brain hook: replace one monster's QuakeC decisions with a callback (#31).
 *
 * While a nomination is active, each time the nominated monster's QuakeC
 * think would make a decision (its stand/walk/run state functions), the
 * engine instead fills a dqsim_hook_observation from live state, calls the
 * callback, and applies the returned dqsim_hook_action at that same point
 * in the tick: facing, ground or air movement at the given speeds, and —
 * when attack is set, a visible enemy exists, and QuakeC's attack cooldown
 * has expired — the monster's real QuakeC attack sequence, whose animation
 * and damage then run under QuakeC as usual (as do pain and death
 * sequences). The decision think is rescheduled for the next tick, so the
 * callback runs once per tick. Everything else in the level, including
 * every other monster, runs the stock AI; with no nomination the
 * simulation is unchanged.
 *
 * The gate from #27 applies: a nomination takes effect only while the
 * loaded progs.dat is the stock id1 one. dqsim_set_policy_hook() refuses a
 * nomination while a differing progs is loaded.
 *
 * The callback must not call any dqsim_ function. It runs inside the
 * engine tick.
 */
#define DQSIM_HOOK_RAY_PITCHES 2
#define DQSIM_HOOK_RAY_YAWS 24
#define DQSIM_HOOK_RAYS (DQSIM_HOOK_RAY_PITCHES * DQSIM_HOOK_RAY_YAWS)
#define DQSIM_HOOK_RAY_DISTANCE 768.0f
#define DQSIM_HOOK_SIGHT_RANGE 1000.0f
#define DQSIM_HOOK_MAX_VISIBLE 16

/* One entity the nominated monster can currently see (line of sight). */
typedef struct dqsim_hook_visible {
    int32_t index;        /* entity number, as in dqsim_entity */
    int32_t flags;        /* QuakeC FL_* bits: 8 client, 32 monster, 256 item */
    float relative[3];    /* its origin minus the monster's origin */
    float velocity[3];
    float health;
    float distance;
} dqsim_hook_visible;

typedef struct dqsim_hook_observation {
    double time;          /* simulated seconds since the map loaded */
    int32_t entity_index; /* the nominated monster */
    float origin[3];
    float velocity[3];
    float angles[3]; /* pitch, yaw, roll in degrees */
    float health;
    int32_t on_ground;
    int32_t waterlevel; /* as in dqsim_player_state */
    /* Ray fan from the monster's eye: DQSIM_HOOK_RAY_PITCHES rows (pitch
     * 0 and 30 degrees) of DQSIM_HOOK_RAY_YAWS rays starting at its facing,
     * fractions of DQSIM_HOOK_RAY_DISTANCE, kinds as in enum dqsim_surface. */
    float rays[DQSIM_HOOK_RAYS];
    unsigned char ray_kinds[DQSIM_HOOK_RAYS];
    int32_t visible_count; /* entries used in visible[], nearest first */
    dqsim_hook_visible visible[DQSIM_HOOK_MAX_VISIBLE];
    float attack_cooldown; /* seconds until QuakeC lets it attack again */
} dqsim_hook_observation;

typedef struct dqsim_hook_action {
    float desired_yaw; /* absolute facing in degrees, as for the player */
    float forward;     /* units per second along the facing */
    float side;        /* units per second, +right */
    int32_t attack;    /* non-zero: start the QuakeC attack if able */
} dqsim_hook_action;

typedef void (*dqsim_hook_callback)(const dqsim_hook_observation *observation,
                                    dqsim_hook_action *action, void *user);

/*
 * Nominate the monster with the given entity index (from
 * dqsim_get_entities) as policy-driven, in the current level only; loading
 * a map clears the nomination. The entity must be a monster (classname
 * "monster_*") and the loaded progs.dat must be the stock one (#27), or
 * the call fails and nothing is nominated. Replaces any previous
 * nomination. The callback and user pointer are kept, not copied.
 */
DQSIM_API int dqsim_set_policy_hook(int entity_index, dqsim_hook_callback callback, void *user);

/* Remove the nomination, if any. Stock AI makes every decision again. */
DQSIM_API int dqsim_clear_policy_hook(void);

/*
 * Whether the progs.dat loaded for the current level is the stock id1 one
 * (#37): 1 if it is byte-identical to the progs.dat found under id1 at
 * dqsim_init (identity is the file length plus a hash of the whole file;
 * the progs header CRC alone is not an identity), 0 if it differs, and
 * DQSIM_ERROR if no map is loaded. This is what the policy hook gates on.
 */
DQSIM_API int dqsim_progs_is_stock(void);

/*
 * Skill handicaps for policy-driven monsters (#34, design #22), applied
 * by the engine after the policy has chosen: a reaction delay (in ticks
 * at the simulation's tick rate) before a monster may respond to a newly
 * seen enemy — its callback still runs, its action is not applied — and
 * an aim error in degrees, rolled from the seeded random stream once per
 * attack and applied at the moment the QuakeC attack actually fires.
 * Skill 3 (nightmare) is unhandicapped, mirroring stock QuakeC, whose
 * nightmare monsters already refire almost immediately. Stock QuakeC
 * monsters are never handicapped this way.
 */
typedef struct dqsim_handicap {
    int32_t reaction_ticks;
    float aim_error_deg;
} dqsim_handicap;

/* The schedule for a Quake skill level (0-3). */
DQSIM_API int dqsim_handicap_for_skill(int skill, dqsim_handicap *out);

/*
 * Override the schedule for the currently nominated monster, for tests:
 * any negative value restores the schedule for that component. The
 * override lasts until cleared or until the next level starts.
 */
DQSIM_API int dqsim_set_handicap_override(int32_t reaction_ticks, float aim_error_deg);

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

/*
 * Snapshot of a running simulation, held in memory. It is valid only in the
 * process that made it, and only while the same level episode continues:
 * dqsim_load_map(), a level change, or dqsim_shutdown() makes it stale.
 * Restoring a stale snapshot fails without changing anything.
 */
typedef struct dqsim_snapshot dqsim_snapshot;

/*
 * Save the complete simulation state, so that restoring it and replaying the
 * same actions gives the same state hashes as the original run. On success
 * *out receives a snapshot that the caller frees with dqsim_snapshot_free().
 * Fails, leaving *out untouched, when no map is loaded.
 */
DQSIM_API int dqsim_snapshot_save(dqsim_snapshot **out);

/*
 * Restore a snapshot taken in this level episode. Afterwards the state hash
 * and dqsim_time() equal what they were when the snapshot was saved. Fails
 * without changing anything if the snapshot is from another episode or the
 * engine has shut down.
 */
DQSIM_API int dqsim_snapshot_restore(const dqsim_snapshot *snapshot);

/* Bytes the snapshot occupies, or 0 for NULL. */
DQSIM_API size_t dqsim_snapshot_size(const dqsim_snapshot *snapshot);

/* Release a snapshot. Accepts NULL. */
DQSIM_API void dqsim_snapshot_free(dqsim_snapshot *snapshot);

/* Stop the engine and release its memory. */
DQSIM_API void dqsim_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* DQSIM_H */

"""Navigation task: move the player to a goal position on a real map.

This is the first training task (roadmap M3). The agent controls the player
with discrete movement actions, sees geometry through ray casts, and is told
where the goal is relative to itself. Monsters are made passive (``notarget``)
and the player invulnerable (``god``), so only movement is being learned.

Tasks come from a *walk graph* built by exploring the map on foot at start-up
(:func:`explore_walk_graph`): starts and goals are points the player has stood
on, and difficulty is the walking distance between them, so every task can be
done. Nearby points of different walks are linked only after the simulator
itself has verified the route (:func:`verify_links`), so walks do not need to
meet to be connected. The graph is a property of the map — it builds from a
load seed derived from the map name, so every process and episode seed agrees
on the pool of starts and goals. The reward tracks progress along the walking
route.

One :class:`NavEnv` uses the process-wide simulator, so there can be only one
per process. :mod:`deepquake.vec` runs many in worker processes.

Observation (float32 vector, length :data:`OBS_DIM`):

====================  =====  ==============================================
slice                 size   meaning
====================  =====  ==============================================
``[0:48]``            48     ray distances, 2 pitch rows x 24 yaw columns,
                             fraction of :data:`RAY_DISTANCE` (see
                             :func:`deepquake.sim.ray_fan`)
``[48:96]``           48     kind of the same rays, in the same order:
                             ``SURFACE_*`` / 6 (0 nothing, 1 world, 2 moving
                             brush, 3 water, 4 slime, 5 lava, 6 sky)
``[96:99]``           3      velocity in the player's frame (forward, left,
                             up) divided by 400
``[99]``              1      1 if standing on ground
``[100:102]``         2      unit vector to the goal in the player's frame
                             (forward, left)
``[102]``             1      goal height difference / 256, clipped to +-1
``[103]``             1      horizontal goal distance / 1024, clipped to 2
====================  =====  ==============================================

Action (4 integers, :data:`ACTION_NVEC`): forward {back, none, forward},
strafe {right, none, left}, turn {-30, -10, 0, +10, +30 degrees}, jump
{no, yes}. Each action is held for :data:`FRAME_SKIP` ticks.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from . import sim

RAY_PITCHES = (0.0, 30.0)
RAY_YAWS = 24
RAY_DISTANCE = 768.0
OBS_DIM = 2 * len(RAY_PITCHES) * RAY_YAWS + 8
ACTION_NVEC = (3, 3, 5, 2)

FRAME_SKIP = 6  # ticks per decision: 12 decisions per second
RUN_SPEED = 400.0
TURN_DEGREES = (-30.0, -10.0, 0.0, 10.0, 30.0)

GOAL_RADIUS = 48.0  # horizontal distance that counts as reaching the goal
GOAL_HEIGHT = 64.0  # and the allowed height difference
MAX_STEPS = 300  # decisions per episode: 25 seconds of game time
RELOAD_EVERY = 50  # episodes between map reloads, to undo opened doors etc.

#: Entities whose positions seed the exploration of the map.
_SEED_PREFIXES = ("info_player_", "item_", "weapon_")

_CELL = (64.0, 64.0, 48.0)  # walk-graph resolution in map units (x, y, z)
_STEP_HEIGHT = 18.0  # the player can step up this much, so flatter links go both ways
_LINK_RANGE = 200.0  # horizontal distance up to which point pairs get verified links
_PROBE_TICKS = 120  # a verification walk gets this many ticks to arrive
_PUSH_CHUNKS = 12  # chunks committed to one blocker; a door opens in about that many
_PLAT_WAIT_TICKS = 288  # straight pushing that lets a lift carry the player (4 s)
_DOOR_PUSH_TICKS = 96  # straight pushing that takes the player through a door (1.3 s)
_DOOR_RANGE = 400.0  # how far from a point a door may still be worth a detour
_TRIGGER_RANGE = 700.0  # a trigger zone is worth a longer detour than a door
_SEAM_RANGE = 900.0  # nearest unreached points this far out are still worth a probe


def hash_name(name: str) -> int:
    """Stable small hash of a string (Python's own ``hash`` varies per process)."""
    value = 2166136261
    for byte in name.encode():
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def explore_walk_graph(
    map_name: str, reload_map: Callable[[], None], *, walk_seconds: float = 60.0
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Explore the loaded map on foot and record where the player can walk.

    Starting from every spawn point and item, the player wanders (heading for
    open space, walking into whatever blocks the way, since most doors open on
    touch). Standing positions are grouped into cells; each cell becomes a
    point, and walking from one cell to the next becomes a directed link
    between them. Links are one-way where the walk fell further down than the
    player could climb back. Deterministic for a given map.

    Returns the points, shape ``(n, 3)``, and the set of links ``(from, to)``.
    The map must already be loaded with ``god`` and ``notarget`` active;
    ``reload_map`` must load it again that way (it is called if a walk
    reaches the level exit, which makes the game load the next level).
    """
    rng = np.random.default_rng(hash_name(map_name))
    cells: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    links: set[tuple[int, int]] = set()

    def cell_of(point) -> int:
        key = tuple(int(point[i] // _CELL[i]) for i in range(3))
        if key not in cells:
            cells[key] = len(points)
            points.append(point)
        return cells[key]

    def link(a: int, b: int) -> None:
        if a != b:
            links.add((a, b))
            # The way back climbs points[a][2] - points[b][2]; dropping is
            # always possible, climbing only within the step height.
            if points[a][2] - points[b][2] <= _STEP_HEIGHT:
                links.add((b, a))

    seeds: list[tuple[float, float, float]] = []
    for entity in sim.entities():
        if entity.classname.startswith(_SEED_PREFIXES):
            x, y, z = entity.origin
            # Items rest on the floor; the player's origin is 24 units above it.
            if sim.teleport((x, y, z + 25.0)):
                sim.step(repeat=24)
                state = sim.player()
                if state.on_ground:
                    seeds.append(state.origin)

    # A yard behind a locked door has no items to seed a walk from, and no
    # walk will chance upon its button; start one at the mover itself, on
    # whichever sides and floor heights take. Probed on a snapshot, so the
    # walks below start from exactly the state they would see without this.
    mover_seeds: list[tuple[float, float, float]] = []
    snapshot = sim.snapshot_save()
    for entity in sim.entities():
        if entity.classname == "trigger_once":
            # Crossing a trigger_once zone can open a door that opens on
            # nothing else (the stairwell doors of e1m1 work this way), so a
            # walk starting inside the zone fires it and may then walk
            # through the door it opened, linking both sides on foot.
            mid = [(entity.absmin[i] + entity.absmax[i]) / 2.0 for i in range(3)]
            for z in (mid[2], entity.absmax[2] - 24.0, entity.absmin[2] + 24.0):
                if sim.teleport((mid[0], mid[1], z)):
                    sim.step(repeat=24)
                    state = sim.player()
                    if state.on_ground or state.waterlevel > 0:
                        mover_seeds.append(state.origin)
                        break
        elif entity.classname in ("door", "func_button"):
            mid = [(entity.absmin[i] + entity.absmax[i]) / 2.0 for i in range(3)]
            taken = 0
            for dx, dy in ((0, 80), (0, -80), (80, 0), (-80, 0)):
                for z in (entity.absmin[2] + 24, mid[2] + 24, entity.absmax[2] + 24):
                    if sim.teleport((mid[0] + dx, mid[1] + dy, z)):
                        sim.step(repeat=24)
                        state = sim.player()
                        if state.on_ground:
                            mover_seeds.append(state.origin)
                            taken += 1
                            break
                if taken >= 2:  # one spot per side of the mover is enough
                    break
    sim.snapshot_restore(snapshot)
    sim.snapshot_free(snapshot)

    column_degrees = 360.0 / RAY_YAWS
    # Mover walks draw from their own stream, so adding them never changes
    # where the item walks went.
    mover_rng = np.random.default_rng(hash_name(map_name) ^ 0x5A5A)
    all_seeds = [(s, walk_seconds, rng) for s in seeds] + [
        (s, walk_seconds / 3, mover_rng) for s in mover_seeds
    ]
    for seed, seconds, walk_rng in all_seeds:
        if not sim.teleport(seed):
            continue
        previous = cell_of(seed)
        position = seed
        yaw = float(walk_rng.uniform(0, 360))
        pushing = False  # walking into something that may be a door
        wet_gap = False  # swum since the last standing position
        wet_previous = None  # last cell linked while swimming
        for _ in range(int(seconds * 72 / FRAME_SKIP)):
            rays = sim.ray_fan((0.0,), RAY_YAWS, RAY_DISTANCE)[0][0]  # distances, sole pitch row
            view = sim.player().view_angles[1] % 360
            blocked = rays[0] * RAY_DISTANCE < 80.0
            pushing = blocked  # most blockers are doors that open on touch
            if not pushing:
                # Head for the frontier: the open direction whose landing spot
                # stands in the least-visited cell. A sweep beats a wander —
                # it finds the one doorway out of a yard instead of milling
                # around in it. The heading sticks unless clearly beaten, so
                # the walk travels instead of shivering between directions.

                def score_of(k: int, rays: np.ndarray, view: float, position: tuple) -> float:
                    column = math.radians(view + k * column_degrees)
                    reach = min(float(rays[k]) * RAY_DISTANCE, 128.0)
                    landing = (
                        position[0] + math.cos(column) * reach,
                        position[1] + math.sin(column) * reach,
                    )
                    key = (
                        int(landing[0] // _CELL[0]),
                        int(landing[1] // _CELL[1]),
                        int(position[2] // _CELL[2]),
                    )
                    return (1.0 if key in cells else 0.0) + (1.0 - float(rays[k]))

                best_k, score_best = 0, math.inf
                for k in range(RAY_YAWS):
                    score = score_of(k, rays, view, position) + walk_rng.random() * 0.1
                    if score < score_best:
                        best_k, score_best = k, score
                if score_best + 0.6 < score_of(
                    k=int(round(((yaw - view) % 360) / column_degrees)) % RAY_YAWS,
                    rays=rays,
                    view=view,
                    position=position,
                ):
                    yaw = (view + best_k * column_degrees) % 360
            sim.step(forward=RUN_SPEED, yaw=yaw, repeat=FRAME_SKIP)
            if sim.level_count() != 0:
                reload_map()
                break
            state = sim.player()
            if math.dist(state.origin, position) < 4.0:
                # Not a door, just a wall: give up on this heading.
                pushing = False
                yaw = float(walk_rng.uniform(0, 360))
            position = state.origin
            if state.on_ground or state.waterlevel > 0:
                # Cells are recorded in water too, so a swim across a canal
                # leaves points along the way; a deep swim never stands, so
                # those points are linked here, not by the standing rules:
                # the way back through the same water is always swimmable,
                # which the climb rule of link() would wrongly deny.
                current = cell_of(state.origin)
                if state.on_ground:
                    # Linked even after time in the air (stairs, a drop):
                    # link() makes it one-way when the landing is lower. A
                    # fall is mostly vertical; a gap swum through is a route
                    # however long it is, up to a bound.
                    gap = math.hypot(
                        points[previous][0] - state.origin[0],
                        points[previous][1] - state.origin[1],
                    )
                    if gap > 400.0:
                        # A teleporter: not a walking route, but a one-way
                        # route the agent can very well take, so keep it —
                        # one-way, because walking back is not an option.
                        if previous != current:
                            links.add((previous, current))
                    elif gap < 200.0 or (wet_gap and gap < 900.0):
                        link(previous, current)
                    if wet_previous is not None and wet_previous != current:
                        # Climbing out of the water where the walk did: the
                        # walk itself swam in from the other side, so both
                        # directions are proven by it.
                        links.add((wet_previous, current))
                        links.add((current, wet_previous))
                    wet_gap = False
                    previous = current
                    wet_previous = None
                else:
                    if wet_previous is None:
                        # Stepping into the water: the last dry cell links to
                        # the first wet one both ways (stepping off the edge
                        # again is always possible).
                        if current != previous:
                            links.add((previous, current))
                            links.add((current, previous))
                    elif current != wet_previous:
                        links.add((wet_previous, current))
                        links.add((current, wet_previous))
                    wet_previous = current
                    wet_gap = True
    return np.array(points, dtype=np.float32), links


def _probe_walk(target: np.ndarray, ticks: int = _PROBE_TICKS) -> bool:
    """Walk the player toward ``target``; return True when it arrives.

    A greedy follower: head for the target, detour through the most open
    direction when a wall is in the way, and walk straight into whatever
    blocks the path — most blockers are doors and ramps, which give way (a
    door opens on touch, a ramp is climbed). A push is committed to for
    :data:`_PUSH_CHUNKS` chunks and judged by net progress, because a door
    takes about a second to open; a wall makes none and earns a cooldown of
    forced detours. The walk stops when no chunk has brought the target
    closer for a while. Aborts (False) at the level exit. Deterministic.
    """
    column_degrees = 360.0 / RAY_YAWS

    def arrived(state: sim.PlayerState) -> bool:
        return (
            math.hypot(state.origin[0] - target[0], state.origin[1] - target[1]) < GOAL_RADIUS
            and abs(state.origin[2] - target[2]) < GOAL_HEIGHT
            and (state.on_ground or state.waterlevel > 0)
        )

    position = sim.player().origin
    best = math.hypot(target[0] - position[0], target[1] - position[1])
    stale = 0
    push_left = 0  # chunks still committed to the current blocker
    push_start = (0.0, 0.0, 0.0)
    cooldown = 0  # chunks of forced detour after a push failed
    detour_yaw = None  # a detour being followed through a bend
    for _ in range(ticks // FRAME_SKIP):
        state = sim.player()
        if arrived(state):
            return True
        if sim.level_count() != 0:
            return False  # walked into an exit; the route cannot be verified
        distance = math.hypot(target[0] - state.origin[0], target[1] - state.origin[1])
        if push_left == 0 and detour_yaw is None:
            stale = stale + 1 if distance > best - 4.0 else 0
            if stale >= 8:
                return False
        best = min(best, distance)
        desired = (
            math.degrees(math.atan2(target[1] - state.origin[1], target[0] - state.origin[0])) % 360
        )
        rays = sim.ray_fan((0.0,), RAY_YAWS, RAY_DISTANCE)[0][0]  # distances, sole pitch row
        view = state.view_angles[1] % 360
        straight = rays[int(round(((desired - view) % 360) / column_degrees)) % RAY_YAWS]
        # In water, swim to the target's depth: the way between two pools of
        # Quake water usually runs along the bottom, not the surface. Blocked
        # and level with the target, the way on runs deep (under a rim), so
        # dive; a target above means surfacing is part of the route.
        dive = 0.0
        if state.waterlevel > 0:
            if target[2] < state.origin[2] - 20.0:
                dive = -RUN_SPEED
            elif target[2] > state.origin[2] + 20.0:
                dive = RUN_SPEED
            elif straight < 0.15:
                dive = -RUN_SPEED
        if push_left > 0:
            yaw = desired  # committed: the blocker may still give way
            push_left -= 1
            if push_left == 0 and math.dist(sim.player().origin, push_start) < 16.0:
                cooldown = 6  # it was a wall after all
        elif straight < 0.15 and cooldown == 0:
            push_left = _PUSH_CHUNKS
            push_start = state.origin
            yaw = desired
        else:
            # Detour. Keep the previous one while it stays open and roughly
            # off-course-bounded: the corridor to a target bends, and choosing
            # anew every chunk makes the walk shiver at the first bend and
            # stall between two headings. Otherwise take the fan direction
            # that is open and least off course.
            k_stick = (
                int(round(((detour_yaw - view) % 360) / column_degrees)) % RAY_YAWS
                if detour_yaw is not None
                else None
            )
            if k_stick is not None and rays[k_stick] <= 0.3:
                k_stick = None  # the corridor closed: choose a new one
            if k_stick is None:
                cost_best = math.inf
                for k in range(RAY_YAWS):
                    column = (view + k * column_degrees) % 360
                    delta = (column - desired + 180.0) % 360.0 - 180.0
                    cost = abs(delta) + (1.0 - rays[k]) * 90.0
                    if abs(delta) > 120.0:
                        cost += 120.0  # away from the target is a last resort
                    if cost < cost_best:
                        k_stick, cost_best = k, cost
            if straight >= 0.3:
                detour_yaw = None  # the way ahead is clear again
                yaw = desired
            else:
                detour_yaw = (view + k_stick * column_degrees) % 360
                yaw = detour_yaw
            if cooldown > 0:
                cooldown -= 1
        sim.step(
            forward=RUN_SPEED,
            up=dive,
            yaw=yaw,
            repeat=FRAME_SKIP,
        )
    return arrived(sim.player())


def _push_walk(target: np.ndarray, ticks: int) -> bool:
    """Walk straight at ``target``, through whatever is in the way.

    Unlike :func:`_probe_walk` this never gives up on a blocked heading: a
    lift platform is exactly the case where the player must keep walking
    into the shaft wall while the platform carries them up, so chunks
    without progress are the point. Returns True on arrival.
    """
    for _ in range(ticks // FRAME_SKIP):
        state = sim.player()
        if (
            math.hypot(state.origin[0] - target[0], state.origin[1] - target[1]) < GOAL_RADIUS
            and abs(state.origin[2] - target[2]) < GOAL_HEIGHT
            and (state.on_ground or state.waterlevel > 0)
        ):
            return True
        if sim.level_count() != 0:
            return False
        yaw = (
            math.degrees(math.atan2(target[1] - state.origin[1], target[0] - state.origin[0])) % 360
        )
        dive = 0.0
        if state.waterlevel > 0:
            if target[2] < state.origin[2] - 20.0:
                dive = -RUN_SPEED
            elif target[2] > state.origin[2] + 20.0:
                dive = RUN_SPEED
        sim.step(forward=RUN_SPEED, up=dive, yaw=yaw, repeat=FRAME_SKIP)
    return _probe_walk(target)


class _Reachability:
    """Which node reaches which through the directed edges known so far.

    BFS results are cached per source. Adding the edge ``a -> b`` can only
    create paths that use it, so it invalidates exactly the cached sources
    that already reached ``a`` (a source's cache always contains itself).
    Clearing the whole cache on every edge instead made corridor-heavy maps,
    whose seam passes query many distinct sources, spend hours re-running
    BFS in Python (#29); the incremental rule keeps every answer identical.
    """

    def __init__(self, n: int, edges: Iterable[tuple[int, int]] = ()):
        self.adjacency: list[set[int]] = [set() for _ in range(n)]
        for a, b in edges:
            self.adjacency[a].add(b)
        self._cache: dict[int, set[int]] = {}

    def reaches(self, a: int, b: int) -> bool:
        if a not in self._cache:
            seen = {a}
            queue = deque([a])
            while queue:
                for nxt in self.adjacency[queue.popleft()]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            self._cache[a] = seen
        return b in self._cache[a]

    def add(self, a: int, b: int) -> None:
        self.adjacency[a].add(b)
        for s in [s for s, seen in self._cache.items() if a in seen]:
            del self._cache[s]


def verify_links(
    points: np.ndarray, reload_map: Callable[[], None], known: set[tuple[int, int]] = frozenset()
) -> set[tuple[int, int]]:
    """Link nearby points by asking the simulator whether the walk works.

    Two exploration walks usually pass through the same area without ever
    standing in the same cell, so their points never link. For every ordered
    pair of points within :data:`_LINK_RANGE` horizontally that no route
    links yet, the player is teleported to one and walked to the other
    (:func:`_probe_walk`); the link is added only if the player actually
    arrives. Falls count (the walk may drop), climbs are settled by the
    physics of stepping, and there are two retries: pairs far apart in height
    with no route may be riding a lift (:func:`_push_walk`), and unrouted
    pairs may be joined through a door, a button or a trigger zone — doors
    are where Quake levels change height or yard, and a straight walk cannot
    aim at one. Each probe starts from a snapshot of the freshly loaded map,
    so the outcome never depends on the probes before it. Deterministic.

    ``reload_map`` must load the map again with ``god`` and ``notarget``
    active and reset the level (it is called if a probe walks into the level
    exit, which invalidates the snapshot).
    """
    links: set[tuple[int, int]] = set()

    # Which point reaches which, through the links known so far. Retries are
    # worth running exactly for ordered pairs with no route yet — gating on
    # parts (weak connectivity) is not enough: a pair can be weakly joined
    # through a one-way drop and still need, say, the lift to be walkable.
    reach = _Reachability(len(points), known)

    def reaches(a: int, b: int) -> bool:
        return reach.reaches(a, b)

    def link(a: int, b: int) -> None:
        links.add((a, b))
        reach.add(a, b)

    def yaw_between(a: int, b: int) -> float:
        return float(
            np.degrees(np.arctan2(points[b][1] - points[a][1], points[b][0] - points[a][0])) % 360
        )

    # Door centres in the closed position, wall buttons, and trigger_once
    # zones, as detour waypoints: a touch opens a door, a touch presses a
    # button, and crossing a trigger zone can open a door that opens on
    # nothing else — locked yards in Quake are usually behind one of the
    # three. The fourth column marks trigger zones, which are worth a longer
    # detour than a door.
    movers = [
        np.array(
            [(e.absmin[i] + e.absmax[i]) / 2.0 for i in range(3)]
            + [1.0 if e.classname == "trigger_once" else 0.0],
            dtype=np.float32,
        )
        for e in sim.entities()
        if e.classname in ("door", "func_button", "trigger_once")
    ]
    # Lift platforms: the push-walk retry makes sense only where one is.
    plats = np.array(
        [
            [(e.absmin[i] + e.absmax[i]) / 2.0 for i in range(3)]
            for e in sim.entities()
            if e.classname == "plat"
        ],
        dtype=np.float32,
    ).reshape(-1, 3)

    def off_segment(a: int, b: int, m: np.ndarray) -> float:
        """How far ``m`` lies from the straight line a->b (a detour bound)."""
        p, q = points[a, :2], points[b, :2]
        t = np.clip(np.dot(m[:2] - p, q - p) / max(float(np.dot(q - p, q - p)), 1.0), 0.0, 1.0)
        return float(np.hypot(*(m[:2] - (p + t * (q - p)))))

    snapshot = sim.snapshot_save()

    def resnapshot_if_left_level() -> None:
        nonlocal snapshot
        if sim.level_count() != 0:
            reload_map()  # new level episode: the old snapshot is stale
            snapshot = sim.snapshot_save()

    def attempt(a: int, b: int, ticks: int = _PROBE_TICKS) -> bool:
        """All probe strategies for the ordered pair ``(a, b)``."""
        sim.snapshot_restore(snapshot)
        if not sim.teleport(tuple(float(v) for v in points[a]), yaw_between(a, b)):
            return False  # blocked on the fresh map; NavEnv drops such points
        if _probe_walk(points[b], ticks):
            return True
        resnapshot_if_left_level()
        if (
            abs(points[a][2] - points[b][2]) > 100.0
            and not reaches(a, b)
            and len(plats)
            and min(
                float(np.linalg.norm(plats[:, :2] - points[a, :2], axis=1).min()),
                float(np.linalg.norm(plats[:, :2] - points[b, :2], axis=1).min()),
            )
            < 400.0
        ):
            # Far above or below, with no route known and a lift platform in
            # range: maybe a lift. Start over and walk straight at the target
            # without giving up, so a platform can carry the player while
            # they push into the shaft wall.
            sim.snapshot_restore(snapshot)
            sim.teleport(tuple(float(v) for v in points[a]), yaw_between(a, b))
            if _push_walk(points[b], _PLAT_WAIT_TICKS):
                return True
            resnapshot_if_left_level()
        if not reaches(a, b):
            # Maybe a locked yard between them. Crossing a trigger_once zone
            # can open a door that opens on nothing else (e1m1's stairwell
            # doors work this way), so the nearest zones come first — the
            # doors they open often shadow them as the nearest movers, and
            # pushing a door that does not open on touch proves nothing.
            # Then the nearest doors and buttons: a touch opens a door, a
            # touch presses a button, which opens its door.
            def near_a(trigger: bool, count: int, limit: float) -> list[tuple[float, int]]:
                return [
                    (distance, k)
                    for distance, k in sorted(
                        (float(np.hypot(m[0] - points[a][0], m[1] - points[a][1])), k)
                        for k, m in enumerate(movers)
                        if (m[3] > 0.5) == trigger
                    )[:count]
                    if distance <= limit
                ]

            retries = [
                (distance, k)
                for distance, k in near_a(True, 2, _TRIGGER_RANGE)
                if off_segment(a, b, movers[k]) < 350.0
            ] + near_a(False, 3, _DOOR_RANGE)
            for _, k in retries:
                sim.snapshot_restore(snapshot)
                if not sim.teleport(
                    tuple(float(v) for v in points[a]),
                    float(
                        np.degrees(
                            np.arctan2(movers[k][1] - points[a][1], movers[k][0] - points[a][0])
                        )
                        % 360
                    ),
                ):
                    continue
                _push_walk(movers[k], _DOOR_PUSH_TICKS)  # through, onto the button, or across
                if _probe_walk(points[b], max(ticks, 2 * _PROBE_TICKS)):
                    return True
                resnapshot_if_left_level()
        return False

    offsets = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for a in range(len(points)):
        near: list[tuple[int, int]] = [
            (b, _PROBE_TICKS)
            for b in range(len(points))
            if b != a
            and 0
            < float(np.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]))
            <= _LINK_RANGE
        ]
        # The nearest points no route reaches across a longer gap: water
        # tunnels, hatches and stairwells join parts farther apart than
        # _LINK_RANGE, and the closest unreached neighbour points straight
        # across the seam. Farther walks get proportionally more ticks — a
        # route 900 units long cannot be walked in the default budget.
        extra = 0
        for b in np.argsort(offsets[a]):
            b = int(b)
            if b == a:
                continue
            if offsets[a][b] > _SEAM_RANGE:
                break
            if reaches(a, b):
                continue
            near.append((b, int(min(offsets[a][b] / 200.0 * 72.0, 480.0))))
            extra += 1
            if extra >= 3:
                break
        for b, ticks in near:
            if reaches(a, b):
                continue  # some route already links them
            if attempt(a, b, ticks):
                link(a, b)
            resnapshot_if_left_level()
    # The same seams from the other side: for each point, the nearest points
    # that cannot reach it yet. A pocket's only access (the trigger that
    # opens its door, the hatch above it) can be far from the pocket while
    # every pocket member sits near it — the per-point pass above then never
    # aims at the pocket, and this pass does.
    for b in range(len(points)):
        extra = 0
        for a in np.argsort(offsets[:, b]):
            a = int(a)
            if a == b:
                continue
            if offsets[a][b] > _SEAM_RANGE:
                break
            if reaches(a, b):
                continue
            if attempt(a, b, int(min(offsets[a][b] / 200.0 * 72.0, 480.0))):
                link(a, b)
            resnapshot_if_left_level()
            extra += 1
            if extra >= 2:
                break
    sim.snapshot_free(snapshot)
    return links


def path_distances(n: int, points: np.ndarray, links: set[tuple[int, int]]) -> np.ndarray:
    """Shortest walking distance between every pair of points; inf if unreachable."""
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b in sorted(links):
        neighbours[a].append((b, float(np.linalg.norm(points[a] - points[b]))))
    distances = np.full((n, n), np.inf, dtype=np.float32)
    for source in range(n):
        best = distances[source]
        best[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > best[node]:
                continue
            for other, length in neighbours[node]:
                candidate = d + length
                if candidate < best[other]:
                    best[other] = candidate
                    heapq.heappush(heap, (candidate, other))
    return distances


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    success: bool


class NavEnv:
    """Goal-reaching on one map. Call :meth:`reset`, then :meth:`step` repeatedly.

    Starts and goals are points of the walk graph, and a task's difficulty is
    the walking distance between them (``path_distance``), so every task is
    reachable on foot. The reward follows progress along the walking route,
    which the agent cannot see: it only knows the straight-line direction.
    """

    def __init__(self, map_name: str = "e1m1", *, seed: int = 0, skill: int = 0):
        self.map_name = map_name
        self.skill = skill
        self.rng = np.random.default_rng(seed)
        self.max_goal_distance = 300.0  # walking distance; raised as a curriculum
        self.min_goal_distance = 100.0
        self._episodes_since_load = 0

        # The walk graph is a property of the map, not of the episode RNG:
        # item positions (hence the item-seeded walks) depend on the load
        # seed, so the graph builds from a fixed one. Every process then
        # agrees on the pool of starts and goals, whatever its episode seed.
        graph_seed = hash_name(map_name) & 0x7FFFFFFF

        def load_fresh() -> None:
            sim.load_map(self.map_name, skill=self.skill, seed=graph_seed)
            sim.client_command("god")
            sim.client_command("notarget")
            sim.step(repeat=2)
            self._episodes_since_load = 0

        load_fresh()
        points, links = explore_walk_graph(map_name, load_fresh)
        load_fresh()  # exploring opened doors and took items
        # Keep only points that are free on a fresh map (not behind a door
        # that starts closed, not on a lift that starts elsewhere).
        keep = np.array([sim.teleport(tuple(float(v) for v in p)) for p in points])
        new_index = np.cumsum(keep) - 1
        self.points = points[keep]
        links = {(int(new_index[a]), int(new_index[b])) for a, b in links if keep[a] and keep[b]}
        # Lift platforms: an exploration walk never stands on one long enough
        # to be carried, so their standing spots never become points. Add one
        # per platform, at the centre of its top face in the fresh position;
        # verify_links() then finds the routes over them.
        for entity in sim.entities():
            if entity.classname == "plat":
                spot = (
                    (entity.absmin[0] + entity.absmax[0]) / 2.0,
                    (entity.absmin[1] + entity.absmax[1]) / 2.0,
                    entity.absmax[2] + 24.0,
                )
                if sim.teleport(spot):
                    self.points = np.append(self.points, np.array([spot], dtype=np.float32), axis=0)
        # Teleporters: a one-way link from the zone to the destination, taken
        # from the entities instead of hoping an exploration walk rode one.
        # (The zone's own standing points link to a new point at the spot the
        # teleport leaves the player.)
        zones = [e for e in sim.entities() if e.classname == "trigger_teleport"]
        dests = [e for e in sim.entities() if e.classname == "info_teleport_destination"]
        if len(zones) == 1 and len(dests) == 1:
            zone, dest = zones[0], dests[0]
            centre = np.array(
                [(zone.absmin[i] + zone.absmax[i]) / 2.0 for i in range(3)], dtype=np.float32
            )
            near_zone = np.flatnonzero(np.linalg.norm(self.points - centre, axis=1) <= 150.0)
            spot = None
            for up in (24.0, 0.0, 8.0, 40.0):
                candidate = np.array(dest.origin, dtype=np.float32) + (0.0, 0.0, up)
                if sim.teleport(tuple(float(v) for v in candidate)):
                    spot = candidate
                    break
            if spot is not None and len(near_zone):
                self.points = np.append(self.points, spot.reshape(1, 3), axis=0)
                dest_index = len(self.points) - 1
                links |= {(int(i), dest_index) for i in near_zone}
        load_fresh()  # the teleports above may have moved a platform
        links |= verify_links(self.points, load_fresh, links)
        self.path_distance = path_distances(len(self.points), self.points, links)
        if len(self.points) < 10:
            raise RuntimeError(f"only {len(self.points)} standing points found on {map_name}")

        self.goal_index = 0
        self.goal = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.steps = 0
        self.remaining = 0.0  # estimated walking distance still to go

    def _load_map(self) -> None:
        sim.load_map(self.map_name, skill=self.skill, seed=int(self.rng.integers(2**31)))
        sim.client_command("god")
        sim.client_command("notarget")
        sim.step(repeat=2)
        self._episodes_since_load = 0

    def set_task(self, start_index: int, goal_index: int, yaw: float) -> np.ndarray:
        """Begin an episode between two points of the walk graph."""
        if self._episodes_since_load >= RELOAD_EVERY or sim.level_count() != 0:
            self._load_map()
        self._episodes_since_load += 1
        start = tuple(float(v) for v in self.points[start_index])
        if not sim.teleport(start, yaw):
            # Something has moved onto the spot. A fresh map fixes doors and
            # lifts; a monster standing there does not, because its wander is
            # seeded by the load seed (blocked at 7 of 20 seeds on e3m6, #30),
            # and _load_map draws a new seed each call — so keep rolling.
            for _ in range(5):
                self._load_map()
                self._episodes_since_load = 1
                if sim.teleport(start, yaw):
                    break
            else:
                raise RuntimeError(f"start point {start} is blocked on a fresh map")
        self.goal_index = int(goal_index)
        self.goal = self.points[goal_index]
        self.yaw = yaw
        self.steps = 0
        sim.step(yaw=self.yaw, repeat=2)
        state = sim.player()
        self.remaining = float(self.path_distance[start_index, goal_index])
        return self._observe(state)

    def sample_task(self, rng: np.random.Generator) -> tuple[int, int, float]:
        """A random ``(start, goal, yaw)`` whose walking distance is in the current range."""
        for _ in range(1000):
            start = int(rng.integers(len(self.points)))
            distances = self.path_distance[start]
            candidates = np.flatnonzero(
                (distances >= self.min_goal_distance) & (distances <= self.max_goal_distance)
            )
            if len(candidates):
                return start, int(rng.choice(candidates)), float(rng.uniform(0, 360))
        # Unlucky draws (or a map whose routes are short): look at every start.
        for start in range(len(self.points)):
            distances = self.path_distance[start]
            candidates = np.flatnonzero(
                (distances >= self.min_goal_distance) & (distances <= self.max_goal_distance)
            )
            if len(candidates):
                return start, int(rng.choice(candidates)), float(rng.uniform(0, 360))
        raise RuntimeError(
            f"no start and goal pair on {self.map_name} within "
            f"{self.min_goal_distance:.0f}-{self.max_goal_distance:.0f} walking units"
        )

    def reset(self) -> np.ndarray:
        """Begin an episode with a random task within the curriculum range."""
        return self.set_task(*self.sample_task(self.rng))

    def step(self, action: np.ndarray) -> StepResult:
        """Apply one action for FRAME_SKIP ticks."""
        forward, strafe, turn, jump = (int(a) for a in action)
        self.yaw = (self.yaw + TURN_DEGREES[turn]) % 360.0
        sim.step(
            forward=(forward - 1) * RUN_SPEED,
            side=(1 - strafe) * RUN_SPEED,
            yaw=self.yaw,
            buttons=sim.BUTTON_JUMP if jump else 0,
            repeat=FRAME_SKIP,
        )
        self.steps += 1
        state = sim.player()

        distance = self._goal_distance(state)
        success = distance < GOAL_RADIUS and abs(self.goal[2] - state.origin[2]) < GOAL_HEIGHT
        left_level = sim.level_count() != 0  # walked into an exit
        remaining = self._remaining(state, distance)
        reward = float(np.clip((self.remaining - remaining) / 100.0, -1.0, 1.0)) - 0.01
        if success:
            reward += 5.0
        self.remaining = remaining
        done = success or left_level or self.steps >= MAX_STEPS
        return StepResult(self._observe(state), float(reward), done, bool(success))

    def _remaining(self, state: sim.PlayerState, straight: float) -> float:
        """Walking distance left: via the nearest graph points, or direct when close."""
        position = np.asarray(state.origin, dtype=np.float32)
        if straight < 2 * GOAL_RADIUS and abs(self.goal[2] - position[2]) < GOAL_HEIGHT:
            return straight
        offsets = np.linalg.norm(self.points - position, axis=1)
        via = offsets + self.path_distance[:, self.goal_index]
        via[offsets > 128.0] = np.inf
        best = float(via.min())
        return best if math.isfinite(best) else self.remaining

    def _goal_distance(self, state: sim.PlayerState) -> float:
        return math.hypot(self.goal[0] - state.origin[0], self.goal[1] - state.origin[1])

    def _observe(self, state: sim.PlayerState) -> np.ndarray:
        obs = np.empty(OBS_DIM, dtype=np.float32)
        n_rays = len(RAY_PITCHES) * RAY_YAWS
        distances, kinds = sim.ray_fan(RAY_PITCHES, RAY_YAWS, RAY_DISTANCE)
        obs[:n_rays] = distances.ravel()
        obs[n_rays : 2 * n_rays] = kinds.ravel() / 6.0

        base = 2 * n_rays
        yaw = math.radians(state.view_angles[1])
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = state.velocity
        obs[base + 0] = (vx * cos + vy * sin) / 400.0
        obs[base + 1] = (-vx * sin + vy * cos) / 400.0
        obs[base + 2] = vz / 400.0
        obs[base + 3] = float(state.on_ground)

        dx, dy = self.goal[0] - state.origin[0], self.goal[1] - state.origin[1]
        distance = math.hypot(dx, dy)
        scale = 1.0 / max(distance, 1e-6)
        obs[base + 4] = (dx * cos + dy * sin) * scale
        obs[base + 5] = (-dx * sin + dy * cos) * scale
        obs[base + 6] = np.clip((self.goal[2] - state.origin[2]) / 256.0, -1.0, 1.0)
        obs[base + 7] = min(distance / 1024.0, 2.0)
        return obs


# ---- baseline policies: functions from observation to action ----


def random_policy(rng: np.random.Generator):
    """Uniformly random actions."""

    def act(obs: np.ndarray) -> np.ndarray:
        return np.array([rng.integers(n) for n in ACTION_NVEC])

    return act


def beeline_policy(obs: np.ndarray) -> np.ndarray:
    """Turn toward the goal and run at it, ignoring walls."""
    base = 2 * len(RAY_PITCHES) * RAY_YAWS
    forward, left = obs[base + 4], obs[base + 5]
    angle = math.degrees(math.atan2(left, forward))  # + means the goal is to the left
    if angle > 20:
        turn = 4
    elif angle > 5:
        turn = 3
    elif angle < -20:
        turn = 0
    elif angle < -5:
        turn = 1
    else:
        turn = 2
    return np.array([2 if abs(angle) < 60 else 1, 1, turn, 0])

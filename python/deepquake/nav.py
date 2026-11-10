"""Navigation task: move the player to a goal position on a real map.

This is the first training task (roadmap M3). The agent controls the player
with discrete movement actions, sees geometry through ray casts, and is told
where the goal is relative to itself. Monsters are made passive (``notarget``)
and the player invulnerable (``god``), so only movement is being learned.

Tasks come from a *walk graph* built by exploring the map on foot at start-up
(:func:`explore_walk_graph`): starts and goals are points the player has stood
on, and difficulty is the walking distance between them, so every task can be
done. The reward tracks progress along the walking route.

One :class:`NavEnv` uses the process-wide simulator, so there can be only one
per process. :mod:`deepquake.vec` runs many in worker processes.

Observation (float32 vector, length :data:`OBS_DIM`):

====================  =====  ==============================================
slice                 size   meaning
====================  =====  ==============================================
``[0:48]``            48     ray distances, 2 pitch rows x 24 yaw columns,
                             fraction of :data:`RAY_DISTANCE` (see
                             :func:`deepquake.sim.ray_fan`)
``[48:51]``           3      velocity in the player's frame (forward, left,
                             up) divided by 400
``[51]``              1      1 if standing on ground
``[52:54]``           2      unit vector to the goal in the player's frame
                             (forward, left)
``[54]``              1      goal height difference / 256, clipped to +-1
``[55]``              1      horizontal goal distance / 1024, clipped to 2
====================  =====  ==============================================

Action (4 integers, :data:`ACTION_NVEC`): forward {back, none, forward},
strafe {right, none, left}, turn {-30, -10, 0, +10, +30 degrees}, jump
{no, yes}. Each action is held for :data:`FRAME_SKIP` ticks.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import sim

RAY_PITCHES = (0.0, 30.0)
RAY_YAWS = 24
RAY_DISTANCE = 768.0
OBS_DIM = len(RAY_PITCHES) * RAY_YAWS + 8
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
    open space, turning away from walls). Standing positions are grouped into
    cells; each cell becomes a point, and walking from one cell to the next
    becomes a directed link between them. Links are one-way where there is a
    drop. Deterministic for a given map.

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
            if abs(points[a][2] - points[b][2]) <= _STEP_HEIGHT:
                links.add((b, a))

    seeds = []
    for entity in sim.entities():
        if entity.classname.startswith(_SEED_PREFIXES):
            x, y, z = entity.origin
            # Items rest on the floor; the player's origin is 24 units above it.
            if sim.teleport((x, y, z + 25.0)):
                sim.step(repeat=24)
                state = sim.player()
                if state.on_ground:
                    seeds.append(state.origin)

    column_degrees = 360.0 / RAY_YAWS
    for seed in seeds:
        if not sim.teleport(seed):
            continue
        previous = cell_of(seed)
        position = seed
        yaw = float(rng.uniform(0, 360))
        for _ in range(int(walk_seconds * 72 / FRAME_SKIP)):
            rays = sim.ray_fan((0.0,), RAY_YAWS, RAY_DISTANCE)[0]
            if rays[0] * RAY_DISTANCE < 80.0 or rng.random() < 0.04:
                # Blocked ahead, or time for a change: head for open space.
                weights = rays.astype(np.float64) ** 2 + 1e-6
                yaw += float(rng.choice(RAY_YAWS, p=weights / weights.sum())) * column_degrees
            elif rng.random() < 0.3:
                yaw += float(rng.choice((-10.0, 10.0)))
            sim.step(forward=RUN_SPEED, yaw=yaw, repeat=FRAME_SKIP)
            if sim.level_count() != 0:
                reload_map()
                break
            state = sim.player()
            if math.dist(state.origin, position) < 4.0:
                yaw = float(rng.uniform(0, 360))  # stuck: try any other way
            position = state.origin
            if state.on_ground:
                # Linked even after time in the air (stairs, a drop): link() makes
                # it one-way when the landing is lower. A long gap means a
                # teleporter, which is not a walking route.
                current = cell_of(state.origin)
                if math.dist(points[previous], state.origin) < 200.0:
                    link(previous, current)
                previous = current
    return np.array(points, dtype=np.float32), links


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

        self._load_map()
        points, links = explore_walk_graph(map_name, self._load_map)
        self._load_map()  # exploring opened doors and took items
        # Keep only points that are free on a fresh map (not behind a door
        # that starts closed, not on a lift that starts elsewhere).
        keep = np.array([sim.teleport(tuple(float(v) for v in p)) for p in points])
        new_index = np.cumsum(keep) - 1
        self.points = points[keep]
        links = {(int(new_index[a]), int(new_index[b])) for a, b in links if keep[a] and keep[b]}
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
            # Something (a door, a lift) has moved onto the spot; a fresh map fixes it.
            self._load_map()
            self._episodes_since_load = 1
            if not sim.teleport(start, yaw):
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
        raise RuntimeError("no start and goal pair found within the distance range")

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
        obs[:n_rays] = sim.ray_fan(RAY_PITCHES, RAY_YAWS, RAY_DISTANCE).ravel()

        yaw = math.radians(state.view_angles[1])
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = state.velocity
        obs[n_rays + 0] = (vx * cos + vy * sin) / 400.0
        obs[n_rays + 1] = (-vx * sin + vy * cos) / 400.0
        obs[n_rays + 2] = vz / 400.0
        obs[n_rays + 3] = float(state.on_ground)

        dx, dy = self.goal[0] - state.origin[0], self.goal[1] - state.origin[1]
        distance = math.hypot(dx, dy)
        scale = 1.0 / max(distance, 1e-6)
        obs[n_rays + 4] = (dx * cos + dy * sin) * scale
        obs[n_rays + 5] = (-dx * sin + dy * cos) * scale
        obs[n_rays + 6] = np.clip((self.goal[2] - state.origin[2]) / 256.0, -1.0, 1.0)
        obs[n_rays + 7] = min(distance / 1024.0, 2.0)
        return obs


# ---- baseline policies: functions from observation to action ----


def random_policy(rng: np.random.Generator):
    """Uniformly random actions."""

    def act(obs: np.ndarray) -> np.ndarray:
        return np.array([rng.integers(n) for n in ACTION_NVEC])

    return act


def beeline_policy(obs: np.ndarray) -> np.ndarray:
    """Turn toward the goal and run at it, ignoring walls."""
    n_rays = len(RAY_PITCHES) * RAY_YAWS
    forward, left = obs[n_rays + 4], obs[n_rays + 5]
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

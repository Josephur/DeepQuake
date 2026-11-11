"""Combat task: the player fights one stock monster on a real map (#32).

This is the experiment environment built to settle the combat-task design
(episode, reward, termination, opponent pool) before a training issue is
written from it. The monster runs its stock QuakeC brain — the player is
not ``god`` and not ``notarget`` here, unlike navigation. Placement reuses
the navigation walk graph: on a fresh load the player is teleported to a
standing point at a fixed distance range from a chosen monster spawn,
with line of sight checked (and, for graph episodes, required broken)
from the ray fan.

Observation (float32 vector, length :data:`OBS_DIM`): the navigation ray
channels first, then combat state.

====================  =====  ==============================================
slice                 size   meaning
====================  =====  ==============================================
``[0:48]``            48     ray distances, 2 pitch rows x 24 yaw columns
``[48:96]``           48     surface kind of the same rays, / 6
``[96:99]``           3      velocity in the player's frame, / 400
``[99]``              1      1 if standing on ground
``[100:102]``         2      unit vector to the enemy in the player's
                             frame, only while visible (else 0)
``[102]``             1      enemy height difference / 256, clipped, ditto
``[103]``             1      enemy distance / 1024, clipped to 2, ditto
``[104]``             1      1 while the enemy is visible
``[105]``             1      own health / 100, clipped to 1
``[106]``             1      enemy health / 100 while visible, else 0
``[107]``             1      decisions remaining, / :data:`timeout_steps`
====================  =====  ==============================================

Action (5 integers, :data:`ACTION_NVEC`): the navigation heads (forward,
strafe, turn, jump) plus fire {no, yes}. Reward is selected by name
(:data:`REWARDS`): ``damage`` pays damage dealt minus taken (per 100)
plus +1 for the kill and -1 for dying; ``survival`` adds a small
per-decision bonus on top; ``sparse`` pays only the terminal +-1.
Timeout pays 0 and ends the episode.
"""

from __future__ import annotations

import math

import numpy as np

from . import nav, sim

OBS_DIM = 2 * len(nav.RAY_PITCHES) * nav.RAY_YAWS + 12
ACTION_NVEC = (3, 3, 5, 2, 2)
REWARDS = ("damage", "survival", "sparse")

FRAME_SKIP = nav.FRAME_SKIP
SIGHT_DISTANCE = 768.0


class CombatEnv:
    """One player against one stock monster, fresh map load per episode."""

    def __init__(
        self,
        map_name: str = "e1m1",
        *,
        seed: int = 0,
        reward: str = "damage",
        monster: str = "monster_army",
        timeout_steps: int = 300,
        min_distance: float = 200.0,
        max_distance: float = 600.0,
    ):
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
        self.map_name = map_name
        self.reward_name = reward
        self.monster = monster
        self.timeout_steps = timeout_steps
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.rng = np.random.default_rng(seed)
        # The walk graph is a property of the map; building it needs the
        # passive-navigation setup, which NavEnv's constructor performs.
        # Combat episodes do their own fresh loads without god/notarget.
        self.points = nav.NavEnv(map_name, seed=seed).points
        self.yaw = 0.0
        self.steps = 0
        self.enemy_index = -1
        self._enemy_cached: sim.Entity | None = None
        self._prev_player_health = 100.0
        self._prev_enemy_health = 0.0

    def _fresh_load(self, load_seed: int) -> list[sim.Entity]:
        sim.load_map(self.map_name, skill=0, seed=load_seed)
        sim.step(repeat=2)
        return [e for e in sim.entities() if e.classname == self.monster]

    def _visible_from(self, state: sim.PlayerState, enemy: sim.Entity, distances) -> bool:
        """Whether the enemy is visible, read off a ray fan already cast."""
        dx = enemy.origin[0] - state.origin[0]
        dy = enemy.origin[1] - state.origin[1]
        distance = math.hypot(dx, dy)
        if distance > SIGHT_DISTANCE:
            return False
        bearing = math.degrees(math.atan2(dy, dx)) % 360.0
        view = state.view_angles[1] % 360.0
        column = int(round(((bearing - view) % 360.0) / (360.0 / nav.RAY_YAWS))) % nav.RAY_YAWS
        return float(distances[0, column]) * SIGHT_DISTANCE >= distance - 48.0

    def _line_of_sight(self, enemy: sim.Entity) -> bool:
        """Whether the enemy is visible from the player's current eye."""
        distances, _kinds = sim.ray_fan(nav.RAY_PITCHES, nav.RAY_YAWS, SIGHT_DISTANCE)
        return self._visible_from(sim.player(), enemy, distances)

    def set_task(self, load_seed: int, opponent_choice: int, task_seed: int):
        """Begin a deterministic episode from the training placement.

        The start point is chosen with ``task_seed`` from the same
        distance/height/line-of-sight-broken candidate set :meth:`reset`
        uses, so a fixed task list measures the task that was trained, not
        a search across the whole map.
        """
        enemies = self._fresh_load(load_seed)
        if not enemies:
            raise RuntimeError(f"no {self.monster} on {self.map_name}")
        enemy = enemies[opponent_choice % len(enemies)]
        distances = np.hypot(
            self.points[:, 0] - enemy.origin[0], self.points[:, 1] - enemy.origin[1]
        )
        height = np.abs(self.points[:, 2] - enemy.origin[2])
        candidates = np.flatnonzero(
            (distances >= self.min_distance) & (distances <= self.max_distance) & (height < 48.0)
        )
        rng = np.random.default_rng(task_seed)
        yaw = float(rng.uniform(0, 360))
        for start_index in rng.permutation(candidates)[:50]:
            start = tuple(float(v) for v in self.points[int(start_index)])
            if not sim.teleport(start, yaw):
                continue
            sim.step(yaw=yaw, repeat=2)
            if self._line_of_sight(enemy):
                continue
            self.enemy_index = enemy.index
            self._enemy_cached = enemy
            self.yaw = yaw
            self.steps = 0
            state = sim.player()
            self._prev_player_health = state.health
            self._prev_enemy_health = enemy.health
            return self._observe(state)
        raise RuntimeError("no line-of-sight-broken start for the fixed task")

    def reset(self) -> np.ndarray:
        for _attempt in range(200):
            load_seed = int(self.rng.integers(2**31))
            enemies = self._fresh_load(load_seed)
            if not enemies:
                raise RuntimeError(f"no {self.monster} on {self.map_name}")
            enemy = enemies[int(self.rng.integers(len(enemies)))]
            distances = np.hypot(
                self.points[:, 0] - enemy.origin[0], self.points[:, 1] - enemy.origin[1]
            )
            height = np.abs(self.points[:, 2] - enemy.origin[2])
            candidates = np.flatnonzero(
                (distances >= self.min_distance)
                & (distances <= self.max_distance)
                & (height < 48.0)
            )
            if not len(candidates):
                continue
            start_index = int(self.rng.choice(candidates))
            start = tuple(float(v) for v in self.points[start_index])
            yaw = float(self.rng.uniform(0, 360))
            if not sim.teleport(start, yaw):
                continue
            sim.step(yaw=yaw, repeat=2)
            if self._line_of_sight(enemy):
                continue  # graph episodes start with line of sight broken
            self.enemy_index = enemy.index
            self._enemy_cached = enemy
            self.yaw = yaw
            self.steps = 0
            state = sim.player()
            self._prev_player_health = state.health
            self._prev_enemy_health = enemy.health
            return self._observe(state)
        raise RuntimeError("no line-of-sight-broken combat start found")

    def _refresh_enemy(self) -> None:
        """Snapshot the tracked enemy (one entity scan per decision)."""
        self._enemy_cached = None
        for entity in sim.entities():
            if entity.index == self.enemy_index:
                self._enemy_cached = entity
                return

    def _enemy(self) -> sim.Entity | None:
        return self._enemy_cached

    def step(self, action: np.ndarray) -> nav.StepResult:
        forward, strafe, turn, jump, fire = (int(a) for a in action)
        self.yaw = (self.yaw + nav.TURN_DEGREES[turn]) % 360.0
        sim.step(
            forward=(forward - 1) * nav.RUN_SPEED,
            side=(1 - strafe) * nav.RUN_SPEED,
            yaw=self.yaw,
            buttons=sim.BUTTON_ATTACK if fire else 0,
            repeat=FRAME_SKIP,
        )
        self.steps += 1
        state = sim.player()
        self._refresh_enemy()
        enemy = self._enemy()
        enemy_health = max(enemy.health, 0.0) if enemy is not None else 0.0
        dealt = max(0.0, self._prev_enemy_health - enemy_health)
        taken = max(0.0, self._prev_player_health - state.health)
        self._prev_enemy_health = enemy_health
        self._prev_player_health = state.health

        won = enemy is None or enemy_health <= 0.0
        lost = state.health <= 0.0
        done = won or lost or self.steps >= self.timeout_steps or sim.level_count() != 0
        if self.reward_name == "sparse":
            reward = 0.0
        else:
            reward = (dealt - taken) / 100.0
            if self.reward_name == "survival":
                reward += 0.001
        if won:
            reward += 1.0
        elif lost:
            reward -= 1.0
        return nav.StepResult(self._observe(state), float(reward), bool(done), bool(won))

    def _observe(self, state: sim.PlayerState) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        n_rays = len(nav.RAY_PITCHES) * nav.RAY_YAWS
        distances, kinds = sim.ray_fan(nav.RAY_PITCHES, nav.RAY_YAWS, nav.RAY_DISTANCE)
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

        enemy = self._enemy()
        if enemy is not None and enemy.health > 0 and self._visible_from(state, enemy, distances):
            dx = enemy.origin[0] - state.origin[0]
            dy = enemy.origin[1] - state.origin[1]
            distance = math.hypot(dx, dy)
            scale = 1.0 / max(distance, 1e-6)
            obs[base + 4] = (dx * cos + dy * sin) * scale
            obs[base + 5] = (-dx * sin + dy * cos) * scale
            obs[base + 6] = np.clip((enemy.origin[2] - state.origin[2]) / 256.0, -1.0, 1.0)
            obs[base + 7] = min(distance / 1024.0, 2.0)
            obs[base + 8] = 1.0
            obs[base + 10] = min(enemy.health / 100.0, 1.0)
        obs[base + 9] = min(max(state.health, 0.0) / 100.0, 1.0)
        obs[base + 11] = 1.0 - self.steps / self.timeout_steps
        return obs


# ---- baseline and evaluation helpers on the same fixed tasks ----


def make_tasks(env: CombatEnv, count: int, seed: int) -> list[tuple[int, int, int]]:
    """Fixed ``(load_seed, opponent_choice, task_seed)`` tasks."""
    rng = np.random.default_rng(seed)
    return [
        (int(rng.integers(2**31)), int(rng.integers(64)), int(rng.integers(2**31)))
        for _ in range(count)
    ]


def evaluate(env: CombatEnv, policy, tasks) -> dict:
    """Run a policy (observation -> action) on every task; win/loss/timeout."""
    wins = losses = timeouts = 0
    lengths = []
    for task in tasks:
        try:
            obs = env.set_task(*task)
        except RuntimeError:
            losses += 1  # an unusable start counts against every policy equally
            continue
        if hasattr(policy, "reset"):
            policy.reset()
        while True:
            result = env.step(policy(obs))
            obs = result.obs
            if result.done:
                break
        lengths.append(env.steps)
        if result.success:
            wins += 1
        elif env._prev_player_health <= 0.0:
            losses += 1
        else:
            timeouts += 1
    total = max(len(tasks), 1)
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "timeout_rate": timeouts / total,
        "mean_decisions": float(np.mean(lengths)) if lengths else 0.0,
    }


def random_policy(rng: np.random.Generator):
    """Uniformly random actions."""

    def act(obs: np.ndarray) -> np.ndarray:
        return np.array([rng.integers(n) for n in ACTION_NVEC])

    return act


def shooter_policy(obs: np.ndarray) -> np.ndarray:
    """Face the visible enemy and fire; otherwise seek it, wall-avoiding."""
    base = 2 * len(nav.RAY_PITCHES) * nav.RAY_YAWS
    if obs[base + 8] > 0:
        angle = math.degrees(math.atan2(obs[base + 5], obs[base + 4]))
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
        return np.array([2, 1, turn, 0, 1])
    if obs[0] < 0.1:  # the ray straight ahead ends at a wall: turn away
        return np.array([2, 1, 4, 0, 0])
    return np.array([2, 1, 2, 0, 0])

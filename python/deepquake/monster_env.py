"""The monster-side combat environment (#35): a hooked monster against a player.

Roles inverted from :mod:`combat`: the policy under training drives the
*monster* through the brain hook, and the environment drives the *player*
opponent (random, the scripted shooter, or a trained player checkpoint —
#38's policy at parity level, per the maintainer's decision there). The
env contract is exactly :class:`combat.CombatEnv`'s, so
:class:`deepquake.vec.ProcVecEnv`, :class:`deepquake.ppo.RolloutActor` and
PPO work unmodified.

Which body the policy wears is a per-episode choice (#48): ``monsters``
lists every class a shared model should learn, sampled in :meth:`reset`,
while a single ``monster`` pins one class for evaluation and for the
single-type runs of #35.

The brain hook fires every tick, but decisions are made every
:data:`nav.FRAME_SKIP` ticks, aligned with the player's decision rate: the
env holds one discrete monster decision across the block and the hook
callback returns it every tick, caching the freshest hook report for the
next observation. During QuakeC attack and pain sequences no think fires,
so a decision may read a report a few ticks (or, mid-sequence, moments)
old — the memory channels of the observation exist for exactly that.

Fresh map loads clear the hook nomination, so every episode re-nominates.
Skill scales the engine's handicaps (reaction delay, aim error; #34) via
``load_map(skill=...)`` and is also an observation input (#33); both sides
of a task always agree on it. Reward is #32's sparse terminal (+1 the
player dies, -1 the monster dies, 0 timeout); a player killed by any
monster counts — the other stock monsters on a map are part of the task,
for the trained and the stock brain alike.

Observation (monster side, length :data:`monster_obs.MONSTER_OBS_DIM`): see
:mod:`deepquake.monster`. Action (4 integers, :data:`MONSTER_ACTION_NVEC`):
forward {back, stop, ahead} and strafe {left, none, right} at the
monster's speed (:data:`MONSTER_SPEED` commanded — the engine's monster
physics cap what actually moves, about 0.55x, measured in the #35
spike), turn in the same 7-step grid the player got from #40
(accumulated into the facing the hook wants), and attack, which starts
the real QuakeC attack sequence when the player is visible and the
QuakeC cooldown has expired.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import combat, nav, sim
from . import monster as monster_obs

MONSTER_ACTION_NVEC = (3, 3, 7, 2)
MONSTER_TURN_DEGREES = combat.TURN_DEGREES
MONSTER_SPEED = nav.RUN_SPEED
DEFAULT_STYLE = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class PlayerOpponent:
    """Who plays the player side: ``random``, ``shooter`` or ``trained``.

    A ``trained`` opponent loads a player checkpoint (#38) and must be
    given its file path; it is a fixed, frozen sparring partner.
    """

    kind: str
    checkpoint: str | None = None

    def build(self, seed: int) -> _Opponent:
        if self.kind == "trained" and not self.checkpoint:
            raise ValueError("a trained opponent needs a checkpoint path")
        if self.kind not in ("random", "shooter", "trained"):
            raise ValueError(f"unknown opponent kind {self.kind!r}")
        return _Opponent(self, seed)


class _Opponent:
    """One player opponent inside one environment process."""

    def __init__(self, spec: PlayerOpponent, seed: int):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self._trained: Callable[[np.ndarray], np.ndarray] | None = None
        if spec.kind == "trained":
            from .eval_nav import TrainedPolicy  # torch: only in training/eval processes

            self._trained = TrainedPolicy(Path(spec.checkpoint), seed)

    def reset(self) -> None:
        if self._trained is not None:
            self._trained.reset()

    def act(
        self,
        state: sim.PlayerState,
        enemy: sim.Entity,
        distances,
        kinds,
        steps: int,
        timeout_steps: int,
    ) -> np.ndarray:
        visible = enemy.health > 0 and combat.visible_from(state, enemy, distances)
        obs = combat.player_observation(
            state, enemy, visible, distances, kinds, steps, timeout_steps
        )
        if self.spec.kind == "random":
            return np.array([int(self.rng.integers(n)) for n in combat.ACTION_NVEC])
        if self.spec.kind == "shooter":
            return combat.shooter_policy(obs)
        return self._trained(obs)


@dataclass
class _Held:
    """The monster decision the hook returns on every tick of a block."""

    yaw: float
    forward: float
    side: float
    attack: bool


class MonsterEnv:
    """One hooked monster against one player opponent, fresh load per episode."""

    def __init__(
        self,
        map_name: str = "e1m1",
        *,
        seed: int = 0,
        monster: str = "monster_army",
        monsters: Sequence[str] | None = None,
        opponent: PlayerOpponent | None = None,
        timeout_steps: int = 600,
        min_distance: float = 200.0,
        max_distance: float = 600.0,
        los_broken_required: bool = True,
        reward: str = "sparse",
        skills: Sequence[int] = (2,),
        styles: Sequence[tuple[float, float, float]] = (DEFAULT_STYLE,),
        brain: str = "policy",
        points: np.ndarray | None = None,
    ):
        if reward not in combat.REWARDS:
            raise ValueError(f"reward must be one of {combat.REWARDS}, got {reward!r}")
        if brain not in ("policy", "stock"):
            raise ValueError(f"brain must be 'policy' or 'stock', got {brain!r}")
        if not skills or any(int(s) not in (0, 1, 2, 3) for s in skills):
            raise ValueError(f"skills must be a non-empty subset of 0-3, got {skills!r}")
        if not styles:
            raise ValueError(
                "styles must be a non-empty sequence of (aggression, caution, stealth)"
            )
        # ``monsters`` is the shared-policy case (#48): the class is sampled
        # per episode, so one network sees both bodies in one run. A single
        # ``monster`` pins the class, which is what evaluation wants.
        self.monsters = tuple(monsters) if monsters is not None else (monster,)
        if not self.monsters:
            raise ValueError("monsters must name at least one monster class")
        self.map_name = map_name
        self.monster = self.monsters[0]
        #: The trained classes the current level stocks (#50); the body of
        #: an episode is sampled from these.
        self.available: tuple[str, ...] = self.monsters
        self.opponent = (opponent or PlayerOpponent("shooter")).build(seed * 1000 + 777)
        self.timeout_steps = timeout_steps
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.los_broken_required = los_broken_required
        self.reward_name = reward
        self.skills = tuple(int(s) for s in skills)
        self.styles = tuple(tuple(float(v) for v in style) for style in styles)
        self.brain = brain
        self.rng = np.random.default_rng(seed)
        # The walk graph is a property of the map; building it needs the
        # passive-navigation setup, which NavEnv's constructor performs.
        # Callers that make many environments for one map (evaluation)
        # pass ``points`` so the graph is built once, not per env.
        self.points = (
            nav.NavEnv(map_name, seed=seed).points if points is None else np.asarray(points)
        )
        self.observer = monster_obs.MonsterObserver(skill=self.skills[0], style=self.styles[0])
        self.skill = self.skills[0]
        self.steps = 0
        self.monster_yaw = 0.0
        self.player_yaw = 0.0
        self.player_pitch = 0.0
        self.monster_index = -1
        self._hook_obs: sim.HookObservation | None = None
        self._held: _Held | None = None
        self._prev_player_health = 100.0
        self._prev_monster_health = 0.0

    # -- brain hook -----------------------------------------------------

    def _hook_policy(self, hook: sim.HookObservation) -> sim.HookAction | None:
        """Runs inside ``sim.step``, once per tick: cache the report, hold the decision."""
        self._hook_obs = hook
        if self._held is None:
            return None
        return sim.HookAction(
            desired_yaw=self._held.yaw,
            forward=self._held.forward,
            side=self._held.side,
            attack=self._held.attack,
        )

    # -- episode lifecycle ----------------------------------------------

    def _fresh_load(self, load_seed: int, skill: int) -> list[sim.Entity]:
        """Load the level and return its monsters.

        All of them, not only the class of the episode: the body is chosen
        from the classes the level stocks (#50), so a run that names a class
        a level does not have still trains on that level.
        """
        sim.load_map(self.map_name, skill=skill, seed=load_seed)
        sim.step(repeat=2)
        return [e for e in sim.entities() if e.classname.startswith("monster_")]

    def _available_classes(self, monsters: list[sim.Entity]) -> list[str]:
        """The trained classes this level stocks, in the run's order."""
        stocked = {e.classname for e in monsters}
        return [name for name in self.monsters if name in stocked]

    def _monster_entity(self) -> sim.Entity | None:
        for entity in sim.entities():
            if entity.index == self.monster_index:
                return entity
        return None

    def _place_player(self, enemy: sim.Entity, rng: np.random.Generator) -> None:
        """Teleport the player to a start 200-600 units from the monster.

        The same candidate set as :meth:`combat.CombatEnv.reset`/``set_task`:
        walk-graph points within the distance band and height bound, line of
        sight broken (training placements always break it, whatever
        ``los_broken_required`` says — the task is to find the fight).
        """
        distances = np.hypot(
            self.points[:, 0] - enemy.origin[0], self.points[:, 1] - enemy.origin[1]
        )
        height = np.abs(self.points[:, 2] - enemy.origin[2])
        candidates = np.flatnonzero(
            (distances >= self.min_distance) & (distances <= self.max_distance) & (height < 48.0)
        )
        yaw = float(rng.uniform(0, 360))
        for start_index in rng.permutation(candidates)[:50] if len(candidates) else ():
            start = tuple(float(v) for v in self.points[int(start_index)])
            if not sim.teleport(start, yaw):
                continue
            sim.step(yaw=yaw, repeat=2)
            if self.los_broken_required and combat.visible_from(
                sim.player(), enemy, sim.ray_fan(nav.RAY_PITCHES, nav.RAY_YAWS, nav.RAY_DISTANCE)[0]
            ):
                continue
            self.player_yaw = yaw
            self.player_pitch = 0.0
            return
        raise RuntimeError("no line-of-sight-broken start for the monster task")

    def _start(
        self,
        load_seed: int,
        monster_choice: int,
        task_seed: int,
        skill: int,
        style: tuple[float, float, float],
        monsters: list[sim.Entity] | None = None,
    ) -> np.ndarray:
        if monsters is None:
            monsters = self._fresh_load(load_seed, skill)
        candidates = [e for e in monsters if e.classname == self.monster]
        if not candidates:
            raise RuntimeError(f"no {self.monster} on {self.map_name}")
        chosen = candidates[monster_choice % len(candidates)]
        self.monster_index = chosen.index
        self._place_player(chosen, np.random.default_rng(task_seed))
        self.steps = 0
        self._hook_obs = None
        self._held = None
        self.observer = monster_obs.MonsterObserver(
            skill=skill, style=style, monster_type=self.monster
        )
        self.skill = skill
        if self.brain == "policy":
            sim.set_policy_hook(self.monster_index, self._hook_policy)
        self._prev_player_health = sim.player().health
        self._prev_monster_health = max(self._monster_entity().health, 0.0)
        self._first_damage_dealt: int | None = None
        self._first_damage_taken: int | None = None
        if self.brain == "policy":
            # Warm-up ticks: the monster's first think fires the hook and the
            # observation is built from that report; the player stands still.
            for _ in range(20):
                if self._hook_obs is not None:
                    break
                sim.step(repeat=2)
            if self._hook_obs is None:
                raise RuntimeError("the hooked monster never thought during the warm-up")
            self.monster_yaw = self._hook_obs.angles[1]
        self.opponent.reset()
        return self._monster_observation()

    def reset(self) -> np.ndarray:
        for _attempt in range(20):
            # The level is loaded first, so the body can be chosen from the
            # classes it stocks: a level without the Enforcer (e1m1) must not
            # spend an episode failing to find one (#50).
            load_seed = int(self.rng.integers(2**31))
            skill = int(self.rng.choice(self.skills))
            style = self.styles[int(self.rng.integers(len(self.styles)))]
            monsters = self._fresh_load(load_seed, skill)
            self.available = tuple(self._available_classes(monsters))
            if not self.available:
                continue
            self.monster = self.available[int(self.rng.integers(len(self.available)))]
            try:
                return self._start(
                    load_seed,
                    int(self.rng.integers(64)),
                    int(self.rng.integers(2**31)),
                    skill,
                    style,
                    monsters,
                )
            except RuntimeError:
                continue  # no usable start: reroll
        raise RuntimeError("no usable monster episode start found")

    def set_task(
        self,
        load_seed: int,
        monster_choice: int,
        task_seed: int,
        skill: int = 2,
        style: tuple[float, float, float] = DEFAULT_STYLE,
    ) -> np.ndarray:
        """Begin a deterministic episode from the training placement."""
        monsters = self._fresh_load(load_seed, int(skill))
        self.available = tuple(self._available_classes(monsters))
        return self._start(load_seed, monster_choice, task_seed, int(skill), tuple(style), monsters)

    # -- one decision ----------------------------------------------------

    def _monster_observation(self) -> np.ndarray:
        if self._hook_obs is None or self.brain == "stock":
            return np.zeros(monster_obs.MONSTER_OBS_DIM, dtype=np.float32)
        return self.observer.observe(self._hook_obs)

    def _player_command(self) -> None:
        """Let the opponent act on the state at the end of the last block."""
        state = sim.player()
        enemy = self._monster_entity()
        distances, kinds = sim.ray_fan(nav.RAY_PITCHES, nav.RAY_YAWS, nav.RAY_DISTANCE)
        forward, strafe, turn, jump, fire, pitch = (
            int(a)
            for a in self.opponent.act(
                state, enemy, distances, kinds, self.steps, self.timeout_steps
            )
        )
        self.player_yaw = (self.player_yaw + combat.TURN_DEGREES[turn]) % 360.0
        self.player_pitch = float(
            np.clip(self.player_pitch + combat.PITCH_DEGREES[pitch], -45.0, 45.0)
        )
        self._player_command_args = dict(
            forward=(forward - 1) * nav.RUN_SPEED,
            side=(1 - strafe) * nav.RUN_SPEED,
            pitch=self.player_pitch,
            yaw=self.player_yaw,
            buttons=sim.BUTTON_ATTACK if fire else 0,
        )

    def step(self, action: np.ndarray | None) -> nav.StepResult:
        if self.brain == "policy" and action is not None:
            forward, strafe, turn, attack = (int(a) for a in action)
            self.monster_yaw = (self.monster_yaw + MONSTER_TURN_DEGREES[turn]) % 360.0
            self._held = _Held(
                yaw=self.monster_yaw,
                forward=(forward - 1) * MONSTER_SPEED,
                side=(1 - strafe) * MONSTER_SPEED,
                attack=bool(attack),
            )
        else:
            self._held = None  # stock brain: QuakeC decides, the hook stays quiet
        self._player_command()
        sim.step(**self._player_command_args, repeat=nav.FRAME_SKIP)
        self.steps += 1
        state = sim.player()
        entity = self._monster_entity()
        player_health = max(state.health, 0.0)
        monster_health = max(entity.health, 0.0) if entity is not None else 0.0
        dealt = max(0.0, self._prev_player_health - player_health)
        taken = max(0.0, self._prev_monster_health - monster_health)
        self._prev_player_health = player_health
        self._prev_monster_health = monster_health
        if dealt > 0.0 and self._first_damage_dealt is None:
            self._first_damage_dealt = self.steps  # the monster side first drew blood
        if taken > 0.0 and self._first_damage_taken is None:
            self._first_damage_taken = self.steps

        won = state.health <= 0.0  # any monster's kill counts: they are one team
        lost = entity is None or entity.health <= 0.0
        done = won or lost or self.steps >= self.timeout_steps or sim.level_count() != 0
        reward = 0.0 if self.reward_name == "sparse" else (dealt - taken) / 100.0
        if won:
            reward += 1.0
        elif lost:
            reward -= 1.0
        return nav.StepResult(self._monster_observation(), float(reward), bool(done), bool(won))


def make_tasks(env: MonsterEnv, count: int, seed: int) -> list[tuple[int, int, int, int]]:
    """Fixed ``(load_seed, monster_choice, task_seed, skill)`` tasks."""
    rng = np.random.default_rng(seed)
    return [
        (
            int(rng.integers(2**31)),
            int(rng.integers(64)),
            int(rng.integers(2**31)),
            int(env.skills[int(rng.integers(len(env.skills)))]),
        )
        for _ in range(count)
    ]


def evaluate(env: MonsterEnv, policy, tasks) -> dict:
    """Run a monster policy (observation -> action) on every task.

    ``policy`` may be ``None`` for the stock QuakeC Grunt: the episode
    then runs with no hook nomination. Returns win/loss/timeout rates
    from the monster's point of view — a win is the player dying.
    """
    wins = losses = timeouts = 0
    lengths = []
    win_lengths: list[int] = []
    damage_times: list[int] = []
    for task in tasks:
        try:
            obs = env.set_task(*task)
        except RuntimeError:
            losses += 1  # an unusable start counts against every policy equally
            continue
        if policy is not None and hasattr(policy, "reset"):
            policy.reset()
        while True:
            action = None if policy is None else policy(obs)
            result = env.step(action)
            obs = result.obs
            if result.done:
                break
        lengths.append(env.steps)
        if result.success:
            wins += 1
            win_lengths.append(env.steps)
        elif env._prev_monster_health <= 0.0:
            losses += 1
        else:
            timeouts += 1
        if env._first_damage_dealt is not None:
            damage_times.append(env._first_damage_dealt)
    total = max(len(tasks), 1)
    # Times are in decisions (12 per second); means cover only the tasks
    # where the event happened, and the companion *_rate says how often.
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "timeout_rate": timeouts / total,
        "mean_decisions": float(np.mean(lengths)) if lengths else 0.0,
        "damage_rate": len(damage_times) / total,
        "mean_time_to_first_damage": (
            float(np.mean(damage_times)) if damage_times else float("nan")
        ),
        "mean_time_to_kill": float(np.mean(win_lengths)) if win_lengths else float("nan"),
    }

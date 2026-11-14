"""The monster's observation: fair senses, skill and style inputs (#33).

The monster-side twin of the navigation observation. A
:class:`MonsterObserver` turns the brain hook's raw per-tick report
(:class:`deepquake.sim.HookObservation`) into the vector a monster policy
will consume. Everything in it is relative to the monster or about the
monster itself — no absolute coordinates, nothing per-map, and nothing
through walls (map-independence rule 4, design #22): enemy features come
only from what the hook currently sees (line of sight) plus a memory of
the last place the enemy *was seen*, which fades after
:data:`MEMORY_SECONDS`. The hook reports no sound events yet, so hearing
adds nothing until it does; when it does, a heard position may refresh
the same memory slot (a sound is a fair sense in the design), never the
currently-visible channels.

Observation (float32 vector, length :data:`MONSTER_OBS_DIM`):

====================  =====  ==============================================
slice                 size   meaning
====================  =====  ==============================================
``[0:48]``            48     ray distances from the monster's eye, 2 pitch
                             rows x 24 yaw columns, fraction of 768
``[48:96]``           48     surface kind of the same rays, / 6
``[96:99]``           3      velocity in the monster's frame (forward,
                             left, up) / 400
``[99]``              1      1 if standing on ground
``[100]``             1      own health / 100, clipped to 1
``[101]``             1      attack cooldown in seconds, clipped to 1
                             (0: QuakeC will let it attack now)
``[102]``             1      water level / 3
``[103:105]``         2      unit vector to the enemy in the monster's
                             frame, only while visible (else 0)
``[105]``             1      enemy height difference / 256, clipped, ditto
``[106]``             1      enemy distance / 1024, clipped to 2, ditto
``[107]``             1      1 while the enemy is visible
``[108]``             1      enemy health / 100 while visible, else 0
``[109:111]``         2      unit vector toward the last seen enemy
                             position, while the memory lasts (else 0)
``[111]``             1      its height difference / 256, clipped, ditto
``[112]``             1      its distance / 1024, clipped to 2, ditto
``[113]``             1      memory age / :data:`MEMORY_SECONDS`, clipped
``[114]``             1      1 while a last-seen position is remembered
``[115]``             1      skill / 3 (Quake's ``skill``, 0-3)
``[116:119]``         3      style: aggression, caution, stealth (0-1)
``[119:122]``         3      monster type, one-hot in :data:`MONSTER_TYPES`
                             order (0: Grunt, 1: Dog, 2: Enforcer)
====================  =====  ==============================================

One observer serves one monster for one episode: call
:meth:`MonsterObserver.reset` when the monster (re)spawns. Building the
vector is deterministic — the same sequence of hook reports always gives
the same vectors.

The type input is what lets one network serve every monster type (#48,
#50): the observation gives the shared trunk the body the policy is wearing,
and the network's per-type output head is selected from the same one-hot. It
is appended rather than prepended so a network trained on the 119-wide layout
warm-starts by copying its input columns unchanged. The Enforcer joins as
the third type: the hook reaches it (#49), so all it needed was its place in
this vocabulary.
"""

from __future__ import annotations

import math

import numpy as np

from . import sim

N_RAYS = len(sim.HOOK_RAY_PITCHES) * sim.HOOK_RAY_YAWS
#: The shared network's type vocabulary, in index order. The client's
#: ``<model>.onnx.monsters`` sidecar maps a classname to this index.
MONSTER_TYPES = ("monster_army", "monster_dog", "monster_enforcer")
N_MONSTER_TYPES = len(MONSTER_TYPES)
MONSTER_OBS_BASE_DIM = 2 * N_RAYS + 23
MONSTER_OBS_DIM = MONSTER_OBS_BASE_DIM + N_MONSTER_TYPES
MEMORY_SECONDS = 5.0


class MonsterObserver:
    """Builds monster observations from brain-hook reports for one monster."""

    def __init__(
        self,
        *,
        skill: int = 1,
        style: tuple[float, float, float] = (0.5, 0.5, 0.5),
        monster_type: str = MONSTER_TYPES[0],
        memory_seconds: float = MEMORY_SECONDS,
    ):
        if skill not in (0, 1, 2, 3):
            raise ValueError(f"skill must be 0-3, got {skill!r}")
        if len(style) != 3 or any(not 0.0 <= value <= 1.0 for value in style):
            raise ValueError(f"style must be three values in [0, 1], got {style!r}")
        if monster_type not in MONSTER_TYPES:
            raise ValueError(f"monster type must be one of {MONSTER_TYPES}, got {monster_type!r}")
        self.skill = skill
        self.style = tuple(float(value) for value in style)
        self.monster_type = monster_type
        self.memory_seconds = float(memory_seconds)
        self.reset()

    def reset(self) -> None:
        """Forget the enemy (a new episode or a respawned monster)."""
        self._last_seen: tuple[float, float, float] | None = None
        self._last_seen_time = 0.0

    def observe(self, hook: sim.HookObservation) -> np.ndarray:
        """The observation vector for one brain-hook report."""
        obs = np.zeros(MONSTER_OBS_DIM, dtype=np.float32)
        obs[:N_RAYS] = hook.rays.ravel()
        obs[N_RAYS : 2 * N_RAYS] = hook.ray_kinds.ravel() / 6.0

        base = 2 * N_RAYS
        yaw = math.radians(hook.angles[1])
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = hook.velocity
        obs[base + 0] = (vx * cos + vy * sin) / 400.0
        obs[base + 1] = (-vx * sin + vy * cos) / 400.0
        obs[base + 2] = vz / 400.0
        obs[base + 3] = float(hook.on_ground)
        obs[base + 4] = min(max(hook.health, 0.0) / 100.0, 1.0)
        obs[base + 5] = min(max(hook.attack_cooldown, 0.0), 1.0)
        obs[base + 6] = hook.waterlevel / 3.0

        enemy = next((v for v in hook.visible if v.flags & sim.FL_CLIENT), None)
        if enemy is not None:
            self._last_seen = tuple(hook.origin[k] + enemy.relative[k] for k in range(3))
            self._last_seen_time = hook.time
            self._enemy_channels(obs, base + 7, enemy.relative, cos, sin)
            obs[base + 11] = 1.0
            obs[base + 12] = min(max(enemy.health, 0.0) / 100.0, 1.0)

        if self._last_seen is not None:
            age = hook.time - self._last_seen_time
            if age <= self.memory_seconds:
                relative = tuple(self._last_seen[k] - hook.origin[k] for k in range(3))
                self._enemy_channels(obs, base + 13, relative, cos, sin)
                obs[base + 17] = min(max(age, 0.0) / self.memory_seconds, 1.0)
                obs[base + 18] = 1.0

        obs[base + 19] = self.skill / 3.0
        obs[base + 20 : base + 23] = self.style
        obs[base + 23 + MONSTER_TYPES.index(self.monster_type)] = 1.0
        return obs

    @staticmethod
    def _enemy_channels(obs, at: int, relative, cos: float, sin: float) -> None:
        """Unit vector (forward, left), height and distance for a relative offset."""
        dx, dy, dz = relative
        distance = math.hypot(dx, dy)
        scale = 1.0 / max(distance, 1e-6)
        obs[at + 0] = (dx * cos + dy * sin) * scale
        obs[at + 1] = (-dx * sin + dy * cos) * scale
        obs[at + 2] = float(np.clip(dz / 256.0, -1.0, 1.0))
        obs[at + 3] = min(distance / 1024.0, 2.0)

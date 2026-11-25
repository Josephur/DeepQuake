"""The monster's observation: fair senses, skill and style inputs (#33).

The monster-side twin of the navigation observation. A
:class:`MonsterObserver` turns the brain hook's raw per-tick report
(:class:`deepquake.sim.HookObservation`) into the vector a monster policy
will consume. Everything in it is relative to the monster or about the
monster itself — no absolute coordinates, nothing per-map, and nothing
through walls (map-independence rule 4, design #22): enemy features come
only from what the hook currently sees (line of sight) plus a memory of
the last place the enemy *was seen*, which fades after
:data:`MEMORY_SECONDS`. Hearing is kept apart from that memory: the last
noise the hook heard has channels of its own and never refreshes the
last-seen place, because a noise says where something was heard, not that
the enemy was seen there. What kind of noise it was is never an input.

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
``[100]``             1      own health as a fraction of its health when
                             nominated (``start_health``), clipped to
                             [0, 1]; a report without a start health
                             divides by 100
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
``[119:121]``         2      unit vector to the last heard noise in the
                             monster's frame, only while it is
                             remembered (else 0)
``[121]``             1      its height difference / 256, clipped, ditto
``[122]``             1      its distance / 1024, clipped to 2, ditto
``[123]``             1      its age / :data:`deepquake.sim.HOOK_HEARD_SECONDS`,
                             clipped, ditto
``[124]``             1      1 while a noise is remembered
``[125:128]``         3      the pace the hook moves the monster at, one-hot:
                             walk, investigate, run
``[128:130]``         2      unit vector to the nearest visible live monster
                             other than itself (the first such in the
                             hook's nearest-first list), else 0
``[130]``             1      its distance / 1024, clipped to 2 (0 when none)
``[131]``             1      visible live monsters other than itself / 4,
                             clipped to 1
``[132:142]``         10     monster type, one-hot in :data:`MONSTER_TYPES`
                             order (0: Grunt, 1: Dog, 2: Enforcer, 3: Ogre,
                             4: Hell Knight, 5: Shalrath, 6: Knight, 7: Fiend,
                             8: Shambler, 9: Scrag)
====================  =====  ==============================================

A unit vector is (forward, left); a height difference is the target's
height minus the monster's own. "Live monster" means the ``FL_MONSTER``
flag and positive health in the hook's ``visible`` list.

One observer serves one monster for one episode: call
:meth:`MonsterObserver.reset` when the monster (re)spawns. Building the
vector is deterministic — the same sequence of hook reports always gives
the same vectors.

The type input is what lets one network serve every monster type (#48,
#50): the observation gives the shared trunk the body the policy is wearing,
and the network's per-type output head is selected from the same one-hot.
The Enforcer through the Shambler joined as the third through ninth types
(#49, #53, #55, #56, #57, #58, #59), and the Scrag is the tenth (retrain
spec parts 0C and 0D): the hook reaches them, so all each needed was its
place in this vocabulary.

The layout changed once, for the retraining (RETRAIN_SPEC part 0D): health
became a fraction of the monster's own starting health instead of a
fraction of 100, and the heard noise, the pace and the visible allies were
added. A checkpoint from the 128-wide layout does not fit this one.
"""

from __future__ import annotations

import math

import numpy as np

from . import sim

N_RAYS = len(sim.HOOK_RAY_PITCHES) * sim.HOOK_RAY_YAWS
#: The shared network's type vocabulary, in index order. The client's
#: ``<model>.onnx.monsters`` sidecar maps a classname to this index.
MONSTER_TYPES = (
    "monster_army",
    "monster_dog",
    "monster_enforcer",
    "monster_ogre",
    "monster_hell_knight",
    "monster_shalrath",
    "monster_knight",
    "monster_demon1",
    "monster_shambler",
    "monster_wizard",
)
N_MONSTER_TYPES = len(MONSTER_TYPES)
MONSTER_OBS_BASE_DIM = 2 * N_RAYS + 36
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
        start = hook.start_health if hook.start_health > 0.0 else 100.0
        obs[base + 4] = min(max(hook.health, 0.0) / start, 1.0)
        obs[base + 5] = min(max(hook.attack_cooldown, 0.0), 1.0)
        obs[base + 6] = hook.waterlevel / 3.0

        enemy = next((v for v in hook.visible if v.flags & sim.FL_CLIENT), None)
        if enemy is not None:
            self._last_seen = tuple(hook.origin[k] + enemy.relative[k] for k in range(3))
            self._last_seen_time = hook.time
            self._target_channels(obs, base + 7, enemy.relative, cos, sin)
            obs[base + 11] = 1.0
            obs[base + 12] = min(max(enemy.health, 0.0) / 100.0, 1.0)

        if self._last_seen is not None:
            age = hook.time - self._last_seen_time
            if age <= self.memory_seconds:
                relative = tuple(self._last_seen[k] - hook.origin[k] for k in range(3))
                self._target_channels(obs, base + 13, relative, cos, sin)
                obs[base + 17] = min(max(age, 0.0) / self.memory_seconds, 1.0)
                obs[base + 18] = 1.0

        obs[base + 19] = self.skill / 3.0
        obs[base + 20 : base + 23] = self.style

        if hook.heard:
            self._target_channels(obs, base + 23, hook.heard_relative, cos, sin)
            obs[base + 27] = min(max(hook.heard_age, 0.0) / sim.HOOK_HEARD_SECONDS, 1.0)
            obs[base + 28] = 1.0
        if hook.pace in (sim.PACE_WALK, sim.PACE_INVESTIGATE, sim.PACE_RUN):
            obs[base + 29 + hook.pace] = 1.0

        allies = [
            v
            for v in hook.visible
            if v.flags & sim.FL_MONSTER and v.health > 0.0 and v.index != hook.entity_index
        ]
        if allies:
            ahead, left, distance = _bearing(allies[0].relative, cos, sin)
            obs[base + 32] = ahead
            obs[base + 33] = left
            obs[base + 34] = min(distance / 1024.0, 2.0)
        obs[base + 35] = min(len(allies) / 4.0, 1.0)

        obs[base + 36 + MONSTER_TYPES.index(self.monster_type)] = 1.0
        return obs

    @staticmethod
    def _target_channels(obs, at: int, relative, cos: float, sin: float) -> None:
        """Unit vector (forward, left), height and distance for a relative offset."""
        dx, dy, dz = relative
        ahead, left, distance = _bearing(relative, cos, sin)
        obs[at + 0] = ahead
        obs[at + 1] = left
        obs[at + 2] = float(np.clip(dz / 256.0, -1.0, 1.0))
        obs[at + 3] = min(distance / 1024.0, 2.0)


def _bearing(relative, cos: float, sin: float) -> tuple[float, float, float]:
    """Where a relative offset lies in the monster's frame: (forward, left)
    as a unit vector, and its flat distance. A zero offset reads (0, 0)."""
    dx, dy, _dz = relative
    distance = math.hypot(dx, dy)
    scale = 1.0 / max(distance, 1e-6)
    return (dx * cos + dy * sin) * scale, (-dx * sin + dy * cos) * scale, distance

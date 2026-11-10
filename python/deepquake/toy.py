"""A tiny environment with the same interface as :class:`deepquake.nav.NavEnv`.

It needs no simulator or game data, so the training code (PPO, the
vectorised runner) can be tested anywhere, including CI. The agent stands on
a line and must walk to a target position.
"""

from __future__ import annotations

import numpy as np

from .nav import StepResult

OBS_DIM = 2
ACTION_NVEC = (3,)  # step left, stay, step right


class LineEnv:
    """Walk along a line from a random start to a random target within 20 steps."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.scale = 1.0  # settable through ProcVecEnv.set_attr, like a curriculum
        self.position = self.target = 0.0
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.position, self.target = self.rng.uniform(-1, 1, size=2) * self.scale
        self.steps = 0
        return self._observe()

    def step(self, action: np.ndarray) -> StepResult:
        before = abs(self.target - self.position)
        self.position += (int(action[0]) - 1) * 0.2
        self.steps += 1
        after = abs(self.target - self.position)
        success = after < 0.15
        reward = (before - after) + (1.0 if success else 0.0)
        return StepResult(self._observe(), float(reward), success or self.steps >= 20, success)

    def _observe(self) -> np.ndarray:
        return np.array([self.position, self.target], dtype=np.float32)


def make_line_env(index: int) -> LineEnv:
    """Factory for :class:`deepquake.vec.ProcVecEnv`."""
    return LineEnv(seed=index)

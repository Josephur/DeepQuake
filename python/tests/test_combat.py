"""The combat experiment environment (#32). Engine tests need game data."""

import numpy as np
import pytest
from test_sim import require_gamedata, run_in_subprocess

from deepquake import combat


class _StubEnv:
    points = np.zeros((10, 3), dtype=np.float32)


def test_reward_names_and_sizes():
    assert combat.REWARDS == ("damage", "survival", "sparse")
    assert combat.OBS_DIM == 108 and combat.ACTION_NVEC == (3, 3, 7, 2, 2, 3)
    with pytest.raises(ValueError, match="reward"):
        combat.CombatEnv("e1m1", reward="nope")


def test_make_tasks_is_deterministic():
    first = combat.make_tasks(_StubEnv(), 5, seed=7)  # type: ignore[arg-type]
    second = combat.make_tasks(_StubEnv(), 5, seed=7)  # type: ignore[arg-type]
    assert first == second and len(first) == 5


COMBAT = """
import json
import numpy as np
from deepquake import combat, sim

sim.init()
env = combat.CombatEnv("e1m1", seed=3)
obs = env.reset()
result = env.step(np.array([2, 1, 3, 0, 0, 1]))
tasks = combat.make_tasks(env, 10, 99)
random = combat.evaluate(env, combat.random_policy(np.random.default_rng(0)), tasks)
shooter = combat.evaluate(env, combat.shooter_policy, tasks)
print(json.dumps({
    "obs_shape": list(obs.shape), "obs_min": float(obs.min()), "obs_max": float(obs.max()),
    "random": random, "shooter": shooter,
}))
"""


@pytest.mark.gamedata
def test_combat_environment():
    require_gamedata()
    # Building the walk graph in the simulator dominates this test.
    out = run_in_subprocess(COMBAT, timeout=300)
    assert out["obs_shape"] == [108]
    assert out["obs_min"] >= -1.0 and out["obs_max"] <= 2.0
    # Every fixed task resolves as a win, a loss or a timeout.
    for name in ("random", "shooter"):
        rates = out[name]
        assert abs(rates["win_rate"] + rates["loss_rate"] + rates["timeout_rate"] - 1.0) < 1e-9

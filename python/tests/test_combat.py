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


def _stage(min_d, max_d, los, until, close=0.0):
    return combat.CurriculumStage(until, combat.Placement(min_d, max_d, los), close)


STAGES = [
    _stage(100.0, 350.0, False, 100),
    _stage(100.0, 600.0, False, 200, 0.5),
    _stage(200.0, 600.0, True, 300, 0.3),
]


def test_curriculum_validation_rejects_bad_schedules():
    with pytest.raises(ValueError, match="at least one stage"):
        combat.validate_stages([])
    with pytest.raises(ValueError, match="close_fraction"):
        combat.validate_stages([_stage(100.0, 350.0, False, 10, 1.5)])
    with pytest.raises(ValueError, match="min_distance"):
        combat.validate_stages([_stage(600.0, 200.0, False, 10)])
    with pytest.raises(ValueError, match="increase"):
        combat.validate_stages([_stage(100.0, 350.0, False, 100), _stage(100.0, 600.0, False, 50)])


def test_stage_placement_progression():
    near, wide, far = (stage.placement for stage in STAGES)
    assert combat._stage_placement(STAGES, 0) is near
    assert combat._stage_placement(STAGES, 99) is near
    assert combat._stage_placement(STAGES, 100) is wide  # until_step is the stage's end
    assert combat._stage_placement(STAGES, 199) is wide
    assert combat._stage_placement(STAGES, 200) is far
    assert combat._stage_placement(STAGES, 10**9) is far  # the last stage holds forever


def test_stage_close_fraction_keeps_the_fight_regime():
    rng = np.random.default_rng(0)
    draws = [combat._stage_placement(STAGES, 150, rng) for _ in range(200)]
    assert all(placement in (STAGES[0].placement, STAGES[1].placement) for placement in draws)
    wide = sum(placement is STAGES[1].placement for placement in draws)
    assert 60 < wide < 140  # close_fraction 0.5 on a seeded stream
    # fraction 0 never falls back (and never draws from the stream)
    strict = np.random.default_rng(0)
    assert all(combat._stage_placement(STAGES, 0, strict) is STAGES[0].placement for _ in range(10))


def test_scale_stages_gives_each_environment_its_share():
    scaled = combat.scale_stages(STAGES, 1.0 / 8)
    assert [stage.until_step for stage in scaled] == [12, 25, 38]  # round(100/8) is 12
    assert scaled[1].close_fraction == 0.5  # the schedule itself is untouched


def test_stages_from_config_round_trip():
    raw = [
        {
            "until_step": 2000000,
            "min_distance": 100.0,
            "max_distance": 350.0,
            "los_broken_required": False,
        },
        {
            "until_step": 10000000,
            "min_distance": 200.0,
            "max_distance": 600.0,
            "los_broken_required": True,
            "close_fraction": 0.3,
        },
    ]
    stages = combat.stages_from_config(raw)
    assert stages[1].placement == combat.Placement(200.0, 600.0, True)
    assert stages[1].close_fraction == 0.3
    typo = {
        "until_step": 5,
        "min_distance": 1.0,
        "max_distance": 2.0,
        "los_broken_required": False,
        "mista": 1,
    }
    with pytest.raises(ValueError, match="unknown curriculum stage settings"):
        combat.stages_from_config([typo])

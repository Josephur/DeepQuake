"""The navigation environment on a real map. Needs game data."""

import pytest
from test_sim import require_gamedata, run_in_subprocess

NAV = """
import json
import numpy as np
from deepquake import nav, sim
from deepquake.eval_nav import evaluate, make_tasks

sim.init()
env = nav.NavEnv("e1m1", seed=3)
obs = env.reset()
result = env.step(np.array([2, 1, 2, 0]))
tasks = make_tasks(env, 40, 100.0, 300.0, seed=1)
random = evaluate(env, nav.random_policy(np.random.default_rng(0)), tasks)
beeline = evaluate(env, nav.beeline_policy, tasks)
rays = sim.ray_fan((0.0, 30.0), 8, 512.0)
classes = sorted({e.classname for e in sim.entities()})
print(json.dumps({
    "pool": len(env.points), "pool_sum": float(env.points.sum()),
    "reachable": float(np.isfinite(env.path_distance).mean()),
    "connected": float((np.isfinite(env.path_distance) | np.isfinite(env.path_distance).T).mean()),
    "obs_shape": list(obs.shape), "obs_dtype": str(obs.dtype),
    "obs_min": float(obs.min()), "obs_max": float(obs.max()),
    "goal_unit": float(np.hypot(obs[52], obs[53])),
    "reward": result.reward, "done": result.done,
    "random": random["success_rate"], "beeline": beeline["success_rate"],
    "rays_shape": list(rays.shape), "classes": classes,
    "teleport_blocked": sim.teleport((1e5, 1e5, 1e5)),
}))
"""


@pytest.mark.gamedata
def test_navigation_environment():
    require_gamedata()
    # Building the walk graph in the simulator dominates this test (~80 s per
    # run); it is slow by design, not hung.
    out = run_in_subprocess(NAV, timeout=300)
    again = run_in_subprocess(NAV, timeout=300)

    assert out["pool"] > 100
    # The walk graph must connect most of the map so tasks can span it.
    # Counted in at least one direction: many of e1m1's routes are real
    # one-way drops (measured: 40/40 sampled reverses fail even with jumps
    # and triple time), so directed pairs cap out far below the two-way
    # connectivity. The reverse floor keeps one-way links honest.
    assert out["connected"] > 0.80
    assert out["reachable"] > 0.70
    assert out["obs_shape"] == [56] and out["obs_dtype"] == "float32"
    assert out["obs_min"] >= -2.0 and out["obs_max"] <= 2.0
    assert abs(out["goal_unit"] - 1.0) < 1e-4
    assert not out["done"]
    assert out["rays_shape"] == [2, 8]
    assert "monster_army" in out["classes"] and "player" in out["classes"]
    assert out["teleport_blocked"] is False
    # Heading for the goal beats moving at random on short tasks.
    assert out["beeline"] > out["random"]
    # The point pool and the evaluation are reproducible across processes.
    assert (out["pool"], out["pool_sum"]) == (again["pool"], again["pool_sum"])
    assert (out["random"], out["beeline"]) == (again["random"], again["beeline"])

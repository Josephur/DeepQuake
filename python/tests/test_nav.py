"""The navigation environment on a real map. Needs game data."""

import numpy as np
import pytest
from test_sim import require_gamedata, run_in_subprocess

from deepquake import nav

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


def _env_without_engine() -> nav.NavEnv:
    """A NavEnv shell for tests of logic that needs no simulator (#30)."""
    env = object.__new__(nav.NavEnv)
    env.map_name = "nowhere"
    env.skill = 0
    env.rng = np.random.default_rng(0)
    env.points = np.array([[0.0, 0.0, 0.0], [839.16, -2833.81, 24.03]], dtype=np.float32)
    env.path_distance = np.array([[0.0, 500.0], [500.0, 0.0]])
    env._episodes_since_load = 0
    return env


def test_set_task_retries_a_start_blocked_by_wander(monkeypatch):
    """A monster on the start spot is seed luck: reload rolls new dice (#30)."""
    env = _env_without_engine()
    calls = {"teleport": 0, "load": 0}

    def teleport(origin, yaw=0.0):
        calls["teleport"] += 1
        return calls["teleport"] > 2  # blocked at first, free from the third roll

    monkeypatch.setattr(nav.sim, "level_count", lambda: 0)
    monkeypatch.setattr(nav.sim, "teleport", teleport)
    monkeypatch.setattr(
        nav.sim, "load_map", lambda *a, **k: calls.__setitem__("load", calls["load"] + 1)
    )
    monkeypatch.setattr(nav.sim, "client_command", lambda *a: None)
    monkeypatch.setattr(nav.sim, "step", lambda **k: None)
    monkeypatch.setattr(nav.sim, "player", lambda: None)
    monkeypatch.setattr(env, "_observe", lambda state: np.zeros(nav.OBS_DIM, np.float32))

    obs = env.set_task(1, 0, 90.0)

    assert obs.shape == (nav.OBS_DIM,) and calls["load"] == 2


def test_set_task_gives_up_on_a_never_free_start(monkeypatch):
    """A spot blocked on every fresh map is a graph bug, and must surface."""
    env = _env_without_engine()
    monkeypatch.setattr(nav.sim, "level_count", lambda: 0)
    monkeypatch.setattr(nav.sim, "teleport", lambda origin, yaw=0.0: False)
    monkeypatch.setattr(nav.sim, "load_map", lambda *a, **k: None)
    monkeypatch.setattr(nav.sim, "client_command", lambda *a: None)
    monkeypatch.setattr(nav.sim, "step", lambda **k: None)

    with pytest.raises(RuntimeError, match="blocked on a fresh map"):
        env.set_task(0, 1, 0.0)

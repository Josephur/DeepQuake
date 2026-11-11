"""The simulator and the navigation environment on a map that is not id's (#28).

The map is generated at run time (deepquake.bspgen) into a temporary basedir
that reuses the user's paks for progs.dat; no map file is ever committed.
"""

import pytest
from test_sim import require_gamedata, run_in_subprocess

CUSTOM_MAP = """
import json, os, tempfile
from pathlib import Path
import numpy as np
from deepquake import bspgen, gamedata, nav, sim
from deepquake.eval_nav import evaluate, make_tasks

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    (base / "id1" / "maps").mkdir(parents=True)
    for pak in gamedata.engine_paks():
        os.symlink(pak, base / "id1" / pak.name)
    (base / "id1" / "maps" / "dqgen1.bsp").write_bytes(bspgen.generate())

    sim.init(base)
    sim.load_map("dqgen1", skill=0, seed=7)
    sim.client_command("god")
    sim.client_command("notarget")
    spawn = sim.player()
    sim.step(forward=100.0, yaw=90.0, repeat=72)  # one second of running, +y
    moved = sim.player()
    rays = sim.ray_fan((0.0,), 8, 2048.0)
    classes = sorted({e.classname for e in sim.entities()})
    first_hash = sim.state_hash()
    blocked = [sim.teleport(point, 0.0) for point in
               ((0.0, 100.0, 96.0), (272.0, 0.0, -50.0), (5000.0, 0.0, 96.0))]
    free = sim.teleport((272.0, 0.0, 24.0), 0.0)
    sim.load_map("dqgen1", skill=0, seed=7)
    sim.client_command("god")
    sim.client_command("notarget")
    sim.step(forward=100.0, yaw=90.0, repeat=72)
    again_hash = sim.state_hash()

    env = nav.NavEnv("dqgen1", seed=3)
    tasks = make_tasks(env, 40, 100.0, 300.0, seed=1)
    random = evaluate(env, nav.random_policy(np.random.default_rng(0)), tasks)
    beeline = evaluate(env, nav.beeline_policy, tasks)

print(json.dumps({
    "spawn": list(spawn.origin), "moved": list(moved.origin),
    "ray_to_wall": float(rays[0, 0]) * 2048.0,
    "classes": classes, "blocked": blocked, "free": free,
    "hash": first_hash, "hash_again": again_hash,
    "points": len(env.points),
    "pairs_in_range": int(np.isfinite(env.path_distance).sum()),
    "random": random["success_rate"], "beeline": beeline["success_rate"],
}))
"""


@pytest.mark.gamedata
def test_generated_map_loads_steps_and_navigates():
    require_gamedata()
    # The walk graph build (a minute of walking) dominates; slow by design.
    out = run_in_subprocess(CUSTOM_MAP, timeout=600)
    again = run_in_subprocess(CUSTOM_MAP, timeout=600)

    # Loads and spawns at the generator's info_player_start, on the floor.
    assert out["spawn"][0] == -280.0 and abs(out["spawn"][2] - 24.5) < 1.0
    # Steps: running one second covers most of the 100 units/s and stays up.
    assert 50 < out["moved"][1] < 110 and abs(out["moved"][2] - 24.0) < 2.0
    # The ray fan sees the wall the generator placed (y = 128).
    assert abs(out["ray_to_wall"] - (128.0 - out["moved"][1])) < 4.0
    # Entities parse: the spawn point is there, the lights did not survive.
    assert "info_player_start" in out["classes"] and "light" not in out["classes"]
    # Hulls: walls, the void and the under-floor are blocked; a room is not.
    assert out["blocked"] == [False, False, False] and out["free"] is True
    # Determinism: same seed and inputs, same state hash — twice.
    assert out["hash"] == out["hash_again"]
    assert out == again
    # The navigation environment builds a usable walk graph on the map.
    assert out["points"] >= 10 and out["pairs_in_range"] > 100
    assert out["beeline"] > out["random"]

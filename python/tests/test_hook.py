"""The brain hook from Python (#31). Needs game data.

The C test (engine/tests/test_hook.c) drives a Grunt on an id map; this
drives one through the Python callback on e1m1 and on the generated map
(bspgen with a monster), which is where generated maps exist.
"""

import pytest
from test_sim import require_gamedata, run_in_subprocess

POLICY = """
import math
from deepquake import sim

state = {"calls": 0, "saw": False, "mode": "run", "start": None, "yaw": 0.0,
         "origin": None}

def policy(obs):
    if state["start"] is None:
        state["start"] = obs.origin
    state["calls"] += 1
    state["yaw"] = obs.angles[1]
    state["origin"] = obs.origin
    for v in obs.visible:
        if v.flags & sim.FL_CLIENT:
            state["saw"] = True
    if state["mode"] == "run":
        return sim.HookAction(desired_yaw=obs.angles[1], forward=200.0)
    if state["mode"] == "attack":
        for v in obs.visible:
            if v.flags & sim.FL_CLIENT:
                return sim.HookAction(
                    desired_yaw=math.degrees(math.atan2(v.relative[1], v.relative[0])),
                    attack=True)
    return None

def drive_grunt(map_name):
    \"\"\"Move the nearest Grunt, then stand in front of it and be shot at.\"\"\"
    import math as _math
    sim.load_map(map_name, skill=0, seed=7)
    player = sim.player()
    grunt = min((e for e in sim.entities() if e.classname == "monster_army"),
                key=lambda e: sum((a - b) ** 2 for a, b in zip(e.origin, player.origin)))
    refused = ""
    try:
        sim.set_policy_hook(player_index(grunt), policy)
    except sim.SimError as exc:
        refused = str(exc)
    state.update(calls=0, saw=False, mode="run", start=None)
    sim.set_policy_hook(grunt.index, policy)
    start = None
    sim.step(repeat=72 * 3)
    moved = _math.dist(state["origin"][:2], state["start"][:2])
    rad = _math.radians(state["yaw"])
    spot = (state["origin"][0] + _math.cos(rad) * 140.0,
            state["origin"][1] + _math.sin(rad) * 140.0,
            state["origin"][2] + 24.0)
    teleported = sim.teleport(spot, state["yaw"] + 180.0)
    state["mode"] = "attack"
    hurt = False
    for _ in range(15):
        sim.step(repeat=72)
        if sim.player().health < 100.0:
            hurt = True
            break
    sim.clear_policy_hook()
    return {"calls": state["calls"], "moved": moved, "saw": state["saw"],
            "teleported": teleported, "hurt": hurt, "refused": refused,
            "hash": sim.state_hash()}

def player_index(grunt):
    return 1  # entity 1 is always the player; the hook must refuse it
"""

ID_MAP = (
    POLICY
    + """
import json
sim.init()
first = drive_grunt("e1m1")
second = drive_grunt("e1m1")
print(json.dumps({"first": first, "second": second}))
"""
)

GENERATED_MAP = (
    POLICY
    + """
import json, os, tempfile
from pathlib import Path
from deepquake import bspgen, gamedata

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    (base / "id1" / "maps").mkdir(parents=True)
    for pak in gamedata.engine_paks():
        os.symlink(pak, base / "id1" / pak.name)
    (base / "id1" / "maps" / "dqgen1.bsp").write_bytes(bspgen.generate(monster=True))
    sim.init(base)
    out = drive_grunt("dqgen1")
print(json.dumps(out))
"""
)


def _check_drive(out: dict) -> None:
    assert out["refused"]  # nominating the player was rejected
    assert out["calls"] > 100
    assert out["moved"] > 40.0
    assert out["teleported"] is True
    assert out["saw"] is True
    assert out["hurt"] is True


@pytest.mark.gamedata
def test_policy_hook_on_id_map():
    require_gamedata()
    out = run_in_subprocess(ID_MAP, timeout=300)
    _check_drive(out["first"])
    # Same seed and inputs (an idle player) give the same state hash.
    assert out["first"]["hash"] == out["second"]["hash"]


@pytest.mark.gamedata
def test_policy_hook_on_generated_map():
    require_gamedata()
    out = run_in_subprocess(GENERATED_MAP, timeout=300)
    _check_drive(out)


@pytest.mark.gamedata
def test_clear_policy_hook_restores_stock_hash():
    """After clearing a nomination the engine is bit-identical to a run
    that never had one - the no-model fallback is exact (#44)."""
    SCENARIO = (
        POLICY
        + """
import json
from deepquake import sim

sim.init()
sim.load_map("e1m1", skill=0, seed=7)
grunt = next(e for e in sim.entities() if e.classname == "monster_army")

# Stock: no nomination at all.
snap = sim.snapshot_save()
for _ in range(12):
    sim.step(repeat=72)
stock_hash = sim.state_hash()

# Nominated with a do-nothing policy, then cleared: same scenario.
sim.snapshot_restore(snap)
sim.set_policy_hook(grunt.index, policy)
for _ in range(12):
    sim.step(repeat=72)
sim.clear_policy_hook()
sim.snapshot_restore(snap)
for _ in range(12):
    sim.step(repeat=72)
cleared_hash = sim.state_hash()
print(json.dumps({"stock": stock_hash, "cleared": cleared_hash}))
"""
    )
    out = run_in_subprocess(SCENARIO, timeout=300)
    assert out["stock"] == out["cleared"]

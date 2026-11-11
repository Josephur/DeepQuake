"""Skill handicaps on the generated map (#34): one Grunt, so the damage
is attributable. Needs game data.
"""

import pytest
from test_sim import require_gamedata, run_in_subprocess

SCENARIO = """
import json, math
from deepquake import sim

def scenario(skill, override=None):
    sim.load_map("dqgen1", skill=skill, seed=7)
    grunt = next(e for e in sim.entities() if e.classname == "monster_army")
    state = {"yaw": 180.0, "origin": grunt.origin}
    def policy(obs):
        state["yaw"] = obs.angles[1]
        state["origin"] = obs.origin
        for v in obs.visible:
            if v.flags & sim.FL_CLIENT:
                return sim.HookAction(
                    desired_yaw=math.degrees(math.atan2(v.relative[1], v.relative[0])),
                    attack=True)
        return None
    sim.set_policy_hook(grunt.index, policy)
    if override is not None:
        sim.set_handicap_override(*override)
    sim.step(repeat=36)
    rad = math.radians(state["yaw"])
    sim.teleport((state["origin"][0] + math.cos(rad) * 140.0,
                  state["origin"][1] + math.sin(rad) * 140.0,
                  state["origin"][2] + 24.0), state["yaw"] + 180.0)
    hurt = None
    for tick in range(72 * 12):
        sim.step(repeat=1)
        if sim.player().health < 100:
            hurt = tick
            break
    sim.clear_policy_hook()
    return {"hurt_tick": hurt, "hash": sim.state_hash()}
"""

GENERATED = (
    SCENARIO
    + """
import os, tempfile
from pathlib import Path
from deepquake import bspgen, gamedata

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    (base / "id1" / "maps").mkdir(parents=True)
    for pak in gamedata.engine_paks():
        os.symlink(pak, base / "id1" / pak.name)
    (base / "id1" / "maps" / "dqgen1.bsp").write_bytes(bspgen.generate(monster=True))
    sim.init(base)
    print(json.dumps({
        "schedule": [sim.handicap_for_skill(s) for s in range(4)],
        "skill0": scenario(0),
        "skill3": scenario(3),
        "skill3_again": scenario(3),
        "huge_reaction": scenario(0, (100000, 0.0)),
        "aim0": scenario(0, (0, 0.0)),
        "aim45": scenario(0, (0, 45.0)),
    }))
"""
)


@pytest.mark.gamedata
def test_handicaps_on_generated_map():
    require_gamedata()
    out = run_in_subprocess(GENERATED, timeout=300)
    assert out["schedule"] == [[36, 12.0], [18, 6.0], [9, 3.0], [0, 0.0]]
    # Same seed and skill give the identical run, handicaps included.
    assert out["skill3"]["hash"] == out["skill3_again"]["hash"]
    # The reaction delay at skill 0 postpones the first damage past skill 3.
    assert out["skill0"]["hurt_tick"] > out["skill3"]["hurt_tick"]
    # A huge reaction delay means no response at all in the window.
    assert out["huge_reaction"]["hurt_tick"] is None
    # A 45-degree aim error postpones the first damage past perfect aim.
    assert out["aim45"]["hurt_tick"] > out["aim0"]["hurt_tick"]

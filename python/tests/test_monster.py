"""The monster observation (#33): fair senses, skill and style inputs."""

import numpy as np
import pytest
from test_sim import require_gamedata, run_in_subprocess

from deepquake import monster, sim

BASE = 2 * monster.N_RAYS


def _hook(time=0.0, origin=(0.0, 0.0, 0.0), visible=(), health=80.0):
    return sim.HookObservation(
        time=time,
        entity_index=12,
        origin=origin,
        velocity=(0.0, 0.0, 0.0),
        angles=(0.0, 0.0, 0.0),
        health=health,
        on_ground=True,
        waterlevel=0,
        rays=np.zeros((2, 24), dtype=np.float32),
        ray_kinds=np.zeros((2, 24), dtype=np.uint8),
        visible=tuple(visible),
        attack_cooldown=0.0,
    )


def _client(relative=(100.0, 0.0, 0.0), health=100.0):
    return sim.HookVisible(
        index=1,
        flags=sim.FL_CLIENT,
        relative=relative,
        velocity=(0.0, 0.0, 0.0),
        health=health,
        distance=float(np.hypot(relative[0], relative[1])),
    )


def test_layout_and_conditioning_inputs():
    observer = monster.MonsterObserver(skill=3, style=(0.0, 0.5, 1.0))
    obs = observer.observe(_hook())
    assert obs.shape == (monster.MONSTER_OBS_DIM,) == (119,)
    assert obs[BASE + 19] == 1.0  # skill 3 / 3
    assert list(obs[BASE + 20 : BASE + 23]) == [0.0, 0.5, 1.0]
    assert obs[BASE + 4] == pytest.approx(0.8)  # own health
    with pytest.raises(ValueError, match="skill"):
        monster.MonsterObserver(skill=4)
    with pytest.raises(ValueError, match="style"):
        monster.MonsterObserver(style=(0.5, 1.5, 0.5))


def test_visible_enemy_and_memory():
    observer = monster.MonsterObserver(skill=1)
    obs = observer.observe(_hook(visible=(_client(),)))
    assert obs[BASE + 11] == 1.0  # currently visible
    assert obs[BASE + 7] == pytest.approx(1.0)  # straight ahead
    assert obs[BASE + 12] == pytest.approx(1.0)  # enemy health

    # The enemy is gone from the senses (behind a wall): current channels
    # read "not visible", but the last seen place is remembered.
    obs = observer.observe(_hook(time=1.0, origin=(10.0, 0.0, 0.0)))
    assert obs[BASE + 11] == 0.0
    assert obs[BASE + 7] == 0.0 and obs[BASE + 12] == 0.0
    assert obs[BASE + 18] == 1.0  # memory valid
    assert obs[BASE + 13] == pytest.approx(1.0)  # still ahead of the monster
    assert obs[BASE + 17] == pytest.approx(0.2)  # 1 s of 5 s old

    # After the memory fades, nothing about the enemy remains.
    obs = observer.observe(_hook(time=10.0, origin=(10.0, 0.0, 0.0)))
    assert obs[BASE + 18] == 0.0 and obs[BASE + 13] == 0.0


def test_deterministic_for_identical_inputs():
    sequence = [_hook(visible=(_client(),)), _hook(time=1.0), _hook(time=2.0)]
    first = [monster.MonsterObserver(skill=2).observe(h) for h in sequence[:1]]
    observer_a, observer_b = monster.MonsterObserver(skill=2), monster.MonsterObserver(skill=2)
    run_a = [observer_a.observe(h) for h in sequence]
    run_b = [observer_b.observe(h) for h in sequence]
    assert first[0].shape == (monster.MONSTER_OBS_DIM,)
    for a, b in zip(run_a, run_b, strict=True):
        assert np.array_equal(a, b)


SCENARIO = """
import json, math
import numpy as np
from deepquake import monster, sim

state = {"visible": [], "observer": monster.MonsterObserver(skill=2, style=(0.25, 0.5, 0.75))}

def policy(hook_obs):
    obs = state["observer"].observe(hook_obs)
    state["visible"].append(float(obs[__BASE__ + 11]))
    state["last"] = obs
    state["origin"] = hook_obs.origin
    state["yaw"] = hook_obs.angles[1]
    return None

def run(map_name, hidden_spot):
    sim.load_map(map_name, skill=0, seed=7)
    player = sim.player()
    grunt = min((e for e in sim.entities() if e.classname == "monster_army"),
                key=lambda e: sum((a - b) ** 2 for a, b in zip(e.origin, player.origin)))
    sim.set_policy_hook(grunt.index, policy)
    sim.step(repeat=36)
    rad = math.radians(state["yaw"])
    front = (state["origin"][0] + math.cos(rad) * 140.0,
             state["origin"][1] + math.sin(rad) * 140.0,
             state["origin"][2] + 24.0)
    seen_teleport = sim.teleport(front, state["yaw"] + 180.0)
    state["visible"] = []
    sim.step(repeat=72)
    saw = max(state["visible"]) if state["visible"] else 0.0
    hidden_teleport = sim.teleport(hidden_spot, 0.0)
    state["visible"] = []
    sim.step(repeat=72)
    hidden = max(state["visible"]) if state["visible"] else 0.0
    sim.clear_policy_hook()
    obs = state["last"]
    return {"saw": saw, "hidden": hidden, "seen_teleport": seen_teleport,
            "hidden_teleport": hidden_teleport, "shape": list(obs.shape),
            "min": float(obs.min()), "max": float(obs.max()),
            "skill": float(obs[__BASE__ + 19]), "hash": sim.state_hash()}
""".replace("__BASE__", str(BASE))

ID_MAP = (
    SCENARIO
    + """
sim.init()
out = run("e1m1", (480.0, -352.0, 89.0))
again = run("e1m1", (480.0, -352.0, 89.0))
print(json.dumps({"out": out, "again": again}))
"""
)

GENERATED_MAP = (
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
    out = run("dqgen1", (-280.0, 100.0, 24.0))
print(json.dumps({"out": out}))
"""
)


@pytest.mark.gamedata
def test_monster_observation_on_id_map():
    require_gamedata()
    result = run_in_subprocess(ID_MAP, timeout=300)
    out = result["out"]
    assert out["shape"] == [monster.MONSTER_OBS_DIM]
    assert out["min"] >= -1.0 and out["max"] <= 2.0
    assert out["saw"] == 1.0  # the player stood in front of the Grunt
    assert out["hidden"] == 0.0  # and read as not visible behind the map
    assert out["skill"] == pytest.approx(2 / 3)
    assert result["again"]["hash"] == out["hash"]  # deterministic scenario


@pytest.mark.gamedata
def test_monster_observation_on_generated_map():
    require_gamedata()
    out = run_in_subprocess(GENERATED_MAP, timeout=300)["out"]
    assert out["shape"] == [monster.MONSTER_OBS_DIM]
    assert out["saw"] == 1.0
    assert out["hidden"] == 0.0

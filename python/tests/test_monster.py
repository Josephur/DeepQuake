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
    assert obs.shape == (monster.MONSTER_OBS_DIM,) == (121,)
    assert obs[BASE + 19] == 1.0  # skill 3 / 3
    assert list(obs[BASE + 20 : BASE + 23]) == [0.0, 0.5, 1.0]
    assert obs[BASE + 4] == pytest.approx(0.8)  # own health
    with pytest.raises(ValueError, match="skill"):
        monster.MonsterObserver(skill=4)
    with pytest.raises(ValueError, match="style"):
        monster.MonsterObserver(style=(0.5, 1.5, 0.5))


def test_monster_type_is_a_one_hot_at_the_end():
    """The type input of the shared model (#48): one-hot, same for every tick."""
    assert monster.MONSTER_TYPES == ("monster_army", "monster_dog")
    army = monster.MonsterObserver(monster_type="monster_army").observe(_hook())
    dog = monster.MonsterObserver(monster_type="monster_dog").observe(_hook())
    type_slice = slice(monster.MONSTER_OBS_BASE_DIM, monster.MONSTER_OBS_DIM)
    assert list(army[type_slice]) == [1.0, 0.0]
    assert list(dog[type_slice]) == [0.0, 1.0]
    # Nothing else depends on the type: the rest of the vector is identical.
    np.testing.assert_array_equal(
        army[: monster.MONSTER_OBS_BASE_DIM], dog[: monster.MONSTER_OBS_BASE_DIM]
    )
    with pytest.raises(ValueError, match="monster type"):
        monster.MonsterObserver(monster_type="monster_ogre")


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


def test_monster_env_validation_and_tasks():
    from deepquake import monster_env

    with pytest.raises(ValueError, match="reward"):
        monster_env.MonsterEnv("e1m1", reward="nope")
    with pytest.raises(ValueError, match="brain"):
        monster_env.MonsterEnv("e1m1", brain="nope")
    with pytest.raises(ValueError, match="skills"):
        monster_env.MonsterEnv("e1m1", skills=(5,))
    with pytest.raises(ValueError, match="opponent kind"):
        monster_env.MonsterEnv("e1m1", opponent=monster_env.PlayerOpponent("nope"))
    with pytest.raises(ValueError, match="checkpoint"):
        monster_env.PlayerOpponent("trained").build(0)
    with pytest.raises(ValueError, match="monsters"):
        monster_env.MonsterEnv("e1m1", monsters=())

    env = monster_env.MonsterEnv.__new__(monster_env.MonsterEnv)
    env.skills = (0, 1, 2, 3)
    first = monster_env.make_tasks(env, 5, seed=7)
    second = monster_env.make_tasks(env, 5, seed=7)
    assert first == second and len(first) == 5
    assert all(task[3] in (0, 1, 2, 3) for task in first)


def test_the_evaluator_pins_the_class_it_measures(monkeypatch):
    """Per-type evaluation: --monster reaches the environment (#48)."""
    from deepquake import eval_monster, monster_env

    seen = {}

    class FakeEnv:
        def __init__(self, map_name, **kwargs):
            seen.update(kwargs)
            seen["map_name"] = map_name

    monkeypatch.setattr(eval_monster, "sim", type("S", (), {"init": staticmethod(lambda: None)}))
    monkeypatch.setattr(eval_monster.nav, "NavEnv", lambda *a, **k: type("N", (), {"points": 0})())
    monkeypatch.setattr(eval_monster.monster_env, "MonsterEnv", FakeEnv)
    monkeypatch.setattr(eval_monster.monster_env, "make_tasks", lambda *a, **k: [])
    monkeypatch.setattr(
        eval_monster.monster_env,
        "evaluate",
        lambda env, policy, tasks: {
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "timeout_rate": 1.0,
            "mean_time_to_first_damage": float("nan"),
            "mean_time_to_kill": float("nan"),
        },
    )
    eval_monster.run_map(
        "e1m1",
        monster_env.PlayerOpponent("shooter"),
        "stock",
        None,
        1,
        0,
        monster="monster_dog",
    )
    assert seen["monster"] == "monster_dog"


DOG_EPISODE = """
import json
import numpy as np
from deepquake import monster, monster_env, sim

sim.init()
env = monster_env.MonsterEnv(
    "e1m1", seed=3, monster="monster_dog",
    opponent=monster_env.PlayerOpponent("shooter"), skills=(2,),
)
obs = env.set_task(*(monster_env.make_tasks(env, 1, 77)[0]))
entity = env._monster_entity()
print(json.dumps({
    "shape": list(obs.shape),
    "type": [float(v) for v in obs[monster.MONSTER_OBS_BASE_DIM:]],
    "classname": entity.classname,
    "index": env.monster_index,
}))
"""


@pytest.mark.gamedata
def test_the_dog_gets_its_own_type_input():
    """A Dog episode is hooked and its observation carries the Dog one-hot (#48)."""
    require_gamedata()
    out = run_in_subprocess(DOG_EPISODE, timeout=300)
    assert out["classname"] == "monster_dog"
    assert out["shape"] == [monster.MONSTER_OBS_DIM]
    assert out["type"] == [0.0, 1.0]


MIXED_EPISODES = """
import json
from deepquake import monster_env, sim

sim.init()
env = monster_env.MonsterEnv(
    "e1m1", seed=4, monsters=("monster_army", "monster_dog"),
    opponent=monster_env.PlayerOpponent("shooter"), skills=(2,),
)
seen = []
for _ in range(6):
    env.reset()
    seen.append(env.monster)
print(json.dumps(seen))
"""


@pytest.mark.gamedata
def test_a_shared_run_samples_both_types():
    """Training samples the body per episode, so one network sees both (#48)."""
    require_gamedata()
    seen = run_in_subprocess(MIXED_EPISODES, timeout=600)
    assert set(seen) == {"monster_army", "monster_dog"}


MONSTER_EPISODE = """
import json
import numpy as np
from deepquake import monster_env, sim

sim.init()
env = monster_env.MonsterEnv(
    "e1m1", seed=3, opponent=monster_env.PlayerOpponent("shooter"), skills=(1, 2, 3)
)
tasks = monster_env.make_tasks(env, 2, 99)
assert tasks[0][3] == tasks[0][3]
obs = env.set_task(*tasks[0])
shape = list(obs.shape)
skill = env.skill
rng = np.random.default_rng(0)
ended = 0
for _ in range(600):
    result = env.step(np.array([rng.integers(n) for n in monster_env.MONSTER_ACTION_NVEC]))
    if result.done:
        ended = env.steps
        break
stock_env = monster_env.MonsterEnv(
    "e1m1", seed=3, opponent=monster_env.PlayerOpponent("shooter"), brain="stock"
)
stock_obs = stock_env.set_task(*tasks[0])
stock_shape = list(stock_obs.shape)
for _ in range(600):
    stock_result = stock_env.step(None)
    if stock_result.done:
        stock_steps = stock_env.steps
        break
print(json.dumps({
    "shape": shape, "skill": skill, "ended": ended,
    "success": bool(result.success), "stock_shape": stock_shape,
    "stock_steps": stock_steps,
}))
"""


@pytest.mark.gamedata
def test_monster_env_episode():
    require_gamedata()
    out = run_in_subprocess(MONSTER_EPISODE, timeout=300)
    assert out["shape"] == [monster.MONSTER_OBS_DIM]
    assert out["skill"] in (1, 2, 3)
    assert out["ended"] > 0  # a random Grunt dies to the shooter well inside the timeout
    assert out["stock_shape"] == [monster.MONSTER_OBS_DIM]
    assert out["stock_steps"] > 0


MONSTER_DETERMINISM = """
import json
import numpy as np
from deepquake import monster_env, sim

sim.init()
results = []
for _run in range(2):
    env = monster_env.MonsterEnv(
        "e1m1", seed=3, opponent=monster_env.PlayerOpponent("shooter"), skills=(2,)
    )
    obs = env.set_task(*(monster_env.make_tasks(env, 1, 55)[0]))
    rng = np.random.default_rng(5)
    first = None
    for _ in range(60):
        action = np.array([rng.integers(n) for n in monster_env.MONSTER_ACTION_NVEC])
        result = env.step(action)
        if result.done:
            break
    results.append({
        "steps": env.steps,
        "player_health": sim.player().health,
        "hash": sim.state_hash(),
    })
print(json.dumps(results))
"""


@pytest.mark.gamedata
def test_monster_env_deterministic():
    """Same seed and same actions: identical episode, hash-verified (#36)."""
    require_gamedata()
    out = run_in_subprocess(MONSTER_DETERMINISM, timeout=300)
    assert out[0] == out[1]

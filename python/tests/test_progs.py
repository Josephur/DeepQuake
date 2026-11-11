"""Game directories and the progs identity (#37). Needs game data.

Each scenario runs in a subprocess because the simulator inits once per
process. Scratch basedirs live in temp directories: symlinks to the
user's paks plus a scratch game directory; no game data is committed.
"""

import pytest
from test_sim import require_gamedata, run_in_subprocess

EXTRACT = """
import struct
from pathlib import Path
from deepquake import gamedata

def stock_progs() -> bytes:
    for pak in reversed(gamedata.engine_paks()):
        data = pak.read_bytes()
        for entry in gamedata.read_pak_directory(pak):
            if entry.name == "progs.dat":
                return data[entry.offset : entry.offset + entry.size]
    loose = gamedata.gamedata_dir() / "id1" / "progs.dat"
    return loose.read_bytes()

def scratch_base(tmp: str, progs: bytes | None) -> Path:
    base = Path(tmp)
    (base / "id1").mkdir(parents=True)
    for pak in gamedata.engine_paks():
        (base / "id1" / pak.name).symlink_to(pak)
    if progs is not None:
        (base / "dqtestmod").mkdir()
        (base / "dqtestmod" / "progs.dat").write_bytes(progs)
    return base
"""

ID1 = """
import json
from deepquake import sim

sim.init()
sim.load_map("e1m1", skill=0, seed=1)
grunt = next(e for e in sim.entities() if e.classname == "monster_army")
nominated = True
try:
    sim.set_policy_hook(grunt.index, lambda obs: None)
except sim.SimError:
    nominated = False
sim.step(repeat=72)
print(json.dumps({"stock": sim.progs_is_stock(), "nominated": nominated}))
"""

COPY = (
    EXTRACT
    + """
import json, tempfile
from deepquake import sim

rejected = []
with tempfile.TemporaryDirectory() as tmp:
    base = scratch_base(tmp, stock_progs())
    for bad in ("../x", "a/b", "a\\\\b", "/abs", "..", ".", "c:x", "-game"):
        try:
            sim.init(base, game=bad)
            rejected.append(bad + ": accepted")
        except sim.SimError:
            pass
    sim.init(base, game="dqtestmod")
    sim.load_map("e1m1", skill=0, seed=1)
    grunt = next(e for e in sim.entities() if e.classname == "monster_army")
    nominated = True
    try:
        sim.set_policy_hook(grunt.index, lambda obs: None)
    except sim.SimError:
        nominated = False
    sim.step(repeat=72)
    print(json.dumps({"stock": sim.progs_is_stock(), "nominated": nominated,
                      "rejected": rejected}))
"""
)

MODIFIED = (
    EXTRACT
    + """
import json, tempfile
from deepquake import sim

with tempfile.TemporaryDirectory() as tmp:
    progs = bytearray(stock_progs())
    ofs_strings = struct.unpack_from("<i", progs, 40)[0]
    at = ofs_strings + 500
    progs[at] = ord("X") if progs[at] != ord("X") else ord("Y")
    base = scratch_base(tmp, bytes(progs))
    sim.init(base, game="dqtestmod")
    sim.load_map("e1m1", skill=0, seed=1)
    grunt = next(e for e in sim.entities() if e.classname == "monster_army")
    nominated = True
    try:
        sim.set_policy_hook(grunt.index, lambda obs: None)
    except sim.SimError:
        nominated = False
    sim.step(repeat=72)  # the level still runs under the stock AI
    print(json.dumps({"stock": sim.progs_is_stock(), "nominated": nominated,
                      "time": sim.time()}))
"""
)


@pytest.mark.gamedata
def test_progs_identity_id1_only():
    require_gamedata()
    out = run_in_subprocess(ID1)
    assert out == {"stock": True, "nominated": True}


@pytest.mark.gamedata
def test_progs_identity_byte_copy():
    require_gamedata()
    out = run_in_subprocess(COPY)
    assert out["stock"] is True and out["nominated"] is True
    assert out["rejected"] == []


@pytest.mark.gamedata
def test_progs_identity_one_byte_change():
    require_gamedata()
    out = run_in_subprocess(MODIFIED)
    assert out["stock"] is False  # one flipped string byte is not stock
    assert out["nominated"] is False  # so the hook gate refuses
    assert out["time"] > 0  # while the level itself still runs

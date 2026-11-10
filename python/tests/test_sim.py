"""Tests for the libdqsim bindings.

The simulator can be started only once per process, so each test that runs
the engine does so in a fresh Python subprocess.
"""

import json
import os
import subprocess
import sys

import pytest

import deepquake
from deepquake import gamedata, sim


def require_library() -> None:
    try:
        sim.find_library()
    except sim.SimLibraryNotFound as exc:
        # CI sets DEEPQUAKE_REQUIRE_SIM so a broken build cannot hide as a skip.
        if os.environ.get("DEEPQUAKE_REQUIRE_SIM"):
            raise
        pytest.skip(str(exc))


def require_gamedata() -> None:
    require_library()
    if not gamedata.inspect().engine_ready:
        pytest.skip("no usable game data; see gamedata/README.md")


def run_in_subprocess(code: str) -> dict:
    """Run code in a new interpreter; it must print one JSON object last."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_version_matches_package():
    require_library()
    assert sim.version() == deepquake.__version__


def test_missing_override_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPQUAKE_SIM_LIB", str(tmp_path / "nope.so"))
    with pytest.raises(sim.SimLibraryNotFound, match="DEEPQUAKE_SIM_LIB"):
        sim.find_library()


def test_errors_are_raised_with_the_engine_message(tmp_path):
    """No game data needed: both calls fail, and say why."""
    require_library()
    out = run_in_subprocess(f"""
import json
from deepquake import sim
errors = []
for call in (sim.step, lambda: sim.init({str(tmp_path)!r})):
    try:
        call()
        errors.append("no error")
    except sim.SimError as exc:
        errors.append(str(exc))
print(json.dumps({{"errors": errors}}))
""")
    assert "dqsim_init has not been called" in out["errors"][0]
    assert out["errors"][1] not in ("no error", "")


EPISODE = """
import json
from deepquake import sim
sim.init()
sim.load_map("e1m1", skill=1, seed={seed})
spawn = sim.player()
hashes = []
for tick in range(500):
    sim.step(forward=400, side=200 if tick % 60 < 30 else -200, yaw=90 + tick % 180,
             buttons=sim.BUTTON_JUMP if tick % 45 == 0 else 0)
    hashes.append(sim.state_hash())
end = sim.player()
try:
    sim.load_map("no_such_map")
    missing = "no error"
except sim.SimError as exc:
    missing = str(exc)
sim.load_map("e1m2", seed=1)
print(json.dumps({{"spawn": spawn.origin, "health": spawn.health, "end": end.origin,
                  "time": sim.time(), "hashes": hashes, "missing": missing}}))
"""


@pytest.mark.gamedata
def test_episode_on_real_map():
    require_gamedata()
    out = run_in_subprocess(EPISODE.format(seed=7))
    assert out["spawn"][:2] == [480.0, -352.0]
    assert out["health"] == 100.0
    assert out["end"] != out["spawn"]
    assert "no_such_map" in out["missing"]
    assert out["time"] > 0


@pytest.mark.gamedata
def test_deterministic_across_processes():
    require_gamedata()
    first = run_in_subprocess(EPISODE.format(seed=7))
    second = run_in_subprocess(EPISODE.format(seed=7))
    other = run_in_subprocess(EPISODE.format(seed=8))
    assert first["hashes"] == second["hashes"]
    assert first["hashes"] != other["hashes"]

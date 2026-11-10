import os

import pytest

import deepquake
from deepquake import sim


def test_version_matches_package():
    try:
        got = sim.version()
    except sim.SimLibraryNotFound as exc:
        # CI sets DEEPQUAKE_REQUIRE_SIM so a broken build cannot hide as a skip.
        if os.environ.get("DEEPQUAKE_REQUIRE_SIM"):
            raise
        pytest.skip(str(exc))
    assert got == deepquake.__version__


def test_missing_override_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPQUAKE_SIM_LIB", str(tmp_path / "nope.so"))
    with pytest.raises(sim.SimLibraryNotFound, match="DEEPQUAKE_SIM_LIB"):
        sim.find_library()

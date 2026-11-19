"""Staging and installing a released model (#45, #48): the manifest is the
contract the download tool verifies, so both sides are tested here."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from deepquake import ppo, release_model

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _download_tool():
    spec = importlib.util.spec_from_file_location("download_model", TOOLS / "download_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset(files: dict[str, bytes], manifest: dict) -> bytes:
    """A release asset as the download tool sees it: a zip of the files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _manifest_for(files: dict[str, bytes], *, corrupt: bool = False) -> dict:
    def digest(name: str) -> str:
        value = hashlib.sha256(files[name]).hexdigest()
        return ("0" + value[1:]) if corrupt else value

    return {
        "format_version": 2,
        "name": "roster",
        "source_run": "roster_pool",
        "total_steps": 3,
        "obs_dim": 121,
        "action_nvec": [3, 3, 7, 2],
        "monsters": ["monster_army", "monster_dog"],
        "onnx_sha256": digest("roster.onnx"),
        "sidecar_sha256": digest("roster.onnx.nvec"),
        "monsters_sha256": digest("roster.onnx.monsters"),
    }


def _write_fight_check(run, classes, *, kill_rate=0.9, facing=0.9, sha=None, passed=None):
    """A fight-check report for ``run``'s checkpoint with the given numbers."""
    from deepquake import fight_check

    side = {
        "usable_tasks": 20,
        "kill_rate": 0.9,
        "damage_rate": 1.0,
        "facing_fraction": 0.95,
        "sight_decisions": 500,
    }
    report = {
        "version": fight_check.REPORT_VERSION,
        "checkpoint_sha256": sha or fight_check.sha256_file(run / "checkpoint.pt"),
        "classes": {
            name: {
                "reference": dict(side),
                "checkpoint": {**side, "kill_rate": kill_rate, "facing_fraction": facing},
            }
            for name in classes
        },
    }
    if passed is not None:
        report["passed"] = passed
    (run / fight_check.REPORT_NAME).write_text(json.dumps(report))


def test_a_monster_model_is_not_released_without_a_passing_fight_check(tmp_path):
    """The release tool refuses every way of not having passed the check (#65)."""
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env

    ppo.seed_everything(65)
    classes = ["monster_army", "monster_dog"]
    model = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM, monster_env.MONSTER_ACTION_NVEC, hidden=8, n_types=2
    )
    run = tmp_path / "pair"
    run.mkdir()
    ppo.save_checkpoint(run / "checkpoint.pt", model, None, {"total_steps": 1, "monsters": classes})
    models = tmp_path / "models"

    def refused(match):
        with pytest.raises(SystemExit, match=match):
            release_model.release_model(run, models, "pair")
        assert not list((models / "pair").glob("*.onnx*")), "a refused model was left staged"

    refused("has not been checked")
    _write_fight_check(run, classes, sha="0" * 64)
    refused("different checkpoint")
    _write_fight_check(run, classes, kill_rate=0.2)
    refused("kills the standing player in 20%")
    _write_fight_check(run, classes, facing=0.3)
    refused("faces the player on 30%")
    _write_fight_check(run, classes[:1])
    refused("monster_dog: not measured")
    # The report's own verdict is not believed: the numbers are judged again.
    _write_fight_check(run, classes, kill_rate=0.2, passed=True)
    refused("kills the standing player")
    _write_fight_check(run, classes)
    assert release_model.release_model(run, models, "pair").is_file()


def test_the_fight_check_cannot_pass_a_class_its_reference_cannot_measure():
    """A weak reference means the probe is blind there: that is a failure, not a pass."""
    from deepquake import fight_check

    good = {"usable_tasks": 20, "kill_rate": 0.9, "facing_fraction": 0.9}
    assert fight_check.judge_class(good, good) == []
    assert "unmeasured" in fight_check.judge_class(good, {**good, "kill_rate": 0.3})[0]
    assert "unmeasured" in fight_check.judge_class(good, {**good, "usable_tasks": 4})[0]
    # 60% of the reference's rate is the bar, whatever the reference's rate is.
    assert fight_check.judge_class({**good, "kill_rate": 0.54}, good) == []
    assert fight_check.judge_class({**good, "kill_rate": 0.5}, good) != []


def test_the_manifest_checksums_the_class_map(tmp_path):
    """A staged shared model names its classes and every file's hash (#48)."""
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env

    ppo.seed_everything(48)
    model = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM,
        monster_env.MONSTER_ACTION_NVEC,
        hidden=8,
        n_types=monster.N_MONSTER_TYPES,
    )
    run = tmp_path / "roster_pool"
    run.mkdir()
    ppo.save_checkpoint(
        run / "checkpoint.pt",
        model,
        None,
        {
            "total_steps": 3,
            "monsters": [
                "monster_dog",
                "monster_army",
                "monster_ogre",
                "monster_enforcer",
                "monster_hell_knight",
                "monster_shalrath",
                "monster_knight",
                "monster_demon1",
                "monster_shambler",
            ],
        },
    )
    with pytest.raises(SystemExit, match="has not been checked"):
        release_model.release_model(run, tmp_path / "models", "roster")
    assert not (tmp_path / "models" / "roster" / "roster.onnx").exists()
    _write_fight_check(run, monster.MONSTER_TYPES)
    manifest_path = release_model.release_model(run, tmp_path / "models", "roster")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_version"] == 2
    assert manifest["monsters"] == [
        "monster_army",
        "monster_dog",
        "monster_enforcer",
        "monster_ogre",
        "monster_hell_knight",
        "monster_shalrath",
        "monster_knight",
        "monster_demon1",
        "monster_shambler",
    ]
    assert manifest["obs_dim"] == monster.MONSTER_OBS_DIM
    assert manifest["action_nvec"] == list(monster_env.MONSTER_ACTION_NVEC)

    out_dir = manifest_path.parent
    class_map = (out_dir / "roster.onnx.monsters").read_bytes()
    expected_map = (
        "monster_army 0\nmonster_dog 1\nmonster_enforcer 2\nmonster_ogre 3\n"
        "monster_hell_knight 4\nmonster_shalrath 5\nmonster_knight 6\n"
        "monster_demon1 7\nmonster_shambler 8\n"
    )
    assert class_map.decode() == expected_map
    assert manifest["monsters_sha256"] == hashlib.sha256(class_map).hexdigest()
    for key, filename in (
        ("onnx_sha256", "roster.onnx"),
        ("sidecar_sha256", "roster.onnx.nvec"),
        ("monsters_sha256", "roster.onnx.monsters"),
    ):
        assert manifest[key] == hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()


def test_a_policy_that_is_not_a_monster_gets_no_class_map(tmp_path):
    """The player model is not routed by classname, so it has no sidecar."""
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    ppo.seed_everything(9)
    model = ppo.ActorCritic(119, (3, 3, 7, 2, 3), hidden=8)
    run = tmp_path / "player_run"
    run.mkdir()
    ppo.save_checkpoint(run / "checkpoint.pt", model, None, {"total_steps": 1})
    manifest_path = release_model.release_model(run, tmp_path / "models", "player")
    manifest = json.loads(manifest_path.read_text())
    assert "monsters" not in manifest and "monsters_sha256" not in manifest
    assert not (manifest_path.parent / "player.onnx.monsters").exists()


def test_the_download_tool_installs_a_shared_model_and_checks_every_file(tmp_path):
    tool = _download_tool()
    files = {
        "roster.onnx": b"the graph",
        "roster.onnx.nvec": b"3 3 7 2\n-30 -15 -5 0 5 15 30\n",
        "roster.onnx.monsters": b"monster_army 0\nmonster_dog 1\n",
    }
    tool.install_archive(_asset(files, _manifest_for(files)), "roster", tmp_path / "models")
    installed = tmp_path / "models" / "roster"
    for name, data in files.items():
        assert (installed / name).read_bytes() == data
    assert json.loads((installed / "manifest.json").read_text())["monsters"] == [
        "monster_army",
        "monster_dog",
    ]

    with pytest.raises(SystemExit, match="checksum mismatch"):
        tool.install_archive(
            _asset(files, _manifest_for(files, corrupt=True)), "roster", tmp_path / "other"
        )


def _player_files_and_manifest() -> tuple[dict[str, bytes], dict]:
    files = {"player.onnx": b"player graph", "player.onnx.nvec": b"3 3 7 2 2 3\n"}
    manifest = {
        "format_version": 2,
        "name": "player",
        "source_run": "combat_grunt_fightstart",
        "total_steps": 1,
        "obs_dim": 108,
        "action_nvec": [3, 3, 7, 2, 2, 3],
        "onnx_sha256": hashlib.sha256(files["player.onnx"]).hexdigest(),
        "sidecar_sha256": hashlib.sha256(files["player.onnx.nvec"]).hexdigest(),
    }
    return files, manifest


def test_the_download_tool_installs_a_format_2_model_without_a_class_map(tmp_path):
    """release_model writes no class map for a player model, so a format 2
    manifest without ``monsters`` must install (models-v2's player, #48)."""
    tool = _download_tool()
    files, manifest = _player_files_and_manifest()
    tool.install_archive(_asset(files, manifest), "player", tmp_path)
    assert (tmp_path / "player" / "player.onnx").read_bytes() == files["player.onnx"]
    assert not (tmp_path / "player" / "player.onnx.monsters").exists()


def test_the_download_tool_refuses_a_class_map_it_cannot_verify(tmp_path):
    """A class map routes monsters to type indices, so it is installed only
    with a checksum, and a manifest that names classes needs the file."""
    tool = _download_tool()
    files, manifest = _player_files_and_manifest()
    stray = dict(files, **{"player.onnx.monsters": b"monster_army 0\n"})
    with pytest.raises(SystemExit, match="no checksum"):
        tool.install_archive(_asset(stray, manifest), "player", tmp_path / "a")

    roster = {
        "roster.onnx": b"the graph",
        "roster.onnx.nvec": b"3 3 7 2\n",
        "roster.onnx.monsters": b"monster_army 0\nmonster_dog 1\n",
    }
    named = _manifest_for(roster)
    del roster["roster.onnx.monsters"]
    with pytest.raises(SystemExit, match="missing roster.onnx.monsters"):
        tool.install_archive(_asset(roster, named), "roster", tmp_path / "b")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_download_tool_still_accepts_a_v1_manifest(tmp_path):
    """models-v1 stays installable: format 1 has no class map (#48)."""
    tool = _download_tool()
    files = {"grunt.onnx": b"old graph", "grunt.onnx.nvec": b"3 3 7 2\n"}
    manifest = {
        "format_version": 1,
        "name": "grunt",
        "source_run": "grunt_pool",
        "total_steps": 1,
        "obs_dim": 119,
        "action_nvec": [3, 3, 7, 2],
        "onnx_sha256": hashlib.sha256(files["grunt.onnx"]).hexdigest(),
        "sidecar_sha256": hashlib.sha256(files["grunt.onnx.nvec"]).hexdigest(),
    }
    tool.install_archive(_asset(files, manifest), "grunt", tmp_path)
    assert (tmp_path / "grunt" / "grunt.onnx").read_bytes() == files["grunt.onnx"]


FIGHT = """
import json
from deepquake import fight_check

fight_check.EPISODES = 6
fight_check.MIN_USABLE = 6
out = {
    brain: fight_check.run_class("monster_army", "e1m1", "", brain)
    for brain in ("reference", "idle")
}
out["idle_reasons"] = fight_check.judge_class(out["idle"], out["reference"])
out["reference_reasons"] = fight_check.judge_class(out["reference"], out["reference"])
print(json.dumps(out))
"""


@pytest.mark.gamedata
def test_the_fight_check_fails_a_grunt_that_does_not_attack():
    """On e1m1 the scripted Grunt kills a standing player in sight; an idle one fails (#65)."""
    from test_sim import require_gamedata, run_in_subprocess

    require_gamedata()
    out = run_in_subprocess(FIGHT, timeout=600)
    assert out["reference"]["usable_tasks"] == 6
    assert out["reference"]["kill_rate"] >= 0.8
    assert out["reference_reasons"] == []
    assert out["idle"]["kill_rate"] == 0.0
    assert any("kills the standing player in 0%" in reason for reason in out["idle_reasons"])

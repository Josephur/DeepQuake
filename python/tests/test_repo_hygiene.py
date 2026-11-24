"""The vendored QuakeC reference (#72). id's original game source is in the
repository to be read, and the hygiene check refuses any change to it."""

import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUAKEC = "engine/third_party/quakec_v101"


def _hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_repo_hygiene", ROOT / "tools" / "check_repo_hygiene.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_quakec_is_vendored_and_has_no_patch_route():
    hygiene = _hygiene()
    assert hygiene.VENDORED_DIRS[QUAKEC] is False
    assert hygiene.vendored_problems(ROOT, hygiene.tracked_files(ROOT), QUAKEC, False) == []


def _copy(tmp_path):
    """The vendored directory under a scratch root, as the check sees it."""
    hygiene = _hygiene()
    root = tmp_path / "repo"
    shutil.copytree(ROOT / QUAKEC, root / QUAKEC)
    tracked = [f"{QUAKEC}/{path.name}" for path in (root / QUAKEC).iterdir()]
    return hygiene, root, tracked


def test_a_changed_file_is_refused_and_named(tmp_path):
    hygiene, root, tracked = _copy(tmp_path)
    source = root / QUAKEC / "wizard.qc"
    pristine = source.read_bytes()
    source.write_bytes(pristine + b"\n// a local edit\n")

    assert hygiene.vendored_problems(root, tracked, QUAKEC, False) == [
        (
            f"{QUAKEC}/wizard.qc",
            "differs from upstream (directory is read-only); "
            f"undo with: git checkout -- {QUAKEC}/wizard.qc",
        )
    ]

    # A patch line in UPSTREAM.md buys nothing in this directory.
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    notes = root / QUAKEC / "UPSTREAM.md"
    notes.write_text(notes.read_text() + f"\n- `wizard.qc` sha256 `{digest}`: a patch\n")
    assert hygiene.PATCH_LINE.findall(notes.read_text())  # the line is a valid patch record
    assert hygiene.vendored_problems(root, tracked, QUAKEC, False)


def test_an_extra_file_and_a_missing_file_are_refused(tmp_path):
    hygiene, root, tracked = _copy(tmp_path)
    (root / QUAKEC / "not_upstream.qc").write_text("void() f = {};\n")
    tracked.append(f"{QUAKEC}/not_upstream.qc")
    (root / QUAKEC / "dog.qc").unlink()

    problems = dict(hygiene.vendored_problems(root, tracked, QUAKEC, False))
    assert problems[f"{QUAKEC}/not_upstream.qc"] == (
        "not an upstream file: vendored code may not gain files"
    )
    assert problems[f"{QUAKEC}/dog.qc"] == "vendored file is missing"

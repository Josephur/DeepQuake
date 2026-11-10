#!/usr/bin/env python3
"""Fail if the git index contains files that must never be published.

Checks every file in the index (so it covers staged files when run as a
pre-commit hook, and the whole tree in CI):

* Quake game data and trained weights, by name and by file signature;
* files larger than the size limit;
* likely secrets files;
* vendored engine code that differs from upstream (see ``vendored_problems``);
* a local fidelity reference checkout that is not pristine.

Standard library only, so it runs before any environment is set up.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1_000_000

FORBIDDEN_SUFFIXES = {
    # Quake data
    ".pak", ".bsp", ".wad", ".mdl", ".spr", ".lmp", ".dem", ".sav",
    # weights and checkpoints
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".npz",
}  # fmt: skip
FORBIDDEN_NAMES = {"progs.dat", ".env"}
# Only these may exist under the local-only directories.
LOCAL_ONLY_DIRS = {"gamedata": {"gamedata/README.md"}, "models": {"models/README.md"}}
LOCAL_ONLY_TOP = {"runs", "checkpoints", "build"}

VENDORED = "engine/third_party/quakespasm"
REFERENCE = "engine/.fidelity/quakespasm"
# A recorded local patch in UPSTREAM.md: - `Quake/file.c` sha256 `<digest>`: reason
PATCH_LINE = re.compile(r"^- `([^`]+)` sha256 `([0-9a-f]{64})`", re.MULTILINE)

# File signatures: PAK archive, WAD2 archive, alias model, sprite.
FORBIDDEN_MAGIC = {b"PACK": "Quake PAK", b"WAD2": "Quake WAD", b"IDPO": "Quake model",
                   b"IDSP": "Quake sprite"}  # fmt: skip


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout
    return [name for name in out.decode("utf-8", errors="surrogateescape").split("\0") if name]


def check(root: Path, name: str) -> str | None:
    """Return a reason the file is not allowed, or None if it is fine."""
    path = Path(name)
    top = path.parts[0]
    if path.name.lower() in FORBIDDEN_NAMES or path.name.startswith(".env."):
        return "forbidden file name"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return f"forbidden file type {path.suffix}"
    if top in LOCAL_ONLY_TOP:
        return f"{top}/ is local-only"
    if top in LOCAL_ONLY_DIRS and name not in LOCAL_ONLY_DIRS[top]:
        return f"{top}/ is local-only"

    full = root / path
    if not full.is_file() or full.is_symlink():
        return None  # deleted in the working tree, or a symlink
    size = full.stat().st_size
    if size > MAX_BYTES:
        return f"{size:,} bytes exceeds the {MAX_BYTES:,} byte limit"
    with full.open("rb") as f:
        magic = f.read(4)
    if magic in FORBIDDEN_MAGIC:
        return f"looks like a {FORBIDDEN_MAGIC[magic]} file"
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vendored_problems(root: Path, tracked: list[str]) -> list[tuple[str, str]]:
    """Vendored engine files must be byte-identical to upstream.

    ``SHA256SUMS`` next to ``UPSTREAM.md`` holds the upstream checksum of
    every vendored file. A file may differ only if it is a recorded local
    patch: ``UPSTREAM.md`` has a line naming the file and the checksum of the
    patched file, as ``- `Quake/<file>` sha256 `<64 hex digits>`: <reason>``,
    and the file contains the marker ``// DeepQuake:``. Trace or debug code
    never qualifies; put it in a scratch copy (see AGENTS.md).
    """
    vendored = root / VENDORED
    manifest_path = vendored / "SHA256SUMS"
    if not manifest_path.is_file():
        return [(f"{VENDORED}/SHA256SUMS", "checksum manifest is missing")]
    manifest = {}
    for line in manifest_path.read_text().splitlines():
        digest, _, name = line.partition("  ")
        manifest[name] = digest
    patches = dict(PATCH_LINE.findall((vendored / "UPSTREAM.md").read_text()))

    problems = []
    prefix = VENDORED + "/"
    present = {name[len(prefix) :] for name in tracked if name.startswith(prefix)}
    present -= {"UPSTREAM.md", "SHA256SUMS"}
    for name in sorted(present - set(manifest)):
        problems.append((prefix + name, "not an upstream file: vendored code may not gain files"))
    for name, digest in sorted(manifest.items()):
        path = vendored / name
        if name not in present or not path.is_file():
            problems.append((prefix + name, "vendored file is missing"))
            continue
        actual = _sha256(path)
        if actual == digest:
            continue
        if patches.get(name) != actual:
            problems.append((prefix + name, "differs from upstream and is not a recorded patch; "
                             f"undo with: git checkout -- {prefix}{name}"))  # fmt: skip
        elif b"// DeepQuake:" not in path.read_bytes():
            problems.append((prefix + name, "recorded patch has no '// DeepQuake:' marker"))
    return problems


def staged_vendored_problems(root: Path) -> list[tuple[str, str]]:
    """The index must hold the same engine files as the working tree.

    ``vendored_problems`` reads the files on disk; this closes the gap where
    a changed file is staged and then restored on disk.
    """
    out = subprocess.run(
        ["git", "diff", "--name-only", "-z", "--", VENDORED],
        cwd=root, check=True, capture_output=True,
    ).stdout  # fmt: skip
    own = {f"{VENDORED}/UPSTREAM.md", f"{VENDORED}/SHA256SUMS"}
    return [(name, "staged content differs from the file on disk; run: git add " + name)
            for name in out.decode().split("\0") if name and name not in own]  # fmt: skip


def reference_problems(root: Path) -> list[tuple[str, str]]:
    """The fidelity reference, when present, must be pristine upstream.

    It is the yardstick for "how stock Quake moves", so it may not carry
    local changes and must be at the commit named in ``UPSTREAM.md``.
    Absent in CI and on machines that have not fetched it: then nothing to check.
    """
    if not (root / REFERENCE / ".git").exists():
        return []
    upstream = (root / VENDORED / "UPSTREAM.md").read_text()
    pinned = re.search(r"\| Commit \| `([0-9a-f]{40})`", upstream)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root / REFERENCE, check=True, capture_output=True, text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.rstrip()  # fmt: skip

    fix = "restore and rebuild it with: tools/fetch_fidelity_reference.sh"
    problems = []
    head = git("rev-parse", "HEAD")
    if pinned is None or head != pinned.group(1):
        problems.append((REFERENCE, f"is at commit {head[:12]}, not the pinned commit; {fix}"))
    changed = git("status", "--porcelain", "--untracked-files=no")
    if changed:
        files = ", ".join(line[3:] for line in changed.splitlines())
        problems.append((REFERENCE, f"has local changes ({files}); undo them with: "
                         f"git -C {REFERENCE} checkout -- . then {fix}"))  # fmt: skip
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tracked = tracked_files(root)
    problems = [(name, reason) for name in tracked if (reason := check(root, name))]
    guard = vendored_problems(root, tracked)
    flagged = {name for name, _ in guard}
    guard += [item for item in staged_vendored_problems(root) if item[0] not in flagged]
    guard += reference_problems(root)
    for name, reason in problems + guard:
        print(f"hygiene: {name}: {reason}", file=sys.stderr)
    if guard:
        print(
            f"\n{len(guard)} problem(s) with vendored engine code. It must match upstream: "
            "see 'Vendored engine code is unmodified' in AGENTS.md.",
            file=sys.stderr,
        )
        if not problems:
            return 1
    if problems:
        print(
            f"\n{len(problems)} file(s) must not be committed. "
            "Unstage them with: git rm --cached <file>",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

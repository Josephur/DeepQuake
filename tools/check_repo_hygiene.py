#!/usr/bin/env python3
"""Fail if the git index contains files that must never be published.

Checks every file in the index (so it covers staged files when run as a
pre-commit hook, and the whole tree in CI):

* Quake game data and trained weights, by name and by file signature;
* files larger than the size limit;
* likely secrets files.

Standard library only, so it runs before any environment is set up.
"""

from __future__ import annotations

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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    problems = [(name, reason) for name in tracked_files(root) if (reason := check(root, name))]
    for name, reason in problems:
        print(f"hygiene: {name}: {reason}", file=sys.stderr)
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

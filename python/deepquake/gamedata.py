"""Locate and inspect the user's local Quake game data.

Run ``python -m deepquake.gamedata`` to check that the data is usable.
Nothing here ever copies game data into the repository.
"""

from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from . import REPO_ROOT

_HEADER = struct.Struct("<4sii")
_ENTRY = struct.Struct("<56sii")

#: Files the simulator cannot run without.
REQUIRED_FILES = ("progs.dat", "maps/start.bsp", "maps/e1m1.bsp")
#: Present only in the full (registered) game.
REGISTERED_MARKER = "maps/e2m1.bsp"


class PakError(ValueError):
    """The file is not a valid Quake PAK archive."""


@dataclass(frozen=True)
class PakEntry:
    name: str
    offset: int
    size: int


def read_pak_directory(path: Path) -> list[PakEntry]:
    """Return the file table of a PAK archive without reading file contents."""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise PakError(f"{path}: too short to be a PAK file")
        magic, dir_offset, dir_length = _HEADER.unpack(header)
        if magic != b"PACK":
            raise PakError(f"{path}: bad magic {magic!r}, expected b'PACK'")
        if dir_offset < 0 or dir_length < 0 or dir_length % _ENTRY.size != 0:
            raise PakError(f"{path}: corrupt directory header")
        if dir_offset + dir_length > file_size:
            raise PakError(f"{path}: directory extends past end of file")
        f.seek(dir_offset)
        raw = f.read(dir_length)

    entries = []
    for raw_name, offset, size in _ENTRY.iter_unpack(raw):
        name = raw_name.split(b"\0", 1)[0].decode("ascii", errors="replace").lower()
        if offset < 0 or size < 0 or offset + size > file_size:
            raise PakError(f"{path}: entry {name!r} extends past end of file")
        entries.append(PakEntry(name, offset, size))
    return entries


def gamedata_dir() -> Path:
    """Directory expected to contain ``id1/`` (``DEEPQUAKE_GAMEDATA`` or ``gamedata/``)."""
    override = os.environ.get("DEEPQUAKE_GAMEDATA")
    return Path(override) if override else REPO_ROOT / "gamedata"


def find_paks(base: Path | None = None) -> list[Path]:
    """PAK files in ``<base>/id1``, in load order. Empty if none are present."""
    id1 = (base or gamedata_dir()) / "id1"
    if not id1.is_dir():
        return []
    return sorted(
        (p for p in id1.iterdir() if p.is_file() and p.suffix.lower() == ".pak"),
        key=lambda p: p.name.lower(),
    )


@dataclass(frozen=True)
class GameDataReport:
    paks: list[Path]
    missing: list[str]
    registered: bool

    @property
    def usable(self) -> bool:
        return bool(self.paks) and not self.missing


def inspect(base: Path | None = None) -> GameDataReport:
    """Check which required files the local PAKs provide."""
    paks = find_paks(base)
    names: set[str] = set()
    for pak in paks:
        names.update(entry.name for entry in read_pak_directory(pak))
    missing = [name for name in REQUIRED_FILES if name not in names]
    return GameDataReport(paks=paks, missing=missing, registered=REGISTERED_MARKER in names)


def main() -> int:
    base = gamedata_dir()
    try:
        report = inspect(base)
    except (PakError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not report.paks:
        print(f"No PAK files found in {base / 'id1'}. See gamedata/README.md.", file=sys.stderr)
        return 1
    for pak in report.paks:
        print(f"found {pak.name}")
    if report.missing:
        print(f"error: missing required files: {', '.join(report.missing)}", file=sys.stderr)
        return 1
    edition = "registered (full game)" if report.registered else "shareware (episode 1 only)"
    print(f"Game data OK: {edition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

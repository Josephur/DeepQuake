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
    """Every PAK file in ``<base>/id1``, whatever its name. Empty if none."""
    id1 = (base or gamedata_dir()) / "id1"
    if not id1.is_dir():
        return []
    return sorted(
        (p for p in id1.iterdir() if p.is_file() and p.suffix.lower() == ".pak"),
        key=lambda p: p.name.lower(),
    )


def engine_paks(base: Path | None = None) -> list[Path]:
    """The PAK files the engine will load, in load order.

    The engine opens ``pak0.pak``, ``pak1.pak``, ... by exactly those
    lower-case names and stops at the first one that is missing.
    """
    id1 = (base or gamedata_dir()) / "id1"
    paks = []
    while (pak := id1 / f"pak{len(paks)}.pak").is_file():
        paks.append(pak)
    return paks


@dataclass(frozen=True)
class GameDataReport:
    paks: list[Path]  # files the engine will load
    ignored: list[Path]  # PAK files present but not named pakN.pak in sequence
    missing: list[str]  # required files not found in ``paks``
    registered: bool

    @property
    def engine_ready(self) -> bool:
        return bool(self.paks) and not self.missing


def inspect(base: Path | None = None) -> GameDataReport:
    """Check that the engine can load the local PAKs and what they provide."""
    paks = engine_paks(base)
    names: set[str] = set()
    for pak in paks:
        names.update(entry.name for entry in read_pak_directory(pak))
    return GameDataReport(
        paks=paks,
        ignored=[p for p in find_paks(base) if p not in paks],
        missing=[name for name in REQUIRED_FILES if name not in names],
        registered=REGISTERED_MARKER in names,
    )


def link(source: Path, base: Path | None = None) -> list[Path]:
    """Symlink the PAK files in ``source`` into ``<base>/id1`` with lower-case names.

    Needed when the files are named ``PAK0.PAK`` (as on the original CD and
    in the rerelease's ``dos/id1``), which the engine does not find on a
    case-sensitive file system. Existing symlinks are replaced; real files
    are never overwritten.
    """
    paks = find_paks(source.parent) if source.name == "id1" else []
    paks = paks or sorted(p for p in source.iterdir() if p.suffix.lower() == ".pak")
    if not paks:
        raise FileNotFoundError(f"no PAK files in {source}")
    id1 = (base or gamedata_dir()) / "id1"
    if id1.is_symlink():
        id1.unlink()
    id1.mkdir(parents=True, exist_ok=True)
    links = []
    for pak in paks:
        dest = id1 / pak.name.lower()
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            raise FileExistsError(f"{dest} is a real file; not replacing it")
        dest.symlink_to(pak.resolve())
        links.append(dest)
    return links


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    base = gamedata_dir()
    try:
        if args[:1] == ["link"] and len(args) == 2:
            for dest in link(Path(args[1]), base):
                print(f"linked {dest.name} -> {dest.resolve()}")
        elif args:
            print("usage: python -m deepquake.gamedata [link <directory-with-pak-files>]")
            return 2
        report = inspect(base)
    except (PakError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for pak in report.paks:
        print(f"found {pak.name}")
    for pak in report.ignored:
        print(f"ignored {pak.name}: the engine only loads pak0.pak, pak1.pak, ... (lower case)")
    if not report.paks:
        print(f"error: no pak0.pak in {base / 'id1'}. See gamedata/README.md.", file=sys.stderr)
        return 1
    if report.missing:
        print(f"error: missing required files: {', '.join(report.missing)}", file=sys.stderr)
        return 1
    edition = "registered (full game)" if report.registered else "shareware (episode 1 only)"
    print(f"Game data OK: {edition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

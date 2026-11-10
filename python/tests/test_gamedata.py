import struct
from pathlib import Path

import pytest

from deepquake import gamedata


def write_pak(path: Path, files: dict[str, bytes]) -> None:
    """Write a minimal synthetic PAK archive (no id Software content)."""
    body = b""
    directory = b""
    offset = 12
    for name, data in files.items():
        directory += struct.pack("<56sii", name.encode("ascii"), offset, len(data))
        body += data
        offset += len(data)
    path.write_bytes(struct.pack("<4sii", b"PACK", offset, len(directory)) + body + directory)


def make_id1(tmp_path: Path, paks: dict[str, dict[str, bytes]]) -> Path:
    id1 = tmp_path / "id1"
    id1.mkdir()
    for pak_name, files in paks.items():
        write_pak(id1 / pak_name, files)
    return tmp_path


def test_read_pak_directory(tmp_path):
    pak = tmp_path / "test.pak"
    write_pak(pak, {"progs.dat": b"abc", "MAPS/E1M1.BSP": b"12345"})
    entries = gamedata.read_pak_directory(pak)
    assert [(e.name, e.size) for e in entries] == [("progs.dat", 3), ("maps/e1m1.bsp", 5)]
    assert pak.read_bytes()[entries[1].offset :][:5] == b"12345"


def test_rejects_bad_magic(tmp_path):
    pak = tmp_path / "bad.pak"
    pak.write_bytes(b"NOPE" + bytes(8))
    with pytest.raises(gamedata.PakError, match="bad magic"):
        gamedata.read_pak_directory(pak)


def test_rejects_truncated_directory(tmp_path):
    pak = tmp_path / "trunc.pak"
    pak.write_bytes(struct.pack("<4sii", b"PACK", 12, 64))
    with pytest.raises(gamedata.PakError, match="past end of file"):
        gamedata.read_pak_directory(pak)


def test_inspect_shareware(tmp_path):
    base = make_id1(tmp_path, {"PAK0.PAK": dict.fromkeys(gamedata.REQUIRED_FILES, b"x")})
    report = gamedata.inspect(base)
    assert report.usable
    assert not report.registered


def test_inspect_registered_across_paks(tmp_path):
    base = make_id1(
        tmp_path,
        {
            "pak0.pak": dict.fromkeys(gamedata.REQUIRED_FILES, b"x"),
            "pak1.pak": {gamedata.REGISTERED_MARKER: b"x"},
        },
    )
    report = gamedata.inspect(base)
    assert report.usable
    assert report.registered
    assert [p.name for p in report.paks] == ["pak0.pak", "pak1.pak"]


def test_inspect_reports_missing(tmp_path):
    base = make_id1(tmp_path, {"pak0.pak": {"progs.dat": b"x"}})
    report = gamedata.inspect(base)
    assert not report.usable
    assert report.missing == ["maps/start.bsp", "maps/e1m1.bsp"]


def test_inspect_no_data(tmp_path):
    report = gamedata.inspect(tmp_path)
    assert report.paks == []
    assert not report.usable

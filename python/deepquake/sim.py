"""Python bindings for libdqsim, the headless Quake simulator.

There is one simulator per process (the engine keeps its state in globals),
so this module exposes functions rather than a class. To run several
simulations at once, use several processes.

    from deepquake import sim

    sim.init()                       # uses gamedata/ by default
    sim.load_map("e1m1", skill=1, seed=0)
    sim.step(forward=400, yaw=90)
    print(sim.player().origin)

The library is found, in order, at:

1. the path in the ``DEEPQUAKE_SIM_LIB`` environment variable;
2. ``build/release/lib/libdqsim.so`` in the source checkout;
3. ``build/dev/lib/libdqsim.so`` in the source checkout.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from . import REPO_ROOT, gamedata

_LIB_NAME = "libdqsim.so"

BUTTON_ATTACK = 1
BUTTON_JUMP = 2

#: Simulated seconds per step unless ``init(tick_seconds=...)`` says otherwise.
DEFAULT_TICK_SECONDS = 1.0 / 72.0


class SimLibraryNotFound(RuntimeError):
    """libdqsim has not been built or could not be located."""


class SimError(RuntimeError):
    """The simulator reported an error. The message comes from the engine."""


class _Config(ctypes.Structure):
    _fields_ = [
        ("basedir", ctypes.c_char_p),
        ("heap_mb", ctypes.c_int),
        ("tick_seconds", ctypes.c_double),
        ("verbose", ctypes.c_int),
    ]


class _Action(ctypes.Structure):
    _fields_ = [
        ("forward", ctypes.c_float),
        ("side", ctypes.c_float),
        ("up", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("buttons", ctypes.c_int32),
        ("impulse", ctypes.c_int32),
    ]


class _PlayerState(ctypes.Structure):
    _fields_ = [
        ("origin", ctypes.c_float * 3),
        ("velocity", ctypes.c_float * 3),
        ("view_angles", ctypes.c_float * 3),
        ("health", ctypes.c_float),
        ("on_ground", ctypes.c_int32),
        ("waterlevel", ctypes.c_int32),
    ]


class _Entity(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_int32),
        ("classname", ctypes.c_char * 32),
        ("origin", ctypes.c_float * 3),
        ("absmin", ctypes.c_float * 3),
        ("absmax", ctypes.c_float * 3),
        ("health", ctypes.c_float),
        ("solid", ctypes.c_int32),
        ("flags", ctypes.c_int32),
    ]


#: Entity.flags bits (QuakeC FL_*).
FL_CLIENT = 8
FL_MONSTER = 32
FL_ITEM = 256


@dataclass(frozen=True)
class Entity:
    index: int  # 0 is the world, 1 is the player
    classname: str  # e.g. "monster_army", "item_health", "info_player_deathmatch"
    origin: tuple[float, float, float]  # zero for brush entities (doors, triggers)
    absmin: tuple[float, float, float]  # world-space bounding box
    absmax: tuple[float, float, float]
    health: float
    solid: int
    flags: int


@dataclass(frozen=True)
class PlayerState:
    origin: tuple[float, float, float]
    velocity: tuple[float, float, float]
    view_angles: tuple[float, float, float]  # pitch, yaw, roll in degrees
    health: float
    on_ground: bool
    waterlevel: int  # 0 dry, 1 ankles, 2 waist (swimming), 3 head under


def find_library() -> Path:
    """Return the path of libdqsim, or raise :class:`SimLibraryNotFound`."""
    override = os.environ.get("DEEPQUAKE_SIM_LIB")
    if override:
        path = Path(override)
        if not path.is_file():
            raise SimLibraryNotFound(f"DEEPQUAKE_SIM_LIB points to a missing file: {path}")
        return path

    candidates = [REPO_ROOT / "build" / preset / "lib" / _LIB_NAME for preset in ("release", "dev")]
    for path in candidates:
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise SimLibraryNotFound(
        f"{_LIB_NAME} not found (looked in {tried}). "
        "Build it with: cmake --preset release && cmake --build --preset release"
    )


@cache
def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(find_library()))
    lib.dqsim_version.restype = ctypes.c_char_p
    lib.dqsim_version.argtypes = []
    lib.dqsim_last_error.restype = ctypes.c_char_p
    lib.dqsim_last_error.argtypes = []
    lib.dqsim_init.argtypes = [ctypes.POINTER(_Config)]
    lib.dqsim_load_map.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
    lib.dqsim_step.argtypes = [ctypes.POINTER(_Action)]
    lib.dqsim_get_player.argtypes = [ctypes.POINTER(_PlayerState)]
    lib.dqsim_step_repeat.argtypes = [ctypes.POINTER(_Action), ctypes.c_int]
    lib.dqsim_client_command.argtypes = [ctypes.c_char_p]
    lib.dqsim_teleport_player.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float]
    lib.dqsim_ray_fan.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    lib.dqsim_get_entities.argtypes = [ctypes.POINTER(_Entity), ctypes.c_int]
    lib.dqsim_map_name.restype = ctypes.c_char_p
    lib.dqsim_map_name.argtypes = []
    lib.dqsim_level_count.restype = ctypes.c_int
    lib.dqsim_level_count.argtypes = []
    lib.dqsim_time.restype = ctypes.c_double
    lib.dqsim_time.argtypes = []
    lib.dqsim_state_hash.restype = ctypes.c_uint64
    lib.dqsim_state_hash.argtypes = []
    lib.dqsim_snapshot_save.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.dqsim_snapshot_restore.argtypes = [ctypes.c_void_p]
    lib.dqsim_snapshot_free.argtypes = [ctypes.c_void_p]
    lib.dqsim_shutdown.restype = None
    lib.dqsim_shutdown.argtypes = []
    return lib


def _check(result: int) -> None:
    if result != 0:
        raise SimError(_load().dqsim_last_error().decode("utf-8", errors="replace"))


def version() -> str:
    """Version string reported by the loaded libdqsim."""
    return _load().dqsim_version().decode("ascii")


def init(
    basedir: Path | str | None = None,
    *,
    heap_mb: int = 0,
    tick_seconds: float = 0.0,
    verbose: bool = False,
) -> None:
    """Start the engine. Allowed once per process.

    ``basedir`` is the directory containing ``id1/``; it defaults to the
    project's game data directory. Zero for ``heap_mb`` or ``tick_seconds``
    selects the library default.
    """
    base = Path(basedir) if basedir is not None else gamedata.gamedata_dir()
    config = _Config(os.fsencode(base), heap_mb, tick_seconds, int(verbose))
    _check(_load().dqsim_init(ctypes.byref(config)))


def load_map(name: str, *, skill: int = 1, seed: int = 0) -> None:
    """Load a map, seed the game's random numbers, and spawn the player."""
    _check(_load().dqsim_load_map(name.encode("ascii"), skill, seed & 0xFFFFFFFF))


def step(
    *,
    forward: float = 0.0,
    side: float = 0.0,
    up: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    buttons: int = 0,
    impulse: int = 0,
    repeat: int = 1,
) -> None:
    """Advance ``repeat`` ticks with the same input.

    Speeds are units per second; angles are absolute degrees.
    """
    action = _Action(forward, side, up, pitch, yaw, buttons, impulse)
    _check(_load().dqsim_step_repeat(ctypes.byref(action), repeat))


def client_command(command: str) -> None:
    """Send a console command as the player; it runs during the next step.

    Includes the cheats ``god``, ``notarget``, ``noclip``, ``fly`` and ``kill``.
    """
    _check(_load().dqsim_client_command(command.encode("ascii")))


def teleport(origin: tuple[float, float, float], yaw: float = 0.0) -> bool:
    """Move the player. Returns False, moving nothing, if the spot is blocked."""
    return _load().dqsim_teleport_player((ctypes.c_float * 3)(*origin), yaw) == 0


#: What a ray hit, mirroring ``enum dqsim_surface`` in dqsim.h. Liquids and
#: sky are reported where the ray enters them, not at their floor.
SURFACE_NOTHING = 0
SURFACE_WORLD = 1
SURFACE_BRUSH = 2
SURFACE_WATER = 3
SURFACE_SLIME = 4
SURFACE_LAVA = 5
SURFACE_SKY = 6


def ray_fan(
    pitches: tuple[float, ...], n_yaw: int, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Distances and surface kinds along a fan of rays from the player's eye.

    Returns ``(distances, kinds)``, each of shape ``(len(pitches), n_yaw)``.
    Distances are float32 in [0, 1] as a fraction of ``max_distance``; kinds
    are uint8 ``SURFACE_*`` values for the same ray. Each row is one pitch
    (degrees, + is down); column 0 points along the view yaw and the rest go
    counter-clockwise.
    """
    out = np.empty((len(pitches), n_yaw), dtype=np.float32)
    kinds = np.empty((len(pitches), n_yaw), dtype=np.uint8)
    _check(
        _load().dqsim_ray_fan(
            (ctypes.c_float * len(pitches))(*pitches),
            len(pitches),
            n_yaw,
            max_distance,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        )
    )
    return out, kinds


def entities() -> list[Entity]:
    """Every entity currently in the level."""
    capacity = 1024
    while True:
        buffer = (_Entity * capacity)()
        count = _load().dqsim_get_entities(buffer, capacity)
        _check(min(count, 0))
        if count <= capacity:
            break
        capacity = count
    return [
        Entity(
            index=e.index,
            classname=e.classname.decode("ascii", errors="replace"),
            origin=tuple(e.origin),
            absmin=tuple(e.absmin),
            absmax=tuple(e.absmax),
            health=e.health,
            solid=e.solid,
            flags=e.flags,
        )
        for e in buffer[:count]
    ]


def player() -> PlayerState:
    """The player's current state."""
    raw = _PlayerState()
    _check(_load().dqsim_get_player(ctypes.byref(raw)))
    return PlayerState(
        origin=tuple(raw.origin),
        velocity=tuple(raw.velocity),
        view_angles=tuple(raw.view_angles),
        health=raw.health,
        on_ground=bool(raw.on_ground),
        waterlevel=int(raw.waterlevel),
    )


def map_name() -> str:
    """Name of the loaded map, or an empty string."""
    return _load().dqsim_map_name().decode("ascii")


def level_count() -> int:
    """How often the game restarted or changed level since :func:`load_map`.

    The game restarts the level when a dead player presses attack or jump,
    and loads the next level at an exit. Stepping continues across both.
    """
    return _load().dqsim_level_count()


def time() -> float:
    """Simulated seconds since the map loaded; negative if no map is loaded."""
    return _load().dqsim_time()


def state_hash() -> int:
    """Hash of the complete game state, for determinism checks."""
    return _load().dqsim_state_hash()


def snapshot_save() -> int:
    """Save the complete simulation state; returns an opaque handle.

    The handle goes to :func:`snapshot_restore` and :func:`snapshot_free`.
    It stays valid only while the same level episode continues: loading a
    map, a level change, or :func:`shutdown` makes it stale, and restoring
    a stale snapshot raises :class:`SimError` without changing anything.
    """
    handle = ctypes.c_void_p()
    _check(_load().dqsim_snapshot_save(ctypes.byref(handle)))
    return int(handle.value)


def snapshot_restore(handle: int) -> None:
    """Put the simulation back into the state :func:`snapshot_save` captured."""
    _check(_load().dqsim_snapshot_restore(ctypes.c_void_p(handle)))


def snapshot_free(handle: int) -> None:
    """Release a snapshot handle. Accepts 0 (no snapshot)."""
    _load().dqsim_snapshot_free(ctypes.c_void_p(handle))


def shutdown() -> None:
    """Stop the engine. It cannot be started again in this process."""
    _load().dqsim_shutdown()

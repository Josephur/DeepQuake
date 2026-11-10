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
    ]


@dataclass(frozen=True)
class PlayerState:
    origin: tuple[float, float, float]
    velocity: tuple[float, float, float]
    view_angles: tuple[float, float, float]  # pitch, yaw, roll in degrees
    health: float
    on_ground: bool


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
    lib.dqsim_map_name.restype = ctypes.c_char_p
    lib.dqsim_map_name.argtypes = []
    lib.dqsim_level_count.restype = ctypes.c_int
    lib.dqsim_level_count.argtypes = []
    lib.dqsim_time.restype = ctypes.c_double
    lib.dqsim_time.argtypes = []
    lib.dqsim_state_hash.restype = ctypes.c_uint64
    lib.dqsim_state_hash.argtypes = []
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
) -> None:
    """Advance one tick. Speeds are units per second; angles are absolute degrees."""
    action = _Action(forward, side, up, pitch, yaw, buttons, impulse)
    _check(_load().dqsim_step(ctypes.byref(action)))


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


def shutdown() -> None:
    """Stop the engine. It cannot be started again in this process."""
    _load().dqsim_shutdown()

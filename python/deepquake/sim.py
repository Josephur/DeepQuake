"""ctypes bindings for libdqsim, the headless Quake simulator.

The library is found, in order, at:

1. the path in the ``DEEPQUAKE_SIM_LIB`` environment variable;
2. ``build/release/lib/libdqsim.so`` in the source checkout;
3. ``build/dev/lib/libdqsim.so`` in the source checkout.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

from . import REPO_ROOT

_LIB_NAME = "libdqsim.so"


class SimLibraryNotFound(RuntimeError):
    """libdqsim has not been built or could not be located."""


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
    return lib


def version() -> str:
    """Version string reported by the loaded libdqsim."""
    return _load().dqsim_version().decode("ascii")

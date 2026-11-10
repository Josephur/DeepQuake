"""Run many environments in parallel, one worker process each.

The simulator allows one instance per process, so parallelism means
processes. Workers are started with the ``spawn`` method: the parent may
hold a CUDA context, which must not be forked.

An environment class used here needs ``reset() -> obs`` and
``step(action) -> result`` where ``result`` has ``obs``, ``reward``, ``done``
and ``success``. When an episode ends the worker resets it straight away and
returns the first observation of the next episode.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import traceback
from collections.abc import Callable

import numpy as np


def _worker(conn, make_env: Callable, index: int) -> None:
    try:
        env = make_env(index)
        conn.send(("ok", env.reset()))
        while True:
            command, payload = conn.recv()
            if command == "step":
                result = env.step(payload)
                obs = env.reset() if result.done else result.obs
                conn.send(("ok", (obs, result.reward, result.done, result.success)))
            elif command == "set":
                name, value = payload
                setattr(env, name, value)
                conn.send(("ok", None))
            elif command == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class WorkerError(RuntimeError):
    """An environment worker raised an exception; the message is its traceback."""


class ProcVecEnv:
    """``n`` environments stepped together. Use as a context manager."""

    def __init__(self, make_env: Callable[[int], object], n: int):
        """``make_env(index)`` is called inside worker ``index`` and must be picklable."""
        context = mp.get_context("spawn")
        self.n = n
        self._conns = []
        self._procs = []
        for index in range(n):
            parent, child = context.Pipe()
            proc = context.Process(target=_worker, args=(child, make_env, index), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        try:
            self.first_obs = np.stack([self._receive(conn) for conn in self._conns])
        except BaseException:
            self.close()
            raise

    def _receive(self, conn):
        try:
            status, payload = conn.recv()
        except EOFError as exc:
            raise WorkerError("an environment worker exited unexpectedly") from exc
        if status == "error":
            raise WorkerError(payload)
        return payload

    def step(self, actions: np.ndarray):
        """Step every environment. Returns ``obs, reward, done, success`` arrays."""
        for conn, action in zip(self._conns, actions, strict=True):
            conn.send(("step", action))
        results = [self._receive(conn) for conn in self._conns]
        obs, reward, done, success = zip(*results, strict=True)
        return (
            np.stack(obs),
            np.array(reward, dtype=np.float32),
            np.array(done, dtype=bool),
            np.array(success, dtype=bool),
        )

    def set_attr(self, name: str, value) -> None:
        """Set an attribute on every environment (used for the curriculum)."""
        for conn in self._conns:
            conn.send(("set", (name, value)))
        for conn in self._conns:
            self._receive(conn)

    def close(self) -> None:
        for conn in self._conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(("close", None))
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

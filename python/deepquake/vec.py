"""Run many environments in parallel, one worker process each.

The simulator allows one instance per process, so parallelism means
processes. Workers are started with the ``spawn`` method: the parent may
hold a CUDA context, which must not be forked.

An environment class used here needs ``reset() -> obs`` and
``step(action) -> result`` where ``result`` has ``obs``, ``reward``, ``done``
and ``success``. When an episode ends the worker resets it straight away and
returns the first observation of the next episode.

Workers can also act for themselves: after :meth:`ProcVecEnv.sync` hands
them a policy, :meth:`ProcVecEnv.rollout` runs many steps inside the worker
(no per-step traffic) and returns the whole buffer — how training collects
its data (#20). The ``step`` interface stays for interactive use.
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
        obs = env.reset()
        actor = None
        conn.send(("ok", obs))
        while True:
            command, payload = conn.recv()
            if command == "step":
                result = env.step(payload)
                obs = env.reset() if result.done else result.obs
                conn.send(("ok", (obs, result.reward, result.done, result.success)))
            elif command == "sync":
                from .ppo import RolloutActor  # torch lives behind the train extra

                spec, state, seed = payload
                if actor is None:
                    actor = RolloutActor(spec, state, seed + index)
                else:
                    actor.sync(state, seed + index)
                conn.send(("ok", None))
            elif command == "rollout":
                result = actor.rollout(env, obs, payload)
                obs = result["next_obs"]
                conn.send(("ok", result))
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

    def sync(self, spec: dict, state: dict, seed: int) -> None:
        """Hand every worker the policy weights and its sampling seed."""
        for conn in self._conns:
            conn.send(("sync", (spec, state, seed)))
        for conn in self._conns:
            self._receive(conn)

    def rollout(self, steps: int) -> dict:
        """Each worker runs ``steps`` decisions locally with its own policy.

        Returns per-step buffers shaped ``(steps, n, ...)`` plus each worker's
        post-rollout observation and memory for bootstrapping, and the
        episode/success counts summed over workers.
        """
        for conn in self._conns:
            conn.send(("rollout", steps))
        results = [self._receive(conn) for conn in self._conns]

        def per_step(key):
            """(worker, step, ...) stacked to the (step, worker, ...) the update uses."""
            stacked = np.stack([result[key] for result in results])
            return stacked.transpose(1, 0, *range(2, stacked.ndim))

        return {
            **{key: per_step(key) for key in ("obs", "hid", "act", "logp", "val", "rew", "done")},
            "episodes": sum(result["episodes"] for result in results),
            "successes": sum(result["successes"] for result in results),
            "next_obs": np.stack([result["next_obs"] for result in results]),
            "next_hidden": np.stack([result["next_hidden"] for result in results]),
        }

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

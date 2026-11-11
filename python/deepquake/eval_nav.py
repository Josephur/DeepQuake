"""Evaluate a navigation policy against the baselines on fixed tasks.

    uv run python -m deepquake.eval_nav runs/nav_maps/checkpoint.pt --maps e1m1,e1m2
    uv run python -m deepquake.eval_nav runs/nav_maps/checkpoint.pt \\
        --maps e1m1,e1m4,e1m7 --held-out e1m4,e1m7

Every policy gets the same list of (start, goal) tasks on a map, generated
from a seed, so results are comparable between runs and checkpoints. Reports
the success rate and the mean time to reach the goal per map, and — when
``--held-out`` names maps — a summary for the training maps and the held-out
maps as their own group. Writes the report to ``eval.json`` next to the
checkpoint.

Maps are evaluated in parallel worker processes (``--jobs``), because
building each map's walk graph takes minutes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

from . import nav, sim


def make_tasks(env: nav.NavEnv, count: int, min_distance: float, max_distance: float, seed: int):
    """Fixed list of ``(start, goal, yaw)`` with walking distance in the given range."""
    rng = np.random.default_rng(seed)
    saved = env.min_goal_distance, env.max_goal_distance
    env.min_goal_distance, env.max_goal_distance = min_distance, max_distance
    try:
        return [env.sample_task(rng) for _ in range(count)]
    finally:
        env.min_goal_distance, env.max_goal_distance = saved


def evaluate(env: nav.NavEnv, policy, tasks) -> dict:
    """Run a policy (observation -> action) on every task."""
    successes = 0
    seconds = []
    for start, goal, yaw in tasks:
        obs = env.set_task(start, goal, yaw)
        if hasattr(policy, "reset"):
            policy.reset()  # a recurrent policy starts every task with no memory
        while True:
            result = env.step(policy(obs))
            obs = result.obs
            if result.done:
                break
        if result.success:
            successes += 1
            seconds.append(env.steps * nav.FRAME_SKIP * sim.DEFAULT_TICK_SECONDS)
    return {
        "success_rate": successes / len(tasks),
        "mean_seconds_to_goal": float(np.mean(seconds)) if seconds else None,
    }


class TrainedPolicy:
    """A checkpoint as an observation -> action policy that remembers.

    The GRU state carries across the steps of one task and is cleared at the
    start of the next (see :func:`evaluate`).
    """

    def __init__(self, checkpoint: Path, seed: int):
        import torch

        from . import ppo

        self.model, _ = ppo.load_model(checkpoint)
        self.model.eval()
        self.generator = torch.Generator().manual_seed(seed)
        self.hidden = self.model.initial_hidden(1)

    def reset(self) -> None:
        self.hidden = self.model.initial_hidden(1)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            self.hidden = self.model.advance(torch.as_tensor(obs).unsqueeze(0), self.hidden)
            dists = self.model.distributions(self.hidden)
            return np.array(
                [torch.multinomial(d.probs[0], 1, generator=self.generator).item() for d in dists]
            )


def evaluate_map(
    map_name: str,
    checkpoint: Path | None,
    episodes: int,
    min_distance: float,
    max_distance: float,
    seed: int,
) -> dict:
    """Build one map's environment and evaluate every policy on it. Runs in a worker."""
    sim.init()
    env = nav.NavEnv(map_name, seed=seed)
    tasks = make_tasks(env, episodes, min_distance, max_distance, seed)
    policies = {
        "random": nav.random_policy(np.random.default_rng(seed)),
        "beeline": nav.beeline_policy,
    }
    if checkpoint is not None:
        policies["trained"] = TrainedPolicy(checkpoint, seed)
    return {
        "map": map_name,
        "points": len(env.points),
        "tasks": episodes,
        "results": {name: evaluate(env, policy, tasks) for name, policy in policies.items()},
    }


def _worker(conn, arguments: dict) -> None:
    try:
        conn.send(("ok", evaluate_map(**arguments)))
    except Exception:  # noqa: BLE001 - the parent decides what to do
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def run_maps(maps: list[str], arguments: dict, jobs: int) -> list[dict]:
    """Evaluate each map, up to ``jobs`` at a time, in separate processes."""
    context = mp.get_context("spawn")
    pending = list(maps)
    running = []
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    while pending or running:
        while pending and len(running) < jobs:
            map_name = pending.pop(0)
            parent, child = context.Pipe()
            proc = context.Process(
                target=_worker, args=(child, {**arguments, "map_name": map_name}), daemon=True
            )
            proc.start()
            child.close()
            running.append((map_name, parent, proc))
        ready = [entry for entry in running if entry[1].poll()]
        if not ready:
            time.sleep(0.2)
            continue
        for entry in ready:
            map_name, conn, proc = entry
            running.remove(entry)
            proc.join(timeout=5)
            status, payload = conn.recv()
            conn.close()
            if status == "error":
                failures[map_name] = payload
            else:
                results[map_name] = payload
                print(f"done: {map_name}", flush=True)
    if failures:
        for map_name, message in failures.items():
            print(f"FAILED: {map_name}\n{message}", flush=True)
        raise SystemExit(1)
    return [results[map_name] for map_name in maps]


def group_summary(reports: list[dict], names: list[str]) -> dict:
    """Mean success per policy over the given maps (a group's own score)."""
    summary = {}
    policies = reports[0]["results"]
    for policy in policies:
        rates = [
            report["results"][policy]["success_rate"]
            for report in reports
            if report["map"] in names
        ]
        times = [
            report["results"][policy]["mean_seconds_to_goal"]
            for report in reports
            if report["map"] in names and report["results"][policy]["mean_seconds_to_goal"]
        ]
        summary[policy] = {
            "maps": [report["map"] for report in reports if report["map"] in names],
            "mean_success_rate": float(np.mean(rates)),
            "mean_seconds_to_goal": float(np.mean(times)) if times else None,
        }
    return summary


def print_table(rows: dict, header: str) -> None:
    print(header)
    print(f"{'policy':<10} {'success':>8} {'mean seconds to goal':>22}")
    for name, result in rows.items():
        seconds = result["mean_seconds_to_goal"]
        shown = f"{seconds:.1f}" if seconds is not None else "-"
        print(f"{name:<10} {result['success_rate']:>8.1%} {shown:>22}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, nargs="?", help="omit to run only the baselines")
    parser.add_argument("--maps", default="e1m1", help="comma-separated map names")
    parser.add_argument("--held-out", default="", help="comma-separated maps kept out of training")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--min-distance", type=float, default=200.0)
    parser.add_argument("--max-distance", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--jobs", type=int, default=0, help="maps at a time (0: one per CPU)")
    args = parser.parse_args()

    maps = [name.strip() for name in args.maps.split(",") if name.strip()]
    held_out = [name.strip() for name in args.held_out.split(",") if name.strip()]
    unknown = [name for name in held_out if name not in maps]
    if unknown:
        parser.error(f"--held-out maps not in --maps: {unknown}")
    jobs = args.jobs or min(len(maps), mp.cpu_count())

    print(
        f"{len(maps)} map(s), {args.episodes} tasks each, goal {args.min_distance:.0f}-"
        f"{args.max_distance:.0f} units away on foot, limit "
        f"{nav.MAX_STEPS * nav.FRAME_SKIP * sim.DEFAULT_TICK_SECONDS:.0f} s, "
        f"{jobs} at a time"
    )
    arguments = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "min_distance": args.min_distance,
        "max_distance": args.max_distance,
        "seed": args.seed,
    }
    reports = run_maps(maps, arguments, jobs)

    for report in reports:
        print()
        print_table(
            report["results"],
            f"map {report['map']} ({report['points']} standing points, {report['tasks']} tasks)",
        )

    training_maps = [name for name in maps if name not in held_out]
    groups = {}
    if training_maps:
        groups["training"] = group_summary(reports, training_maps)
    if held_out:
        groups["held_out"] = group_summary(reports, held_out)
        print()
        print(f"held-out group: {held_out}")
        print_table(
            {
                name: {
                    "success_rate": value["mean_success_rate"],
                    "mean_seconds_to_goal": value["mean_seconds_to_goal"],
                }
                for name, value in groups["held_out"].items()
            },  # fmt: skip
            "held-out maps as one group",
        )
    if len(maps) > 1:
        print()
        print_table(
            {
                name: {
                    "success_rate": value["mean_success_rate"],
                    "mean_seconds_to_goal": value["mean_seconds_to_goal"],
                }
                for name, value in group_summary(reports, maps).items()
            },  # fmt: skip
            "all maps as one group",
        )

    if args.checkpoint:
        report = {
            "checkpoint": str(args.checkpoint),
            "maps": maps,
            "held_out": held_out,
            "episodes": args.episodes,
            "seed": args.seed,
            "min_distance": args.min_distance,
            "max_distance": args.max_distance,
            "per_map": reports,
            "groups": groups,
        }
        out = args.checkpoint.parent / "eval.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

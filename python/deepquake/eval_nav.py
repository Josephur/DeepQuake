"""Evaluate a navigation policy against the baselines on fixed tasks.

    uv run python -m deepquake.eval_nav runs/nav_e1m1/checkpoint.pt

Every policy gets the same list of (start, goal) tasks, generated from a
seed, so results are comparable between runs and checkpoints. Reports the
success rate and the mean time to reach the goal, and writes them to
``eval.json`` next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
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


def trained_policy(checkpoint: Path, seed: int):
    """Wrap a checkpoint as an observation -> action function (sampled actions)."""
    import torch

    from . import ppo

    model, _ = ppo.load_model(checkpoint)
    model.eval()
    generator = torch.Generator().manual_seed(seed)

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dists = model.distributions(torch.as_tensor(obs).unsqueeze(0))
            return np.array(
                [torch.multinomial(d.probs[0], 1, generator=generator).item() for d in dists]
            )

    return act


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, nargs="?", help="omit to run only the baselines")
    parser.add_argument("--map", default="e1m1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--min-distance", type=float, default=200.0)
    parser.add_argument("--max-distance", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    sim.init()
    env = nav.NavEnv(args.map, seed=args.seed)
    tasks = make_tasks(env, args.episodes, args.min_distance, args.max_distance, args.seed)

    policies = {
        "random": nav.random_policy(np.random.default_rng(args.seed)),
        "beeline": nav.beeline_policy,
    }
    if args.checkpoint:
        policies["trained"] = trained_policy(args.checkpoint, args.seed)

    results = {name: evaluate(env, policy, tasks) for name, policy in policies.items()}
    print(f"map {args.map}, {args.episodes} tasks, goal {args.min_distance:.0f}-"
          f"{args.max_distance:.0f} units away on foot, limit "
          f"{nav.MAX_STEPS * nav.FRAME_SKIP * sim.DEFAULT_TICK_SECONDS:.0f} s")  # fmt: skip
    print(f"{'policy':<10} {'success':>8} {'mean seconds to goal':>22}")
    for name, result in results.items():
        seconds = result["mean_seconds_to_goal"]
        shown = f"{seconds:.1f}" if seconds is not None else "-"
        print(f"{name:<10} {result['success_rate']:>8.1%} {shown:>22}")

    if args.checkpoint:
        report = {"map": args.map, "episodes": args.episodes, "seed": args.seed,
                  "min_distance": args.min_distance, "max_distance": args.max_distance,
                  "results": results}  # fmt: skip
        out = args.checkpoint.parent / "eval.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

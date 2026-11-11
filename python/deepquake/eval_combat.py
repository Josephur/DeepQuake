"""Evaluate a combat policy against the baselines on fixed tasks (#32).

    uv run python -m deepquake.eval_combat runs/<run>/checkpoint.pt
    uv run python -m deepquake.eval_combat --monster monster_ogre --map e1m3

Every policy gets the same list of deterministic tasks, so results are
comparable between checkpoints, reward variants and opponent types.
Reports win/loss/timeout rates; writes them next to the checkpoint (or
prints only, without one).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import combat, sim
from .eval_nav import TrainedPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, nargs="?", help="omit to run only the baselines")
    parser.add_argument("--map", default="e1m1")
    parser.add_argument("--monster", default="monster_army")
    parser.add_argument("--reward", default="damage", choices=combat.REWARDS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout-steps", type=int, default=300)
    args = parser.parse_args()

    sim.init()
    env = combat.CombatEnv(
        args.map,
        seed=args.seed,
        reward=args.reward,
        monster=args.monster,
        timeout_steps=args.timeout_steps,
    )
    tasks = combat.make_tasks(env, args.episodes, args.seed)
    policies = {
        "random": combat.random_policy(np.random.default_rng(args.seed)),
        "shooter": combat.shooter_policy,
    }
    if args.checkpoint:
        policies["trained"] = TrainedPolicy(args.checkpoint, args.seed)

    print(f"{args.episodes} tasks on {args.map} vs {args.monster}")
    print(f"{'policy':<10} {'win':>6} {'loss':>6} {'timeout':>8} {'mean decisions':>15}")
    results = {}
    for name, policy in policies.items():
        results[name] = combat.evaluate(env, policy, tasks)
        r = results[name]
        print(
            f"{name:<10} {r['win_rate']:6.1%} {r['loss_rate']:6.1%} "
            f"{r['timeout_rate']:8.1%} {r['mean_decisions']:15.0f}"
        )

    if args.checkpoint:
        out = args.checkpoint.parent / f"eval_combat_{args.map}_{args.monster}.json"
        out.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "map": args.map,
                    "monster": args.monster,
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

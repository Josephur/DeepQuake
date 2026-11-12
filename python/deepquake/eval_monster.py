"""Evaluate the monster-side policy against the stock Grunt (#35, #36).

    uv run python -m deepquake.eval_monster runs/grunt_pool/checkpoint.pt
    uv run python -m deepquake.eval_monster runs/grunt_pool/checkpoint.pt --skill 2

For every skill level and every map, both brains play the same fixed
tasks (``(load_seed, monster_choice, task_seed, skill)`` from the
training placement) against the same player opponent: the **stock**
QuakeC Grunt (no hook; stock AI decides) and the **trained** policy
through the brain hook. The reported win rate is the player's death
rate — how often the monster side kills the player — so the comparison
answers #35's question: does the trained policy beat the stock Grunt?

The player opponent is fixed per invocation (``--opponent``): a trained
player checkpoint (the default; #38's option-E sparring partner), the
scripted shooter, or a random player. Training and held-out maps are
reported as separate groups (rule 3 of map independence). One process
per map builds the walk graph once and runs all its cells; writes
``eval_monster.json`` next to the checkpoint, or prints only without
one.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import monster_env, nav, sim
from .eval_nav import TrainedPolicy

SKILLS = (0, 1, 2, 3)


def run_map(
    map_name: str,
    opponent: monster_env.PlayerOpponent,
    brain: str,
    policy_path: str | None,
    episodes: int,
    seed: int,
) -> dict:
    """All skill cells for one map; the walk graph builds once here."""
    try:
        sim.init()
    except sim.SimError as exc:
        if "may only be called once" not in str(exc):
            raise  # pool workers are reused across maps: already initialised
    points = nav.NavEnv(map_name, seed=12345).points
    env = monster_env.MonsterEnv(
        map_name, seed=12345, opponent=opponent, brain=brain, points=points
    )
    policy = None
    if brain == "policy" and policy_path is not None:
        policy = TrainedPolicy(Path(policy_path), seed)
    stub = monster_env.MonsterEnv.__new__(monster_env.MonsterEnv)
    stub.skills = SKILLS
    cells = {}
    for skill in SKILLS:
        tasks = monster_env.make_tasks(stub, episodes, seed)
        chosen = [(t[0], t[1], t[2], skill) for t in tasks]
        if brain == "stock":
            cells[f"s{skill}"] = monster_env.evaluate(env, None, chosen)
        else:
            cells[f"s{skill}"] = monster_env.evaluate(env, policy, chosen)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "checkpoint", type=Path, nargs="?", help="omit to run only the stock baseline"
    )
    parser.add_argument(
        "--opponent",
        default="trained",
        choices=("trained", "shooter", "random"),
        help="who plays the player side (default: a trained player checkpoint)",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=Path("runs/combat_grunt_fightstart/checkpoint.pt"),
        help="player checkpoint for --opponent trained",
    )
    parser.add_argument(
        "--maps",
        type=str,
        nargs="*",
        default=["e1m1", "e2m1", "e4m1"],
        help="training maps to evaluate on",
    )
    parser.add_argument(
        "--held-out",
        type=str,
        nargs="*",
        default=["e3m1"],
        help="held-out maps, reported as their own group",
    )
    parser.add_argument("--episodes", type=int, default=50, help="tasks per (map, skill) cell")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--jobs", type=int, default=4, help="maps evaluated in parallel")
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="sanity run: stock Grunt vs the scripted shooter (the #32 stock "
        "player stand-in; see the #36 comments for why both sides cannot "
        "literally be stock), asserting the rates stay symmetric",
    )
    args = parser.parse_args()

    if args.sanity:
        args.opponent = "shooter"
        args.checkpoint = None
    if args.opponent == "trained" and not args.opponent_checkpoint.exists():
        raise SystemExit(f"{args.opponent_checkpoint} does not exist")
    opponent = monster_env.PlayerOpponent(
        args.opponent,
        checkpoint=str(args.opponent_checkpoint) if args.opponent == "trained" else None,
    )
    brains = ["stock"] + (["policy"] if args.checkpoint is not None else [])

    work = []
    for group, maps in (("training", args.maps), ("held-out", args.held_out)):
        for map_name in maps:
            for brain in brains:
                work.append((group, map_name, brain))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            (group, map_name, brain): pool.submit(
                run_map,
                map_name,
                opponent,
                brain,
                str(args.checkpoint) if args.checkpoint else None,
                args.episodes,
                args.seed,
            )
            for group, map_name, brain in work
        }
        raw = {key: future.result() for key, future in futures.items()}

    results: dict = {}
    for (group, map_name, brain), cells in raw.items():
        group_map = results.setdefault(group, {}).setdefault(map_name, {})
        for skill_key, rates in cells.items():
            group_map[f"{brain}@{skill_key}"] = rates

    summary: dict = {}
    for group, maps in results.items():
        summary[group] = {}
        for skill in SKILLS:
            for brain in brains:
                rates = [maps[m][f"{brain}@s{skill}"]["win_rate"] for m in maps]
                summary[group][f"{brain}@s{skill}"] = round(float(np.mean(rates)), 4)
                damage = [maps[m][f"{brain}@s{skill}"]["mean_time_to_first_damage"] for m in maps]
                summary[group][f"{brain}@s{skill}"] = {
                    "win_rate": round(float(np.mean(rates)), 4),
                    "mean_time_to_first_damage": round(float(np.nanmean(damage)), 1),
                }

    print(
        f"opponent: {args.opponent}"
        + (f" ({args.opponent_checkpoint})" if args.opponent == "trained" else "")
        + f"  tasks per cell: {args.episodes}"
    )
    print(
        f"{'group':<10} {'skill':<7} {'stock win':>10} {'trained win':>12}"
        f" {'stock t/dmg':>12} {'trained t/dmg':>14} {'stock t/kill':>13} {'trained t/kill':>15}"
    )
    for group, maps in results.items():
        for skill in SKILLS:
            cells = {
                brain: {
                    field: float(np.nanmean([maps[m][f"{brain}@s{skill}"][field] for m in maps]))
                    for field in ("win_rate", "mean_time_to_first_damage", "mean_time_to_kill")
                }
                for brain in brains
            }
            row = [f"{group:<10}", f"s{skill:<6}"]
            row.append(f"{cells['stock']['win_rate']:>10.1%}")
            row.append(
                f"{cells['policy']['win_rate']:>12.1%}" if "policy" in cells else f"{'--':>12}"
            )
            for field in ("mean_time_to_first_damage", "mean_time_to_kill"):
                row.append(f"{cells['stock'][field]:>12.0f}")
                row.append(
                    f"{cells['policy'][field]:>14.0f}" if "policy" in cells else f"{'--':>14}"
                )
            print(" ".join(row))

    out = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "opponent": {"kind": args.opponent, "checkpoint": str(args.opponent_checkpoint)},
        "episodes": args.episodes,
        "seed": args.seed,
        "sanity": args.sanity,
        "results": results,
        "summary": summary,
    }
    if args.sanity:  # the three outcomes are counted from one event stream: they must fill 100%
        for group, maps in results.items():
            for map_name, cells in maps.items():
                for key, rates in cells.items():
                    total = rates["win_rate"] + rates["loss_rate"] + rates["timeout_rate"]
                    assert abs(total - 1.0) < 1e-9, (group, map_name, key, total)
    if args.checkpoint:
        out_path = args.checkpoint.parent / "eval_monster.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

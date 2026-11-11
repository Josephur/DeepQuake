"""Train the navigation policy with PPO.

    uv run python -m deepquake.train_nav configs/nav_e1m1.toml
    uv run python -m deepquake.train_nav configs/nav_e1m1.toml --resume

Writes to ``runs/<name>/``: ``progress.csv`` (one row per update),
``checkpoint.pt`` (latest weights, optimiser and progress, for ``--resume``
and for :mod:`deepquake.eval_nav`), and a copy of the config.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
import tomllib
from collections import deque
from functools import partial
from pathlib import Path

from . import REPO_ROOT, nav, ppo
from .vec import ProcVecEnv


def make_env(maps: list[str], seed: int, index: int) -> nav.NavEnv:
    """Runs inside a worker process: start its simulator and build the environment.

    Workers take maps in turn, so a training run draws from every map in the
    config and each map gets an equal share of the environments.
    """
    from . import sim

    sim.init()
    return nav.NavEnv(maps[index % len(maps)], seed=seed * 1000 + index)


def validate_maps(config: dict) -> tuple[list[str], list[str]]:
    """Training and held-out map lists, checked before any simulator starts."""
    if "maps" in config:
        maps = list(config["maps"])
    elif "map" in config:
        maps = [config["map"]]  # the single-map configs that predate multi-map runs
    else:
        raise ValueError("the config needs 'maps' (the training maps)")
    held_out = list(config.get("held_out", []))
    if not maps:
        raise ValueError("the config needs at least one training map")
    if len(set(maps)) != len(maps):
        raise ValueError(f"duplicate training maps: {maps}")
    overlap = set(maps) & set(held_out)
    if overlap:
        raise ValueError(f"maps both trained on and held out: {sorted(overlap)}")
    envs = config.get("envs")
    if envs is not None and len(maps) > envs:
        raise ValueError(
            f"{len(maps)} training maps but only {envs} environments: workers take "
            "maps in turn, so every map needs at least one worker or it is "
            "silently never trained on"
        )
    return maps, held_out


def train(config: dict, *, resume: bool = False, max_updates: int | None = None) -> Path:
    """Run training. Returns the run directory."""
    maps, held_out = validate_maps(config)
    run_dir = REPO_ROOT / "runs" / config["name"]
    checkpoint_path = run_dir / "checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise SystemExit(f"{checkpoint_path} exists; pass --resume or choose another name")
    run_dir.mkdir(parents=True, exist_ok=True)

    ppo.seed_everything(config["seed"])
    device = ppo.pick_device(config.get("device") or None)
    ppo_config = ppo.config_from_dict(config.get("ppo", {}))
    curriculum = config["curriculum"]
    goal_distance = curriculum["start"]

    model = ppo.ActorCritic(nav.OBS_DIM, nav.ACTION_NVEC, ppo_config.hidden).to(device)
    start_steps = 0
    saved = None
    if resume:
        model, saved = ppo.load_model(checkpoint_path, device)
        start_steps = saved["total_steps"]
        goal_distance = saved["goal_distance"]

    print(f"device: {device}  envs: {config['envs']}  run: {run_dir}")
    print(f"maps: {', '.join(maps)}  held out: {', '.join(held_out) or 'none'}", flush=True)
    with ProcVecEnv(partial(make_env, maps, config["seed"]), config["envs"]) as env:
        env.set_attr("max_goal_distance", goal_distance)
        learner = ppo.PPO(env, model, ppo_config, device)
        learner.total_steps = start_steps
        if saved:
            learner.optimizer.load_state_dict(saved["optimizer"])

        log_path = run_dir / "progress.csv"
        new_log = not log_path.exists()
        with log_path.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            if new_log:
                writer.writerow(
                    ["total_steps", "seconds", "goal_distance", "success_rate", "episodes",
                     "mean_reward", "entropy", "value_loss"]
                )  # fmt: skip
            started = time.time()
            update = 0
            # Success is judged over the last few updates: one update alone sees
            # too few finished episodes, and mostly the short, successful ones.
            recent: deque[tuple[int, int]] = deque(maxlen=4)
            while learner.total_steps < config["total_steps"]:
                stats = learner.update()
                update += 1
                recent.append((stats["successes"], stats["episodes"]))
                success = sum(s for s, _ in recent) / max(sum(e for _, e in recent), 1)
                writer.writerow(
                    [stats["total_steps"], round(time.time() - started, 1), goal_distance,
                     round(success, 4), stats["episodes"], round(stats["mean_reward"], 5),
                     round(stats["entropy"], 4), round(stats["value_loss"], 5)]
                )  # fmt: skip
                log_file.flush()
                rate = (stats["total_steps"] - start_steps) / (time.time() - started)
                print(
                    f"steps {stats['total_steps']:>9,}  goal<= {goal_distance:6.0f}  "
                    f"success {success:5.2f}  entropy {stats['entropy']:5.2f}  {rate:,.0f} steps/s",
                    flush=True,
                )
                # Curriculum: allow more distant goals once near ones are reliably reached.
                if (
                    len(recent) == recent.maxlen
                    and success >= curriculum["threshold"]
                    and goal_distance < curriculum["max"]
                ):
                    goal_distance = min(goal_distance + curriculum["step"], curriculum["max"])
                    env.set_attr("max_goal_distance", goal_distance)
                    recent.clear()

                last = learner.total_steps >= config["total_steps"] or update == max_updates
                if last or update % config.get("checkpoint_every", 20) == 0:
                    ppo.save_checkpoint(
                        checkpoint_path,
                        model,
                        learner.optimizer,
                        {"total_steps": learner.total_steps, "goal_distance": goal_distance,
                         "maps": maps, "held_out": held_out},
                    )  # fmt: skip
                if last:
                    break
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="TOML config file, e.g. configs/nav_e1m1.toml")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<name>")
    parser.add_argument("--updates", type=int, help="stop after this many updates (for testing)")
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    run_dir = train(config, resume=args.resume, max_updates=args.updates)
    shutil.copy(args.config, run_dir / "config.toml")
    print(f"done: {run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()

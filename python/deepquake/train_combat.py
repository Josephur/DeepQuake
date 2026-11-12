"""Train the combat policy with PPO (#32 experiments).

    uv run python -m deepquake.train_combat configs/combat_<name>.toml

Writes to ``runs/<name>/``: ``progress.csv``, ``checkpoint.pt`` (loads with
:func:`deepquake.ppo.load_model`), and a copy of the config. A run trains
against one reward variant and either a single ``map``/``monster`` pair or
an ``opponents`` list of them, which the workers take in turn.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
import tomllib
from functools import partial
from pathlib import Path

from . import REPO_ROOT, combat, ppo
from .vec import ProcVecEnv


def opponent_list(config: dict) -> list[tuple[str, str]]:
    """The (map, monster) opponents of a run, in worker order."""
    if "opponents" in config:
        return [(entry["map"], entry["monster"]) for entry in config["opponents"]]
    return [(config.get("map", "e1m1"), config.get("monster", "monster_army"))]


def make_env(config: dict, seed: int, index: int, decisions_so_far: int = 0) -> combat.CombatEnv:
    """Runs inside a worker process: start its simulator and build the environment."""
    from . import sim

    sim.init()
    map_name, monster = opponent_list(config)[index % len(opponent_list(config))]
    envs = int(config["envs"])
    common: dict = {
        "reward": config.get("reward", "damage"),
        "monster": monster,
        "timeout_steps": int(config.get("timeout_steps", 300)),
    }
    if "stages" in config:
        # A curriculum owns the placement; it starts where a resumed run
        # left off, so extending a run does not replay early stages.
        common["stages"] = combat.scale_stages(
            combat.stages_from_config(config["stages"]), 1.0 / envs
        )
        common["curriculum_offset"] = round(decisions_so_far / envs)
    else:
        common.update(
            min_distance=float(config.get("min_distance", 200.0)),
            max_distance=float(config.get("max_distance", 600.0)),
            los_broken_required=bool(config.get("los_broken_required", True)),
        )
    return combat.CombatEnv(map_name, seed=seed * 1000 + index, **common)


def train(config: dict, *, resume: bool = False, max_updates: int | None = None) -> Path:
    """Run training. Returns the run directory."""
    run_dir = REPO_ROOT / "runs" / config["name"]
    checkpoint_path = run_dir / "checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise SystemExit(f"{checkpoint_path} exists; pass --resume or choose another name")
    run_dir.mkdir(parents=True, exist_ok=True)

    ppo.seed_everything(config["seed"])
    device = ppo.pick_device(config.get("device") or None)
    ppo_config = ppo.config_from_dict(config.get("ppo", {}))
    opponents = opponent_list(config)

    model = ppo.ActorCritic(combat.OBS_DIM, combat.ACTION_NVEC, ppo_config.hidden).to(device)
    start_steps = 0
    saved = None
    if resume:
        model, saved = ppo.load_model(checkpoint_path, device)
        start_steps = saved["total_steps"]

    print(
        f"device: {device}  envs: {config['envs']}  run: {run_dir}\n"
        f"opponents: {opponents}  reward: {config.get('reward', 'damage')}",
        flush=True,
    )
    with ProcVecEnv(partial(make_env, config, config["seed"], start_steps), config["envs"]) as env:
        learner = ppo.PPO(env, model, ppo_config, device, seed=config["seed"] * 1000)
        learner.total_steps = start_steps
        if saved:
            learner.optimizer.load_state_dict(saved["optimizer"])
        log_path = run_dir / "progress.csv"
        new_log = not log_path.exists()
        with log_path.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            if new_log:
                writer.writerow(
                    ["total_steps", "seconds", "success_rate", "episodes", "mean_reward", "entropy"]
                )
            started = time.time()
            update = 0
            while learner.total_steps < config["total_steps"]:
                stats = learner.update()
                update += 1
                writer.writerow(
                    [
                        stats["total_steps"],
                        round(time.time() - started, 1),
                        round(stats["successes"] / max(stats["episodes"], 1), 4),
                        stats["episodes"],
                        round(stats["mean_reward"], 5),
                        round(stats["entropy"], 4),
                    ]
                )
                log_file.flush()
                rate = (stats["total_steps"] - start_steps) / (time.time() - started)
                print(
                    f"steps {stats['total_steps']:>9,}  win "
                    f"{stats['successes'] / max(stats['episodes'], 1):5.2f}  "
                    f"entropy {stats['entropy']:5.2f}  {rate:,.0f} steps/s",
                    flush=True,
                )
                last = learner.total_steps >= config["total_steps"] or update == max_updates
                if last or update % config.get("checkpoint_every", 25) == 0:
                    ppo.save_checkpoint(
                        checkpoint_path,
                        model,
                        learner.optimizer,
                        {
                            "total_steps": learner.total_steps,
                            "opponents": opponents,
                            "reward": config.get("reward", "damage"),
                        },
                    )
                if last:
                    break
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--updates", type=int, help="stop after this many updates (for testing)")
    parser.add_argument("--resume", action="store_true", help="continue from the checkpoint")
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    run_dir = REPO_ROOT / "runs" / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.toml")

    train(config, resume=args.resume, max_updates=args.updates)
    print(f"done: {run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()

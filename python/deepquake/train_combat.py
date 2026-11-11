"""Train the combat policy with PPO (#32 experiments).

    uv run python -m deepquake.train_combat configs/combat_<name>.toml

Writes to ``runs/<name>/``: ``progress.csv``, ``checkpoint.pt`` (loads with
:func:`deepquake.ppo.load_model`), and a copy of the config. One map, one
monster type and one reward variant per run — the #32 comparisons hold
everything else fixed.
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


def make_env(config: dict, seed: int, index: int) -> combat.CombatEnv:
    """Runs inside a worker process: start its simulator and build the environment."""
    from . import sim

    sim.init()
    return combat.CombatEnv(
        config.get("map", "e1m1"),
        seed=seed * 1000 + index,
        reward=config.get("reward", "damage"),
        monster=config.get("monster", "monster_army"),
        timeout_steps=int(config.get("timeout_steps", 300)),
    )


def train(config: dict, *, max_updates: int | None = None) -> Path:
    """Run training. Returns the run directory."""
    run_dir = REPO_ROOT / "runs" / config["name"]
    checkpoint_path = run_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        raise SystemExit(f"{checkpoint_path} exists; choose another name")
    run_dir.mkdir(parents=True, exist_ok=True)

    ppo.seed_everything(config["seed"])
    device = ppo.pick_device(config.get("device") or None)
    ppo_config = ppo.config_from_dict(config.get("ppo", {}))
    model = ppo.ActorCritic(combat.OBS_DIM, combat.ACTION_NVEC, ppo_config.hidden).to(device)

    print(
        f"device: {device}  envs: {config['envs']}  run: {run_dir}\n"
        f"map: {config.get('map', 'e1m1')}  monster: {config.get('monster', 'monster_army')}  "
        f"reward: {config.get('reward', 'damage')}",
        flush=True,
    )
    with ProcVecEnv(partial(make_env, config, config["seed"]), config["envs"]) as env:
        learner = ppo.PPO(env, model, ppo_config, device, seed=config["seed"] * 1000)
        log_path = run_dir / "progress.csv"
        with log_path.open("w", newline="") as log_file:
            writer = csv.writer(log_file)
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
                rate = stats["total_steps"] / (time.time() - started)
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
                            "map": config.get("map", "e1m1"),
                            "monster": config.get("monster", "monster_army"),
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
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    run_dir = REPO_ROOT / "runs" / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.toml")

    train(config, max_updates=args.updates)
    print(f"done: {run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()

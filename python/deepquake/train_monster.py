"""Train the monster-side policy with PPO through the brain hook (#35).

    uv run python -m deepquake.train_monster configs/grunt_pool.toml

The roles are inverted from :mod:`train_combat`: workers drive a hooked
Grunt (via :class:`deepquake.monster_env.MonsterEnv`) against player
opponents, rotating ``(map, opponent)`` per worker like the combat
trainer rotates monsters. Skills and styles are sampled per episode
inside the environment, independent of the opponent, so the policy's
skill and style inputs stay unconfounded. Writes to ``runs/<name>/``:
``progress.csv``, ``checkpoint.pt`` (loads with
:func:`deepquake.ppo.load_model`), and a copy of the config.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
import tomllib
from functools import partial
from pathlib import Path

from . import REPO_ROOT, monster, monster_env, ppo
from .vec import ProcVecEnv


def opponent_list(config: dict) -> list[monster_env.PlayerOpponent]:
    """The player opponents of a run, in worker order."""
    entries = config.get("opponents")
    if not entries:
        return [monster_env.PlayerOpponent("shooter")]
    opponents = []
    for entry in entries:
        if isinstance(entry, str):
            opponents.append(monster_env.PlayerOpponent(entry))
        else:
            opponents.append(
                monster_env.PlayerOpponent(entry["kind"], checkpoint=entry.get("checkpoint"))
            )
    return opponents


def map_list(config: dict) -> list[str]:
    """The training maps, round-robin over workers (the #19 machinery)."""
    maps = config.get("maps") or [config.get("map", "e1m1")]
    if "envs" in config and int(config["envs"]) < len(maps):
        raise ValueError(f"envs ({config['envs']}) must cover every map ({len(maps)})")
    return list(maps)


def monster_list(config: dict) -> list[str]:
    """The monster classes a run trains, in type order (#48).

    ``monsters`` lists every body a shared model learns; ``monster`` (the
    single-type runs of #35) pins one. A class outside
    :data:`monster.MONSTER_TYPES` is a config mistake, not a runtime one, so
    it fails here.
    """
    classes = config.get("monsters")
    if classes is None:
        classes = [config.get("monster", monster.MONSTER_TYPES[0])]
    classes = list(classes)
    unknown = [name for name in classes if name not in monster.MONSTER_TYPES]
    if unknown:
        raise ValueError(
            f"unknown monster classes {unknown}; known types are {monster.MONSTER_TYPES}"
        )
    if not classes:
        raise ValueError("monsters must name at least one monster class")
    return classes


def build_model(config: dict, ppo_config: ppo.PPOConfig, device) -> ppo.ActorCritic:
    """The network of a run: shared type inputs, warm started if asked.

    A run that lists more than one class, or any run at all once the shared
    layout exists, builds :class:`ppo.ActorCritic` with one head per type.
    ``init_from`` names a checkpoint to warm start from — the Grunt's
    ``runs/grunt_pool/checkpoint.pt`` in the roster run, the two-type roster
    model in the three-type run (#50) — so the shared network starts as the
    old brain and the fine-tune moves it. The source may be narrower (its
    extra observation columns are zeroed and its heads carry over), but not
    wider.
    """
    model = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM,
        monster_env.MONSTER_ACTION_NVEC,
        ppo_config.hidden,
        n_types=monster.N_MONSTER_TYPES,
    ).to(device)
    init_from = config.get("init_from")
    if init_from:
        source_path = Path(init_from)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        if not source_path.exists():
            raise SystemExit(
                f"init_from {source_path} does not exist; warm starts need the "
                f"source run's checkpoint.pt (omit init_from to train from scratch)"
            )
        source, _saved = ppo.load_model(source_path, device)
        if source.obs_dim > model.obs_dim:
            raise SystemExit(
                f"init_from {source_path} is wider (obs_dim {source.obs_dim}) than "
                f"this run's observation ({model.obs_dim}); warm start from a "
                f"narrower or equal checkpoint"
            )
        model.init_from(source)
    return model


def make_env(config: dict, seed: int, index: int) -> monster_env.MonsterEnv:
    """Runs inside a worker process: start its simulator and build the environment."""
    from . import sim

    sim.init()
    maps = map_list(config)
    opponents = opponent_list(config)
    styles = [tuple(entry["style"]) for entry in config.get("styles", [])] or None
    return monster_env.MonsterEnv(
        maps[index % len(maps)],
        seed=seed * 1000 + index,
        monsters=monster_list(config),
        opponent=opponents[index % len(opponents)],
        timeout_steps=int(config.get("timeout_steps", 600)),
        min_distance=float(config.get("min_distance", 200.0)),
        max_distance=float(config.get("max_distance", 600.0)),
        reward=config.get("reward", "sparse"),
        skills=tuple(int(s) for s in config.get("skills", [2])),
        styles=styles if styles is not None else monster_env.DEFAULT_STYLE,
        brain="policy",
    )


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
    maps = map_list(config)
    opponents = opponent_list(config)
    if config.get("held_out"):
        overlap = set(maps) & set(config["held_out"])
        if overlap:
            raise ValueError(f"training maps and held-out maps overlap: {sorted(overlap)}")

    model = build_model(config, ppo_config, device)
    start_steps = 0
    saved = None
    if resume:
        model, saved = ppo.load_model(checkpoint_path, device)
        start_steps = saved["total_steps"]

    print(
        f"device: {device}  envs: {config['envs']}  run: {run_dir}\n"
        f"maps: {maps}  monsters: {monster_list(config)}  "
        f"opponents: {opponents}  skills: {config.get('skills', [2])}",
        flush=True,
    )
    with ProcVecEnv(partial(make_env, config, config["seed"]), config["envs"]) as env:
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
                            "maps": maps,
                            "monsters": monster_list(config),
                            "held_out": list(config.get("held_out", [])),
                            "opponents": [(o.kind, o.checkpoint) for o in opponents],
                            "skills": list(config.get("skills", [2])),
                            "reward": config.get("reward", "sparse"),
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

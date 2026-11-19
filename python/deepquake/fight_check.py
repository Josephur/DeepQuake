"""Does a trained monster model fight a player it can see? (#65)

    uv run python -m deepquake.fight_check runs/<name>/checkpoint.pt

The check a model must pass before :mod:`deepquake.release_model` will
stage it. It exists because win rates did not catch models-v9: its
monsters turned away from a player in plain view and did not attack, and
still "won" against a weak sparring opponent (#63).

Every class of the checkpoint is put in front of the easiest possible
target: a player standing still, in sight, 150 to 350 units away, who
never moves, turns or shoots, with the level's other monsters kept out of
it. The same fixed tasks are played twice in the same body: by the
checkpoint, and by a scripted **reference** that does the obvious thing
(turn to the player, walk up, hold attack). The reference shows what the
body can do under the current engine rules, so the bar moves with the
rules instead of being a guessed number.

A class passes when all of these hold (the constants below):

- the reference itself kills the player in at least
  :data:`REFERENCE_FLOOR` of the tasks, otherwise the probe says nothing
  about this class and the class **fails as unmeasured**;
- the checkpoint kills in at least :data:`KILL_FRACTION` of the
  reference's rate;
- while the player is in sight, the monster's shown facing is within
  :data:`FACING_DEGREES` of the player on at least :data:`FACING_FRACTION`
  of decisions.

The report also plays the same tasks with **random** actions. That column
is not part of the verdict, and it is the most important thing to read:
the engine glue turns the monster to a visible player, charges and runs
the stock attack whenever attack is pressed, so on this probe random
button presses kill about as often as the reference (measured on
models-v11, #65). Passing therefore shows only that the model is not
*worse* than the body's defaults. It does not show the model learned
anything. A model whose numbers match the random column has not.

The model passes when every class passes. The report,
``fight_check.json`` beside the checkpoint, carries the checkpoint's
SHA-256, so it vouches for that file and no other.

The thresholds and the probe are the maintainer's to change, not the
trainer's: a run that fails has a model that does not fight, and the fix
is in the training, never here (see "Training a model" in AGENTS.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import monster_env, nav, sim

REPORT_NAME = "fight_check.json"
REPORT_VERSION = 1

#: The reference must kill the standing player this often for a class to be
#: measurable at all.
REFERENCE_FLOOR = 0.5
#: The checkpoint's kill rate, as a fraction of the reference's.
KILL_FRACTION = 0.6
#: "Facing the player": shown facing within this many degrees...
FACING_DEGREES = 45.0
#: ...on at least this fraction of the decisions with the player in sight.
FACING_FRACTION = 0.6
#: Tasks that start with the player in the monster's sight, per class; fewer
#: usable ones than this and the class fails as unmeasured.
EPISODES = 20
MIN_USABLE = 10
#: Fixed tasks tried per usable one wanted.
TASK_POOL = 10
SKILL = 2
SEED = 2026
MIN_DISTANCE = 150.0
MAX_DISTANCE = 350.0
#: One decision in twelve a second: 30 seconds to kill a target that does
#: not fight back.
TIMEOUT_STEPS = 360

#: A level that stocks each class, in order of preference.
CLASS_MAPS = {
    "monster_army": ("e1m1", "e1m2"),
    "monster_dog": ("e1m1", "e1m2"),
    "monster_enforcer": ("e2m1", "e2m2"),
    "monster_ogre": ("e1m2", "e1m4"),
    "monster_hell_knight": ("e3m6", "e2m5"),
    "monster_shalrath": ("e4m7", "e4m8"),
    "monster_knight": ("e1m4", "e1m5"),
    "monster_demon1": ("e2m5", "e2m4"),
    "monster_shambler": ("e1m8", "e1m3"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _angle(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _client_in_sight(hook: sim.HookObservation | None) -> sim.HookVisible | None:
    if hook is None:
        return None
    for seen in hook.visible:
        if seen.flags & sim.FL_CLIENT and seen.health > 0:
            return seen
    return None


def reference_action(env: monster_env.MonsterEnv) -> np.ndarray:
    """The scripted reference: turn to the player, walk up, hold attack."""
    stop = np.array([1, 1, monster_env.MONSTER_TURN_DEGREES.index(0.0), 0])
    client = _client_in_sight(env._hook_obs)
    if client is None:
        return stop
    bearing = math.degrees(math.atan2(client.relative[1], client.relative[0]))
    wanted = (bearing - env.monster_yaw + 180.0) % 360.0 - 180.0
    turn = int(np.argmin([abs(wanted - step) for step in monster_env.MONSTER_TURN_DEGREES]))
    forward = 2 if client.distance > 120.0 else 1
    return np.array([forward, 1, turn, 1])


def run_class(
    classname: str,
    map_name: str,
    checkpoint: str,
    brain: str,
    placement: str = "sight",
    opponent: str = "still",
) -> dict:
    """One class on one map, played by ``brain``.

    'checkpoint', the scripted 'reference', 'random' (uniform actions) or
    'idle' (never moves, never attacks).
    """
    try:
        sim.init()
    except sim.SimError as exc:
        if "may only be called once" not in str(exc):
            raise  # pool workers are reused: already initialised
    points = nav.NavEnv(map_name, seed=SEED).points
    env = monster_env.MonsterEnv(
        map_name,
        seed=SEED,
        monster=classname,
        opponent=monster_env.PlayerOpponent(opponent),
        timeout_steps=TIMEOUT_STEPS,
        min_distance=MIN_DISTANCE,
        max_distance=MAX_DISTANCE,
        skills=(SKILL,),
        points=points,
        placement=placement,
        others_passive=True,
    )
    policy = None
    if brain == "idle":

        def policy(_obs):
            return np.array([1, 1, monster_env.MONSTER_TURN_DEGREES.index(0.0), 0])

    if brain == "random":
        rng = np.random.default_rng(SEED)

        def policy(_obs):
            return np.array([int(rng.integers(n)) for n in monster_env.MONSTER_ACTION_NVEC])

    if brain == "checkpoint":
        import torch

        from .eval_nav import TrainedPolicy  # torch: only when a checkpoint is measured

        torch.set_num_threads(1)  # one class per process: many threads only fight
        policy = TrainedPolicy(Path(checkpoint), SEED)
    # Not every stocked monster has a spot in the band it can see: the fixed
    # task list is long, and the first EPISODES usable tasks are played.
    tasks = monster_env.make_tasks(env, EPISODES * TASK_POOL, SEED)
    usable = kills = damaged = 0
    sight_decisions = facing_decisions = attack_presses = 0
    closest: list[float] = []
    for task in tasks:
        if usable == EPISODES:
            break
        try:
            obs = env.set_task(*task)
        except RuntimeError:
            continue  # the level has no such start: not a usable task
        if placement == "sight" and _client_in_sight(env._hook_obs) is None:
            continue  # the monster cannot see the player from here
        usable += 1
        if hasattr(policy, "reset"):
            policy.reset()
        nearest = float("inf")
        while True:
            client = _client_in_sight(env._hook_obs)
            entity = env._monster_entity()
            if client is not None and entity is not None:
                sight_decisions += 1
                bearing = math.degrees(math.atan2(client.relative[1], client.relative[0]))
                if _angle(entity.yaw, bearing) <= FACING_DEGREES:
                    facing_decisions += 1
                nearest = min(nearest, client.distance)
            action = reference_action(env) if policy is None else policy(obs)
            attack_presses += int(action[3])
            result = env.step(action)
            obs = result.obs
            if result.done:
                break
        kills += int(result.success)
        damaged += int(env._first_damage_dealt is not None)
        if math.isfinite(nearest):
            closest.append(nearest)
    return {
        "map": map_name,
        "usable_tasks": usable,
        "kill_rate": kills / usable if usable else 0.0,
        "damage_rate": damaged / usable if usable else 0.0,
        "facing_fraction": facing_decisions / sight_decisions if sight_decisions else 0.0,
        "sight_decisions": sight_decisions,
        "attack_presses": attack_presses,
        "mean_closest_distance": float(np.mean(closest)) if closest else None,
    }


def judge_class(policy: dict, reference: dict) -> list[str]:
    """Why this class fails; empty when it passes."""
    reasons = []
    if reference["usable_tasks"] < MIN_USABLE:
        return [
            f"unmeasured: only {reference['usable_tasks']} tasks start with the player in "
            f"sight (need {MIN_USABLE})"
        ]
    if reference["kill_rate"] < REFERENCE_FLOOR:
        return [
            f"unmeasured: the scripted reference kills in only {reference['kill_rate']:.0%} of "
            f"tasks (need {REFERENCE_FLOOR:.0%}), so the probe says nothing about this class"
        ]
    needed = KILL_FRACTION * reference["kill_rate"]
    if policy["kill_rate"] < needed:
        reasons.append(
            f"kills the standing player in {policy['kill_rate']:.0%} of tasks; the reference "
            f"kills in {reference['kill_rate']:.0%}, so at least {needed:.0%} is required"
        )
    if policy["facing_fraction"] < FACING_FRACTION:
        reasons.append(
            f"faces the player on {policy['facing_fraction']:.0%} of the decisions it can see "
            f"it (need {FACING_FRACTION:.0%} within {FACING_DEGREES:.0f} degrees)"
        )
    return reasons


def judge(report: dict, classes: list[str]) -> dict[str, list[str]]:
    """Failure reasons per class, for the classes a model must cover."""
    failures: dict[str, list[str]] = {}
    for classname in classes:
        measured = report.get("classes", {}).get(classname)
        if measured is None:
            failures[classname] = ["not measured"]
            continue
        reasons = judge_class(measured["checkpoint"], measured["reference"])
        if reasons:
            failures[classname] = reasons
    return failures


def require_passing_report(checkpoint: Path, classes: list[str]) -> None:
    """Raise SystemExit unless ``checkpoint`` has a passing fight check.

    The verdict is recomputed here from the report's numbers with this
    build's thresholds: the report's own ``passed`` field is not trusted.
    """
    path = checkpoint.parent / REPORT_NAME
    command = f"uv run python -m deepquake.fight_check {checkpoint}"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist: this model has not been checked. Run: {command}")
    report = json.loads(path.read_text())
    if report.get("version") != REPORT_VERSION:
        raise SystemExit(f"{path} is from another version of the check. Run: {command}")
    if report.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise SystemExit(f"{path} was made for a different checkpoint file. Run: {command}")
    failures = judge(report, classes)
    if failures:
        lines = [f"  {name}: {reason}" for name, reasons in failures.items() for reason in reasons]
        raise SystemExit(
            f"{checkpoint} does not pass the fight check and will not be released:\n"
            + "\n".join(lines)
            + "\nThe fix is in the training, not in the check (AGENTS.md, 'Training a model')."
        )


def main() -> None:
    from . import ppo  # torch

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--jobs", type=int, default=6, help="classes measured in parallel")
    args = parser.parse_args()

    _, checkpoint = ppo.load_model(args.checkpoint)
    classes = list(checkpoint.get("monsters") or ["monster_army"])
    unknown = [name for name in classes if name not in CLASS_MAPS]
    if unknown:
        raise SystemExit(f"no probe map is listed for {unknown}: add it to CLASS_MAPS")

    brains = ("reference", "random", "checkpoint")
    work = [(name, brain) for name in classes for brain in brains]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            (name, brain): pool.submit(
                run_class, name, CLASS_MAPS[name][0], str(args.checkpoint), brain
            )
            for name, brain in work
        }
        measured = {key: future.result() for key, future in futures.items()}

    report = {
        "version": REPORT_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "thresholds": {
            "reference_floor": REFERENCE_FLOOR,
            "kill_fraction": KILL_FRACTION,
            "facing_degrees": FACING_DEGREES,
            "facing_fraction": FACING_FRACTION,
            "episodes": EPISODES,
            "skill": SKILL,
            "seed": SEED,
        },
        "classes": {
            name: {
                "checkpoint": measured[(name, "checkpoint")],
                "reference": measured[(name, "reference")],
                "random": measured[(name, "random")],
            }
            for name in classes
        },
    }
    failures = judge(report, classes)
    report["failures"] = failures
    report["passed"] = not failures

    print(
        f"{'class':<22} {'kills':>7} {'reference':>10} {'random':>7} "
        f"{'facing':>8} {'reference':>10} {'random':>7}  verdict"
    )
    for name in classes:
        cells = report["classes"][name]
        mine, ref, rnd = cells["checkpoint"], cells["reference"], cells["random"]
        print(
            f"{name:<22} {mine['kill_rate']:>7.0%} {ref['kill_rate']:>10.0%} "
            f"{rnd['kill_rate']:>7.0%} {mine['facing_fraction']:>8.0%} "
            f"{ref['facing_fraction']:>10.0%} {rnd['facing_fraction']:>7.0%}  "
            + ("FAIL" if name in failures else "pass")
        )
    print(
        "The random column is random button presses in the same body. A model whose "
        "numbers match it has learned nothing this probe can see."
    )
    for name, reasons in failures.items():
        for reason in reasons:
            print(f"  {name}: {reason}")
    out_path = args.checkpoint.parent / REPORT_NAME
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")
    print("PASSED" if report["passed"] else "FAILED: this model must not be released")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

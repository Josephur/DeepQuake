# DeepQuake — agent guide

DeepQuake replaces Quake's hand-coded QuakeC monster AI with trained neural
policies, so enemies navigate, flank, retreat, ambush and coordinate instead of
running at the player. Read `docs/ARCHITECTURE.md` for the design and
`docs/ROADMAP.md` for the current milestone before starting work.

## Every session: resume first, then one issue at a time

A session can be cut off at any moment, without warning. Nothing you remember
survives; only what is pushed to GitHub does. So work is saved continuously,
and every session starts by finding out where the last one stopped.

**Start of session**

1. `git status` and `git branch --show-current`. If there are uncommitted
   changes, they are the last session's unsaved work: read the diff, and
   commit them to the current issue branch as `WIP` if they belong to it.
   **Exception:** changes under `engine/third_party/` are never committed
   as WIP. They are leftover trace or debug code. Read the diff to learn
   what the last session was looking into, then undo it with
   `git checkout -- engine/third_party`, unless the issue's last Checkpoint
   names that file as a deliberate patch.
2. `gh run list --branch main --limit 1`. If CI on `main` is failing, fixing
   it is your task for this session (open a `bug` issue for it).
3. `gh issue list --label in-progress`. There is at most one.
   - **One exists:** that is your task. Read the whole issue, and above all
     its last `Checkpoint` comment. Check out its branch
     (`issue-<number>-<short-name>`), pull, run the checks, and continue from
     the checkpoint's "Next" line. Do not start over and do not pick another
     issue.
   - **None exists:** read `docs/ROADMAP.md`, take the lowest-numbered open
     issue in the current milestone that is not labelled `blocked` or
     `needs-human`, and start it as described under "Work tracking". If the
     current milestone has no open issues left, see "Starting a milestone".
4. If a training run was left running (the checkpoint says so), check whether
   it finished before doing anything else: look for
   `runs/<name>/checkpoint.pt` and the end of `runs/<name>/train.log`.

**During the session**

- Work on the issue's branch. Commit after every step that leaves the code
  working (a test added and passing, a function finished), and push the
  branch after every commit. WIP commits on the branch are fine.
- After every pushed commit, and before anything slow (a build from scratch,
  a training run), write a checkpoint comment on the issue, in exactly this
  form:

  ```
  Checkpoint
  Done: <what is finished and committed, with commit hashes>
  State: <branch name; do the checks pass right now? anything running in the background?>
  Next: <the single next step, specific enough to act on without rereading everything>
  Notes: <decisions made and why, dead ends and why they failed, numbers measured>
  ```

  The next session, which remembers nothing, will act on "Next". Write it
  for that reader.
- Start long commands so they outlive the session, and log to a file:
  `nohup uv run python -m deepquake.train_nav configs/<name>.toml > runs/<name>.log 2>&1 &`
  Put the run name in the checkpoint.

**End of an issue**

1. All checks under "Build, test, check" pass on the branch, including the
   tests that need game data.
2. `git checkout main && git pull && git merge --squash <branch>`, commit
   with a proper message (`Refs #N`), push, and wait for CI to pass on `main`.
3. Close the issue with its evidence comment, remove the `in-progress`
   label, delete the branch locally and on GitHub.
4. Only then start the next issue.

**Starting a milestone**

When the current milestone has no open issues, tick it off in
`docs/ROADMAP.md`, then create the issues for the next milestone: one per
roadmap item, each small enough to finish in a few hours, each with a
**Goal** and a **Done when** checklist that can be checked by running
something. If a roadmap item is too vague to write that checklist, create a
`research` issue to settle it first. If a choice would change the design in
`docs/ARCHITECTURE.md`, label the issue `needs-human` and stop.

**Stop the session and leave a checkpoint when**

- every open issue is `blocked` or `needs-human`;
- something needs `sudo`, a system package, a purchase, an account, or
  access to another machine;
- the same step has failed three times with different fixes. Write what you
  tried in the checkpoint, label the issue `blocked`, and take another.

**Never, to make progress**

- weaken, skip or delete a test, or loosen a threshold, so that checks pass;
- change the evaluation tasks, seeds or baselines so that numbers look
  better (changing them for a stated reason, re-measuring everything, is
  fine);
- close an issue whose **Done when** items are not all demonstrated;
- report a result you did not measure in this session or cannot point to in
  a file or an issue comment.

## Hard rules

1. **Never commit Quake game data.** `.pak`, `.bsp`, `.wad`, `.mdl`, `.spr`,
   `.lmp`, `progs.dat`, demos, and anything extracted from them are id
   Software's proprietary assets. They live only in `gamedata/` (gitignored).
   This also covers screenshots of copyrighted levels in docs, and any file
   from the Quake Enhanced (KEX) rerelease other than its `id1` pak files used
   locally.
2. **Never commit weights, checkpoints, or run output.** They live in
   `models/` and `runs/` (gitignored) and are published as release assets.
3. **Stay inside this repository.** Do not read, search, or write elsewhere in
   the user's home directory. If you need game files, ask the user to place or
   symlink them at `gamedata/id1/`.
4. **Only GPL-compatible code goes in.** The project is GPL-2.0-or-later
   because it links id's GPL engine source. Do not copy code from the KEX
   rerelease or any source without a compatible licence. Record the origin of
   vendored code in `engine/third_party/<name>/UPSTREAM.md` (URL, commit,
   licence, local patches).
5. **No secrets, tokens, absolute home paths, or personal details** in tracked
   files. Machine-specific settings go in `.env` or `*.local.*` (gitignored).
6. **Run the checks before every commit** (see below). Do not bypass the
   pre-commit hook with `--no-verify`.

## Layout

```
engine/                 C. Headless simulator (libdqsim) and, later, the playable client
  include/dqsim/        Public C API — the only contract Python and tools depend on
  src/                  Our code: API (dqsim.c), system layer, fake network, client stubs
  src/shim/             Stand-ins for SDL and renderer headers that are not vendored
  tests/                C tests, run by ctest
  tools/                C programs built with the library (dqsim_bench)
  third_party/quakespasm/  Vendored QuakeSpasm server code, unmodified; see its UPSTREAM.md
python/deepquake/       Python package
  sim.py                Bindings for libdqsim (one simulator per process)
  gamedata.py           Find, check and link the user's game data
  nav.py                Navigation task: observation, actions, reward, baselines
  vec.py                Run many environments in worker processes
  ppo.py                PPO and the policy network (knows nothing about Quake)
  train_nav.py          Training entry point; eval_nav.py: evaluation entry point
  toy.py                Simulator-free environment for testing the training code
python/tests/           pytest
configs/                Training configs (TOML), one per experiment
tools/                  Repo maintenance scripts (stdlib-only Python or POSIX sh)
docs/                   ARCHITECTURE.md, ROADMAP.md
gamedata/               LOCAL ONLY. User's Quake data: gamedata/id1/pak0.pak, pak1.pak
models/                 LOCAL ONLY. Downloaded or trained weights
runs/                   LOCAL ONLY. Training logs and checkpoints
```

Add new top-level directories only when the roadmap calls for them (for
example `qc/` for QuakeC hooks, `configs/` for training configs). Do not create
empty placeholder directories or stub modules ahead of need.

## Build, test, check

```bash
git config core.hooksPath .githooks       # once per clone

cmake --preset dev && cmake --build --preset dev && ctest --preset dev
cmake --preset release && cmake --build --preset release && ctest --preset release
uv sync --extra train                      # --extra train adds PyTorch (needed to train)
uv run ruff check . && uv run ruff format --check .
uv run pytest
python3 tools/check_repo_hygiene.py        # forbidden files, size limits
uv run python -m deepquake.gamedata        # verify local game data is usable
build/release/bin/dqsim_bench gamedata e1m1 20000 24   # simulator throughput
```

Train and evaluate (output goes to `runs/<name>/`, which is gitignored):

```bash
uv run python -m deepquake.train_nav configs/nav_e1m1.toml            # add --resume to continue
uv run python -m deepquake.eval_nav runs/nav_e1m1/checkpoint.pt       # compare with baselines
uv run python -m deepquake.eval_nav                                   # baselines only
```

Python loads the **release** build of the library first, so rebuild the
release preset before running pytest after any engine change.

CI (`.github/workflows/ci.yml`) runs the same commands with no game data and
no GPU. Anything needing game data or a GPU must be a pytest test marked
`gamedata` or `gpu` (deselected in CI), or a ctest test that exits with code
77 when the data is absent. Such tests must skip cleanly, never fail, when
the resource is missing. Because CI cannot run them, **you must run them
locally before closing any issue that touches `engine/`**, and say in the
closing comment that you did.

## Work tracking: every change has a GitHub issue

The issue tracker is the project's history. Use the `gh` CLI.

1. **Before starting work**, find the issue for it (`gh issue list`,
   `docs/ROADMAP.md` names them). If none exists, create one first:
   `gh issue create --title "..." --label <type>,<area> --milestone "<milestone>" --body "..."`
   with a body stating the **Goal** and a **Done when** checklist.
   - Type label, exactly one: `bug`, `feature`, `task`, `research`, `documentation`.
   - Area label, at least one: `area:engine`, `area:python`, `area:training`, `area:infra`.
2. **When you start**, add the `in-progress` label
   (`gh issue edit N --add-label in-progress`), create the branch
   `issue-<number>-<short-name>` from an up-to-date `main`, and comment that
   you are starting, with your approach in one or two sentences.
3. **While working**, comment on anything a later reader would need: a
   decision and its reason, a dead end and why it failed, a measurement.
   Bugs you find along the way get their own `bug` issue with steps to
   reproduce, expected and actual behaviour; do not fix them silently.
4. **Commits** reference the issue in the message body as `Refs #N`. Do not
   use `Closes #N` or `Fixes #N`: issues are closed by hand, with a comment.
5. **When done**, and only after the checks pass and the change is pushed,
   close with a comment:
   `gh issue close N --comment "..."` stating what was done, the commit
   hashes, and the evidence that each **Done when** item is met (test names,
   command output, numbers). If an item was not met, do not close; say what
   remains.
6. **If you cannot proceed**, write a checkpoint, push the branch, remove
   `in-progress`, and add the `blocked` label and a comment with the
   reason. If it needs the maintainer (a decision, a purchase, access, a
   licence question), add `needs-human` as well, then move to another issue.
7. Never close an issue as completed for work that is partial, untested, or
   failing. Never delete issues or edit other people's comments.

## Conventions

- **C**: C11, `-Wall -Wextra -Werror` for our code (not for `third_party/`).
  Public symbols are prefixed `dqsim_`.
- **Vendored engine code is unmodified.** When the engine needs to behave
  differently, first try a stub in `engine/src/client_stubs.c`, a shim header
  in `engine/src/shim/`, or a compile definition in `engine/CMakeLists.txt`.
  Edit a vendored file only when none of those can work, mark the change with
  a `// DeepQuake:` comment, and list it in `UPSTREAM.md` with its checksum,
  in the form that file describes. The hygiene check compares every
  vendored file with `SHA256SUMS` and fails on any other difference.
- **Instrumenting the engine** (adding prints or traces to see what it does
  internally) is done in a scratch copy, never in `engine/third_party/` and
  never in the fidelity reference `engine/.fidelity/`:

  ```bash
  mkdir -p engine/.scratch && cp -r engine/third_party/quakespasm engine/.scratch/quakespasm
  # edit files under engine/.scratch/quakespasm/Quake/, then build a separate tree:
  cmake -S . -B build/trace -DDQSIM_QUAKESPASM_DIR=$PWD/engine/.scratch/quakespasm
  cmake --build build/trace
  ```

  To instrument the stock engine the same way, copy
  `engine/.fidelity/quakespasm` to `engine/.scratch/reference` and build
  that copy with its own Makefile. `engine/.scratch/` and `build/` are
  gitignored. Delete the scratch copy when you have your answer. The
  hygiene check refuses any commit while a vendored file differs from
  upstream or the fidelity reference has local changes, and never measure
  fidelity against anything but the untouched reference.
- **One simulator per process.** The engine's state is global. Never try to
  run two simulations in one process or call it from two threads; use worker
  processes. `dqsim_init` can be called once per process, which is why Python
  tests that run the engine do so in a subprocess.
- **Python**: 3.11+, managed with `uv`, formatted and linted by `ruff`. Type
  hints on public functions. The core package depends only on numpy; PyTorch
  and friends sit behind the `train` extra.
- **Determinism**: the simulator must give identical results for identical
  seed and inputs. Never introduce wall-clock time or unseeded randomness into
  `engine/` or environment code.
- **Training hardware**: use the RTX 3090 (select it by name or UUID, not by
  index, since the machine has two GPUs). Simulation runs on CPU across many
  processes; the GPU is for the learner only. Keep training code device
  agnostic (take the device from configuration). Do not install or clone
  tooling for other accelerators (for example Intel Gaudi) on this machine or
  into this repository; that is a `needs-human` decision.
- **Experiments**: every training run has a config file in `configs/` and a
  unique `name`. Never overwrite a previous run's directory; pick a new name.
  Report results as numbers from `eval_nav` on its fixed task list next to
  the random and bee-line baselines, in the issue. A training curve going up
  is not a result; the evaluation is.
- **Changing the observation or action layout** invalidates every saved
  checkpoint. Do it deliberately, update the table in `nav.py`'s docstring
  and `docs/ARCHITECTURE.md`, and retrain.
- **Commits**: imperative subject line. On an issue branch, commit often.
  On `main`, one squashed commit per issue. `main` must always build and
  pass CI.
- **Docs**: when a decision changes the design, update
  `docs/ARCHITECTURE.md` in the same commit. When a milestone item is finished,
  tick it in `docs/ROADMAP.md`.

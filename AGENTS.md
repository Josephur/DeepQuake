# DeepQuake — agent guide

DeepQuake replaces Quake's hand-coded QuakeC monster AI with trained neural
policies, so enemies navigate, flank, retreat, ambush and coordinate instead of
running at the player. Read `docs/ARCHITECTURE.md` for the design and
`docs/ROADMAP.md` for the current milestone before starting work.

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
  src/                  Our code
  tests/                C tests, run by ctest
  third_party/          Vendored GPL engine source (added in M1), minimally patched
python/deepquake/       Python package: sim bindings, environments, models, training
python/tests/           pytest
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
uv sync                                    # add --extra train for PyTorch
uv run ruff check . && uv run ruff format --check .
uv run pytest
python3 tools/check_repo_hygiene.py        # forbidden files, size limits
uv run python -m deepquake.gamedata        # verify local game data is usable
```

CI (`.github/workflows/ci.yml`) runs the same commands with no game data and
no GPU. Anything needing game data or a GPU must be a pytest test marked
`gamedata` or `gpu`; those are deselected in CI and must skip cleanly when the
resource is absent.

## Work tracking: every change has a GitHub issue

The issue tracker is the project's history. Use the `gh` CLI.

1. **Before starting work**, find the issue for it (`gh issue list`,
   `docs/ROADMAP.md` names them). If none exists, create one first:
   `gh issue create --title "..." --label <type>,<area> --milestone "<milestone>" --body "..."`
   with a body stating the **Goal** and a **Done when** checklist.
   - Type label, exactly one: `bug`, `feature`, `task`, `research`, `documentation`.
   - Area label, at least one: `area:engine`, `area:python`, `area:training`, `area:infra`.
2. **When you start**, comment that you are starting and your approach in one
   or two sentences.
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
6. **If you cannot proceed**, add the `blocked` label and a comment with the
   reason. If it needs the maintainer (a decision, a purchase, access, a
   licence question), add `needs-human` as well, then move to another issue.
7. Never close an issue as completed for work that is partial, untested, or
   failing. Never delete issues or edit other people's comments.

## Conventions

- **C**: C11, `-Wall -Wextra -Werror` for our code (not for `third_party/`).
  Public symbols are prefixed `dqsim_`. Keep patches to vendored code small and
  list each one in that directory's `UPSTREAM.md`.
- **Python**: 3.11+, managed with `uv`, formatted and linted by `ruff`. Type
  hints on public functions. The core package depends only on numpy; PyTorch
  and friends sit behind the `train` extra.
- **Determinism**: the simulator must give identical results for identical
  seed and inputs. Never introduce wall-clock time or unseeded randomness into
  `engine/` or environment code.
- **Training hardware**: use the RTX 3090 (select it by name or UUID, not by
  index, since the machine has two GPUs). Simulation runs on CPU across many
  processes; the GPU is for the learner only.
- **Commits**: small, imperative subject line, one logical change each. `main`
  must always build and pass CI.
- **Docs**: when a decision changes the design, update
  `docs/ARCHITECTURE.md` in the same commit. When a milestone item is finished,
  tick it in `docs/ROADMAP.md`.

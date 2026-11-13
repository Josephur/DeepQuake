# DeepQuake

Neural-network brains for Quake's monsters.

Quake's enemies run fixed QuakeC routines: see player, run at player, attack.
DeepQuake trains a policy for each monster type with reinforcement learning so
they can navigate levels properly, use cover, flank, retreat, ambush, and work
together — then plugs those policies into a playable engine so you can fight
them.

**Status:** the headless simulator runs real Quake maps deterministically at over 100,000 ticks per second per core, and a first policy trained in five minutes navigates E1M1 (96% of goals reached, against 73% for a run-straight-at-it baseline). Monster brains come next. See [docs/ROADMAP.md](docs/ROADMAP.md).

## How it works

1. **Headless simulator** (`engine/`): Quake's server-side game code (physics,
   collision, QuakeC VM) built from id Software's GPL source as a library with
   no graphics, sound, or network, so thousands of game ticks per second can
   run per CPU core.
2. **Training** (`python/deepquake/`): many simulator instances feed a PyTorch
   learner on one GPU. Agents first learn to move and fight as a player, then
   monster policies are trained against them.
3. **Play**: the same engine code, built as a normal client, runs the exported
   policies in place of the QuakeC AI.

Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## You need your own copy of Quake

This repository contains **no game data**. You must own Quake and supply its
`id1` pak files (from the original release, Steam/GOG, or the 2021 rerelease).
See [gamedata/README.md](gamedata/README.md).

## Build

Requirements: a C11 compiler, CMake ≥ 3.21, Ninja, Python ≥ 3.11, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Josephur/DeepQuake.git
cd DeepQuake
git config core.hooksPath .githooks

cmake --preset release
cmake --build --preset release
ctest --preset release

uv sync                    # add --extra train for PyTorch
uv run pytest
uv run python -m deepquake.gamedata link /path/to/Quake/id1   # link your game data
ctest --preset release     # now includes the tests that need game data
uv run pytest
```

Try it:

```python
from deepquake import sim

sim.init()
sim.load_map("e1m1", skill=1, seed=0)
for _ in range(72):  # one second of game time
    sim.step(forward=400, yaw=90)
print(sim.player())
```

Train and evaluate the navigation policy (needs an NVIDIA GPU or patience):

```bash
uv sync --extra train
uv run python -m deepquake.train_nav configs/nav_e1m1.toml
uv run python -m deepquake.eval_nav runs/nav_e1m1/checkpoint.pt
```

## Trained models

Weights are published as GitHub release assets, not stored in git. Download a
trained monster brain and play against it:

    python3 tools/download_model.py grunt
    build/release/bin/deepquake -basedir gamedata -userdir <a-writable-dir> \
        -window +set dq_model models/grunt/grunt.onnx +map e1m1

Every Grunt in the level is driven by the model (the `dq_max_monsters` cvar
caps the count; 0, the default, is all of them). See
[models/README.md](models/README.md).

## Licence

GPL-2.0-or-later, see [LICENSE](LICENSE). DeepQuake builds on the Quake engine
source released by id Software under the GPL. Quake is a trademark of id
Software; this project is not affiliated with or endorsed by id Software or
Bethesda.

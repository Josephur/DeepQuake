# Architecture

## Goal

Replace the decision-making of Quake's monsters with learned policies, while
keeping their bodies (speed, health, attacks, animations) exactly as the game
defines them. A DeepQuake Grunt has the same gun and the same legs as a stock
Grunt; it just decides better.

## Components

```
            training                                   play
 ┌──────────────────────────────┐        ┌──────────────────────────────┐
 │ PyTorch learner (RTX 3090)   │        │ Playable client              │
 │   ▲ observations  │ actions  │        │  (same engine source + GL)   │
 │   │               ▼          │ export │   brain hook ── ONNX runtime │
 │ N × libdqsim workers (CPU)   │ ─────► │   (CUDA or CPU), batched     │
 │   headless Quake server core │ .onnx  │   over all monsters per tick │
 └──────────────────────────────┘        └──────────────────────────────┘
```

### 1. Headless simulator — `engine/`, `libdqsim`

Built from an existing GPL Quake engine rather than rewritten, because exact
physics matters: a policy trained on almost-Quake movement will not transfer.

- Keep the server side: BSP loading and hull collision, `SV_Physics`, player
  and monster movement, and the QuakeC VM running the stock `progs.dat` (so
  weapons, items, doors, triggers, and damage behave as in the real game).
- Remove the client side: renderer, sound, input, networking, console UI.
- Expose a small C API in `engine/include/dqsim/`: create, load map, reset
  with seed, apply actions, step one tick, read observations, snapshot and
  restore.
- One simulator instance per process. The Quake server keeps its state in
  globals (`sv`, `svs`, the QuakeC VM); rather than rewrite that, training
  runs one worker process per environment.
- Deterministic: same seed and actions give the same trajectory. Every
  `rand()` call in the engine is redirected at compile time to a seeded
  generator (`engine/src/rng.c`), and there is no wall clock.
- Fixed tick: one step is 1/72 s of game time by default (Quake's maximum
  frame rate, at which its physics behaves as designed). It is configurable
  at start-up up to 0.1 s; larger ticks are faster but change movement
  slightly, so policies must be trained and played at the same tick.
- The player is a client on a fake network connection (`net_fake.c`). The
  API queues the same bytes a real client sends, and the unmodified server
  code parses them, so player input takes exactly the path it takes in the
  real game. When the game restarts the level (death) or changes level
  (exit), the simulator signs the player on again and counts it.
- Engine errors (`Sys_Error`) do not exit the process; they return an error
  to the caller and mark the simulator dead.

**Engine base: QuakeSpasm** (github.com/sezero/quakespasm, GPL-2.0-or-later),
vendored unmodified under `engine/third_party/quakespasm/`. Chosen after surveying seven
engines in November 2026:

| Engine | Verdict |
| --- | --- |
| QuakeSpasm | **Chosen.** Plain C, 64-bit clean, actively maintained. Its server core (`sv_*.c`, `pr_*.c`, `world.c`, about 8,800 lines) contains no references to client, renderer, or SDL code, so it separates cleanly. Faithful to original physics. |
| Ironwail | QuakeSpasm fork with a faster renderer. Its server core has picked up client-side QuakeC, drawing calls, and SDL threading, so it is harder to strip. Physics files differ from QuakeSpasm by only a few lines, so it remains an option for the play client later. |
| vkQuake, QuakeSpasm-Spiked | Server code is heavily entangled with the client, particles, sound, and (vkQuake) multithreading. |
| id's original WinQuake | Ships null drivers, but is 32-bit only (QuakeC strings are stored as pointer offsets) and unmaintained since 1999. QuakeSpasm is this code with those problems fixed. |
| FTEQW, DarkPlaces | Have dedicated servers, but are far larger (790k and 200k lines) and deliberately change physics and QuakeC behaviour. |

No existing project wraps the Quake 1 engine as a training environment, so
there is nothing to reuse there. The same vendored server source is compiled
into both the headless library and the playable client, so training and play
share physics by construction. Quake Enhanced (KEX) is closed source and is
only a source of `id1` data.

### 2. Brain hook

The point where a policy takes over from QuakeC. Monster think functions
(`ai_stand`, `ai_walk`, `ai_run`, attack selection) are intercepted in the
engine (monster movement lives in `sv_move.c`: `SV_movestep`, `SV_NewChaseDir`,
`SV_MoveToGoal`); the policy chooses movement and when to start an attack, and the
existing QuakeC code still plays the animation and applies the damage. With no
model loaded for a monster type the stock AI runs unchanged, so the game is
always playable.

### 3. Observations and actions

Chosen to be cheap to compute in the simulator and identical in the client:

- **Observations**: own state (health, velocity, on-ground, attack cooldown);
  a fan of BSP ray casts giving distance and surface type (the agent's
  "vision" of geometry); relative position, velocity, and line-of-sight of
  visible enemies and allies; recent sound events (monsters in Quake hear);
  a short memory of the last known enemy position.
- **Actions**: discrete move direction, turn, jump where the body allows it,
  and attack choice. Player agents use the normal player command (forward,
  side, yaw, pitch, jump, fire, weapon).
- **Style conditioning**: an extra input vector (aggression, caution, stealth)
  that shifts reward weights during training, so one network can be told to
  play offensively, defensively, or covertly at run time.

**Implemented so far** (navigation only, `python/deepquake/nav.py`, whose
docstring holds the exact layout): 48 ray distances, own velocity and
on-ground flag, and the direction and distance to the goal; four discrete
action heads (forward/back, strafe, turn, jump) held for 6 ticks. The engine
provides the primitives (`dqsim_ray_fan`, `dqsim_get_entities`); the
observation vector is assembled in Python. Enemy, ally and sound features
are not implemented yet.

### 4. Models

Small recurrent policies trained from scratch with PPO, on the order of 10⁵ to
10⁷ parameters. A language model is the wrong tool: the engine thinks at 10 to
72 Hz for dozens of monsters, so each decision has a budget well under a
millisecond, and the inputs are geometry and kinematics, not text.

The first policy (navigation) is a plain two-layer MLP with no memory,
about 150,000 parameters; a recurrent core is the next step.

**One shared network for all monster types, not one network each.** Most of
what a monster must learn is the same for every type: reading geometry from
ray casts, path-finding, keeping or breaking line of sight, judging the
player's movement. Separate networks would each relearn that from scratch
and could not learn to cooperate in mixed groups. So:

- a shared trunk (encoder and recurrent core) used by every monster;
- a monster-type embedding as an extra input, so behaviour can differ by
  body (a Fiend leaps, a Scrag flies, an Ogre lobs grenades);
- a small output head per type, because the available actions differ.

The first monster (Grunt) is trained alone, which is the same thing as a
per-monster model. The shared design starts to matter with the second type,
which begins from the Grunt's trunk instead of from nothing. One network also
suits play: every monster in the level is evaluated in a single batched
inference call per tick. If a type later proves to be held back by sharing,
it can be split off and fine-tuned on its own; that direction is easy, the
reverse is not.

### 5. Training — `python/deepquake/`

- Many `libdqsim` workers on CPU cores produce experience; a single learner
  on the RTX 3090 consumes it. Simulation throughput, not GPU memory, is the
  expected bottleneck.
- Order of work: a **player agent** learns movement and navigation, then
  combat. Monster policies are then trained against a pool of player agents
  (and scripted opponents) so they do not overfit to one opponent.
- Curriculum: empty rooms, then real maps; single monster, then groups.
- Checkpoints are ordinary PyTorch files, which load on CUDA or CPU
  regardless of where they were trained. For the client they will be
  exported to ONNX and run through ONNX Runtime with the CUDA provider when
  a GPU is present, falling back to CPU.
- Measured on the first run: about 6,800 decisions per second with 20
  simulator processes. The limit is stepping the environments and passing
  data between processes, not the GPU, so a larger accelerator would not
  make this faster; fewer, cheaper round trips would.

### 6. Distribution

Source in git. Weights as GitHub release assets with a checksum manifest,
fetched into `models/`. Game data never leaves the user's machine.

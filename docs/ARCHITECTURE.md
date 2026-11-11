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
- Snapshots are in-memory copies of the mutable server state: `sv`, `svs`,
  the live edicts, the client block, the QuakeC globals, the clocks, the RNG,
  and the fake network's pending input. Restoring one and replaying the same
  actions reproduces the original state hashes. The map hunk (models,
  `progs.dat`, the string table) is never rewound because it does not change
  during play, so restore cannot leave dangling model or string pointers. The
  world's spatial index is rebuilt from the restored edicts' area links in
  their exact order, so collision tie-breaking and trigger order match the
  original run. A snapshot is valid only in the process and level episode
  that made it; a new map, a level change, or a death restart makes it stale
  and restore refuses it.

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

**Fidelity is measured, not assumed.** `engine/tests/test_fidelity.c` (ctest
name `dqsim.fidelity`) starts a second, unmodified QuakeSpasm built from the
same vendored commit by `tools/fetch_fidelity_reference.sh`, and drives both
engines with the same 775-tick input script. A minimal NetQuake client in the
test performs the real sign-on, sends each tick as the same `clc_move` bytes
libdqsim queues, and reads the player's origin back off the wire. Origins must
agree within the protocol's 1/16-unit quantisation at every tick, and the
health and on-ground flags exactly. A change to the observation or action
layout is cheap; a change to the movement path is not, because this test
fails.

The reference runs from a scratch basedir (a temp directory with `id1`
symlinked to the user's data and a `test` game directory holding an
`autoexec.cfg`), never from `+` commands: QuakeSpasm rebuilds the command
line into a 256-byte buffer and `stuffcmds()` parses that truncated copy, so
with long absolute paths a trailing `+map` is silently dropped and the server
falls back to `map start`. The test asserts the world model announced in
`svc_serverinfo` is `maps/e1m1.bsp` for the same reason.

### 2. Brain hook

The point where a policy takes over from QuakeC. Monster think functions
(`ai_stand`, `ai_walk`, `ai_run`, attack selection) are intercepted in the
engine (monster movement lives in `sv_move.c`: `SV_movestep`, `SV_NewChaseDir`,
`SV_MoveToGoal`); the policy chooses movement and when to start an attack, and the
existing QuakeC code still plays the animation and applies the damage. With no
model loaded for a monster type the stock AI runs unchanged, so the game is
always playable.

#### Custom game code (mods) (#27)

Many custom maps ship a mod directory whose `progs.dat` replaces id1's: a
`monster_army` there may have different health, attacks or think functions,
or sit next to monsters no policy knows. Classname alone therefore cannot
mean "a monster we know how to control". Decided:

- **A monster is policy-driven only when both hold:** its classname has a
  trained policy, **and** the `progs.dat` loaded for the level is
  byte-identical to the stock id1 `progs.dat` the policies were trained
  with. Identity is the loaded file's length and a hash of the whole file
  (the engine already CRCs the whole file in `PR_LoadProgs`); the
  `progs.dat` header CRC is not an identity — it only versions the system
  variables. Byte identity guarantees the entity fields, think functions
  and attack behaviour the hook and the training relied on, with no
  per-function guessing. A mod that ships an unmodified copy of the stock
  `progs.dat` passes, as do the many custom maps that ship none.
- **The gate is per level, not per monster**, because `progs.dat` is loaded
  per level: if it differs, no monster in that level is policy-driven, even
  a familiar classname. Everything else is untouched — the mod's game code
  still runs exactly as the stock engine would run it (its weapons, rules
  and new monsters included); only the decision replacement is withheld,
  and unknown classnames keep the stock AI in every case. This is map
  independence rule 4 below, made concrete. It is a compatibility check,
  not a security boundary: a mod is code the user chose to run, exactly as
  with the stock engine.
- **The simulator must be able to load a mod at all** before any of this
  can be tested: today `dqsim_init` passes only `-basedir`, so only `id1`
  is ever searched. M4 therefore adds an optional game directory to
  `dqsim_config` (a single directory name under the basedir, passed to the
  engine as `-game`; unset means `id1` only) and a way to ask libdqsim
  whether the loaded `progs.dat` is the stock one. Training stays on `id1`
  by configuration, not by engine limitation.
- **Split:** the game directory, the identity query and the gate itself
  are M4 (the hook, #31, refuses or ignores a nomination the gate rejects,
  so the stock think runs). M5 needs no new mechanism: the client already
  starts with the user's `-game` directory as stock Quake does, and the
  same gate in the shared engine code withholds the policy there. Relaxing
  byte identity later (for example to "the QuakeC functions and fields the
  hook uses are present and unmodified") is possible, but only with
  evidence that real mods need it; it is not M4 work.

**Implemented (#31):** `dqsim_set_policy_hook` / `dqsim_clear_policy_hook`
(the Python form is `sim.set_policy_hook`) nominate one monster of the
current level. `sv_phys.c` is compiled with `PR_ExecuteProgram` renamed
to the hook's entry point — the vendored file itself is unmodified — so
when the nominated monster's scheduled stand/walk/run think would run,
the callback instead receives an observation built from live state (its
own origin, velocity, facing, health and ground/water state; a 2×24 ray
fan with surface kinds from its eye; the entities it has line of sight
to, nearest first) and returns an action: absolute facing, forward/side
speeds, and an attack flag. Movement is applied as ground steps or air
velocity at that same point in the tick and the decision think is
rescheduled one tick later, so the policy decides every tick; the
animation frames come from the QuakeC state functions' own frames. An
attack action with a visible player and QuakeC's `attack_finished`
expired starts the monster's real QuakeC attack sequence, and attack,
pain and death sequences always run under QuakeC. Until #37 adds game
directories the simulator can only load id1, so the progs gate compares
against the first progs loaded in the process. With no nomination the
simulation is bit-identical (the fidelity test guards this); the
nomination refuses non-monsters and, per the gate above, a non-stock
progs. Turning observations into the fair-senses monster observation of
section 3 is #33, and training a policy on this hook is #35.

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
- **Skill conditioning**: one more input, Quake's own `skill` setting
  (0 easy, 1 normal, 2 hard, 3 nightmare). See "Difficulty" below.

#### Map independence

A trained model must work on any map, including a custom map nobody on this
project has seen, as long as its monsters are ones the model knows how to
control. The player picks a map and it works; nothing is prepared per map.
Four rules keep that true:

1. **Inputs come only from live engine state, relative to the agent.** Ray
   casts from where it stands, its own velocity, what it can currently see
   and hear, its own memory. No absolute coordinates, no map name, no
   per-map embedding, and nothing looked up from a file made for that map.
2. **Per-map artifacts are scaffolding for training and evaluation only.**
   The walk graph in `nav.py` chooses tasks and shapes the reward; the
   policy never sees it, and play never needs it. Feeding the policy a hint
   derived from it (the route's direction, say) would break this rule, even
   though it would make the training numbers better.
3. **Generalisation is measured, not assumed.** A result says something
   about new maps only if it was measured on maps kept out of training.
   Training draws from many maps, and evaluation reports held-out maps as
   their own group (#19).
4. **Unknown content falls back to stock AI.** A monster type with no model,
   or a map whose game code changes a monster we thought we knew, keeps its
   QuakeC brain (#27).

Rule 1 has a cost: with only local senses, finding a route across a level it
has never seen takes memory and exploration, which is why the recurrent core
comes before combat. It also puts weight on what the rays report. Today that
is distance only, so lava looks like floor; the surface kind per ray (#26)
goes in before the first monster is trained.

#### Difficulty

Difficulty follows Quake's `skill` cvar; there is no second difficulty
setting. A trained policy is not limited by the weapons (a Grunt's shotgun
still spreads, a grenade still takes time to fly) but it can react
instantly, track perfectly and lead projectiles exactly, which is neither
fair nor fun. Difficulty therefore comes from three places:

1. **Handicaps outside the model, scaled by skill.** A reaction delay
   before a monster can respond to a newly seen enemy, and aim error added
   to the direction it fires in. They are applied by the engine after the
   policy has chosen, the same way in the simulator and the client, so they
   can be tuned without retraining. Both are seeded, never wall-clock.
2. **Fair senses.** A monster's observation holds only what that monster
   could see or hear: enemies in line of sight, sounds within hearing
   range, and its own memory of where the enemy was last seen. The player's
   position through walls is never an input.
3. **A skill input to the network.** The policy is told the skill level and
   is trained at every level, against opponents of different strength, so
   that easy means simpler play (less flanking, slower to press an
   advantage) and not only a strong monster with shaky aim.

The skill input exists from the first monster trained (the Grunt, M4), next
to the style input. Adding an input later changes the observation layout
and invalidates every checkpoint.

The training objective is not to kill the player as often as possible; that
produces the monster this section exists to avoid. Monsters are rewarded
for being hard to beat at the given skill, and evaluated at each skill
level against the stock QuakeC monster at the same level.

**Implemented so far** (navigation only, `python/deepquake/nav.py`, whose
docstring holds the exact layout): 48 ray distances and the 48 matching
ray surface kinds (what each ray hit: world, moving brush, water, slime,
lava, sky, or nothing), own velocity and
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

The first policy (navigation) is a GRU actor-critic of about 245,000
parameters: one recurrent cell from the observation to a hidden state,
with a categorical head per action group and a value head reading the same
state. The hidden state starts at zero each episode and is carried across
the rollout; PPO updates it through backpropagation through time over
fixed-length chunks of the rollout, replayed from their recorded starting
states. It was trained on 22 `id1` levels and evaluated on 8 it never saw
(#19): the held-out group is where map independence (below) is measured.

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
  expected bottleneck. Workers act for themselves: each holds a CPU copy of
  the policy (one torch thread each) and runs a whole rollout locally, so an
  update costs one weights push and one buffer read, not a round trip per
  decision (#20). The recurrent state lives in the worker, zeroed at episode
  ends, and persists across weight pushes exactly as it did when the learner
  stepped the environments itself.
- Order of work: a **player agent** learns movement and navigation, then
  combat. Monster policies are then trained against a pool of player agents
  (and scripted opponents) so they do not overfit to one opponent.
- Curriculum: empty rooms, then real maps; single monster, then groups.
- Checkpoints are ordinary PyTorch files, which load on CUDA or CPU
  regardless of where they were trained. For the client they will be
  exported to ONNX and run through ONNX Runtime with the CUDA provider when
  a GPU is present, falling back to CPU.
- Measured (same task, 20 workers): 2,500 decisions/s when every decision
  was a pipe round trip plus a GPU forward; 8,300/s with worker-side acting
  and a packed BPTT replay (#20). The limit is now the workers' own stepping
  and inference, not process traffic.

### 6. Distribution

Source in git. Weights as GitHub release assets with a checksum manifest,
fetched into `models/`. Game data never leaves the user's machine.

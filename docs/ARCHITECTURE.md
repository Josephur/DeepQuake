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

**Implemented (#42, #43):** the vendored tree now carries the client side of
upstream too, and one CMake tree builds the headless `libdqsim`, the tests,
and the `deepquake` client executable — the same sources, selected for their
role at compile time (upstream's own `isDedicated` switch gates every client
subsystem; the simulator stubs the client, the client compiles it). The
brain hook lives in `brain_hook.c`, linked by both; in the client,
`brain_onnx.c` nominates a Grunt when the `dq_model` cvar names a model,
runs it through the system ONNX Runtime (CUDA provider, CPU fallback,
measured 78-87 microseconds per decision at the hook's 72 Hz tick), and
maps greedy per-head actions to the same discrete decisions training used.
A verification build (`-DDQSIM_CLIENT_SELFTEST="model,report"`) replays
synthetic hook reports through the client's own observation builder and
model and writes them for comparison against the Python policy — measured
deviations: observation 3.6e-7, probabilities 6.0e-8, identical greedy
actions. Without a model the client is stock Quake.

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
(the Python form is `sim.set_policy_hook`) nominate a monster of the
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
pain and death sequences always run under QuakeC.

**The attack entry (#47).** The entry is the first frame of the class's
attack animation, the function the spawn built-in assigns to the monster's
`th_melee` field and the stock AI runs when it decides to attack: the
Grunt's `army_atk1` (chained to `army_fire`) and the Dog's `dog_atta1`, a
lunge whose chain bites from `dog_bite`. It is found by name — `<prefix>atk1`,
then `<prefix>atta1`, `attack`, `melee1`, and the other shapes the stock
classes use (see below). `<Class>CheckAttack` is *not* an
entry: it is the AI's range classifier, it reads the global `enemy_range`
that only the stock AI sets, and it starts nothing. Because the bite, not
the hook, is what checks the range, a policy that attacks can make the Dog
lunge across a room, but damage lands only at contact distance (measured:
4 bites in 3 s from 50 units, first after 0.4 s, 7-21 damage each; from
300 units the dog charges to 73 and lands its first bite there).
The simulator loads an
optional game directory (`dqsim_config.game`, #37), and the progs gate
compares the level's loaded `progs.dat` against the stock identity read
from id1 at start-up (`dqsim_progs_is_stock`). With no nomination the
simulation is bit-identical (the fidelity test guards this); the
nomination refuses non-monsters and, per the gate above, a non-stock
progs. Turning observations into the fair-senses monster observation of
section 3 is #33, and training a policy on this hook is #35.

**Any class, not just the Grunt and the Dog (#49).** A class does not
necessarily name its QuakeC functions after itself: in the stock id1 program
the Enforcer's states are `enf_stand1`, `enf_run1` and `enf_atk1`, the
Shambler's are `sham_*`, the Hell Knight's `hknight_*`, the Tarbaby's
`tbaby_*`, the Wizard's `wiz_*`, the Shalrath's `shal_*` and the Fish's
`f_*`, while the Grunt's are `army_*`, the Dog's `dog_*`, the Knight's
`knight_*`, the Demon's `demon1_*` and the Zombie's `zombie_*`. So the hook
does not derive a prefix from the classname. It reads the class's prefix off
the monster's own decision think — `<prefix>_stand<n>`, `<prefix>_walk<n>`
or `<prefix>_run<n>`, the states the stock AI schedules — and resolves
everything else from it: the stand and run frames it animates with, the
attack entry and the shot. Every class that stocks an id1 map has that
frame pattern except `monster_oldone` and `monster_boss`, whose states are
`idle`/`thrash` and `idle`/`shock` instead. No class names are in the code.

The attack entry is found by shape, since the classes disagree on it: `atk1`
(the Grunt's `army_atk1`, the Enforcer's `enf_atk1`), `atta1` (the Dog's
`dog_atta1`), `attack`, `attack1` (the Shalrath's `shal_attack1`), `melee`
(the Ogre's `ogre_melee`) and `melee1`. The shape probe is the fallback: the
entries the hook actually runs are the spawn's own `th_melee` and
`th_missile` fields, which the stock AI picks between by range — melee below
120 units, the ranged entry beyond (#53). The ranged entry is the spawn's
`th_missile` only when its chain reaches a shot: the Dog's `th_missile` is a
one-shot pounce (`dog_leap1`), and one entry execution per decision cannot
charge with it, so the Dog keeps the lunge the probe finds at every range.
The shot is found from what the attack chain calls, because the shot is not
always named after the class either: the Grunt's is `army_fire`, the
Enforcer's `enforcer_fire`, and a function whose name contains `fire` or
`shot` is one (the Ogre's grenade launch is `OgreFireGrenade`, which contains
`Fire` without ending in it; the engine's own `SUB_CheckRefire` is excluded).
The chains are walked from the attack entries, so the throw frames the name
scan could never reach (`ogre_nail4`) are found too. Measured on the local
game data, before and after: the Enforcer went from 0 hook calls in 72
ticks (the stock AI kept playing it) to 67, and from no damage under the
policy to 45; the Ogre went from 149 calls and no damage to 65 calls and 30
damage. The Grunt, Dog, Knight, Demon and Zombie are unchanged. Under #53
the Ogre's grenades reach the player at 300 units (a clean throw deals 26),
which the single melee entry could never do.

**The aim error covers every shot of an attack (#49, #53).** The error is
rolled once, when the policy asks for the attack, and applied where the shot
is aimed — the enemy's origin, from the firing frame. A class whose chain
fires more than once (the Enforcer fires at `enf_atk6` and again at
`enf_atk10`) would otherwise keep an exact second shot, and its difficulty
would not scale with `skill`. The error therefore stays pending until the
monster decides again, and every frame that fires is offset. Shots that aim
from the monster's own facing rather than from the enemy's origin — the
Ogre's grenade throws with `makevectors(self.angles)` and only takes its
lead from the enemy — cannot be bent by displacing the enemy, so the
monster's facing is rolled by the same error too, and restored after the
frame (#53). The stock AI's sight globals (`enemy_range`, `enemy_vis`,
`enemy_infront`) are set before the attack runs, since the chains read them
(the grenade's lead comes from `enemy_range`) and the hook replaced the
think that would have set them. Measured: with the aim error overridden to
45 degrees the Enforcer's shots all miss (0 damage in 3 s), and with no
error the same shot hits; the Ogre's throw window hashes differ with and
without the same override, which is the flight-path difference the sim's
entity copy cannot show.

**A whole level at once (#46, M6).** The hook keeps a fixed table of 64
nominations, each with its own callback, user pointer, QuakeC names, animation
clock, skill and handicap state; the state hash and snapshots carry them all
in nomination order, and a restore is refused when the set of nominations
differs. The play client nominates every monster of a class that a loaded
model covers (`<model>.monsters`, falling back to the Grunt for the sidecar-less
models-v1 export), each with its own recurrent state, last-seen memory and
facing, capped by `dq_max_monsters` (0 = all). Measured on e1m1 at 72 Hz: 34
Grunts driven at 35-40 microseconds per decision with the CUDA provider (41
with CPU), about 1.3 ms per tick inside a 13.9 ms frame, so per-monster calls
stay and batched inference waits for a measurement that needs it. id1 gates
monster counts by skill (e1m1 stocks 18 Grunts at skill 1 and 34 at skill 2),
which is why the console line, not the map, says how many are driven.

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

**Implemented for the monster (#33)** (`python/deepquake/monster.py`,
whose docstring holds the exact layout): the brain hook's ray fan and
surface kinds and the monster's own velocity, ground and water state,
health and QuakeC attack cooldown come first; then the enemy's bearing,
height and distance and its health — only while the hook sees it — and a
last-seen memory (bearing, height, distance, age) that fades after five
seconds; finally `skill / 3` and the style vector (aggression, caution,
stealth). Nothing in the vector is absolute, per-map, or visible through
walls: a test places the player behind a wall on an id map and on the
generated map and asserts the enemy channels read "not visible".

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
   **Implemented (#34)** for policy-driven monsters in
   `engine/src/handicap.c` (a unit the client will compile unchanged):
   the schedule is 36/18/9/0 ticks of reaction delay and 12°/6°/3°/0° of
   aim error for skill 0–3 (`dqsim_handicap_for_skill`). During the delay
   the callback still runs but its action is not applied; the aim error
   is rolled from the seeded stream once per attack and applied where
   the QuakeC attack actually aims — at its fire frame, from the enemy's
   position — and pending handicap state is part of snapshots and the
   state hash. Nightmare is unhandicapped because stock QuakeC already
   refires almost immediately there.
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

**Implemented (#48, #50, #53, #55)** for the Grunt, the Dog, the Enforcer,
the Ogre and the Hell Knight, with
three simplifications of the design above:

- **The type is a one-hot, appended to the observation**
  (`deepquake.monster.MONSTER_TYPES = (monster_army, monster_dog,
  monster_enforcer, monster_ogre, monster_hell_knight)`, so
  `MONSTER_OBS_DIM` is 119 + 5 = 124). The recurrent
  cell's input weights play the role of the embedding, and the model keeps a
  single observation input, which is what the client passes.
- **A policy head per type, one shared value head.** The types have
  the same action layout today, so the heads differ only in weights.
  The head is selected by multiplying each head's logits by the
  observation's one-hot and summing, not by an index gather, so the ONNX
  graph has no second input derived from the observation.
- **Inference is per monster, not batched.** The client runs one session
  for every Grunt, Dog and Enforcer in the level, but one inference call per
  decision, each monster with its own recurrent state and type index.
  Batching all monsters into one call per tick is still to do.

A new type **warm-starts** from the previous roster: `ActorCritic.init_from`
copies the source's recurrent cell (with zeroed input columns for the new
one-hot entries), each source head into the head of the same index — the
last source head when the source had fewer types — and its value head, so
the shared model starts out playing exactly like the model it came from (a
unit test checks actions, log-probabilities, value and hidden state are
identical for the carried-over one-hots). Training then samples the body per
episode, so all the types train together. The exported model ships a class
map, `<model>.onnx.monsters` (`classname type_index` per line), which tells
the client which classes it drives; a 119-wide v1 model without one still
loads as the Grunt policy.

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

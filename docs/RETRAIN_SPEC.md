# Monster retraining spec

This document says how to retrain the monster models so that the policy adds
something the engine glue does not already do. It is written for a session
that remembers nothing about how it came to be. Follow it literally. Where it
gives a number, use that number. Where it says "stop and ask", stop: write a
Checkpoint, label the issue `needs-human`, and wait.

`AGENTS.md` applies throughout, and above all its section "Training a model".
Nothing here relaxes it. Where this spec is stricter, this spec applies.

## Why this exists

The models released as models-v8 to v11 looked good on win rates and were
worthless. Issue #65 measured it: on a player standing still in plain view,
the released roster11 model, a scripted "turn to the player and hold attack"
reference, and uniformly random button presses all kill at the same rate,
class by class. On a player hidden 200 to 600 units away, all three kill in 0
to 20% of tasks.

The reason is the engine glue in `engine/src/brain_hook.c`. Whenever the
policy presses attack with the player in view, the glue turns the monster
toward the player, charges a melee monster into reach (and leaps where the
class has a leap), and runs the stock QuakeC attack. So in a visible fight
the policy decides almost nothing, and the old training gave it nothing else
to learn.

## What the maintainer decided

These decisions are binding. Do not change them, and do not reinterpret them.

1. **What monsters hear:** gunfire, explosions, and other monsters' pain and
   death cries. Hearing everything the player does (jumping, landing, item
   pickups) is a later option that must be switchable. Hearing as a whole
   must be switchable on and off, and its level is tied to the game's
   difficulty setting (`skill`).
2. **Hearing range:** adjustable, default 1000 units (the distance at which
   Quake stops playing a normal-attenuation sound to a listener). Sound passes
   through walls.
3. **What a noise tells a monster:** where the noise came from, not where the
   player is. The monster investigates that spot.
4. **Pace after hearing a noise:** chosen at random, once per alert, between
   running and an "investigation" pace slower than a run. Legs must not slide.
   The random choice is seeded and deterministic.
5. **Skills are taught one at a time**, in this order: hearing and hunting an
   unseen player; retreating when hurt; using cover; flanking; group play.
   Each stage has its own pass/fail test, and the next stage does not start
   until the previous one has passed. Group play is the hardest and least
   certain. It is its own last stage and is not promised by any earlier one.
6. **The glue keeps doing the basic fight** (face a visible player, charge,
   stock attack). The policy's job is everything around it.

Two consequences the maintainer asked for, which this spec makes true by
design:

- Hearing on/off, what is heard, and the range are **settings of the engine
  glue** (cvars in the play client, parameters in the simulator and the
  environment). They are not inputs to the model. The model only receives "a
  noise was heard at this relative position, this long ago" (plus the pace the
  glue chose). Training samples a spread of those settings so that they can be
  changed after training without retraining (values in "Hearing settings
  sampled in training" under Stage 1).
- The mapping from `skill` to hearing settings is a small table (Stage 0,
  part 0B), and the maintainer can override every entry by cvar.

## How to use this spec

1. Always work the **lowest unfinished stage**. A stage is finished only when
   its tracking issue is closed. Never start a part of stage N+1 while stage
   N is open.
2. Each stage has **one tracking issue**, titled `Retrain stage N: <name>`.
   Stages that list parts (0A, 0B, ... and "code" and "training" parts in
   Stages 1-5) get **one issue per part** as well, titled
   `Retrain stage <part>: <what>`, because AGENTS.md wants issues small enough
   to finish in a few hours. Each part issue has its own branch and its own
   squash merge to `main`. The tracking issue lists the part issues as a
   checklist and is closed last, after the maintainer's sign-off (see "The
   go-ahead and the sign-off").
3. Use milestone `M6 Full roster` and the labels given in "How to start".
4. When the maintainer has asked for the retraining, the stage issues are the
   work. The other open issues (#38, #54, #63, #65) are not worked on
   separately: #63 (what the policy should add) and #65 (a check that cannot
   tell a model from random) are answered by Stages 0 and 1. Comment on both
   when Stage 1 closes, linking its closing comment; the maintainer closes
   them.
5. The player navigation model and the player combat model are not part of
   this work. Do not retrain them, do not change `nav.py`'s observation, and
   do not change `train_nav.py` or `eval_nav.py`.

---

## Rules that apply to every stage

### Never change these to make something pass

- A test, a threshold, a probe, its seed, its maps, its task count, its time
  limit, or a scripted reference. This includes every constant in
  `python/deepquake/fight_check.py`. The only changes this spec allows to
  `fight_check.py` are listed per stage under "Changes to `fight_check.py`".
  A change not on that list is forbidden even if it looks like a fix. If you
  believe a probe is wrong, write why on the issue, label it `needs-human`,
  and stop.
- `fight_check.json`: never written or edited by hand, never copied from
  another run.
- A config's reward terms, coefficients, tasks, opponents, maps, skills,
  hearing spread or `total_steps`, once the maintainer has given the go-ahead
  for it. A run with any difference is a different run: new name, new
  go-ahead.
- The observation and action layout, except in part 0D. If a later stage
  seems to need a new input, stop and ask.
- The decisions above.

Updating a test that asserts the observation layout (for example the
`(128,)` shape check in `python/tests/test_monster.py`) to the new layout in
part 0D is required and is not weakening a test. Replace each such assertion
with the equivalent for the new table; never delete one.

### The two-bar rule

Every stage from 1 on adds a **probe** to `fight_check.py`: fixed tasks
(maps, distances, seed, time limit, task count) on which the stage's skill
shows. On each probe the checkpoint is measured next to two bars, played in
the same bodies on the same tasks:

- **random**: uniform random actions (`MONSTER_ACTION_NVEC`), seeded with the
  probe's seed;
- **script**: the stage's simple scripted policy, defined exactly in the
  stage section.

The checkpoint passes a probe only if it beats **both** bars by the margins
the stage states. The bars are measured **before** the training run starts,
with `fight_check --bars` (added in part 0E), and pasted into the stage's
issue. The probe is valid only if, in that measurement, the script beats
random by the stage's stated gate and leaves headroom. If it does not, the
probe cannot measure the skill: stop and ask.

Probes accumulate. A Stage N checkpoint must pass the sight probe and every
probe of Stages 1 to N. A later stage never relaxes an earlier probe.

Numbers marked **(proposed)** are this spec's first guesses. They become
binding when the stage's code part merges them into `fight_check.py`, after
the bars have been measured and recorded, and before the go-ahead. They are
never changed after a training run has started.

### The go-ahead and the sign-off

Agents and the maintainer post to GitHub under the same account, so a
comment's author does not tell you who wrote it. The rule is therefore about
wording, and it relies on you:

- **Go-ahead.** A training run (the smoke run included) starts only after the
  stage's issue has a comment whose first line is exactly
  `Go-ahead: <run name>`, written by the maintainer. An agent never writes a
  line that starts with `Go-ahead:`. If the maintainer gives the go-ahead in
  chat instead, quote their words on the issue under a first line
  `Maintainer go-ahead (chat, <date>): <run name>`.
- **Sign-off.** A stage is finished only after the maintainer has played the
  result and said so, in a comment whose first line is
  `Played: <run name> OK`, or in chat, quoted on the issue as above. An agent
  never writes a line that starts with `Played:`. For Stage 0, which has no
  model to play, the sign-off is `Reviewed: stage 0 OK`.
- Before asking for a go-ahead, the issue must already show: the config file
  (committed on `main`), the bars measured on the probe, and the expected
  steps per second. Ask in a comment that ends with the exact line the
  maintainer should write. Then label the issue `needs-human`, remove
  `in-progress`, and stop the session if no other stage work is open.

### Running a training run

Do these in order, every time.

1. `main` is up to date and every check under "Build, test, check" in
   AGENTS.md passes on it, including the game-data tests and
   `ctest --preset release`. Never train from a branch.
2. Check the GPU is free:
   `nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv`.
   Paste the output in the issue. If the RTX 3090 shows more than 10%
   utilization before you start, something else is using it (#54): stop and
   ask.
3. Smoke run. Copy the config to `configs/<name>_smoke.toml` and change only
   `name` to `"<name>_smoke"`. Both files are committed on the training
   part's branch and squash-merged to `main` before you ask for the go-ahead.
   On `main`, run:

   ```bash
   nohup uv run python -m deepquake.train_monster configs/<name>_smoke.toml --updates 10 > runs/<name>_smoke.log 2>&1 &
   ```

   When it ends, check: `runs/<name>_smoke/progress.csv` has 10 data rows;
   the last `steps/s` figure in `runs/<name>_smoke.log` is at least half the
   figure the stage expects; the log has no `Traceback`, no `nan`, and no
   `WARNING` line. Then run
   `uv run python -m deepquake.fight_check runs/<name>_smoke/checkpoint.pt`
   once. It is expected to fail; it must run to the end and write
   `runs/<name>_smoke/fight_check.json`. Paste both results.
4. The real run:

   ```bash
   nohup uv run python -m deepquake.train_monster configs/<name>.toml > runs/<name>.log 2>&1 &
   ```

   Write a Checkpoint naming the run before doing anything else.
5. While it runs, do nothing that loads the CPU or GPU heavily. Do not start
   a second run. Do not kill a run you did not start.
6. Stop and write a Checkpoint if the log shows an error, NaN losses, or
   steps per second below half the expected figure.
7. When it ends: `uv run python -m deepquake.fight_check runs/<name>/checkpoint.pt`.
   Paste the whole printed table into the issue, pass or fail.

### What to report in the issue

For every training run, in one comment:

- the run name, the config path, the `main` commit it ran on;
- total steps, wall time, the steps per second at the start and at the end;
- the bars measured before the run (link the comment);
- the whole `fight_check` table and its verdict;
- for each probe: the model's numbers next to random and script, and the
  sentence "The model beats random by X and the script by Y on <probe>" with
  the real numbers;
- if the model's numbers are within the stage's margin of the random column
  on the stage's probe, the sentence **"This model matches the random column:
  it has learned nothing this probe can see."** Do not release it;
- which probe maps are training maps and which are held out (the report says
  so per class); claim nothing about new maps from training-map numbers.

### Stop and ask when

Write a Checkpoint, label the issue `needs-human` (and `blocked` where no
other part can proceed), and wait, when any of these happens:

1. A training run would start without a go-ahead in the form above.
2. The bars measurement shows the script does not beat random by the stage's
   gate, or leaves less headroom than the stage requires.
3. On a probe map the `idle` control kills or damages the player above the
   limits in "The idle control" (part 0E), on both maps listed for a class.
4. You want to change `fight_check.py` in any way the stage does not list.
5. You want to change anything in a config after its go-ahead, or a reward
   term, coefficient, task, opponent, map list or step budget before it.
6. A model fails a probe, or matches the random column. Report and wait; do
   not retrain with changed settings on your own, and do not `--resume` or
   `init_from` a checkpoint that failed.
7. A training run shows an error, NaN, or less than half the expected steps
   per second.
8. The same step has failed three times with different fixes (AGENTS.md).
9. A change to a vendored file in `engine/third_party/` looks necessary.
10. For any roster class, the investigation speed (half the run speed) is not
    at least 1.2 times its walk speed (part 0B, `class_speeds` ctest).
11. A hearing test shows a sound the categories in part 0B do not describe
    (a sound heard that should not be, or not heard that should be).
12. The Scrag cannot be made to move, attack, and take the aim error under the
    hook (part 0C).
13. The observation needs anything not in the table of part 0D.
14. Neither listed map makes the Fiend's or the Knight's sight-probe
    reference measurable (part 0E).
15. The maintainer's play test reports a problem. Do not close the stage.
16. Stage 5's multi-monster environment design (it needs the maintainer's
    approval before any code; see Stage 5).
17. Anything in this spec contradicts the code in a way you cannot resolve by
    reading the code. Say what and where.

---

## Facts this spec builds on

Checked against the code at the commit that added this file. If the code has
changed since, read it again before acting on a detail.

- `engine/third_party/quakespasm/` is never modified. Engine behaviour is
  changed in `engine/src/`, or by compiling a vendored file with a renamed
  symbol: `sv_phys.c` is compiled as its own object library
  (`quakespasm_phys` in `engine/CMakeLists.txt`) with
  `PR_ExecuteProgram=dqsim_hook_execute_program`. In the client target the
  same define is set on `sv_phys.c` with `set_source_files_properties`.
- QuakeC `sound()` is `PF_sound` in `pr_cmds.c`, which calls `SV_StartSound`
  (defined in `sv_main.c`). `sv_phys.c` also calls `SV_StartSound` directly
  (water splashes, the Fiend's landing). `pr_cmds.c` is compiled inside the
  `quakespasm_server` object library today.
- **Explosions do not go through `SV_StartSound`.** QuakeC writes a temp
  entity (`svc_temp_entity`, 23, then `TE_EXPLOSION`, 3, or
  `TE_TAREXPLOSION`, 4, then three coordinates) to `MSG_BROADCAST`, which is
  `sv.datagram`; the client plays the explosion sound itself
  (`cl_tent.c`). `PF_WriteByte` and `PF_WriteCoord` in `pr_cmds.c` call
  `MSG_WriteByte (sizebuf_t *, int)` and
  `MSG_WriteCoord (sizebuf_t *, float, unsigned int)`.
- `pr_xfunction` (declared in `progs.h`) is the QuakeC function executing
  when a builtin runs.
- A driven monster's speed: `DQ_STATE_INTERVAL` (0.1 s) and the distances
  its walk and run chains pass to `ai_walk` and `ai_run` give `walk_speed`
  and `run_speed` in `dq_hook_slot`. In `hook_apply`,
  `running = ent->v.enemy || slot->walk_len == 0 || slot->walk_speed <= 0`,
  and `body_speed` is `run_speed` or `walk_speed`. A monster gets an enemy
  when the hook sees a live player, or when QuakeC's damage code sets it. The
  animation advances one frame every 7 ticks (`slot->anim_tick / 7`) through
  `walk_seq`, `run_seq` or `stand_seq`.
- **The hook's animation clock is not in snapshots or the state hash.**
  `dqsim_state_hash` and the snapshot header in `engine/src/dqsim.c` carry
  only the handicap states (`dq_brain_hook_handicaps`), not `anim_tick`.
- The policy's facing is kept in the edict: `v_angle[YAW]` is the policy's
  facing, `angles[YAW]` the shown facing, `v_angle[PITCH]` the shown facing
  as the hook last wrote it (comment at `DQ_SHOWN_TURN_RATE`). `policy_yaw()`
  adopts a turn QuakeC made. The observation's `angles[1]` is the policy's
  facing at that moment.
- Both callers accumulate an absolute desired yaw and never resync it:
  `MonsterEnv.monster_yaw` (`python/deepquake/monster_env.py`, `step`) and
  `dq_monster_state.yaw` (`engine/src/brain_onnx.c`, `onnx_policy`). After an
  attack in which QuakeC turned the monster, the hook turns it back to the
  stale heading. `fight_check.reference_action` reads `env.monster_yaw` too,
  so the reference suffers the same defect.
- The play client builds the observation in C (`dq_build_observation` in
  `brain_onnx.c`) with a fixed style of 0.25, 0.5, 0.75. A `dq_style` cvar is
  registered but never read.
- In training, styles were sampled but no reward term reads them, so the
  style inputs never meant anything.
- The observation (`python/deepquake/monster.py`) encodes own health as
  `health / 100` clipped to 1: no signal for a class with more than 100
  health until it is badly hurt (an Ogre has 200, a Shambler 600).
- The hook's `visible` list already contains visible monsters
  (`FL_MONSTER`), but `monster.py` ignores them.
- `train_monster.make_env` does not pass `placement` or `others_passive`, so
  every past training run had the level's other monsters active, and "any
  monster's kill counts". The roster11 log shows training win rates of 0.75
  to 0.91.
- The Scrag (`monster_wizard`) does not attack under the hook and is not in
  `MONSTER_TYPES`. `entry_is_ranged` cannot follow its dispatcher
  `Wiz_Missile` (comment at `leap_think` in `dq_brain_hook_nominate`).
- `fight_check.py` today: one probe (player standing still in sight, 150 to
  350 units, 20 usable tasks per class, skill 2, seed 2026, 360 decisions,
  other monsters passive), verdict from `REFERENCE_FLOOR`, `KILL_FRACTION`,
  `FACING_DEGREES`, `FACING_FRACTION`; the random column is printed but not
  judged. `release_model.py` refuses a monster model without a passing report
  for that exact checkpoint file.
- Measured (#61, #65, #63, roster11):
  - a Grunt walks at 18.3 units/s and runs at 108.8; a Dog runs at about 330;
  - hidden standing player, 200-600 units, 50 s: every brain kills in 0-20%
    of tasks;
  - on e1m8 the player takes damage from the level itself (60% "damage" with
    no monster in sight), so e1m8 is not clean for kill-rate probes;
  - the Fiend's reference kills in 40% of tasks on e2m5; the Knight's
    reference faces the player on 65-66% of decisions on e1m4;
  - roster11 trained at about 2,720 steps/s with 19 workers on the RTX 3090.
- Machine: train on the RTX 3090 chosen by name (`device = "3090"`); #54
  (throughput halved while another service shares the GPU) is open and
  `needs-human`. The full pytest takes about 2 minutes warm and about 15
  after an engine change; walk graphs are cached in `runs/walk_graphs/`.

---

## Stage 0: groundwork (no training)

**Goal.** Fix the defects that shaped the old models, add hearing to the
engine glue with its settings, tests and skill table, and change the
observation layout once, for every later stage.

Stage 0 has five parts, each its own issue and branch, done in this order:
0A, 0B, 0C, 0D, 0E. No training run of any kind happens in Stage 0.

### 0A. The facing is not turned back after an attack

Labels: `bug`, `area:python`, `area:engine`.

The callers must stop accumulating an absolute yaw. Instead, the turn of a
decision is spread over its six ticks, each tick measured from the facing the
hook reports, so a turn made by QuakeC is kept automatically.

1. Write the test first, and run it on unchanged code. It must fail there;
   paste its output in the issue.
   `python/tests/test_monster.py::test_the_facing_is_not_turned_back_after_an_attack`
   (new, marked `gamedata`, runs the simulator in a subprocess like the other
   tests in that file):
   - `MonsterEnv("e1m1", seed=2026, monster="monster_army", placement="sight",
     opponent=PlayerOpponent("still"), others_passive=True, min_distance=150,
     max_distance=350, timeout_steps=360, skills=(2,))`, with
     `sim.set_handicap_override(0, 0.0)` after each `set_task`;
   - take the first task from `make_tasks(env, 200, 2026)` whose start has the
     player in the hook's sight **and** a bearing at least 60 degrees from the
     monster's reported facing;
   - every decision: action `[1, 1, 3, 1]` (no move, no strafe, turn 0 — index
     3 of `MONSTER_TURN_DEGREES` — attack);
   - from the decision after the first attack started (`attack_cooldown`
     becomes positive) until the episode ends, count decisions with the player
     in sight, and those whose shown facing (`Entity.yaw` of the monster) is
     within 45 degrees of the bearing to the player;
   - assert the second count is at least 90% of the first.
2. `python/deepquake/monster_env.py`:
   - `_Held` holds `turn: float` (degrees for the whole decision) instead of
     `yaw`;
   - `_hook_policy(hook)` returns
     `desired_yaw = hook.angles[1] + self._held.turn / nav.FRAME_SKIP`;
   - `step()` sets `turn = MONSTER_TURN_DEGREES[turn_index]` and no longer
     updates an accumulator;
   - `monster_yaw` becomes a read-only property returning
     `self._hook_obs.angles[1]` (0.0 before the first report), so
     `fight_check.reference_action` keeps working without a change;
   - update the module docstring ("turn ... accumulated into the facing the
     hook wants" becomes "turn by this many degrees over the decision, from
     the facing the hook reports").
3. `engine/src/brain_onnx.c`: remove `dq_monster_state.yaw` and its
   accumulation in `onnx_policy`; add `float held_turn`; on a decision tick
   set `held_turn = model->turn[actions[2]]`; on every tick (decision and
   held) set `action->desired_yaw = hook->angles[1] + state->held_turn /
   DQ_FRAME_SKIP`. In `nominate_pass` drop the `state->yaw = ent->v.angles[YAW]`
   line. In `dq_selftest_run` drop `state.yaw`.
4. Why the split is safe: the hook turns at most `DQ_MONSTER_TURN_RATE` (360
   degrees/s) times the tick, 5 degrees at 72 Hz; the largest turn, 30
   degrees over 6 ticks, is 5 per tick. Write this in the comment.
5. Update `docs/ARCHITECTURE.md` section 2 ("Implemented (#31)" paragraph):
   the policy's turn is relative to its reported facing, so turns made by
   QuakeC attack states are kept.

**Done when**

- [ ] The new test failed on the old code (output in the issue) and passes.
- [ ] The release client builds (`cmake --build --preset release`), and
      `grep -n "state->yaw" engine/src/brain_onnx.c` prints nothing.
- [ ] All checks under "Build, test, check" pass, game-data tests included.

### 0B. Hearing in the engine glue

Labels: `feature`, `area:engine`, `area:python`.

#### How sounds are caught

Try this route first; it changes no vendored file.

1. Move `${QS_DIR}/pr_cmds.c` out of `quakespasm_server` into a new object
   library `quakespasm_cmds`, built exactly like `quakespasm_phys`
   (same includes, `USE_SDL2`, `rand=dqsim_rand`, `-w -fno-strict-aliasing`,
   `C_EXTENSIONS ON`) with three more definitions:
   `SV_StartSound=dq_hear_start_sound`, `MSG_WriteByte=dq_hear_write_byte`,
   `MSG_WriteCoord=dq_hear_write_coord`. Add
   `$<TARGET_OBJECTS:quakespasm_cmds>` to `dqsim`.
2. Add `SV_StartSound=dq_hear_start_sound` to `quakespasm_phys`'s definitions.
3. In the client target, inside the existing `if(DQSIM_ONNXRUNTIME_FOUND)`
   block: append `SV_StartSound=dq_hear_start_sound` to `sv_phys.c`'s
   `COMPILE_DEFINITIONS`, and set `pr_cmds.c`'s `COMPILE_DEFINITIONS` to the
   three definitions of step 1. Without ONNX Runtime the client has no hook
   and no renames.
4. In `engine/src/brain_hook.c` (new functions, compiled without the renames,
   so their calls reach the real ones):
   - `void dq_hear_start_sound(edict_t *entity, int channel, const char
     *sample, int volume, float attenuation)`: classify (below), deliver,
     then call the real `SV_StartSound` with the same arguments;
   - `void dq_hear_write_byte(sizebuf_t *sb, int c)` and
     `void dq_hear_write_coord(sizebuf_t *sb, float f, unsigned int flags)`:
     call the real function, and run a small state machine that only looks at
     writes to `&sv.datagram`: byte 23 then byte 3 or 4 then three coordinates
     is an explosion at those coordinates; any other write to `sv.datagram`
     in between resets it.
5. If this route cannot work, apply AGENTS.md "Conventions" in order (stub,
   shim, compile definition). Editing a vendored file is stop-and-ask
   condition 9.

The wrappers must not change anything when no monster is nominated: no
random draws, no writes. `dqsim.fidelity` and
`python/tests/test_hook.py::test_clear_policy_hook_restores_stock_hash` must
still pass unchanged.

#### What is heard (the categories)

A sound is a **noise** if it falls in one of these categories. Everything
else (idle and sight sounds, doors, lifts, ambient sounds, splashes) is
ignored.

| Kind (new constant in `dqsim.h`) | Rule | Origin |
| --- | --- | --- |
| `DQSIM_NOISE_GUNFIRE` 1 | `SV_StartSound` with `(channel & 7) == 1` (`CHAN_WEAPON`) from an emitter with `FL_CLIENT` or `FL_MONSTER` | the emitter's origin |
| `DQSIM_NOISE_EXPLOSION` 2 | the temp-entity explosion write described above | the written coordinates |
| `DQSIM_NOISE_CRY` 3 | `SV_StartSound` with `(channel & 7) == 2` (`CHAN_VOICE`) from an `FL_MONSTER` emitter, when the emitter's health is 0 or less, or when the executing function (`pr_xfunction - pr_functions`) is the emitter's own `th_pain` field | the emitter's origin |
| `DQSIM_NOISE_PLAYER` 4 | only when player noises are on: any other `SV_StartSound` from an `FL_CLIENT` emitter | the emitter's origin |

Gunfire includes the monsters' own weapons, not only the player's: a Grunt
firing at the player is a noise its allies hear. A monster never hears its own
sounds (emitter is the listening monster).

#### Who hears it, and what is kept

For every nominated, live monster other than the emitter, while hearing is
on: if the straight-line 3D distance from the noise origin to the monster's
origin is at most the monster's hearing range, the monster hears it. There is
no line-of-sight test (sound passes through walls) and attenuation is
ignored.

Each nominated monster keeps one heard noise: the most recent one. Two noises
in the same tick: the nearer wins. The heard noise is remembered for
`DQSIM_HOOK_HEARD_SECONDS` = 10 seconds (new constant in `dqsim.h`), then
forgotten.

#### The alert and the pace

- A monster with an enemy runs, as today.
- A monster without an enemy that hears a noise while it has no remembered
  noise starts an **alert**: draw `dqsim_rand() / (double)RAND_MAX` once; below
  the run chance the alert's pace is `DQSIM_PACE_RUN` (2), otherwise
  `DQSIM_PACE_INVESTIGATE` (1). A noise heard during an alert refreshes the
  heard noise and its time but keeps the pace. When the remembered noise is
  forgotten, the alert ends and the pace is `DQSIM_PACE_WALK` (0).
- Speeds: walk = `walk_speed`; run = `run_speed`; **investigate =
  0.5 x `run_speed`**. A class with no walk chain (`walk_len == 0` or
  `walk_speed <= 0`) runs at every pace, as today.
- Animation while moving: walk and run keep today's 7-tick clock and frames.
  **Investigate plays the walk frames, advanced by distance travelled:** a
  float phase advances by `moved_distance / (walk_speed * DQ_STATE_INTERVAL)`
  per tick and the frame is `walk_seq[(unsigned)phase % walk_len]`. A stock
  walk covers `walk_speed * 0.1` units per frame; the investigation pace covers
  the same, so the feet keep pace with the ground.

#### Settings and the skill table

Settings are process-wide, apply to every nomination, and survive map loads.

New in `engine/include/dqsim/dqsim.h` (and `dqsim.c`):

```c
typedef struct dqsim_hearing {
    int32_t enabled;       /* 0: no monster hears anything; 1: hearing on (default) */
    float range;           /* units; 0 or less: the skill table's (default 0) */
    float run_chance;      /* 0-1: chance an alert is run, not investigated;
                              below 0: the skill table's (default -1) */
    int32_t player_noises; /* 1: also hear every sound the player makes (default 0) */
} dqsim_hearing;

DQSIM_API int dqsim_set_hearing(const dqsim_hearing *settings);
DQSIM_API int dqsim_hearing_for_skill(int skill, dqsim_hearing *out);
```

Internally `brain_hook.c` gets `dq_brain_set_hearing(...)` and
`dq_brain_hearing_for_skill(...)`, declared in `brain_hook.h`, which
`dqsim.c` wraps and the client calls directly. The skill table lives in
`brain_hook.c` (proposed; the maintainer can override every entry by cvar):

| `skill` | range (units) | run chance |
| --- | --- | --- |
| 0 (easy) | 500 | 0.25 |
| 1 (normal) | 1000 | 0.5 |
| 2 (hard) | 1000 | 0.5 |
| 3 (nightmare) | 1500 | 0.75 |

The skill used is the slot's own (`slot->skill`, set at nomination).

Python (`python/deepquake/sim.py`, new): `set_hearing(*, enabled: bool = True,
range: float = 0.0, run_chance: float = -1.0, player_noises: bool = False)
-> None` and `hearing_for_skill(skill: int) -> tuple[float, float]` (range,
run chance), plus `HOOK_HEARD_SECONDS = 10.0`, `PACE_WALK = 0`,
`PACE_INVESTIGATE = 1`, `PACE_RUN = 2`, `NOISE_GUNFIRE = 1`,
`NOISE_EXPLOSION = 2`, `NOISE_CRY = 3`, `NOISE_PLAYER = 4`.

#### New observation fields

Append to `dqsim_hook_observation`, after `attack_cooldown`, and mirror in
`_HookObservation` and `HookObservation` in `sim.py` (same order; Python
field names identical):

```c
    float start_health;      /* health when nominated */
    int32_t heard;           /* 1 while a heard noise is remembered */
    int32_t heard_kind;      /* DQSIM_NOISE_*; for tests and logs, never a policy input */
    float heard_relative[3]; /* where the noise was, minus the monster's origin */
    float heard_age;         /* seconds since it was heard */
    int32_t pace;            /* DQSIM_PACE_*: the pace the hook moves the monster at now */
```

`start_health` is the monster's health at `dq_brain_hook_nominate`, kept in
the slot.

#### State that must survive snapshots

Add a struct `dq_slot_state` to `engine/src/dq_internal.h` (plain types only,
like `dq_handicap_state`) holding everything the hook changes per monster
while it runs: the animation clock (`anim_tick`), the investigation phase, the
heard noise (valid flag, origin, time, kind), and the alert pace. Keep it in
`dq_hook_slot`. Add `dq_brain_hook_slot_states(dq_slot_state *out, size_t
max)` and `dq_brain_set_slot_states(const dq_slot_state *in, size_t count)`
to `brain_hook.h`, mirroring the handicap pair, and carry them in
`dqsim_state_hash` and in the snapshot header next to the handicaps. This
also closes the gap that the animation clock is not carried today.

#### Tests (all in `engine/tests/test_hook.c` unless named otherwise)

Each new ctest uses the existing helpers (`driven_start`, `place_player_ahead`,
`driven_entity`) and records the observation fields from its callback. Use
`dqsim_set_handicap_override(0, 0.0f)` where a reaction delay would get in the
way.

1. `hearing_gunfire`: e1m1, a driven Grunt; the player placed 300 to 900 units
   away with line of sight broken; the player fires once. Within 2 ticks the
   Grunt reports `heard == 1`, `heard_kind == DQSIM_NOISE_GUNFIRE`, and
   `heard_relative` within 2 units of the player's origin at the shot minus the
   Grunt's origin. The player then moves 200 units: the heard position does not
   follow (decision 3).
2. `hearing_range`: range 500 set by `dqsim_set_hearing`; a shot from 400 units
   away is heard, one from 700 is not; both with line of sight broken.
3. `hearing_switch`: `enabled = 0`: a shot from 300 units is not heard; back to
   1: it is.
4. `hearing_explosion`: the player gets all weapons (`impulse 9`) and the
   rocket launcher (`impulse 7`) and fires at a wall at least 250 units away
   from itself, within range of the Grunt. The Grunt's last heard noise is
   `DQSIM_NOISE_EXPLOSION`, at least 200 units from the player's position.
5. `hearing_cries`: a driven Grunt and a stock Grunt out of its sight but in
   range; the player shoots the stock Grunt until it dies. The driven Grunt
   hears `DQSIM_NOISE_CRY` with `heard_relative` within 2 units of the stock
   Grunt's position, at least once while the stock Grunt is alive (pain) and
   once on its death. Record in the issue which sound kinds were heard and
   when.
6. `hearing_player_noises`: with `player_noises = 0`, the player jumping
   (`DQSIM_BUTTON_JUMP`) 200 units away is not heard; with 1 it is, as
   `DQSIM_NOISE_PLAYER`.
7. `investigate_pace` (modelled on `body_speed`): run chance 0 (always
   investigate); a shot alerts the Grunt without it seeing the player; drive
   it forward 144 ticks. Every moving tick shows a walk frame (90-113); the
   speed is within 10% of 0.5 x the measured run speed; the number of frame
   changes per 100 units travelled is within 15% of the same figure measured
   in stock walk (phase 0 of `body_speed`); `pace == DQSIM_PACE_INVESTIGATE`.
   With run chance 1 the same alert gives run frames (73-80) and
   `pace == DQSIM_PACE_RUN`.
8. `alert_pace_seeded`: run chance 0.5; 20 alerts in a row (alert, let the
   noise be forgotten, alert again) give at least one of each pace; the same
   load seed gives the same sequence of paces twice.
9. `hearing_snapshot_replay`: a driven Grunt; the player fires, so an alert is
   rolled; save a snapshot; step 144 ticks recording the state hash every
   tick, with a second shot in the window; restore; replay the same inputs;
   every hash matches. Then repeat the 144 ticks from the snapshot with hearing
   off: the hashes must differ after the second shot (the hash covers the
   hearing state).
10. `class_speeds`: for each roster class on the map `animation_frames` uses
    for it, print walk, run and investigation speeds measured from movement,
    and check `0.5 * run >= 1.2 * walk`. Paste the printed table in the issue.
    A class that fails is stop-and-ask condition 10.
11. `engine/tests/test_handicap.c`: `dqsim_hearing_for_skill` returns the skill
    table above for 0-3 and fails for 4.
12. `python/tests/test_hook.py::test_hearing_settings_round_trip` (new):
    `sim.hearing_for_skill` matches the table; `sim.set_hearing` accepts the
    defaults.

Update `docs/ARCHITECTURE.md` section 2 with a "Hearing" paragraph (route,
categories, range, alert and pace, settings, skill table) and section 3's
"Fair senses" note.

**Done when**

- [ ] Tests 1-12 exist and pass; their printed figures are in the issue.
- [ ] `dqsim.fidelity` and every existing ctest and pytest pass unchanged.
- [ ] `python3 tools/check_repo_hygiene.py` passes (no vendored file changed).
- [ ] All checks under "Build, test, check" pass, game-data tests included.

### 0C. The Scrag attacks under the hook

Labels: `bug`, `area:engine`.

1. Which id1 maps stock `monster_wizard` was measured when this spec was
   written (skill 2): e3m5 20, e3m6 19, e4m3 18, e2m4 17, e1m4 16, e2m7 16,
   e4m4 14, e1m5 10, e2m5 10, e3m3 9, e1m3 7, e1m8 7, e1m2 6, e3m2 5, e3m7 5,
   e2m6 4, e1m6 3, e3m4 2. Confirm the top of that list with a scratch script
   (load each map, count entities; not committed) and paste it in the issue.
   Use e3m5 as the Scrag's probe and training map and e4m3 as its second
   probe map; e3m6 and e2m4 are already held out for other classes and stay
   held out.
2. New ctests, written first and shown failing: `wizard_drive` (a driven
   Scrag ordered forward moves at least 50 units in 2 s and its height changes
   by less than 32 units), `wizard_attack` (in sight at 300 units, attack
   held: the player takes damage within 6 s), `wizard_aim_error` (modelled on
   `enforcer_aim_error`: with the aim error overridden to 45 degrees its shots
   do measurably less damage than with 0).
3. The Scrag's shot is likely spawned as another entity's think (its
   dispatcher starts a chain whose frame calls a function that stores a
   shot-named function as a new entity's `think`). Extend the chain walk in
   `brain_hook.c` so such a chain counts as ranged, and so the aim error
   reaches the shot. The aim error code today acts only when `self` is the
   nominated monster; a shot run as another entity's think needs its own
   rule.
4. If any of the three tests cannot pass after three different attempts,
   stop and ask (condition 12). The Scrag then stays out of training; its
   type entry in part 0D is still added, unused.

**Done when**

- [ ] The three ctests failed before (output in the issue) and pass.
- [ ] The two Scrag maps are named in the issue.
- [ ] All checks under "Build, test, check" pass, game-data tests included.

### 0D. The new observation layout (changed once, here)

Labels: `feature`, `area:python`, `area:engine`.

This invalidates every monster checkpoint, which are being discarded anyway.
The player models are not affected.

#### The layout

`monster.py` builds it; `brain_onnx.c` builds the same in C. Rows marked
**changed** or **new** differ from today; every other row is unchanged.

| Slice | Size | Meaning |
| --- | --- | --- |
| `[0:48]` | 48 | ray distances, 2 pitch rows x 24 yaw columns, fraction of 768 |
| `[48:96]` | 48 | surface kind of the same rays, / 6 |
| `[96:99]` | 3 | velocity in the monster's frame (forward, left, up) / 400 |
| `[99]` | 1 | 1 if on ground |
| `[100]` | 1 | **changed:** health / `start_health`, clipped to [0, 1] |
| `[101]` | 1 | attack cooldown in seconds, clipped to 1 |
| `[102]` | 1 | water level / 3 |
| `[103:105]` | 2 | unit vector to the enemy in the monster's frame, while visible |
| `[105]` | 1 | enemy height difference / 256, clipped, ditto |
| `[106]` | 1 | enemy distance / 1024, clipped to 2, ditto |
| `[107]` | 1 | 1 while the enemy is visible |
| `[108]` | 1 | enemy health / 100 while visible |
| `[109:111]` | 2 | unit vector to the last seen enemy position, while remembered (5 s) |
| `[111]` | 1 | its height difference / 256, clipped |
| `[112]` | 1 | its distance / 1024, clipped to 2 |
| `[113]` | 1 | memory age / 5, clipped |
| `[114]` | 1 | 1 while a last-seen position is remembered |
| `[115]` | 1 | skill / 3 |
| `[116:119]` | 3 | style: aggression, caution, stealth (0-1) |
| `[119:121]` | 2 | **new:** unit vector to the heard noise in the monster's frame, while `heard` |
| `[121]` | 1 | **new:** its height difference / 256, clipped, ditto |
| `[122]` | 1 | **new:** its distance / 1024, clipped to 2, ditto |
| `[123]` | 1 | **new:** `heard_age` / `HOOK_HEARD_SECONDS`, clipped to 1, ditto |
| `[124]` | 1 | **new:** 1 while `heard` |
| `[125:128]` | 3 | **new:** pace one-hot: walk, investigate, run |
| `[128:130]` | 2 | **new:** unit vector to the nearest visible live monster other than itself |
| `[130]` | 1 | **new:** its distance / 1024, clipped to 2 (0 when none) |
| `[131]` | 1 | **new:** visible live monsters other than itself / 4, clipped to 1 |
| `[132:142]` | 10 | monster type one-hot, `MONSTER_TYPES` order; **changed:** `monster_wizard` appended as index 9 |

`MONSTER_OBS_BASE_DIM` = 132, `MONSTER_OBS_DIM` = 142. `heard_kind` is never
encoded. The heard noise has its own channels; it does not refresh the
last-seen memory (this replaces the plan in today's `monster.py` docstring).
"Live monster" means `flags & FL_MONSTER` and `health > 0` in the hook's
`visible` list.

The action layout does not change: `MONSTER_ACTION_NVEC = (3, 3, 7, 2)`.

#### Changes

1. `python/deepquake/monster.py`: the table above in the docstring; the
   builder; `MONSTER_TYPES` gains `"monster_wizard"`.
2. `engine/src/brain_onnx.c`: `DQ_MONSTER_OBS_BASE_DIM` 132,
   `DQ_MONSTER_MAX_TYPES` 10, `DQ_MAX_CLASSES` 10; `dq_build_observation`
   builds the same vector. Refuse every model whose width is not
   132 + n (1 <= n <= 10), with the existing log line; remove the 119-wide
   path. Style comes from a new `float style[3]` in `dq_monster_state`, filled
   at nomination from the `dq_style` cvar (three numbers, each clipped to
   [0, 1]; default `0.5 0.5 0.5`, the value training uses). `dq_selftest_run`
   sets `state.style` to 0.25, 0.5, 0.75 explicitly, so its cases do not
   change. Extend `dq_selftest_make_hook` so one case has a heard noise and a
   visible monster.
3. Hearing cvars in `brain_onnx.c` (new, registered with `adopt_cvar` and
   added to the `adopted` table of `register_cvars`), pushed to
   `dq_brain_set_hearing` every client tick:

   | Cvar | Default | Meaning |
   | --- | --- | --- |
   | `dq_hearing` | `1` | 0: monsters hear nothing; 1: on |
   | `dq_hear_range` | `0` | 0: the skill table's range; above 0: that many units |
   | `dq_hear_run` | `-1` | below 0: the skill table's run chance; 0-1: that chance |
   | `dq_hear_player` | `0` | 1: also hear the player's jumps, landings, pickups and pain |

4. Tests: update every layout assertion in `python/tests/test_monster.py` and
   `python/tests/test_export_onnx.py` to the new table (142 wide, one-hot at
   132). New: `test_monster.py::test_heard_noise_channels` (a synthetic
   `HookObservation` with a noise 300 units ahead and 100 to the left gives the
   expected values in `[119:125]`), `test_heard_kind_is_not_in_the_observation`
   (two reports that differ only in `heard_kind` give identical vectors),
   `test_health_is_a_fraction_of_start_health`, `test_nearest_ally_channels`,
   `test_pace_one_hot`.
5. Client parity: build the verification client
   (`-DDQSIM_CLIENT_SELFTEST="<model>,<report>"`, see ARCHITECTURE.md section
   1) with an ONNX export of an untrained 142-wide `ActorCritic`, run it under
   `xvfb-run` exactly as AGENTS.md describes, and compare its observation lines
   with `MonsterObserver` on the same synthetic reports (a scratch script).
   Paste the largest deviation; it must be below 1e-5.
6. `docs/ARCHITECTURE.md`: replace the "Implemented for the monster (#33)"
   description and the type list in section 4 with the new layout; note that
   style is fed from `dq_style` and trained at its default.

**Done when**

- [ ] The layout in `monster.py`'s docstring is the table above, and the new
      tests pass.
- [ ] The client refuses a 128-wide model and loads a 142-wide one (log lines
      from an `xvfb-run` launch in the issue).
- [ ] Client and Python observations agree within 1e-5 (number in the issue).
- [ ] All checks under "Build, test, check" pass, game-data tests included.

### 0E. Fight check groundwork

Labels: `task`, `area:python`.

#### Changes to `fight_check.py` allowed in 0E (and nothing else)

1. `REPORT_VERSION` = 2. The report records the checkpoint's `stage` and, per
   class, whether its probe map is a training map or held out (from the
   checkpoint's `maps` and `held_out`).
2. `judge` and `require_passing_report` refuse a checkpoint without a `stage`
   key ("trained before docs/RETRAIN_SPEC.md").
3. **The idle control.** The sight probe also plays the `idle` brain (already
   in `run_class`). A class's map is unmeasured, and the class fails as
   unmeasured, when idle's kill rate is above 0.05 or its damage rate above
   0.10 on it. Every later probe applies the same control.
4. `CLASS_MAPS` may change only like this, and only with the measurements in
   the issue: for `monster_demon1`, `monster_knight` and `monster_shambler`,
   the second listed map may move first, if on it the reference kills in at
   least `REFERENCE_FLOOR` of tasks, its facing is at least 0.75, and the idle
   control passes. `monster_wizard` is added with the two maps from part 0C,
   if 0C passed.
5. New command-line options: `--bars STAGE` (play only the `reference`,
   `random` and `idle` brains on the probes up to `STAGE`, no checkpoint
   needed, and write `runs/fight_bars/stage<STAGE>.json`), and
   `--classes NAME ...` (the classes to measure with `--bars`). With `--bars`
   the checkpoint argument is not given.
6. Unit tests in `python/tests/test_release.py` for points 2 and 3 (synthetic
   reports, no game data).

No threshold constant changes: `REFERENCE_FLOOR`, `KILL_FRACTION`,
`FACING_DEGREES`, `FACING_FRACTION`, `EPISODES`, `MIN_USABLE`, `TASK_POOL`,
`SKILL`, `SEED`, `MIN_DISTANCE`, `MAX_DISTANCE` and `TIMEOUT_STEPS` stay as
they are.

#### Steps

1. Run `uv run python -m deepquake.fight_check --bars 0 --classes
   monster_army monster_dog monster_enforcer monster_ogre monster_hell_knight
   monster_shalrath monster_knight monster_demon1 monster_shambler` (add
   `monster_wizard` if 0C passed). Paste the table. Part 0A changed the
   reference's facing, so these are new numbers.
2. For the Fiend, the Knight and the Shambler, if their row is unmeasurable on
   the first map, measure the second map with the same command after a
   temporary local swap, paste it, and make the swap permanent only if point 4
   holds. If neither map works, stop and ask (condition 14).

**Done when**

- [ ] The bars table for every roster class is in the issue, with the idle
      column, and every class is measurable on its listed map.
- [ ] The new unit tests pass, and the existing ones in `test_release.py` pass
      unchanged apart from the report version.
- [ ] All checks under "Build, test, check" pass, game-data tests included.

### Stage 0 done when

- [ ] Parts 0A-0E are closed with evidence.
- [ ] `docs/ARCHITECTURE.md` describes the yaw rule, hearing, and the new
      layout.
- [ ] The maintainer has written `Reviewed: stage 0 OK`.

---

## Machinery shared by Stages 1-5

Stage 1's code part builds this; later stages extend it. Everything here is
new unless it names something that exists.

### Tasks

A config's training episodes are drawn from **tasks**. New dataclass
`TaskSpec` in `monster_env.py`:

| Field | Meaning |
| --- | --- |
| `name` | task name, the key under `[tasks]` in the config |
| `weight` | relative chance of this task per episode |
| `placement` | `"hidden"` or `"sight"` (existing meanings) |
| `min_distance`, `max_distance` | flat distance band of the player's start (existing meaning) |
| `timeout_steps` | episode length in decisions (12 per second) |
| `opponents` | opponent kinds; one is drawn uniformly per episode |
| `start_health` | `[lo, hi]`: the monster's health is set to a uniform fraction of `start_health` in that range at episode start (`[1.0, 1.0]` = untouched) |
| `armed` | the player starts with `impulse 9` then `impulse 2` (all weapons and ammo, shotgun selected) |
| `classes` | optional: the classes this task may be drawn for |

`MonsterEnv` gains `tasks: Sequence[TaskSpec] | None = None`. When `None`, a
single task is built from the existing arguments, so the sight probe and the
existing tests keep working. Per episode, `reset()` draws the class (as today),
then a task among those whose `classes` allow it, by weight, then an opponent.
`set_task(...)` (used by probes) takes the task explicitly.

`MonsterEnv` also gains `path_distance: np.ndarray | None` next to `points`
(both from `nav.NavEnv(map_name, seed=seed)` when not given). Callers that pass
`points` must pass `path_distance` too.

### Opponents

Add to `PlayerOpponent` and `_Opponent` in `monster_env.py`. The existing
`still`, `random` and `shooter` keep their behaviour. The `trained` opponent is
not used by this spec (AGENTS.md: wins against it are not evidence).

| Kind | Behaviour |
| --- | --- |
| `camper` (new) | Never moves or turns. Fires one shot (fire = 1 for one decision) at decision 12 and then every N decisions, N drawn uniformly from 36-60 by the opponent's own seeded generator. Otherwise `STILL_ACTION`. |
| `noisy_shooter` (new) | `combat.shooter_policy`; in addition, while the monster is not visible, fires on the camper's schedule. |
| `turret` (new) | Never moves. While the monster is visible, turns and fires exactly as `shooter_policy` does, with forward and strafe held at 1 (none). While it is not visible, holds its facing. |
| `watcher` (new) | `turret`, plus one shot at decision 12. Its starting facing points at the monster's start position. |

### Hearing settings in the environment

New dataclass `HearingSampling` in `monster_env.py`, fields `on_probability`,
`range_min`, `range_max`, `player_noises_probability`, `run_chances`.
`MonsterEnv(hearing=...)`: when given, `reset()` draws per episode, from
`self.rng`, hearing on with `on_probability`, a range uniform in
`[range_min, range_max]`, player noises on with `player_noises_probability`,
and a run chance uniformly from `run_chances`, and calls `sim.set_hearing`.
When not given, and always in `set_task`, `sim.set_hearing()` is called with its
defaults (on, the skill table). `MonsterEnv(deaf=True)` calls
`sim.set_hearing(enabled=False)` instead; probes use it for the hearing-off
column.

### Reward terms

`reward = "staged"` (new value; `MonsterEnv` accepts it next to
`combat.REWARDS`) computes, every decision, the sum over the config's
`[reward_terms]` of coefficient x term. Terms:

| Term | Value for one decision |
| --- | --- |
| `kill` | 1 if the player's health went from above 0 to 0 or below during this decision |
| `death` | 1 if the driven monster died during this decision |
| `damage` | `dealt - taken`, where `dealt` = player health lost this decision / 100 and `taken` = monster health lost this decision / `start_health` (both 0 or more) |
| `approach` | `gamma * phi_next - phi`, with `phi = -min(path, 3000) / 1000` and `gamma` the config's `[ppo] gamma`; 0 when `path` is unknown. Only counted while the monster's health fraction is at least `approach_min_health` (default 0). |
| `hidden_hurt` | 1 while the monster's health fraction is below 0.5 and the player cannot see it |
| `exposed_cooling` | on tasks named `cover` only: 1 while the player can see the monster and its `attack_cooldown` is above 0 |
| `flank` | `dealt` (as above) when, at this decision, the angle between the player's view yaw and the bearing from the player to the monster is above 60 degrees; else 0 |

Definitions:

- `path`: the walking distance from the monster to the player on the walk
  graph. If their flat distance is under 128 and their height difference
  under 64, it is the straight 3D distance. Otherwise take the graph points
  within 128 units of the monster (set I) and of the player (set J); `path` is
  the minimum over i in I, j in J of `|monster - P_i| + path_distance[i, j] +
  |P_j - player|`. Empty I or J, or no finite value: unknown, and `phi` keeps
  its last value. The walk graph is built for the player's hull, so it is only
  an approximation for large monsters; it is reward shaping, never an input
  (map independence rule 2).
- "The player can see the monster": `combat.visible_from(player_state,
  monster_entity, player_rays)`, the function the opponents already use
  (768-unit sight distance).
- Health fraction: health / `start_health`.

`won` (the player died) and `lost` (the monster died) end the episode as
today.

### Configs

New top-level keys read by `train_monster.py`: `stage` (required for
`reward = "staged"`; saved into the checkpoint dict next to `monsters`),
`others_passive`, `[tasks.<name>]` tables (fields of `TaskSpec`),
`[hearing]` (fields of `HearingSampling`), `[reward_terms]` (coefficients
and `approach_min_health`). With `[tasks]`, the old top-level `opponents`,
`placement`, `min_distance`, `max_distance` and `timeout_steps` must be
absent; `train_monster.py` refuses a config that has both. `make_env` passes
`others_passive`, `tasks`, `hearing` and the walk graph's `path_distance`.
Save `stage`, the tasks, the hearing spread and the reward terms in the
checkpoint dict.

### Probes in `fight_check.py`

Each stage adds one probe function in the shape of `run_class` (same task
selection: `make_tasks(env, EPISODES_<PROBE> * TASK_POOL, SEED)`, first usable
tasks played) and one `judge_<probe>` function. `main` runs every probe up to
the checkpoint's `stage` with the brains the stage lists, prints one table per
probe with the bars beside the model, and the verdict requires every probe up
to that stage. Every probe uses `others_passive=True`, skill 2, seed 2026, and
the idle control. Probe maps are `CLASS_MAPS` after part 0E.

---

## Stage 1: hunting by sound

**Goal.** A monster that hears a player it cannot see goes to where the noise
was, finds the player and engages it, more often than a bee-line script and
than random actions.

Parts: `1-code` (labels `feature`, `area:python`, `area:training`) builds the
shared machinery above and this stage's probe; `1-train` (labels `task`,
`area:training`) measures the bars, asks for the go-ahead and runs
`s1_hunt_grunt` and then `s1_hunt_roster`.

### Environment

- Task `hunt`: placement `hidden` (line of sight broken), 300-1200 units,
  timeout 720 decisions (60 s), opponents `camper`, `camper`,
  `noisy_shooter` (so two thirds of episodes are campers), full health, not
  armed.
- `others_passive = true`, so no other monster's kill or damage is counted.
- Training starts **from scratch** (no `init_from`): the old checkpoints have
  another layout.

### Hearing settings sampled in training

`on_probability = 0.85`, `range_min = 400`, `range_max = 1600`,
`player_noises_probability = 0.25`, `run_chances = [0.0, 0.25, 0.5, 0.75,
1.0]`. This covers the skill table (500-1500) with margin, hearing off, player
noises on, and every pace mix. Settings outside these ranges are untested.
Every later stage uses the same spread.

### Reward

`kill = 1.0`, `death = -1.0`, `damage = 1.0`, `approach = 1.0`.

### The script (`hunt_reference_action`, new in `fight_check.py`)

Reads only the observation vector (new layout; indices from the table in part
0D). "Turn toward angle a" means the index of `MONSTER_TURN_DEGREES` nearest
to a, where a = `degrees(atan2(left, forward))` of a unit-vector pair.

1. Enemy visible (`obs[107] == 1`): turn toward `obs[103:105]`; forward 2 if
   `obs[106] * 1024 > 120` else 1; strafe 1; attack 1.
2. Else, noise remembered (`obs[124] == 1`) and `obs[122] * 1024 > 64`: turn
   toward `obs[119:121]`; forward 2; strafe 1; attack 0. If `obs[0] < 0.1` (a
   wall within 77 units straight ahead), turn index 6 (+30 degrees) instead.
3. Else, last seen remembered (`obs[114] == 1`) and `obs[112] * 1024 > 64`:
   as step 2, toward `obs[109:111]`.
4. Else `[1, 1, 3, 0]` (stand still and wait).

### The probe `hunt`

| Setting | Value |
| --- | --- |
| Classes and maps | every class of the checkpoint, its `CLASS_MAPS` map after part 0E |
| Placement | hidden, 400-900 units flat (proposed) |
| Opponent | `camper` |
| Hearing | defaults for skill 2 (on, 1000 units, player noises off, run chance 0.5) |
| Tasks | `EPISODES_HUNT = 40` usable tasks per class (proposed), `TASK_POOL` 10, `SEED` 2026, skill 2 |
| Time limit | 720 decisions (60 s) (proposed) |
| Brains | `checkpoint`, `script` (`hunt_reference_action`), `random`, `idle`, `deaf` (the checkpoint with `MonsterEnv(deaf=True)`) |
| Measured per brain | `found_rate`: share of tasks in which the hook's `visible` list held the live player at some decision; `kill_rate`; median decisions to first sight over found tasks |

"Aggregate" means the mean over the measured classes.

**Gate, from the bars measurement before training:** aggregate
`found_rate(script) - found_rate(random) >= 0.15`, and aggregate
`found_rate(script) <= 0.80` (room for the model to beat it). Otherwise stop
and ask.

**Pass bars (proposed):**

| Bar | Condition |
| --- | --- |
| H1 | aggregate `found_rate(model) >= found_rate(script) + 0.15` |
| H2 | aggregate `found_rate(model) >= found_rate(random) + 0.30` |
| H3 | every class: `found_rate(model) >= found_rate(script) - 0.10` |
| H4 | aggregate `kill_rate(model) >= kill_rate(script)` |
| H5 | aggregate `found_rate(model) - found_rate(deaf) >= 0.10` (it uses what it hears) |
| H6 | the sight probe passes |

### Changes to `fight_check.py` allowed in Stage 1

`hunt_reference_action`; the `hunt` probe with the constants above
(`EPISODES_HUNT`, its distances and time limit); the `script` and `deaf`
brains; `judge_hunt` with bars H1-H5; `REPORT_VERSION` = 3. Unit tests in
`test_release.py`: `judge_hunt` fails a model equal to the script, fails a
model equal to random, fails when the gate fails (unmeasured), and passes
numbers that clear every bar.

### Runs

Run `1a` first; `1b` only after `1a` has passed and has its own go-ahead.

`configs/s1_hunt_grunt.toml` (expected about 2,700 steps/s with 18 workers,
as roster11 measured; stop below 1,350; about 1 hour):

```toml
# Stage 1a of docs/RETRAIN_SPEC.md: hunting by sound, the Grunt alone.
#   uv run python -m deepquake.train_monster configs/s1_hunt_grunt.toml
name = "s1_hunt_grunt"
stage = 1
monsters = ["monster_army"]
# The id1 maps that stock monster_army at skill 2 (measured 2026-11-19:
# e1m1 34, e1m2 16, e2m1 13, e3m1 35, e4m1 31); e3m1 is held out (#35).
maps = ["e1m1", "e1m2", "e2m1", "e4m1"]
held_out = ["e3m1"]
envs = 18
skills = [0, 1, 2, 3]
# Style has no reward term: train it fixed at the client's dq_style default.
styles = [{ style = [0.5, 0.5, 0.5] }]
others_passive = true
reward = "staged"
seed = 0
total_steps = 10_000_000
device = "3090"
checkpoint_every = 25

[tasks.hunt]
weight = 1.0
placement = "hidden"
min_distance = 300.0
max_distance = 1200.0
timeout_steps = 720
opponents = ["camper", "camper", "noisy_shooter"]
start_health = [1.0, 1.0]
armed = false

[hearing]
on_probability = 0.85
range_min = 400.0
range_max = 1600.0
player_noises_probability = 0.25
run_chances = [0.0, 0.25, 0.5, 0.75, 1.0]

[reward_terms]
kill = 1.0
death = -1.0
damage = 1.0
approach = 1.0

[ppo]
steps_per_env = 256
epochs = 4
minibatches = 4
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip = 0.2
entropy_coef = 0.01
hidden = 256
bptt = 32
```

`configs/s1_hunt_roster.toml`: the same file with these differences (about 3
hours):

```toml
name = "s1_hunt_roster"
monsters = [
  "monster_army", "monster_dog", "monster_enforcer", "monster_ogre",
  "monster_hell_knight", "monster_shalrath", "monster_knight",
  "monster_demon1", "monster_shambler",
]  # add "monster_wizard" only if part 0C passed
init_from = "runs/s1_hunt_grunt/checkpoint.pt"
# Every type's head starts from the Grunt's, the only trained one.
init_head_from = { monster_dog = "monster_army", monster_enforcer = "monster_army", monster_ogre = "monster_army", monster_hell_knight = "monster_army", monster_shalrath = "monster_army", monster_knight = "monster_army", monster_demon1 = "monster_army", monster_shambler = "monster_army" }
# roster11's training maps without e1m8, where the level itself hurts the player.
maps = [
  "e1m1", "e2m1", "e4m1", "e1m2", "e2m2", "e4m2", "e2m3", "e2m5", "e4m5",
  "e2m6", "e4m6", "e4m7", "e1m4", "e4m3", "e4m4", "e1m6", "e2m7", "e3m4",
]  # plus the Scrag's training map from part 0C, if it passed
held_out = ["e3m1", "e3m2", "e3m6", "e4m8", "e1m5", "e2m4", "e1m3"]
envs = 18  # one per map at least: raise to the map count if a map is added
total_steps = 30_000_000
```

(Write the whole file out; the snippet lists only what differs. With the
Scrag, add `monster_wizard = "monster_army"` to `init_head_from` and its held-out
map to `held_out`.)

### Done when

- [ ] `1-code` is merged with tests: the opponents, tasks, hearing sampling,
      reward terms (one unit test per term with synthetic states), config
      validation, a determinism test for a `hunt` task
      (`test_monster.py::test_hunt_task_is_deterministic`, game data: two runs
      of the same task give the same per-decision state hashes), and the
      `judge_hunt` tests.
- [ ] The bars (`fight_check --bars 1 --classes monster_army`, then with every
      roster class) are in the issue and pass the gate.
- [ ] `s1_hunt_grunt` passed `fight_check` (sight and hunt) on its checkpoint;
      table in the issue.
- [ ] `s1_hunt_roster` passed `fight_check` on its checkpoint; table in the
      issue.
- [ ] `uv run python -m deepquake.release_model runs/s1_hunt_roster models
      s1_hunt_roster` succeeded; a 60 s `xvfb-run` launch with that model on
      e1m1 (AGENTS.md command) logs "monster(s) driven by the model".
- [ ] The maintainer has written `Played: s1_hunt_roster OK`.

---

## Stage 2: retreating when hurt

**Goal.** A hurt monster facing a player who out-guns it breaks line of sight
and survives, instead of standing and dying, while a healthy monster still
hunts and fights as in Stage 1.

Parts: `2-code` and `2-train`, as in Stage 1.

### Environment

- New simulator call for setting a monster's health:
  `DQSIM_API int dqsim_set_entity_health(int entity_index, float health)` in
  `dqsim.h` (new; changes only that field: no pain, no enemy), and
  `sim.set_entity_health(index, health)` (new). A ctest checks that the
  monster's health reads back as set (`dqsim_get_entities`), and that setting
  the old value again restores the old state hash.
- Task `retreat` (new): placement `sight`, 250-500 units, timeout 240 (20 s),
  opponent `shooter`, `start_health = [0.15, 0.5]`, armed.
- Task `hunt` as in Stage 1.
- `init_from = "runs/s1_hunt_roster/checkpoint.pt"` (same layout).

### Reward

`kill = 1.0`, `death = -2.0`, `damage = 1.0`, `approach = 1.0` with
`approach_min_health = 0.5`, `hidden_hurt = 0.002`. A hurt monster is no
longer paid to approach and is paid a little for staying out of sight.

### The scripts

- `flee_reference_action` (new): if `obs[100] < 0.5` and the enemy is visible,
  turn toward the angle of `(-obs[103], -obs[104])` (directly away), forward
  2, strafe 1, attack 0, with the same wall rule as the hunt script; if
  `obs[100] < 0.5` and the enemy is not visible, `[1, 1, 3, 0]`; otherwise
  `hunt_reference_action`.
- `hunt_reference_action` as a second reference ("fight"), to show that
  retreating is what matters on this probe.

### The probe `retreat`

| Setting | Value |
| --- | --- |
| Classes and maps | every class of the checkpoint, its `CLASS_MAPS` map |
| Placement | sight, 250-450 units (proposed) |
| Opponent | `shooter`, armed |
| Monster health | 0.25 x `start_health`, set at task start (proposed) |
| Tasks | `EPISODES_RETREAT = 40` per class, seed 2026, skill 2 |
| Time limit | 240 decisions (20 s) (proposed) |
| Brains | `checkpoint`, `flee` script, `fight` script, `random` |
| Measured | `survival_rate`: share of tasks in which the monster is alive at the end (a dead player counts as survived); `dealt`: mean player health lost per task |

**Gate:** aggregate `survival(flee) - survival(fight) >= 0.15`,
`survival(flee) - survival(random) >= 0.10`, and `survival(flee) <= 0.80`.

**Pass bars (proposed):** R1 aggregate `survival(model) >= survival(flee) +
0.10`; R2 aggregate `survival(model) >= survival(random) + 0.25`; R3 aggregate
`survival(model) >= survival(fight) + 0.25`; R4 the sight and hunt probes pass.

### Changes to `fight_check.py` allowed in Stage 2

`flee_reference_action`; the `retreat` probe with the constants above;
`judge_retreat` with R1-R3; `REPORT_VERSION` = 4; unit tests as in Stage 1.

### Config `configs/s2_retreat.toml`

```toml
# Stage 2 of docs/RETRAIN_SPEC.md: retreating when hurt.
#   uv run python -m deepquake.train_monster configs/s2_retreat.toml
name = "s2_retreat"
stage = 2
monsters = [
  "monster_army", "monster_dog", "monster_enforcer", "monster_ogre",
  "monster_hell_knight", "monster_shalrath", "monster_knight",
  "monster_demon1", "monster_shambler",
]  # exactly the classes s1_hunt_roster trained
init_from = "runs/s1_hunt_roster/checkpoint.pt"
maps = [
  "e1m1", "e2m1", "e4m1", "e1m2", "e2m2", "e4m2", "e2m3", "e2m5", "e4m5",
  "e2m6", "e4m6", "e4m7", "e1m4", "e4m3", "e4m4", "e1m6", "e2m7", "e3m4",
]  # exactly s1_hunt_roster's
held_out = ["e3m1", "e3m2", "e3m6", "e4m8", "e1m5", "e2m4", "e1m3"]
envs = 18
skills = [0, 1, 2, 3]
styles = [{ style = [0.5, 0.5, 0.5] }]
others_passive = true
reward = "staged"
seed = 0
total_steps = 20_000_000  # proposed
device = "3090"
checkpoint_every = 25

[tasks.hunt]
weight = 0.5
placement = "hidden"
min_distance = 300.0
max_distance = 1200.0
timeout_steps = 720
opponents = ["camper", "camper", "noisy_shooter"]
start_health = [1.0, 1.0]
armed = false

[tasks.retreat]
weight = 0.5
placement = "sight"
min_distance = 250.0
max_distance = 500.0
timeout_steps = 240
opponents = ["shooter"]
start_health = [0.15, 0.5]
armed = true

[hearing]
on_probability = 0.85
range_min = 400.0
range_max = 1600.0
player_noises_probability = 0.25
run_chances = [0.0, 0.25, 0.5, 0.75, 1.0]

[reward_terms]
kill = 1.0
death = -2.0
damage = 1.0
approach = 1.0
approach_min_health = 0.5
hidden_hurt = 0.002

[ppo]
steps_per_env = 256
epochs = 4
minibatches = 4
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip = 0.2
entropy_coef = 0.01
hidden = 256
bptt = 32
```

### Done when

- [ ] `2-code` merged with tests (`dqsim_set_entity_health` ctest, the new
      reward terms, `judge_retreat`).
- [ ] Bars measured and pass the gate (issue comment).
- [ ] `s2_retreat` passes `fight_check` (sight, hunt, retreat); table in the
      issue.
- [ ] Released locally and launched under `xvfb-run` as in Stage 1.
- [ ] The maintainer has written `Played: s2_retreat OK`.

---

## Stage 3: using cover

**Goal.** A ranged monster under fire from a player who stays put trades
better than standing in the open: it stays out of the player's sight while it
cannot shoot and comes back into view to shoot.

Melee classes are not trained or probed on cover: they must close the
distance. Ranged classes here: `monster_army`, `monster_enforcer`,
`monster_ogre`, `monster_hell_knight`, `monster_shalrath`, `monster_shambler`,
and `monster_wizard` if trained.

Parts: `3-code` and `3-train`.

### Environment

Task `cover` (new): placement `sight`, 300-600 units, timeout 360 (30 s),
opponent `turret`, full health, armed, `classes` = the ranged classes above.
Tasks `hunt` and `retreat` as in Stage 2.

### Reward

Stage 2's terms plus `exposed_cooling = -0.002`.

### The script

The sight probe's existing `reference_action` ("stand and shoot": turn to the
player, walk up to 120 units, hold attack).

### The probe `cover`

| Setting | Value |
| --- | --- |
| Classes and maps | the ranged classes of the checkpoint, their `CLASS_MAPS` map |
| Placement | sight, 300-600 units (proposed) |
| Opponent | `turret`, armed |
| Tasks | `EPISODES_COVER = 40` per class, seed 2026, skill 2 |
| Time limit | 360 decisions (30 s) (proposed) |
| Brains | `checkpoint`, `script` (`reference_action`), `random` |
| Measured | `net`: mean over tasks of (player health lost / 100 - monster health lost / `start_health`); `exposure`: share of decisions in which the player can see the monster |

**Gate:** aggregate `net(script) - net(random) >= 0.15`.

**Pass bars (proposed):** C1 aggregate `net(model) >= net(script) + 0.15`; C2
aggregate `net(model) >= net(random) + 0.30`; C3 aggregate `exposure(model) <=
exposure(script) - 0.15`; C4 every earlier probe passes.

### Changes to `fight_check.py` allowed in Stage 3

The `cover` probe with the constants above, `judge_cover` with C1-C3,
`REPORT_VERSION` = 5, unit tests as in Stage 1.

### Config `configs/s3_cover.toml`

`configs/s2_retreat.toml` with: `name = "s3_cover"`, `stage = 3`,
`init_from = "runs/s2_retreat/checkpoint.pt"`, `total_steps = 20_000_000`
(proposed), `[tasks.hunt] weight = 0.4`, `[tasks.retreat] weight = 0.3`, and:

```toml
[tasks.cover]
weight = 0.3
placement = "sight"
min_distance = 300.0
max_distance = 600.0
timeout_steps = 360
opponents = ["turret"]
start_health = [1.0, 1.0]
armed = true
classes = [
  "monster_army", "monster_enforcer", "monster_ogre",
  "monster_hell_knight", "monster_shalrath", "monster_shambler",
]

[reward_terms]
kill = 1.0
death = -2.0
damage = 1.0
approach = 1.0
approach_min_health = 0.5
hidden_hurt = 0.002
exposed_cooling = -0.002
```

(Write the whole file out.)

### Done when

- [ ] `3-code` merged with tests (the `turret` opponent, `exposed_cooling`,
      task `classes` filtering, `judge_cover`); a test that an armed `turret`
      fires for the whole 30 s (at least 50 `DQSIM_NOISE_GUNFIRE` noises heard
      by a driven monster in range).
- [ ] Bars measured and pass the gate.
- [ ] `s3_cover` passes `fight_check` (sight, hunt, retreat, cover).
- [ ] Released locally and launched under `xvfb-run`.
- [ ] The maintainer has written `Played: s3_cover OK`.

---

## Stage 4: flanking

**Goal.** Against a player who watches the way the monster is coming, the
monster lands its first hit from outside the player's view more often than a
monster that walks straight in.

Parts: `4-code` and `4-train`.

### Environment

Task `flank` (new): placement `hidden`, 400-900 units, timeout 600 (50 s),
opponent `watcher`, full health, armed. Earlier tasks as in Stage 3.

The probe records, per task, the angle between the player's view yaw and the
bearing from the player to the monster at the decision of the monster side's
first damage (record it in `MonsterEnv.step` where `_first_damage_dealt` is
set).

### Reward

Stage 3's terms plus `flank = 0.5`.

### The script

`hunt_reference_action` (Stage 1): it walks straight toward the noise and so
into the watcher's view.

### The probe `flank`

| Setting | Value |
| --- | --- |
| Classes and maps | every class of the checkpoint, its `CLASS_MAPS` map |
| Placement | hidden, 400-900 units (proposed) |
| Opponent | `watcher`, armed |
| Tasks | `EPISODES_FLANK = 40` per class, seed 2026, skill 2 |
| Time limit | 600 decisions (50 s) (proposed) |
| Brains | `checkpoint`, `script` (`hunt_reference_action`), `random` |
| Measured | `flank_rate`: share of tasks whose first damage came from more than 60 degrees off the player's view; `kill_rate` |

**Gate:** aggregate `kill_rate(script) - kill_rate(random) >= 0.10`, and
aggregate `flank_rate(script) <= 0.60`.

**Pass bars (proposed):** F1 aggregate `flank_rate(model) >= flank_rate(script)
+ 0.15`; F2 aggregate `flank_rate(model) >= flank_rate(random) + 0.15`; F3
aggregate `kill_rate(model) >= kill_rate(script)`; F4 every earlier probe
passes.

### Changes to `fight_check.py` allowed in Stage 4

The `flank` probe with the constants above, `judge_flank` with F1-F3,
`REPORT_VERSION` = 6, unit tests as in Stage 1.

### Config `configs/s4_flank.toml`

`configs/s3_cover.toml` with: `name = "s4_flank"`, `stage = 4`,
`init_from = "runs/s3_cover/checkpoint.pt"`, `total_steps = 20_000_000`
(proposed), weights `hunt = 0.3`, `retreat = 0.2`, `cover = 0.2`, and:

```toml
[tasks.flank]
weight = 0.3
placement = "hidden"
min_distance = 400.0
max_distance = 900.0
timeout_steps = 600
opponents = ["watcher"]
start_health = [1.0, 1.0]
armed = true

[reward_terms]
kill = 1.0
death = -2.0
damage = 1.0
approach = 1.0
approach_min_health = 0.5
hidden_hurt = 0.002
exposed_cooling = -0.002
flank = 0.5
```

(Write the whole file out.)

### Done when

- [ ] `4-code` merged with tests (`watcher`, the `flank` term, `judge_flank`).
- [ ] Bars measured and pass the gate.
- [ ] `s4_flank` passes `fight_check` (sight, hunt, retreat, cover, flank).
- [ ] Released locally and launched under `xvfb-run`.
- [ ] The maintainer has written `Played: s4_flank OK`.

---

## Stage 5: group play

**Goal.** Two to four monsters driven by one model kill a player together
more often, and lose fewer of their own, than the same monsters driven by the
Stage 4 model, which plays each monster as if it were alone.

This stage is the least certain. Nothing in Stages 1-4 promises it, and
Stage 4's model is releasable without it.

Parts: `5-design` (label `research`, `needs-human`), `5-code`, `5-train`.

### 5-design: approve the environment first

The trainer has one monster per environment today
(`ProcVecEnv`, `ppo.RolloutActor`). Group play needs one environment that
drives k monsters (one nomination each, the same weights, each its own
recurrent state and observation) and a learner that treats each monster as
its own rollout slot. Write a design comment on the `5-design` issue that
says: how `MonsterGroupEnv` (new, in `monster_env.py`) returns k observations
and takes k actions; how `vec.py` and `ppo.py` flatten envs x k; how an
episode ends when monsters die at different times; and what the reward of
each monster is (below). Then stop and ask. No code before the maintainer
approves the design.

### Environment (after approval)

Task `group` (new): k drawn uniformly from 2-4 monsters of the episode's class,
the k nearest to each other within 600 units; placement `hidden`, 400-900
units from the nearest of them; timeout 540 (45 s); opponent `noisy_shooter`,
armed; full health. Earlier tasks keep running with k = 1.

### Reward (per monster i)

`kill = 1.0` (the team's kill, given to every monster), `death = -1.0` (own),
`team_damage = 0.5` (new term: player health lost this decision / 100, given
to every monster), `taken = -1.0` (new term: own health lost / own
`start_health`), plus Stage 4's `approach`, `hidden_hurt`, `exposed_cooling`
and `flank` on each monster's own state.

### The script and the baselines

Each monster runs `hunt_reference_action` on its own observation. The extra
baseline is the **Stage 4 checkpoint** driving every monster.

### The probe `group`

| Setting | Value |
| --- | --- |
| Classes and maps | the checkpoint's classes whose `CLASS_MAPS` map stocks at least 3 within 600 units of each other |
| Group | k = 3 |
| Placement | hidden, 400-900 units from the nearest (proposed) |
| Opponent | `noisy_shooter`, armed |
| Tasks | `EPISODES_GROUP = 40` per class, seed 2026, skill 2 |
| Time limit | 540 decisions (45 s) (proposed) |
| Brains | `checkpoint`, `stage4` (the Stage 4 checkpoint), `script`, `random` |
| Measured | `kill_rate`; `losses`: mean monsters dead per task |

**Gate:** aggregate `kill_rate(script) - kill_rate(random) >= 0.15`.

**Pass bars (proposed):** G1 aggregate `kill_rate(model) >= kill_rate(stage4)
+ 0.10`; G2 `>= kill_rate(script) + 0.15`; G3 `>= kill_rate(random) + 0.30`;
G4 aggregate `losses(model) <= losses(stage4)`; G5 every earlier probe passes.

### Config `configs/s5_group.toml`

`configs/s4_flank.toml` with `name = "s5_group"`, `stage = 5`,
`init_from = "runs/s4_flank/checkpoint.pt"`, `total_steps = 30_000_000`
(proposed), the earlier task weights each scaled by 0.6, and:

```toml
[tasks.group]
weight = 0.4
placement = "hidden"
min_distance = 400.0
max_distance = 900.0
timeout_steps = 540
opponents = ["noisy_shooter"]
start_health = [1.0, 1.0]
armed = true
group_size = [2, 4]  # new TaskSpec field, defined by 5-design

[reward_terms]
kill = 1.0
death = -1.0
damage = 0.0
team_damage = 0.5
taken = -1.0
approach = 1.0
approach_min_health = 0.5
hidden_hurt = 0.002
exposed_cooling = -0.002
flank = 0.5
```

The approved design may change the keys `group_size`, `team_damage` and
`taken`; nothing else.

### Done when

- [ ] The maintainer approved the `5-design` comment.
- [ ] `5-code` merged with tests (k monsters nominated and stepped
      deterministically; per-monster rewards; `judge_group`).
- [ ] Bars measured (including the `stage4` column) and pass the gate.
- [ ] `s5_group` passes `fight_check` (every probe).
- [ ] Released locally and launched under `xvfb-run`.
- [ ] The maintainer has written `Played: s5_group OK`.

---

## What is uncertain

Say which of these happened, in the issue, when it happens.

- **Catching sounds by renaming symbols.** It should catch every QuakeC
  `sound()` and every QuakeC explosion. A mod that writes temp entities
  another way would be missed; mods are not trained on (the progs gate).
- **The cry rule.** It assumes every class plays its pain sound from its own
  `th_pain` function. A class that plays it from a later frame would have no
  pain cry (its death cry still counts). Test 5 of part 0B checks the Grunt;
  record any class found otherwise.
- **The investigation pace's look.** Walk frames advanced by distance are
  this spec's choice (the maintainer chose only that the pace is random
  between a run and something slower); they keep the feet matched to the
  ground. If it looks
  wrong in play, the fallback is the run frames played at half rate (the same
  distance rule with `run_speed`); that is the maintainer's call.
- **The Scrag** flies; the hook's air movement was never tested with a flyer,
  and its aim error may need a new rule. It may stay out of the roster.
- **Probe power.** 40 tasks per class give about plus or minus 0.15 on one
  class's rate; the bars are therefore mostly on aggregates. A class can look
  worse than the script by chance; H3's 0.10 allowance is for that. If results
  sit on a bar's edge, do not rerun with new seeds to get a pass: report and
  ask.
- **The walk graph** is built for the player's body. Path shaping for an Ogre
  or a Shambler is approximate, and some routes a player fits through they do
  not.
- **Retreat** only delays: Quake monsters do not heal. A monster that hides
  forever is a possible result; the earlier probes guard against cowardice
  when healthy, but "comes back later" is not taught here.
- **Cover** needs geometry near the fight. On open maps the script may already
  be as good as anything; then the gate fails and the probe needs the
  maintainer.
- **Flanking** needs a second route. Many tasks have only one; the flank rate
  of every brain may be low. Fallback: a `research` issue to restrict the
  probe to tasks whose walk graph has two routes, which is the maintainer's
  decision.
- **Group play** needs new training plumbing and has a credit-assignment
  problem; it may not beat the Stage 4 model. Fallback: release the Stage 4
  model; monsters still react to each other's noises through hearing.
- **Throughput.** Longer episodes and more scripted opponents may change steps
  per second; #54 can halve it at any time.
- **Training budgets** (`total_steps`) are guesses. Running longer is a new run
  with a new name and a new go-ahead.

**Out of scope:** hearing the player's footsteps continuously; stealth and
ambush behaviour; giving the style inputs a meaning; batched inference in the
client; monsters outside the roster (Zombie, Spawn, Rotfish, bosses); changing
the glue's basic fight (decision 6); per-class handicap tuning; the player
models (#38).

---

## How to start

Run these from the repository root, in this order, in a fresh session.

1. The AGENTS.md start of session:

   ```bash
   git status && git branch --show-current
   gh run list --branch main --limit 1
   gh issue list --label in-progress
   ```

   If an issue is in progress, continue it from its last Checkpoint and stop
   reading here.
2. Find the stage issues:

   ```bash
   gh issue list --state all --search "Retrain stage in:title" --limit 50
   ```

3. If none exist, create the Stage 0 tracking issue and the 0A issue. Write
   each body to a file under `runs/` (gitignored) first: a **Goal** line and
   the part's **Done when** list copied from this spec, plus
   `Spec: docs/RETRAIN_SPEC.md, part 0A`.

   ```bash
   gh issue create --title "Retrain stage 0: groundwork" --label task,area:engine,area:python --milestone "M6 Full roster" --body-file runs/issue_stage0.md
   gh issue create --title "Retrain stage 0A: keep the facing QuakeC gives a monster after an attack" --label bug,area:python,area:engine --milestone "M6 Full roster" --body-file runs/issue_stage0a.md
   ```

   Create the other part issues of a stage only when you start that part, so
   the lowest open one is always the next one. Titles:

   | Part | Title | Labels |
   | --- | --- | --- |
   | 0B | Retrain stage 0B: hearing in the engine glue | `feature,area:engine,area:python` |
   | 0C | Retrain stage 0C: the Scrag attacks under the hook | `bug,area:engine` |
   | 0D | Retrain stage 0D: the new monster observation layout | `feature,area:python,area:engine` |
   | 0E | Retrain stage 0E: fight check groundwork | `task,area:python` |
   | 1-code | Retrain stage 1-code: hunting environment and probe | `feature,area:python,area:training` |
   | 1-train | Retrain stage 1-train: s1_hunt_grunt and s1_hunt_roster | `task,area:training` |
   | 2-code, 2-train | Retrain stage 2-code / 2-train: retreating when hurt | as Stage 1 |
   | 3-code, 3-train | Retrain stage 3-code / 3-train: using cover | as Stage 1 |
   | 4-code, 4-train | Retrain stage 4-code / 4-train: flanking | as Stage 1 |
   | 5-design | Retrain stage 5-design: the multi-monster environment | `research,area:training,needs-human` |
   | 5-code, 5-train | Retrain stage 5-code / 5-train: group play | as Stage 1 |

   Each stage's tracking issue is titled `Retrain stage N: <name>` with labels
   `task,area:training`, created when its first part starts.
4. Start the lowest open part as AGENTS.md "Work tracking" says:

   ```bash
   gh issue edit <N> --add-label in-progress
   git checkout main && git pull
   git checkout -b issue-<N>-<short-name>
   gh issue comment <N> --body "Starting. Approach: <one or two sentences from this spec>."
   ```

5. For part 0A, the first code you write is the failing test of its step 1.

# Roadmap

Milestone overview. The unit of work is a GitHub issue; each item below names
its issue. Tick an item here in the same commit that completes it. Issues for
M4 onward are created when that milestone starts.

## M0 — Scaffold

- [x] Repository layout, licence, agent guide
- [x] CMake build of `libdqsim` (version stub) with a C test
- [x] Python package loading `libdqsim`, with tests
- [x] Game-data checker and repository hygiene guard (pre-commit and CI)

## M1 — Headless simulator

- [x] Choose engine base: QuakeSpasm (see ARCHITECTURE.md) (#2)
- [x] Vendor QuakeSpasm under `engine/third_party/quakespasm/` with `UPSTREAM.md` (#3)
- [x] Strip to server core; build with no SDL, GL, audio, or network (#4)
- [x] C API: create, load map, reset(seed), step, destroy (#5)
- [x] Seeded random numbers and fixed frame time (no wall clock) (#6)
- [x] Determinism test: same seed and inputs give identical state hashes (#7)
- [x] Fidelity test: recorded player inputs replay to the same positions as the unmodified engine (#8)
- [x] Snapshot and restore (#9)
- [x] Throughput benchmark (ticks per second per core) recorded in this file (#10)

Throughput (2026-11-09, 24-thread desktop CPU, release build, 1/72 s ticks,
`dqsim_bench gamedata <map> 20000 <processes>`, player running and firing):

| Map | 1 process | 24 processes, total |
| --- | --- | --- |
| e1m1 | 114,000 ticks/s (1,580x real time) | 1,540,000 ticks/s |
| e1m3 | not measured | 800,000 ticks/s |

## M2 — Environment

- [x] Simulator primitives for observations: ray casts, entity list, teleport, client commands (#11)
- [x] Python vectorised environment over worker processes (#12)
- [x] Random and bee-line baseline agents; real-map test (`gamedata` marker) (#13)

## M3 — Player agent: movement and navigation

- [x] PPO training loop on the RTX 3090 (#14)
- [x] Goal-reaching task on a walk graph, with a distance curriculum (#15)
- [x] Evaluation on fixed tasks against the baselines (#16)

First result (2026-11-09, `configs/nav_e1m1.toml`, 2,000,000 decisions, about
5 minutes on the RTX 3090 with 20 simulator processes, 6,800 decisions/s).
`eval_nav` on E1M1, 300 fixed tasks, goal 300 to 2,000 units away on foot,
25 s limit:

| Policy | Success | Mean time to goal |
| --- | --- | --- |
| Random actions | 11.0% | 12.4 s |
| Bee-line (turn to goal, run) | 72.7% | 2.3 s |
| Trained | 96.0% | 1.7 s |

### Handoff gate: passed

A trained policy reaches goals on a real map far more often than both
baselines. From here the work is tracked in the open GitHub issues. Known
limits of this first policy, each with an issue:

- The policy has no memory and was trained on one map (#19). Models must
  work on maps they have never seen (ARCHITECTURE.md, "Map independence"),
  so #19 trains on many maps and reports held-out maps separately.
  **Fixed 2026-11-10 (#19): see the multi-map result below.**
- No map outside id's own has ever been loaded (#28). **Fixed 2026-11-11
  (#28): a generated map loads, runs and navigates — see \`bspgen\`.**
- Monsters are passive and the player invulnerable during navigation
  training; combat comes in M4.

Re-measured after the walk graph was connected (2026-11-10, #18:
`nav_e1m1_g3`, same 2,000,000 decisions; 89% of point pairs joined in at
least one direction, up from 17%, longest route 6,790 units up from 2,150;
many routes are real one-way drops, so 73% of ordered pairs are walkable).
The tasks now span the level, which makes them harder — the first table's
tasks were almost all short because fragmentation left nothing longer:

| Policy | Success (goal 300–2,000) | Mean time | Success (goal 200–1,000) |
| --- | --- | --- | --- |
| Random actions | 4.3% | 14.2 s | 15.3% |
| Bee-line | 35.7% | 4.0 s | 67.3% |
| Trained | 75.3% | 5.4 s | 88.3% |

Memory and many maps (2026-11-10, #19: `configs/nav_maps.toml`, GRU
actor-critic with bptt-chunked PPO, 10,000,000 decisions, about 75 minutes
with 22 simulator processes — one per training map — at 2,756 decisions/s).
Trained on 22 of the 30 `id1` levels whose walk graph builds in sane time
(the other six, e1m2/e1m3/e4m2/e4m5/e4m8/dm3, are #29); held out, by a rule
fixed before training, the 3rd level of each episode that builds (e2m3,
e3m3, e4m3) and every buildable deathmatch map (dm1, dm2, dm4, dm5, dm6).
`eval_nav`, 300 fixed tasks per map, goal 200 to 1,000 units away on foot,
25 s limit:

| Policy | Training maps (22) | Held-out maps (8) | Held-out mean time |
| --- | --- | --- | --- |
| Random actions | 11.9% | 11.2% | 12.0 s |
| Bee-line | 63.6% | 58.5% | 2.3 s |
| Trained | 84.7% | **81.6%** | 2.7 s |

The trained policy beats bee-line on every one of the eight held-out maps
(per-map results in #19), and the gap between training and held-out maps is
about 3 points: the first evidence that the model reads geometry, not the
map. Two bugs found on the way, both fixed: some start points are blocked
by seeded monster wander (#30), and six maps' walk graphs take over an hour
to build (#29; fixed 2026-11-12 — every `verify_links` link cleared the
whole per-source BFS cache, so corridor-heavy maps re-ran full BFS in
Python for hours; invalidating only the sources that reach the new edge's
tail is exact and builds all six in 31–1,626 s with byte-identical
graphs on the maps that already built).

Ray kinds in the observation (2026-11-11, #26: `configs/nav_maps2.toml`,
same maps, curriculum, PPO settings and 10,000,000 decisions as `nav_maps`;
the observation gained one surface-kind channel per ray, OBS_DIM 56 → 104,
so the old checkpoint could not be reused). Training ran at 7,598
decisions/s after the walk graphs built (the slowest, e2m7, took about an
hour and gated the first update). `eval_nav`, 300 fixed tasks per map, goal
200 to 1,000 units away on foot, 25 s limit — the same task list as above:

| Policy | Training maps (22) | Held-out maps (8) | Held-out mean time |
| --- | --- | --- | --- |
| Random actions | 11.4% | 12.1% | 11.7 s |
| Bee-line | 64.0% | 57.5% | 2.5 s |
| Trained | 70.9% | **72.7%** | 7.0 s |

The retrained policy still beats both baselines on the held-out group and
on seven of its eight maps (dm4 is the exception: 57.7% against bee-line's
60.0%), but it is below `nav_maps` (84.7% / 81.6%) and much slower to the
goal (7.0 s against 2.7 s). That is the measured cost of this run, not a
reason to re-roll it: the kind channel is in place before any monster is
trained, which is what #26 required. Whether a longer or re-tuned run
recovers the old scores is future work, not part of #26.

Combat training, first attempt (2026-11-11, #38, still open:
`configs/combat_pool.toml`, sparse reward and 600-decision timeout from
#32, the e1m1 Grunt and e1m4 Ogre as opponents, 10,000,000 decisions at
about 5,400 steps/s after extending the first 5M run). `eval_combat`,
200 fixed tasks per opponent:

| Policy | Grunt, e1m1 (trained) | Ogre, e1m4 (trained) | Knight, e1m4 (held-out type) | Ogre, e1m5 (held-out map) |
| --- | --- | --- | --- | --- |
| Random actions | 15.5% | 0.0% | 3.0% | 2.5% |
| Scripted shooter | 25.0% | 9.5% | 9.0% | 22.5% |
| Trained | 18.5% | 0.0% | 5.0% | 5.0% |

The trained policy beats random on the Grunt but not the scripted
shooter, and wins nothing against the trained Ogre, so #38's bar (beat
both baselines on the trained opponents) is **not met** and the issue
stays open; the numbers stand as measured rather than re-rolled. The
working branch is `issue-38-combat-training`.

Second attempt (2026-11-11, #38 after the #39 diagnosis:
`configs/combat_pool2.toml`, training episodes start at 100–350 units
with line of sight allowed instead of 200–600 with it broken, same
opponents and sparse reward, 10,000,000 decisions). Same fixed tasks:

| Policy | Grunt, e1m1 (trained) | Ogre, e1m4 (trained) | Knight, e1m4 (held-out type) | Ogre, e1m5 (held-out map) |
| --- | --- | --- | --- | --- |
| Random actions | 15.5% | 0.0% | 3.0% | 2.5% |
| Scripted shooter | 25.0% | 9.5% | 9.0% | 22.5% |
| Trained | 16.0% | 0.0% | 4.5% | 3.5% |

Placement was not the lever either: the bar is still not met, and the
next suspect named by #39 is the action representation itself (turn
quantisation, no pitch aim), which is a design question rather than
more training budget.

Third attempt (2026-11-11, #38 after the #40 aim spike:
`configs/combat_pool3.toml`, finer turn steps {-30,-15,-5,0,+5,+15,+30}
and a pitch aim head in the action layout, same opponents, sparse
reward and original placement, 10,000,000 decisions). Same fixed
tasks; baselines use the same new action heads:

| Policy | Grunt, e1m1 (trained) | Ogre, e1m4 (trained) | Knight, e1m4 (held-out type) | Ogre, e1m5 (held-out map) |
| --- | --- | --- | --- | --- |
| Random actions | 13.5% | 0.0% | 2.5% | 5.0% |
| Scripted shooter | 23.0% | 9.5% | 9.0% | 22.5% |
| Trained | 10.5% | 0.0% | 1.5% | 5.0% |

Aim fidelity in the scripted policy was worth wins (#40), but the
trained policy with the finer heads scored *worse*, so representation
alone does not explain the gap either. #38 stays open and blocked;
three measured attempts now bound the problem to optimisation and
task structure rather than any single input the policy is missing.

Optimisation and task-structure grid (2026-11-11/12, #38: eight
grunt-only 10M-decision runs, each a single lever, all evaluated on
the same fixed 200 Grunt tasks where random scores 13.5% and the
scripted shooter 23.0%):

| Run | Lever | Eval win | Own-placement train win |
| --- | --- | --- | --- |
| `combat_grunt_only` | none (old placement) | 11.5% | 16.9% |
| `combat_grunt_entlow` | entropy_coef 0.001 | 12.5% | 15.9% |
| `combat_grunt_lrhigh` | lr 1e-3 | 17.0% | 15.5% |
| `combat_grunt_ep8` | epochs 8 | 9.0% | 15.2% |
| `combat_grunt_fightstart` | close in-sight starts | 20.5% | 61.3% |
| `combat_grunt_fightstart_entlow` | + entropy_coef 0.001 | 9.0% | 55.6% |
| `combat_grunt_fightstart_damage` | + dense damage reward | 15.0% | 57.9% |
| `combat_grunt_mixed` | 100–600 mixed band | 12.0% | ~40% |

The fight-start placement trains the fight itself well (61–65% windows
against random's 39% and the shooter's 73% on that placement) — the
learning problem on a fixed distribution is solved. What no run does
is transfer to the fixed evaluation tasks, which all begin far away
with line of sight broken: the committed variant times out on 59.5%
of them (it never learned to search), and every variant lands at or
below the old-layout like-for-like 18.5%.

Placement curriculum (2026-11-12, #38 after the maintainer chose to
anneal the placement outward: `configs/combat_grunt_curriculum.toml`,
close fights 0–2M → wide band 2–6M → the evaluation placement itself
6–10M with 30% of episodes kept close so the fighting signal survives;
the trainer divides the global schedule across environments exactly).
Stage 3 trained at 22–32% and was still improving at the budget end,
but the evaluation scored **19.0%** — below even pure fight-start's
20.5%, which remains the best of nine measured variants (9.0–20.5%,
shooter 23.0%, random 13.5%). No training-side lever has closed the
last few points on the far, sight-broken tasks, where the trained
policy sits at statistical near-parity (z ≈ −0.8 at n=200). The
decision — iterate the schedule, change the action heads, change the
evaluation's task definition, or accept parity and move to the
monster-side training (#35/#36) — is with the maintainer; #38 is
parked as `blocked` + `needs-human`.

## M4 — First monster brain

- [x] Ray casts report what they hit (liquids, doors, sky), before any monster is trained (#26)
- [x] Decide how the hook treats maps that ship their own game code (#27)
- [x] Brain hook: policy replaces QuakeC decisions for one monster type (Grunt) (#31)
- [x] Research the combat task (episode, reward, termination, opponent pool); the training issue follows from it (#32)
- [ ] Train the player agent for combat (#38)
- [x] Monster observation with fair senses only, and a skill input (Quake's `skill`, 0 to 3) next to the style input (design: #22) (#33)
- [x] Engine handicaps scaled by skill: reaction delay and aim error, seeded, identical in simulator and client (#22) (#34)
- [x] Train the Grunt against the pool at every skill level; style conditioning (aggressive, defensive, covert) (#35)
- [x] Evaluation against the stock QuakeC Grunt under identical conditions, at each skill level (#36; the both-side-stock sanity row became stock-vs-scripted-shooter — no stock player AI exists, see the M4 notes)

Monster-side training (2026-11-12, #35: `configs/grunt_pool.toml`, the
roles inverted — the policy drives the nominated Grunt through the brain
hook, one held discrete decision per 6-tick block, while the
environment drives the player opponent; `MonsterEnv` keeps the combat
environment's contract so PPO runs unmodified). Only four id1 maps stock
`monster_army` (measured): e1m1, e2m1, e4m1 train, e3m1 held out.
Opponents rotate per worker: the scripted shooter, a random player, and
#38's trained player checkpoint at parity level (the option-E decision
there). Skills 0–3 and four style vectors are sampled per episode into
the engine handicaps and the observation. 10,000,070 decisions, sparse
terminal reward. Evaluation (`eval_monster`): fixed tasks per map and
skill, the same player opponent for both brains — win rate is the
player's death rate:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps, stock Grunt | 41.3% | 49.3% | 62.7% | 76.7% |
| Training maps, trained Grunt | 43.3% | 58.7% | **67.3%** | 70.0% |
| Held-out e3m1, stock Grunt | 58.0% | 80.0% | 78.0% | 78.0% |
| Held-out e3m1, trained Grunt | 58.0% | 82.0% | **82.0%** | 74.0% |

The trained Grunt beats the stock QuakeC Grunt at skill 2 (the #35
Done-when bar) on both groups, and at skill 1; at skill 3 the stock
nightmare brain, unhandicapped and refiring near-instantly, stays
ahead. Per map at skill 2: e1m1 52→66%, e2m1 66→68%, e4m1 70→68%,
held-out e3m1 78→82% — differences of 2–4 points on 150 tasks are
about 1 standard error, so the win is a consistent direction, not a
proven margin. Determinism is hash-checked per episode (test in
`test_monster.py`).

The M4 handoff metrics (#36; decisions, 12 per second — means over the
tasks where the event happened):

| Metric, training maps | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Time to first damage, stock | 97 | 80 | 53 | 31 |
| Time to first damage, trained | 87 | 78 | **44** | 29 |
| Time to kill, stock (won tasks) | 278 | 252 | 203 | 171 |
| Time to kill, trained (won tasks) | 303 | 276 | 212 | 163 |

At skill 2 the trained Grunt engages sooner (first damage at 44
decisions vs the stock brain's 53) and kills slightly more often but a
little slower. The sanity row of the harness (`eval_monster --sanity`)
runs the stock Grunt against the scripted shooter — the project's stock
player stand-in since #32 — because a literal both-side-stock mirror
does not exist: the player has no stock AI, the hook drives one
monster, and allied Grunts never fight each other. Measured: 42–58%
monster wins on training maps (57.3% at skill 2), first damage 27–81
decisions, and the outcome rates are asserted to fill exactly 100%.

## M5 — Play against it

- [x] Playable client built from the same vendored engine (#42: the `deepquake` target; client sources vendored from the same upstream commit, no vendored-file edits)
- [x] ONNX inference in the client (CUDA with CPU fallback) (#43: the shared brain hook drives one Grunt from the ONNX export; 78-87 microseconds per decision measured; batching per tick arrives with multi-monster, M6)
- [x] Stock AI fallback when no model is present (#44: the fallback is hash-proven bit-identical to never nominating; corrupt and wrong-architecture models are refused with clear logs)
- [x] Model manifest, download tool, first public release (#45: reproducible checksum manifests, `tools/download_model.py`, release `models-v1` published with the Grunt and player policies)

## M6 — Full roster

- [ ] Remaining monster types, simplest first (#46: every monster of a class driven; #47: the Dog driven; #48: the Dog joins the Grunt in one shared model, release `models-v2`)
- [ ] Group coordination between monsters
- [ ] Style settings exposed as cvars; difficulty handicaps tuned per monster type (difficulty itself follows `skill` from M4, see ARCHITECTURE.md)

The hook reaches any stock class (2026-11-13, #49): a class does not name
its QuakeC functions after itself — the Enforcer's states are `enf_stand1`,
`enf_run1`, `enf_atk1`, the Shambler's are `sham_*`, the Hell Knight's
`hknight_*` — so the hook reads the prefix off the monster's own decision
think instead of deriving one from the classname, and finds the attack entry
by shape (`ogre_melee`, `shal_attack1`) and the shot from what the attack
chain calls (`enforcer_fire`, not `enf_fire`). Measured on the local game
data: the Enforcer went from 0 hook calls in 72 ticks and no damage under
the policy to 67 calls and 45 damage, the Ogre from no damage to 30; the
Grunt, Dog, Knight, Demon and Zombie are unchanged. The aim error now
covers every shot of an attack, so a class that fires twice (the Enforcer)
is handicapped on both shots instead of keeping an exact second one.

The **Remaining monster types** item is **not** ticked: the Enforcer's hook
path is proved (#49) and the Enforcer has joined the shared model (#50,
below), but the item covers every remaining type — the Ogre, the Fiend, the
Shambler, the Hell Knight and the rest are still stock.

Shared model, second type (2026-11-13, #48: `configs/roster_pool.toml`,
one network for the Grunt and the Dog, type one-hot appended to the
observation, a policy head per type; warm-started from `grunt_pool` and
fine-tuned 10,000,896 decisions on e1m1/e2m1/e4m1 with both bodies sampled
per episode, e3m1 held out; everything else as `grunt_pool`). Same
evaluation as M4 (50 fixed tasks per map and skill, trained player
opponent); the stock Grunt row reproduces M4's exactly:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps, stock Dog | 46.7% | 52.0% | 63.3% | 65.3% |
| Training maps, shared model's Dog | 51.3% | 56.0% | **69.3%** | 72.7% |
| Held-out e3m1, stock Dog | 72.0% | 90.0% | 76.0% | 74.0% |
| Held-out e3m1, shared model's Dog | 76.0% | 90.0% | **88.0%** | 84.0% |
| Training maps, single-type Grunt (M4) | 43.3% | 58.7% | 67.3% | 70.0% |
| Training maps, shared model's Grunt | 45.3% | 56.0% | 65.3% | 72.0% |
| Held-out e3m1, single-type Grunt (M4) | 58.0% | 82.0% | 82.0% | 74.0% |
| Held-out e3m1, shared model's Grunt | 58.0% | 84.0% | 84.0% | 78.0% |

The Dog is ahead of the stock Dog in 7 of 8 cells (equal in one) and the
Grunt does not regress after sharing (within -2.7 to +4.0 points of the
single-type model, 5 cells up, 2 down, 1 equal; about one standard error).
In the client, `roster` drives all 42 Grunts and Dogs of e1m1 from one
session at 45-46 microseconds per decision (the v1 Grunt model: 32).

Shared model, third type (2026-11-13, #50: `configs/roster3_pool.toml`,
`MONSTER_TYPES` = army, dog, enforcer, the observation 119 + 3 = 122 wide,
warm-started from the two-type roster model and fine-tuned 10,000,896
decisions on e1m1/e2m1/e4m1 with all three bodies sampled per episode,
e3m1 held out). An episode's body is sampled from the classes the level
stocks — e1m1 has no Enforcer, so sampling one there only wasted episodes
(#50). Same evaluation as M4 (50 fixed tasks per map and skill, trained
player opponent); the stock rows reproduce the table above exactly, which
confirms the protocol did not move:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps, stock Grunt | 41.3% | 49.3% | 62.7% | 76.7% |
| Training maps, shared model's Grunt | 40.0% | 51.3% | 61.3% | 70.7% |
| Training maps, stock Dog | 46.7% | 52.0% | 63.3% | 65.3% |
| Training maps, shared model's Dog | 57.3% | 64.7% | 70.7% | 76.0% |
| Training maps (e2m1/e4m1), stock Enforcer | 42.0% | 55.0% | 69.0% | 69.0% |
| Training maps (e2m1/e4m1), shared model's Enforcer | 46.0% | 53.0% | 70.0% | 73.0% |
| Held-out e3m1, stock Grunt | 58.0% | 80.0% | 78.0% | 78.0% |
| Held-out e3m1, shared model's Grunt | 68.0% | 76.0% | 86.0% | 74.0% |
| Held-out e3m1, stock Dog | 72.0% | 90.0% | 76.0% | 74.0% |
| Held-out e3m1, shared model's Dog | 84.0% | 88.0% | 84.0% | 84.0% |
| Held-out e3m1, stock Enforcer | 82.0% | 74.0% | 84.0% | 88.0% |
| Held-out e3m1, shared model's Enforcer | 68.0% | 74.0% | 82.0% | 86.0% |

Against the two-type roster model, per cell of the difference: the Dog is
up in 5 of 8 cells (+6.0 to +8.7 on the training maps, +8.0 held out at
skill 0), equal in one, down in two (-2.0, -4.0 held out). The Grunt's
deltas run -5.3 to +10.0, mixed in sign within every map and skill (e2m1
up at skill 0/1 and down at 2/3, e4m1 the reverse), -2.9 points on average
over the 16 per-map cells (-1.4 over the 8 group cells); every cell of
both classes is within about one
standard error of the difference, which is the noise floor here, not a
regression with a direction. The Enforcer beats the stock Enforcer at
skill 2 on the training maps — 70.0% against 69.0%, the #35 bar — by one
task in a hundred, so call it parity with the stock Enforcer rather than a
win: ahead of stock at skills 0 and 3 on the training maps, behind only at
held-out skill 0 (-14.0, one standard error at 50 tasks). The Enforcer's
mean time to first damage is shorter than stock's in 5 of 8 cells (it
engages sooner). In the client, the model drives e3m1's 64 monsters of all
three classes from one session (34 Grunts, 9 Dogs, 21 Enforcers) at 50-51
microseconds per decision. Released as `models-v3` with the three-entry
class map (`monster_army 0`, `monster_dog 1`, `monster_enforcer 2`).

The melee commitment was fixed first (2026-11-14, #51: `configs/roster4_damage.toml`,
the same three-type roster trained with the dense `damage` reward instead of
the sparse terminal one). Under the sparse reward the Dog never committed to
closing — its back commands outweighed ahead commands 2.9x (the two-type
model) and 7.6x (the three-type one), the attack bit did all the closing and
the explicit movement cancelled it between lunges, and it crept at the player
without biting. Under the damage reward the sign flips (114:213), the probe
kills rise from 1/8 to 5/8, and the Dog is ahead of the stock Dog in all 8
evaluation cells (held-out 96%/98% at skills 1/3); the Grunt and Enforcer
bars still pass. Released unchanged as part of `models-v4`.

Shared model, fourth type (2026-11-14, #53: `configs/roster5_ogre.toml`,
`MONSTER_TYPES` = army, dog, enforcer, ogre, the observation 119 + 4 = 123
wide, warm-started from the damage-reward three-type model and fine-tuned
10,000,896 decisions on the union of the two class pools - e1m1/e2m1/e4m1
for the Grunt, Dog and Enforcer, e1m2/e2m2/e4m2 for the Ogre (no Ogre map
overlaps the previous four), e3m1 and e3m2 held out; reward stays `damage`,
everything else as before). The hook change that made the Ogre drivable:
the attack entries come from the spawn's own `th_melee`/`th_missile` fields
(the throw lived only in `th_missile`, unreachable from a name-shaped
probe), the ranged entry is the spawn's `th_missile` only when its chain
reaches a shot (the Dog's is a one-shot pounce), fire frames are found by
walking the attack chains, a shot function is one whose name contains `fire`
or `shot`, the aim error also rolls the monster's facing (the grenade aims
from `makevectors(self.angles)`), and the stock AI's sight globals are set
before the attack. Same M4 evaluation, the Ogre on its own maps:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps (e1m2/e2m2/e4m2), stock Ogre | 73.3% | 78.7% | 84.7% | 92.0% |
| Training maps, shared model's Ogre | 69.3% | 82.0% | **87.3%** | 89.3% |
| Held-out e3m2, stock Ogre | 66.0% | 72.0% | 86.0% | 86.0% |
| Held-out e3m2, shared model's Ogre | 68.0% | 74.0% | **92.0%** | **92.0%** |

The Ogre beats the stock Ogre at skill 2 on the training maps (the #35 bar),
is ahead in 3 of 4 held-out cells and faster to first damage in 6 of 8. The
existing classes do not regress against the damage-reward three-type model:
the Dog is ahead of the stock Dog in all 8 cells again (74.0% vs 63.3% at
training skill 2, held-out 98% vs 74% at skill 3) and level with the #51
model; the Grunt is cell-for-cell level with it; the Enforcer is at 74.0% vs
stock 69.0% at training skill 2 (the #50 bar, margin unchanged). In the
creep probe the Dog now charges forward relentlessly (back:ahead 15:584,
against the original models' 2.9-7.6x backward). No single id1 map stocks
all four classes, so the client check drives them across two maps: e2m1's
46 monsters (13 Grunts, 7 Dogs, 26 Enforcers, types 0-2) at 47 microseconds
per decision and e1m2's 28 (16 Grunts, 12 Ogres, types 0 and 3) at 50.
Released as `models-v4` with the four-entry class map (`monster_army 0`,
`monster_dog 1`, `monster_enforcer 2`, `monster_ogre 3`).

Shared model, fifth type (2026-11-14, #55: `configs/roster7_hknight.toml`,
`MONSTER_TYPES` = army, dog, enforcer, ogre, hell knight, the observation
119 + 5 = 124 wide, warm-started from the four-type roster model with the
new type's head taken from the **Enforcer's** — a new optional
`init_head_from` maps a new type to the source type whose mechanical shape
matches; the default rule (the last source head) gave the Hell Knight the
Ogre's head and 15,000,000 decisions bought only stock parity (74.0% vs
74.0% at skill 2, `roster6_hknight`, kept as the negative result), while
the mapped head beat stock within the standard 10,000,896). Trained on the
union pool plus e2m3/e2m5/e4m5 (the Hell Knight's stocked maps), e3m1/e3m2/
e3m6 held out; reward stays `damage`. Same M4 evaluation, the Hell Knight
on its own maps:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps (e2m3/e2m5/e4m5), stock Hell Knight | 64.7% | 68.7% | 74.0% | 76.0% |
| Training maps, shared model's Hell Knight | 55.3% | 66.7% | **77.3%** | **80.7%** |
| Held-out e3m6, stock Hell Knight | 74.0% | 90.0% | 86.0% | 86.0% |
| Held-out e3m6, shared model's Hell Knight | 76.0% | **92.0%** | 84.0% | 86.0% |

The Hell Knight beats the stock Hell Knight at skill 2 on the training maps
(the #35 bar) and at skill 3; its weak cell is skill 0 (-9.4, the stock
knight is a 74-92% baseline everywhere). The other four classes do not
regress: the Dog is ahead of the stock Dog in all 8 cells at both eval
seeds tried (and the same checkpoint's cells swing up to 14 points between
seeds, which is the eval's own variance - the seed defines the task set and
the policy's sampling); the Ogre beats stock at skill 2 and held-out skills
2/3 (94%/92%); the Enforcer is at 73.0% vs stock 69.0%; the Grunt is at
62.7% vs stock 62.7% at skill 2 (level, as #51/#53 left it). In the client,
the five classes are driven across two maps: e2m1's 46 monsters (types 0-2)
at 51 microseconds per decision and e2m5's 42 (24 Ogres type 3, 18 Hell
Knights type 4) at 54. Released as `models-v5` with the five-entry class
map (`monster_army 0`, `monster_dog 1`, `monster_enforcer 2`,
`monster_ogre 3`, `monster_hell_knight 4`).

Shared model, sixth type (2026-11-14, #56: `configs/roster8_shalrath.toml`,
`MONSTER_TYPES` = army, dog, enforcer, ogre, hell knight, shalrath, the
observation 119 + 6 = 125 wide, warm-started from the five-type roster model
with the Shalrath's head taken from the Enforcer's, the #55 mechanism). Two
hook generalizations the Shalrath needed: the decision-think pattern accepts
the digitless single-state stand (its stand think is `shal_stand`, so a
standing Shalrath stayed stock-controlled until the pattern allowed it - the
ctest shows 144 policy decisions in 2 s while standing), and a shot function
is one whose name contains fire, shot or missile (the spike launch is
`ShalMissile`). Trained on the union pool plus e2m6/e4m6/e4m7, e4m8 held
out; reward stays `damage`. Same M4 evaluation, the Shalrath on its own maps:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps (e2m6/e4m6/e4m7), stock Shalrath | 34.0% | 26.7% | 28.0% | 28.0% |
| Training maps, shared model's Shalrath | 38.7% | 29.3% | **32.0%** | **32.7%** |

The Shalrath beats the stock Shalrath in every training cell (the #35 bar at
skill 2 by 4.0 points; the stock Shalrath's own win rates are low - its
homong spikes are strong but its placement is sparse and the trained player
opponent punishes it). The held-out e4m8 cells at skill 0 are void (0% both
sides with no damage timings: skill 0 stocks too few Shalraths there for
placement to succeed; not a result), and at skills 1-3 the held-out picture
is level to slightly behind (56/76/82 stock vs 56/76/82/82 trained at
skills 1-3). The other five classes do not regress beyond the eval's own
seed variance (the #55 measurement): the Dog is ahead of the stock Dog in
all 8 cells again (76.0% vs 63.3% at training skill 2), the Ogre beats stock
at skill 2 (86.7% vs 84.7%), the Hell Knight at 77.3% vs 74.0%, the Enforcer
at 66.0% vs 69.0% (-3.0, 0.4 standard errors at 100 tasks) and the Grunt at
64.7% vs 62.7%. In the client, the six classes are driven across three maps:
e2m1's 46 monsters (types 0-2) at 52 microseconds per decision, e1m2's 28
(types 0 and 3) at 56, and e4m7's 22 (18 Hell Knights type 4, 10 Shalraths
type 5) at 55. Released as `models-v6` with the six-entry class map
(`monster_army 0`, `monster_dog 1`, `monster_enforcer 2`, `monster_ogre 3`,
`monster_hell_knight 4`, `monster_shalrath 5`).

Shared model, seventh type (2026-11-15, #57: `configs/roster9_knight.toml`,
`MONSTER_TYPES` = army, dog, enforcer, ogre, hell knight, shalrath, knight,
the observation 119 + 7 = 126 wide, warm-started from the six-type roster
model with the Knight's head taken from the **Dog's** — a melee closer, the
closest mechanical shape). The increment measured what the Knight is in this
progs first: a **pure melee monster** — the spawn's entries both point at the
sword chain (knight_atk1..10, ai_melee), the knight_bow chain is vestigial
(ai_turn only) and no knight function spawns anything, so the stock-lore
fireball does not exist in this engine and the ctests pin the sword instead
(22 hits in 8 s at 50 units, chain gated). Trained on the union pool plus
e1m4/e4m3/e4m4, e1m5 held out; reward stays `damage`. Same M4 evaluation,
the Knight on its own maps:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps (e1m4/e4m3/e4m4), stock Knight | 70.7% | 78.7% | 77.3% | 88.0% |
| Training maps, shared model's Knight | 72.7% | 81.3% | **83.3%** | **89.3%** |
| Held-out e1m5, stock Knight | 58.0% | 84.0% | 80.0% | 86.0% |
| Held-out e1m5, shared model's Knight | 64.0% | **86.0%** | **90.0%** | 86.0% |

The Knight beats the stock Knight in every training cell (the #35 bar at
skill 2 by 6.0 points) and in 3 of 4 held-out cells. The other six classes
do not regress beyond the eval's own variance (the #55/#56 seed-swing
reference): the Dog is ahead of the stock Dog in all 8 cells (71.3% vs
65.3% at training skill 3), the Ogre at 82.0% vs 84.7% at skill 2 (-2.7,
0.5 standard errors) and ahead in all 4 held-out cells, the Hell Knight at
76.0% vs 74.0%, the Enforcer at 72.0% vs 69.0%, the Grunt at 60.0% vs
62.7% (-2.7, 0.5 SE, with its known skill-3 trade-off -8.0), and the
Shalrath holds its #56 result (32.7% vs 28.0% at skill 2). In the client,
the seven classes are driven across four maps: e2m1's 46 monsters (types
0-2) at 60 microseconds per decision, e2m5's 42 (types 3-4) at 58, e1m4's
40 (21 Ogres type 3, 19 Knights type 6) at 59, and e4m7's 22 (types 4-5) at
65 — the per-decision cost creeps with the model's width (models-v3's
two-class client ran at 50-51). Released as `models-v7` with the
seven-entry class map (`monster_army 0`, `monster_dog 1`, `monster_enforcer 2`,
`monster_ogre 3`, `monster_hell_knight 4`, `monster_shalrath 5`,
`monster_knight 6`).

Shared model, eighth type (2026-11-15, #58: `configs/roster10_demon.toml`,
`MONSTER_TYPES` = army, dog, enforcer, ogre, hell knight, shalrath, knight,
demon1, the observation 119 + 8 = 127 wide, warm-started from the seven-type
roster model with the Fiend's head taken from the **Dog's** — a melee closer,
per the #55 mechanism). The Fiend is a pure melee monster with a leap, and
the ctests pin it (the leap closes 100 to 48 units and the scratch lands, 7
hits in 8 s). Trained on the union pool plus e1m6/e2m7, e2m4 held out;
reward stays `damage`. Same M4 evaluation, the Fiend on its own maps:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps (e1m6/e2m7), stock Fiend | 46.0% | 95.0% | 99.0% | 99.0% |
| Training maps, shared model's Fiend | 48.0% | 97.0% | 99.0% | 100.0% |
| Held-out e2m4, stock Fiend | 62.0% | 84.0% | 86.0% | 88.0% |
| Held-out e2m4, shared model's Fiend | 56.0% | 82.0% | 86.0% | 86.0% |

**The stock Fiend saturates the task**: at skills 1-3 it kills the trained
player opponent 95-99% of the time, so the cells are at ceiling and the
#35 bar's "beats" reduces to "ties at ceiling" at skill 2 (99.0 vs 99.0)
while skills 0/1/3 are ahead (+2.0/+2.0/+1.0). The Fiend is never behind
stock on the training maps; on the held-out map it is 2-6 points behind in
every cell, within the eval's own seed variance (the #55/#56 measurement,
up to 14 points per cell on the same checkpoint). The other seven classes
do not regress: the Dog is ahead of the stock Dog in all 8 cells (82.0% vs
65.3% at training skill 3), the Enforcer at 78.0% vs 69.0% (+9.0), the
Knight at 81.3% vs 77.3%, the Ogre at 86.0% vs 84.7%, the Hell Knight at
75.3% vs 74.0%, the Shalrath holds its #56 result (32.0% vs 28.0%), and the
Grunt at 60.0% vs 62.7% (-2.7, 0.5 SE, with its known skill-3 trade-off).
In the client, the eight classes are driven across five maps: e2m1's 46
monsters (types 0-2) at 52 microseconds per decision, e2m5's 48 (types 3-4,
7) at 51, e1m4's 43 (types 3, 6-7) at 52, e4m7's 30 (types 4-5, 7) at 55,
and e2m4's 44 (types 3-4, 12 Fiends type 7) at 53. Released as `models-v8`
with the eight-entry class map (`monster_army 0`, `monster_dog 1`,
`monster_enforcer 2`, `monster_ogre 3`, `monster_hell_knight 4`,
`monster_shalrath 5`, `monster_knight 6`, `monster_demon1 7`).

Shared model, ninth type (2026-11-16, #59: `configs/roster11_shambler.toml`,
`MONSTER_TYPES` = army, dog, enforcer, ogre, hell knight, shalrath, knight,
demon1, shambler, the observation 119 + 9 = 128 wide, warm-started from the
eight-type roster model with the Shambler's head taken from the **Hell
Knight's** — a melee-and-beam hybrid, the closest mechanical shape among the
trained types, per the #55 mechanism). The Shambler needs no new hook rule:
its attack entry is the `sham_melee` dispatcher (the claw at melee range,
the lightning beam at range, chain `sham_magic1..12` launching through
`CastLightning`), which the #53 rules reach unchanged — and although
`CastLightning` matches no fire/shot/missile name rule, the skill handicap
still moves the beam, proven by state hash (the clean and 45°-error windows
differ). The ctests pin the drive (92 units in 2 s), the claw (2 hits in
8 s at 60 units, health 100 → -22) and the beam (9 hits in 8 s at 300
units). Trained on the union pool plus e1m8/e3m4 (e1m6 already in it; 19
maps, 19 envs, 10,000,384 decisions at about 2,720 steps/s), e1m3 held out;
reward stays `damage`. Same M4 evaluation, the Shambler on its own maps:

| Player-death rate | skill 0 | skill 1 | skill 2 | skill 3 |
| --- | --- | --- | --- | --- |
| Training maps (e1m8/e3m4/e1m6), stock Shambler | 62.0% | 94.0% | 72.7% | 75.3% |
| Training maps, shared model's Shambler | 59.3% | 88.7% | **74.7%** | 74.7% |
| Held-out e1m3, stock Shambler | void | void | 84.0% | 90.0% |
| Held-out e1m3, shared model's Shambler | void | void | 80.0% | **94.0%** |

The Shambler beats the stock Shambler at skill 2 on the training maps (the
#35 bar) by 2.0 points; its other training cells sit within about one
standard error (skills 0/1/3: -2.7/-5.3/-0.7). The held-out e1m3 cells at
skills 0/1 are void (0% both sides with no damage timings: skills 0/1 stock
too few Shamblers there for placement to succeed — sparse stocking, the
Shalrath's recorded property from #56); at skill 3 the held-out cell is
ahead (+4.0) and at skill 2 behind (-4.0), both within the eval's own seed
variance (the #55/#56 reference, up to 14 points per cell). The other eight
classes do not regress beyond that variance: the Dog is ahead of the stock
Dog in 7 of its 8 cells and equal in the eighth (76.0% vs 65.3% at training
skill 3), the Knight at 81.3% vs 77.3% (holding #57), the Ogre at 87.3% vs
84.7%, the Hell Knight at 74.7% vs 74.0%, the Enforcer at 70.0% vs 69.0%,
the Fiend ties its saturated baseline (99.0 vs 99.0 at skill 2, the #58
ruling), the Shalrath holds #56 (32.0% vs 28.0%), and the Grunt at 58.7%
vs 62.7% (-4.0, 0.7 SE, with its known skill-3 trade-off). The client's
accepted shared-model width had to grow with the vocabulary
(`DQ_MONSTER_MAX_TYPES`/`DQ_MAX_CLASSES` 8 → 9): the first client check
refused the 128-wide model with a clear log line, which is the bound doing
its job. In the client, the nine classes are driven across three maps:
e1m8's 27 monsters (24 Ogres and 3 Shamblers, type 8) at 67-74
microseconds per decision, e2m6's 55 monsters of five classes at once (16
Ogres, 14 Hell Knights, 2 Shalraths, 18 Fiends, 5 Shamblers) at 72-73, and
e2m1's 46 (types 0-2) at 72-73.
Released as `models-v9` with the nine-entry class map (`monster_army 0`,
`monster_dog 1`, `monster_enforcer 2`, `monster_ogre 3`,
`monster_hell_knight 4`, `monster_shalrath 5`, `monster_knight 6`,
`monster_demon1 7`, `monster_shambler 8`).

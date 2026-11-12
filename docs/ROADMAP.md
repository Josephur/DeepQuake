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

- [ ] Playable client built from the same vendored engine
- [ ] ONNX inference in the client (CUDA with CPU fallback), batched per tick
- [ ] Stock AI fallback when no model is present
- [ ] Model manifest, download tool, first public release

## M6 — Full roster

- [ ] Remaining monster types, simplest first
- [ ] Group coordination between monsters
- [ ] Style settings exposed as cvars; difficulty handicaps tuned per monster type (difficulty itself follows `skill` from M4, see ARCHITECTURE.md)

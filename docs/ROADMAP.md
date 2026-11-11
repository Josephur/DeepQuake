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
to build (#29, still open).

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

## M4 — First monster brain

- [x] Ray casts report what they hit (liquids, doors, sky), before any monster is trained (#26)
- [x] Decide how the hook treats maps that ship their own game code (#27)
- [ ] Brain hook: policy replaces QuakeC decisions for one monster type (Grunt) (#31)
- [ ] Research the combat task (episode, reward, termination, opponent pool); the training issue follows from it (#32)
- [ ] Monster observation with fair senses only, and a skill input (Quake's `skill`, 0 to 3) next to the style input (design: #22) (#33)
- [ ] Engine handicaps scaled by skill: reaction delay and aim error, seeded, identical in simulator and client (#22) (#34)
- [ ] Train the Grunt against the pool at every skill level; style conditioning (aggressive, defensive, covert) (#35)
- [ ] Evaluation against the stock QuakeC Grunt under identical conditions, at each skill level (#36)

## M5 — Play against it

- [ ] Playable client built from the same vendored engine
- [ ] ONNX inference in the client (CUDA with CPU fallback), batched per tick
- [ ] Stock AI fallback when no model is present
- [ ] Model manifest, download tool, first public release

## M6 — Full roster

- [ ] Remaining monster types, simplest first
- [ ] Group coordination between monsters
- [ ] Style settings exposed as cvars; difficulty handicaps tuned per monster type (difficulty itself follows `skill` from M4, see ARCHITECTURE.md)

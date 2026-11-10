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

- The walk graph is fragmented, so no task is longer than about 2,100 units
  and only about 17% of point pairs are connected (#18).
- The policy has no memory and was trained on one map (#19).
- Monsters are passive and the player invulnerable during navigation
  training; combat comes in M4.

## M4 — First monster brain

- [ ] Brain hook: policy replaces QuakeC decisions for one monster type (Grunt)
- [ ] Player agent learns combat; opponent pool
- [ ] Monster observation with fair senses only, and a skill input (Quake's `skill`, 0 to 3) next to the style input (design: #22)
- [ ] Engine handicaps scaled by skill: reaction delay and aim error, seeded, identical in simulator and client (#22)
- [ ] Train the Grunt against the pool at every skill level; style conditioning (aggressive, defensive, covert)
- [ ] Evaluation against the stock QuakeC Grunt under identical conditions, at each skill level

## M5 — Play against it

- [ ] Playable client built from the same vendored engine
- [ ] ONNX inference in the client (CUDA with CPU fallback), batched per tick
- [ ] Stock AI fallback when no model is present
- [ ] Model manifest, download tool, first public release

## M6 — Full roster

- [ ] Remaining monster types, simplest first
- [ ] Group coordination between monsters
- [ ] Style settings exposed as cvars; difficulty handicaps tuned per monster type (difficulty itself follows `skill` from M4, see ARCHITECTURE.md)

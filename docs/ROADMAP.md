# Roadmap

Tick items as they land. Each milestone ends with `main` green in CI.

## M0 — Scaffold

- [x] Repository layout, licence, agent guide
- [x] CMake build of `libdqsim` (version stub) with a C test
- [x] Python package loading `libdqsim`, with tests
- [x] Game-data checker and repository hygiene guard (pre-commit and CI)

## M1 — Headless simulator

- [ ] Confirm engine base; vendor it under `engine/third_party/` with `UPSTREAM.md`
- [ ] Strip to server core; build with no SDL, GL, audio, or network
- [ ] C API: create, load map, reset(seed), step, destroy
- [ ] Multiple independent instances per process (remove global state)
- [ ] Determinism test: same seed and inputs give identical state hashes
- [ ] Fidelity test: recorded player inputs replay to the same positions as the unmodified engine
- [ ] Snapshot and restore
- [ ] Throughput benchmark (ticks per second per core) recorded in this file

## M2 — Environment

- [ ] Observation and action spec implemented in the engine (ray casts, entity features, sounds)
- [ ] Python vectorised environment over worker processes
- [ ] Random and scripted baseline agents; smoke test on a real map (`gamedata` marker)

## M3 — Player agent: movement and navigation

- [ ] PPO training loop on the RTX 3090
- [ ] Reach-the-goal and exploration tasks, curriculum over maps
- [ ] Evaluation suite: success rate and time-to-goal per map

## M4 — First monster brain

- [ ] Brain hook: policy replaces QuakeC decisions for one monster type (Grunt)
- [ ] Player agent learns combat; opponent pool
- [ ] Train the Grunt against the pool; style conditioning (aggressive, defensive, covert)
- [ ] Evaluation against the stock QuakeC Grunt under identical conditions

## M5 — Play against it

- [ ] Playable client built from the same vendored engine
- [ ] ONNX inference in the client (CUDA with CPU fallback), batched per tick
- [ ] Stock AI fallback when no model is present
- [ ] Model manifest, download tool, first public release

## M6 — Full roster

- [ ] Remaining monster types, simplest first
- [ ] Group coordination between monsters
- [ ] Difficulty and style settings exposed as cvars

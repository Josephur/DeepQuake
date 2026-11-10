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
- [ ] Vendor QuakeSpasm under `engine/third_party/quakespasm/` with `UPSTREAM.md` (#3)
- [ ] Strip to server core; build with no SDL, GL, audio, or network (#4)
- [ ] C API: create, load map, reset(seed), step, destroy (#5)
- [ ] Seeded random numbers and fixed frame time (no wall clock) (#6)
- [ ] Determinism test: same seed and inputs give identical state hashes (#7)
- [ ] Fidelity test: recorded player inputs replay to the same positions as the unmodified engine (#8)
- [ ] Snapshot and restore (#9)
- [ ] Throughput benchmark (ticks per second per core) recorded in this file (#10)

## M2 — Environment

- [ ] Observation and action spec implemented in the engine (ray casts, entity features, sounds) (#11)
- [ ] Python vectorised environment over worker processes (#12)
- [ ] Random and scripted baseline agents; smoke test on a real map (`gamedata` marker) (#13)

## M3 — Player agent: movement and navigation

- [ ] PPO training loop on the RTX 3090 (#14)
- [ ] Reach-the-goal and exploration tasks, curriculum over maps (#15)
- [ ] Evaluation suite: success rate and time-to-goal per map (#16)

### Handoff gate

M0 to M3 are built interactively. Once a trained policy demonstrably does
something in the game (for example, reaches a goal on a real map more often
than a random agent), the remaining milestones are worked autonomously.

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

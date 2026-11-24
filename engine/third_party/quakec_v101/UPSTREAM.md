# id Software's QuakeC, version 1.01 (vendored, read-only reference)

| | |
| --- | --- |
| Upstream | https://github.com/id-Software/Quake-Tools |
| Commit | `c0d1b91c74eb654365ac7755bc837e497caaca73` (master, 2012-01-31, "Source release of qcc and the qutils, updated long ago to be GPL.") |
| Licence | GPL-2.0-or-later: every `.qc` file opens with that licence header. The text is `COPYING`, upstream's `qcc/COPYING`. |
| Vendored | 2026-11-23 (#72) |

## What is here

The game source id shipped with version 1.01 of Quake: upstream's whole
`qcc/v101qc` directory, 35 `.qc` files and `progs.src` (the order they are
compiled in), byte for byte, plus the licence. Nothing is compiled from it
and nothing includes it. It is here to be read.

It is the reference for how the stock `progs.dat` was built: which state
calls which function, what a function stores on an entity it spawns, where
a shot is aimed. The monster hook in `engine/src/brain_hook.c` resolves
those structures from the compiled progs at run time, so when the hook does
not follow a class's attack, this is where to look before changing it.
Three attempts on the Scrag's attack (#70) were spent on guesses about its
chain before anyone read this.

Where a class's behaviour lives:

- Grunt (`monster_army`): `soldier.qc`
- Rottweiler (`monster_dog`): `dog.qc`
- Enforcer (`monster_enforcer`): `enforcer.qc`
- Ogre (`monster_ogre`): `ogre.qc`
- Fiend (`monster_demon1`): `demon.qc`
- Knight (`monster_knight`): `knight.qc`
- Hell Knight (`monster_hell_knight`): `hknight.qc`
- Scrag (`monster_wizard`): `wizard.qc`
- Vore (`monster_shalrath`): `shalrath.qc`
- Shambler (`monster_shambler`): `shambler.qc`
- Zombie, Rotfish, Tarbaby and the bosses: `zombie.qc`, `fish.qc`,
  `tarbaby.qc`, `boss.qc` and `oldone.qc`

Shared code every class's file calls: `ai.qc` (the `ai_stand`, `ai_walk`
and `ai_run` states, and the range and sight globals the attack states
read), `fight.qc` (sight and attack checks), `monsters.qc` (the spawns),
`subs.qc` (`SUB_AttackFinished` and the like) and `combat.qc` (`T_Damage`
and `Killed`). `defs.qc` holds the constants and the entity fields.

Two things to know when reading it:

- This is the only original game source id has published. The simulator
  runs whatever `progs.dat` is in the user's game data, and whether the
  later retail build differs from this source in any detail is not recorded
  here. Where a detail decides a test, the compiled progs decide.
- id's other game source on GitHub, `quake-rerelease-qc`, is the 2021
  rerelease's code. It stays out of this repository (AGENTS.md, hard
  rule 1).

## Local patches

**None, and none are allowed.** This directory is a reference. Every file
is byte-identical to upstream at the commit above, and
`tools/check_repo_hygiene.py` (pre-commit hook and CI) enforces it with no
recorded-patch route: a changed file fails the check until it is restored
with `git checkout -- engine/third_party/quakec_v101/<file>`.

## Updating

Upstream has not changed since 2012 and is unlikely to. To move the commit
anyway:

1. Check the new upstream commit out elsewhere (outside this repository).
2. Copy `qcc/v101qc/*` and `qcc/COPYING` over, replacing the files here.
3. Update the commit and date above, and regenerate the checksums:
   `sha256sum COPYING *.qc progs.src > SHA256SUMS` (run in this directory).
4. Run `python3 tools/check_repo_hygiene.py`.

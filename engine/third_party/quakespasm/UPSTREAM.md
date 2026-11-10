# QuakeSpasm (vendored)

| | |
| --- | --- |
| Upstream | https://github.com/sezero/quakespasm |
| Commit | `e8795a79273f2823661727265d0617106e282bea` (master, 2026-10-29, after tag `quakespasm-0.97.0`) |
| Licence | GPL-2.0-or-later, see `LICENSE.txt` |
| Vendored | 2026-11-09 |

## What is here

Only the server side of the engine and the support code it needs, copied
from upstream's `Quake/` directory:

- **Server core:** `sv_main.c`, `sv_move.c`, `sv_phys.c`, `sv_user.c`,
  `world.c`, and the QuakeC VM `pr_cmds.c`, `pr_edict.c`, `pr_exec.c`.
- **Host:** `host.c`, `host_cmd.c` (start-up, map loading, client sign-on).
- **Support:** `cmd.c`, `common.c`, `crc.c`, `cvar.c`, `gl_model.c` (BSP
  loading and collision hulls), `mathlib.c`, `zone.c`, `strlcat.c`,
  `strlcpy.c`.
- **Headers:** only those the files above include, plus `progdefs.q1`.

Not vendored: renderer, sound, input, menus, networking drivers, platform
code, bundled libraries, and build files for other platforms.

## How it is built

`engine/CMakeLists.txt` compiles these files into `libdqsim` together with
our own code in `engine/src/`:

- `src/shim/SDL.h`, `SDL_opengl.h`, `glquake.h` stand in for headers that are
  not vendored. They are found because `src/shim` is first on the include
  path.
- `src/client_stubs.c`, `src/platform.c`, `src/net_fake.c` define the
  client, system, and network symbols these files reference.
- `-Drand=dqsim_rand` routes every `rand()` call to the seeded generator in
  `src/rng.c`.

## Local patches

**None.** Every file is byte-identical to upstream at the commit above. Keep
it that way where possible: prefer a shim, a stub, or a compile definition
over editing a vendored file. If an edit is unavoidable, list it here with
the file, the reason, and a marker comment `// DeepQuake:` at the change.

## Updating

1. Check out the new upstream commit elsewhere (outside this repository).
2. Copy over the files listed above, re-apply any local patches.
3. Build; if the link fails on a new undefined symbol, add a stub.
4. Run the full test suite including the `gamedata` tests. The determinism
   and fidelity tests decide whether the update is acceptable.
5. Update the commit and date in this file.

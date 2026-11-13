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
- **Client side (added 2026-11-12, #42, same commit):** everything the
  playable client needs, still from upstream's `Quake/` directory at the
  commit above — client game logic (`cl_*.c`, `view.c`, `keys.c`,
  `chase.c`), renderer (`gl_*.c`, `r_*.c`, `gl_vidsdl.c`, `image.c`),
  UI (`console.c`, `menu.c`, `sbar.c`), sound core (`snd_dma.c`,
  `snd_mix.c`, `snd_mem.c`, `snd_codec.c`, `snd_wave.c`, `snd_sdl.c`,
  `bgmusic.c`), networking (`net_main.c`, `net_dgrm.c`, `net_loop.c`,
  `net_udp.c`, `net_bsd.c`), platform (`main_sdl.c`, `sys_sdl_unix.c`,
  `in_sdl.c`, `cd_sdl.c` — which includes `cd_null.c` under SDL2 —
  `cfgfile.c`, `pl_linux.c`, `wad.c`), the bundled libraries `miniz.c`
  and `lodepng.c`, and their headers (`glquake.h`, `anorms.h`,
  `anorm_dots.h`, `gl_warp_sin.h`, `qs_bmp.h`, `stb_image_write.h`,
  `cfgfile.h`, `lodepng.h`, `net_dgrm.h`, `net_loop.h`, `net_udp.h`,
  `snd_codec.h`, `snd_codeci.h`, `snd_wave.h`, the codec headers
  `snd_codec.c` includes unconditionally (`snd_flac.h`, `snd_mikmod.h`,
  `snd_modplug.h`, `snd_mp3.h`, `snd_opus.h`, `snd_umx.h`,
  `snd_vorbis.h`, `snd_xmp.h`), and `resource.h` (the icon declarations
  `gl_vidsdl.c` includes).

Not vendored: the optional music codec backends (snd_flac/mp3/mikmod/
modplug/mpg123/opus/umx/vorbis/xmp — each needs a system library),
Windows and mac platform files (sys_sdl_win.c, pl_win.c, pl_osx.m,
net_win.c, net_wins.c, net_wipx.c, wsaerror.h, resource.h), and
upstream's build files.

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

**None.** Every file is byte-identical to upstream at the commit above, and
`tools/check_repo_hygiene.py` (pre-commit hook and CI) enforces it: each
file's checksum must match `SHA256SUMS` in this directory.

Keep it that way where possible: prefer a shim, a stub, or a compile
definition over editing a vendored file. Trace and debug code is never a
patch; it goes in a scratch copy (see "Instrumenting the engine" in
`AGENTS.md`). If a permanent edit is unavoidable:

1. Mark the change in the file with a comment starting `// DeepQuake:`.
2. Add a line to the list below, exactly in this form, with the checksum of
   the patched file (`sha256sum Quake/<file>`):

   ```
   - `Quake/<file>` sha256 `<64 hex digits>`: <what was changed and why>
   ```

Do not edit `SHA256SUMS` for a patch: it records upstream, not our changes.

Patches:

(none)

## Updating

1. Check out the new upstream commit elsewhere (outside this repository).
2. Copy over the files listed above, re-apply any local patches.
3. Build; if the link fails on a new undefined symbol, add a stub.
4. Run the full test suite including the `gamedata` tests. The determinism
   and fidelity tests decide whether the update is acceptable.
5. Update the commit and date in this file, and regenerate the checksums:
   `sha256sum LICENSE.txt Quake/* > SHA256SUMS` (run in this directory).
   Update the checksum of every patch line as well.

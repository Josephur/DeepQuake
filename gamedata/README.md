# Game data (local only)

Everything in this directory except this file is ignored by git and must stay
that way. Quake's data files are copyrighted and cannot be redistributed.

The simulator needs the `id1` pak files, named in lower case:

```
gamedata/
  id1/
    pak0.pak     shareware episode
    pak1.pak     registered episodes
```

The easiest way is to let the tool create symlinks to your copy. It also
fixes upper-case names (`PAK0.PAK`), which the engine cannot find on Linux:

```bash
uv run python -m deepquake.gamedata link "/path/to/your/Quake/id1"
```

Run `uv run python -m deepquake.gamedata` at any time to check what was found
and whether it contains what the simulator needs.

If you have the 2021 rerelease (Steam or GOG), link the original data it
ships in `dos/id1` rather than its own `id1`: that is the 1996 game with the
stock `progs.dat`, which is what the simulator is developed against. The
rerelease's `id1/pak0.pak` contains a newer `progs.dat` built for the KEX
engine.

To keep the data somewhere else entirely, set `DEEPQUAKE_GAMEDATA` to a
directory that contains `id1/`.

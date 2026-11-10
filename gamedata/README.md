# Game data (local only)

Everything in this directory except this file is ignored by git and must stay
that way. Quake's data files are copyrighted and cannot be redistributed.

Place or symlink your `id1` directory here:

```
gamedata/
  id1/
    pak0.pak     shareware episode
    pak1.pak     registered episodes (original release; the rerelease ships one pak0.pak)
```

For example:

```bash
ln -s "/path/to/your/Quake/id1" gamedata/id1
uv run python -m deepquake.gamedata
```

If you have the 2021 rerelease (Steam or GOG), prefer the original data it
ships in `dos/id1` (`PAK0.PAK` and `PAK1.PAK`): that is the 1996 game with the
stock `progs.dat`, which is what the simulator is developed against. The
rerelease's own `id1/pak0.pak` contains a newer `progs.dat` built for the KEX
engine.

The checker reports which paks were found and whether they contain what the
simulator needs (`progs.dat` and the maps). To keep the data somewhere else,
set `DEEPQUAKE_GAMEDATA` to a directory that contains `id1/`.

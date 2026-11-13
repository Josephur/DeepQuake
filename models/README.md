# Models (local only)

Everything in this directory except this file is ignored by git. Trained
weights are published as GitHub release assets and downloaded here; your own
training runs export here too.

## Downloading a model

    python3 tools/download_model.py model-name --tag models-v1

installs `models/<name>/` (the ONNX model, its action-layout sidecar, a
checksum manifest and, for a roster model, the monster class map) from the
project's GitHub release, verifying every sha256 checksum. Then play against
it:

    build/release/bin/deepquake -basedir gamedata -userdir <a-writable-dir> \
        -window +set dq_model models/grunt/grunt.onnx +map e1m1

Without `dq_model` (or with a model that fails to load) monsters play their
stock QuakeC AI. **`models-v1`** holds `grunt` (the single-type Grunt policy,
observation width 119). **`models-v2`** holds `roster`, one shared policy for
the Grunt **and** the Dog (width 121: the type one-hot is appended): load it
and **every** Grunt and Dog in the level is driven by that one session, each
with its own recurrent state, skill handicaps and type index. The class map
(`<model>.onnx.monsters`) is what tells the client which class is which
(`deepquake.monster.MONSTER_TYPES`); a model without one is the Grunt policy.
A v1 model still loads unchanged.

Measured on a release build with `roster` on e1m1 at skill 2: 42 monsters
driven (34 Grunts at type 0, 8 Dogs at type 1), 45-46 microseconds per
decision (`deepquake: N monster(s) driven by the model: ...` in the console);
the v1 `grunt` model takes 32 on the same map (34 Grunts driven). Set the Quake `skill`
cvar as usual; difficulty follows it. `dq_max_monsters` caps how many are
driven (0, the default, is all of them). Player-side combat models are
published the same way, without a class map.

## Releasing models (maintainers)

`sh tools/stage_release_assets.sh` regenerates `models/` from the run
directories (export, layout sidecars, checksum manifest — reproducible, never
hand-edited) and zips the release assets under `build/release_assets/`.

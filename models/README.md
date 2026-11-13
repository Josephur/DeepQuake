# Models (local only)

Everything in this directory except this file is ignored by git. Trained
weights are published as GitHub release assets and downloaded here; your own
training runs export here too.

## Downloading a model

    python3 tools/download_model.py grunt

installs `models/grunt/` (the ONNX model, its action-layout sidecar, and a
checksum manifest) from the project's GitHub release, verifying the manifest's
sha256 checksums. Then play against it:

    build/release/bin/deepquake -basedir gamedata -userdir <a-writable-dir> \
        -window +set dq_model models/grunt/grunt.onnx +map e1m1

Without `dq_model` (or with a model that fails to load) the Grunt plays its
stock QuakeC AI. With it, **every** Grunt in the level is driven by the model,
each with its own recurrent state and skill handicaps — measured 34 on e1m1 at
skill 2, 35 on e3m1, 31 on e4m1 — at about 35-40 microseconds per decision
(`deepquake: N monster(s) driven by the model` in the console). Set the Quake
`skill` cvar as usual; difficulty follows it. `dq_max_monsters` caps how many
are driven (0, the default, is all of them). Models for the player-side combat
policy are published the same way.

## Releasing models (maintainers)

`sh tools/stage_release_assets.sh` regenerates `models/` from the run
directories (export, layout sidecar, checksum manifest — reproducible, never
hand-edited) and zips the release assets under `build/release_assets/`.

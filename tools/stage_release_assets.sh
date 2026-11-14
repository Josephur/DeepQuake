#!/bin/sh
# Stage the model release assets from the run directories (#45).
# Regenerates models/ and zips the release assets; reproducible - the
# manifests come from the pipeline, never by hand.
#
# The first argument is the run the "roster" model is exported from: the
# two-type roster by default (models-v2), the three-type one for models-v3.
#   sh tools/stage_release_assets.sh runs/roster3_pool
set -e
cd "$(dirname "$0")/.."
roster_run="${1:-runs/roster_pool}"
uv run python -m deepquake.release_model "$roster_run" models roster
uv run python -m deepquake.release_model runs/combat_grunt_fightstart models player
mkdir -p build/release_assets
for name in roster player; do
  files="$name.onnx $name.onnx.nvec manifest.json"
  if [ -f "models/$name/$name.onnx.monsters" ]; then
    files="$files $name.onnx.monsters"
  fi
  (cd models/"$name" && zip -q "../../build/release_assets/$name.zip" $files)
done
ls -la build/release_assets/

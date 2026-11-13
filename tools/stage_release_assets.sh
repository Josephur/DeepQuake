#!/bin/sh
# Stage the model release assets from the run directories (#45).
# Regenerates models/ and zips the release assets; reproducible - the
# manifests come from the pipeline, never by hand.
set -e
cd "$(dirname "$0")/.."
uv run python -m deepquake.release_model runs/roster_pool models roster
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

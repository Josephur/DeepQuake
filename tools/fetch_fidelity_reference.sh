#!/bin/sh
# Build the pristine upstream QuakeSpasm used by the fidelity test
# (ctest name dqsim.fidelity, engine/tests/test_fidelity.c).
#
# Fetches the exact commit vendored in engine/third_party/quakespasm
# (see its UPSTREAM.md) into engine/.fidelity/ (gitignored) and builds it
# with upstream's own Makefile. Unmodified: this binary is the reference
# "how stock Quake moves" for the whole repository.
#
# Needs: git, gcc, make, SDL2 (sdl2-config), about 1 minute.
# Rerun after any engine patch to re-verify fidelity.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dest="$repo_root/engine/.fidelity/quakespasm"
url=https://github.com/sezero/quakespasm

# The commit vendored in third_party/quakespasm. Keep in sync with
# engine/third_party/quakespasm/UPSTREAM.md.
commit=e8795a79273f2823661727265d0617106e282bea

if [ ! -x "$(command -v sdl2-config)" ]; then
    echo "error: sdl2-config not found; install SDL2 development files" >&2
    exit 1
fi

if [ ! -d "$dest/.git" ]; then
    echo "fetching $url at $commit"
    mkdir -p "$(dirname "$dest")"
    git init -q "$dest"
    git -C "$dest" remote add origin "$url"
fi

git -C "$dest" fetch -q --depth 1 origin "$commit"
git -C "$dest" checkout -q FETCH_HEAD
head=$(git -C "$dest" rev-parse HEAD)
if [ "$head" != "$commit" ]; then
    echo "error: checked out $head, expected $commit" >&2
    exit 1
fi

# Codecs off: the reference runs headless; no music libraries needed.
make -C "$dest/Quake" -s USE_SDL2=1 USE_CODEC_MP3=0 USE_CODEC_VORBIS=0 \
    USE_CODEC_WAVE=0 quakespasm

echo "reference binary: $dest/Quake/quakespasm"
echo "run: ctest --preset dev -R dqsim.fidelity"

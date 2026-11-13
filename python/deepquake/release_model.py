"""Stage a trained policy for release (#45, M5).

    uv run python -m deepquake.release_model runs/grunt_pool models grunt

Exports the run's checkpoint to ONNX (deepquake.export_onnx: graph,
layout sidecars, inline parity check), then writes ``<models_dir>/<name>/``
with the model, its sidecars, and a manifest:

- ``format_version`` — the manifest schema (2; 1 is the v1 layout, no
  class map — models-v1 stays installable);
- ``name``, ``source_run``, ``total_steps`` — provenance;
- ``obs_dim``, ``action_nvec`` — the layout identity a stale model is
  refused by;
- ``monsters`` — the monster classes the model drives, in type order, and
  the type index each one's observation one-hot uses (#48);
- ``onnx_sha256``, ``sidecar_sha256``, ``monsters_sha256`` — what the
  download tool verifies.

Re-running over the same run directory reproduces every byte: the
manifest carries no timestamps and is never hand-edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import REPO_ROOT, export_onnx

MANIFEST_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_model(run_dir: Path, models_dir: Path, name: str) -> Path:
    """Export ``run_dir``'s checkpoint into ``models_dir/name`` with a manifest."""
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} does not exist")
    out_dir = models_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    spec = export_onnx.export_onnx(checkpoint, onnx_path)
    sidecar = Path(str(onnx_path) + ".nvec")
    class_map = Path(str(onnx_path) + ".monsters")

    manifest = {
        "format_version": MANIFEST_VERSION,
        "name": name,
        "source_run": run_dir.name,
        "total_steps": spec["total_steps"],
        "obs_dim": spec["obs_dim"],
        "action_nvec": spec["action_nvec"],
        "onnx_sha256": sha256_file(onnx_path),
        "sidecar_sha256": sha256_file(sidecar),
    }
    if spec["monsters"]:
        manifest["monsters"] = spec["monsters"]
        manifest["monsters_sha256"] = sha256_file(class_map)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run directory with checkpoint.pt")
    parser.add_argument("models_dir", type=Path, nargs="?", default=REPO_ROOT / "models")
    parser.add_argument("name", type=str, help="model name for the directory and manifest")
    args = parser.parse_args()
    manifest_path = release_model(args.run_dir, args.models_dir, args.name)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download a DeepQuake model from a GitHub release and verify it (#45).

    python3 tools/download_model.py grunt
    python3 tools/download_model.py grunt --repo Josephur/DeepQuake --tag models-v1

Fetches the release asset ``<name>.zip`` (model, layout sidecar, manifest),
verifies every checksum in the manifest against the downloaded bytes, and
installs the three files under ``models/<name>/``. Standard library only;
GitHub API access needs no auth for public releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "deepquake-download"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="model name (the release asset <name>.zip)")
    parser.add_argument("--repo", default="Josephur/DeepQuake")
    parser.add_argument(
        "--tag", default=None, help="release tag (default: the repo's latest model release)"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path(__file__).resolve().parent.parent / "models"
    )
    args = parser.parse_args()

    if args.tag:
        release = {
            "tag_name": args.tag,
            "assets": [
                {"name": a["name"], "browser_download_url": a["browser_download_url"]}
                for a in json.loads(
                    fetch(f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}")
                )["assets"]
            ],
        }
    else:
        release = json.loads(fetch(f"https://api.github.com/repos/{args.repo}/releases/latest"))
    asset = next((a for a in release["assets"] if a["name"] == f"{args.name}.zip"), None)
    if asset is None:
        known = ", ".join(a["name"] for a in release["assets"]) or "(none)"
        raise SystemExit(
            f"no {args.name}.zip in release {release.get('tag_name')}; assets: {known}"
        )

    data = fetch(asset["browser_download_url"])
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{args.name}.zip"
        zip_path.write_bytes(data)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if "manifest.json" not in names:
                raise SystemExit("the asset has no manifest.json; refusing to install")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format_version") != 1:
                raise SystemExit(f"unsupported manifest format {manifest.get('format_version')}")
            for key, filename in (
                ("onnx_sha256", f"{args.name}.onnx"),
                ("sidecar_sha256", f"{args.name}.onnx.nvec"),
            ):
                if filename not in names:
                    raise SystemExit(f"the asset is missing {filename}")
                actual = hashlib.sha256(archive.read(filename)).hexdigest()
                if actual != manifest[key]:
                    raise SystemExit(
                        f"checksum mismatch for {filename}: "
                        f"manifest {manifest[key]} vs downloaded {actual}"
                    )
            out_dir = args.models_dir / args.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for filename in names:
                (out_dir / filename).write_bytes(archive.read(filename))
    manifest_path = args.models_dir / args.name / "manifest.json"
    onnx_path = args.models_dir / args.name / f"{args.name}.onnx"
    digest = sha256_file(onnx_path)[:12]
    print(f"installed {args.name} (onnx sha256 {digest}...) under {manifest_path.parent}")


if __name__ == "__main__":
    main()

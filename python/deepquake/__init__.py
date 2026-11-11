"""DeepQuake: neural-network brains for Quake's monsters."""

from pathlib import Path

__version__ = "0.2.0"

#: Repository root when running from a source checkout (editable install).
REPO_ROOT = Path(__file__).resolve().parents[2]

"""Where the pipeline's data lives: the musiclib-rs `data/` directory (DB
snapshots, gold sets, model outputs, bundles). Set MUSICLIB_DATA to use
another one; the default assumes musiclib-rs is checked out next to this repo."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MUSICLIB_DATA") or REPO.parent / "musiclib-rs/data").resolve()

from __future__ import annotations

import os
from pathlib import Path

def project_root() -> Path:
    # repo root is current working dir inside container (/app)
    return Path.cwd()

def data_root() -> Path:
    # If DATA_ROOT is set, use it. Otherwise default to repo/data
    v = os.environ.get("DATA_ROOT")
    return Path(v) if v else (project_root() / "data")

def artifact_root() -> Path:
    # If ARTIFACT_ROOT is set, use it. Otherwise default to repo/artifacts
    v = os.environ.get("ARTIFACT_ROOT")
    return Path(v) if v else (project_root() / "artifacts")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

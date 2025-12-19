from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.utils.paths import ensure_dir

@dataclass
class NormalizeFxConfig:
    bronze_dir: Path
    silver_dir: Path

def run_normalize_fx(cfg: NormalizeFxConfig) -> Path:
    out_dir = ensure_dir(Path(cfg.silver_dir))
    p = cfg.bronze_dir / "source=ecb_exr" / "fx_majors_raw.parquet"
    if not p.exists():
        out = out_dir / "fx_majors.parquet"
        pd.DataFrame(columns=["date"]).to_parquet(out, index=False)
        return out

    df = pd.read_parquet(p)
    # Keep as a wide table: date + majors columns
    out = out_dir / "fx_majors.parquet"
    df.to_parquet(out, index=False)
    return out

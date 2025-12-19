from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.ingest.ecb_sdmx import fetch_ecb_csv, EcbQuery
from src.utils.paths import ensure_dir

@dataclass
class EcbYieldCurveConfig:
    bronze_dir: Path
    series_keys: list[str]
    start: str
    end: str

def run_ecb_yield_curve(cfg: EcbYieldCurveConfig) -> Path:
    out_dir = ensure_dir(cfg.bronze_dir / "source=ecb_yc")
    frames: list[pd.DataFrame] = []

    for key in cfg.series_keys:
        df = fetch_ecb_csv(EcbQuery(flow="YC", key=key, start=cfg.start, end=cfg.end))
        df["series_key"] = key
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    out = out_dir / "yc_raw.parquet"
    all_df.to_parquet(out, index=False)
    return out

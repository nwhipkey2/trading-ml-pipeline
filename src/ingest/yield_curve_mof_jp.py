from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pandas as pd

from src.utils.http import get_bytes
from src.utils.paths import ensure_dir

@dataclass
class JapanMofConfig:
    bronze_dir: Path
    csv_url: str
    start: str
    end: str

def run_japan_mof(cfg: JapanMofConfig) -> Path:
    out_dir = ensure_dir(cfg.bronze_dir / "source=japan_mof_jgb")
    raw = get_bytes(cfg.csv_url)

    # MOF CSV is often NOT UTF-8 (commonly Shift_JIS / CP932).
    # Try a small set of encodings, then fall back to latin-1.
    df = None
    last_err = None
    for enc in ("cp932", "shift_jis", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(BytesIO(raw), encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None

    if df is None:
        raise RuntimeError("Failed to parse Japan MOF CSV with common encodings") from last_err

    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize("UTC")

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    out = out_dir / "japan_mof_jgb_raw.parquet"
    df.to_parquet(out, index=False)
    return out

from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pandas as pd

from src.utils.http import get_bytes
from src.utils.paths import ensure_dir

@dataclass
class RbnzWholesaleConfig:
    bronze_dir: Path
    data_url: str
    start: str
    end: str

def run_rbnz_wholesale(cfg: RbnzWholesaleConfig) -> Path:
    out_dir = ensure_dir(cfg.bronze_dir / "source=rbnz_wholesale")

    raw = get_bytes(cfg.data_url)

    # RBNZ often serves an Excel/CSV via .ashx. Let pandas infer:
    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception:
        df = pd.read_excel(BytesIO(raw))

    # Attempt to find a date column
    date_col = df.columns[0]
    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize("UTC")

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    out = out_dir / "rbnz_wholesale_raw.parquet"
    df.to_parquet(out, index=False)
    return out

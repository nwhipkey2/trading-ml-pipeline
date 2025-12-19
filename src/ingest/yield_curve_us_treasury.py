from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pandas as pd

from src.utils.http import get_bytes
from src.utils.paths import ensure_dir

@dataclass
class UsTreasuryCurveConfig:
    bronze_dir: Path
    archive_urls: list[str]
    start: str
    end: str

def _read_treasury_csv(url: str) -> pd.DataFrame:
    raw = get_bytes(url)
    df = pd.read_csv(BytesIO(raw))
    return df

def run_us_treasury_curve(cfg: UsTreasuryCurveConfig) -> Path:
    out_dir = ensure_dir(cfg.bronze_dir / "source=us_treasury_yc")
    frames = []
    for url in cfg.archive_urls:
        df = _read_treasury_csv(url)
        df["source_url"] = url
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Treasury CSV archives typically have "Date" + maturity columns like "1 Mo", "2 Yr", etc.
    # Standardize:
    all_df.rename(columns={"Date": "date"}, inplace=True)
    all_df["date"] = pd.to_datetime(all_df["date"], format="%Y-%m-%d", errors="coerce")
    mask = all_df["date"].isna()
    if mask.any():
        all_df.loc[mask, "date"] = pd.to_datetime(all_df.loc[mask, "date"], format="%m/%d/%Y", errors="coerce")
    all_df["date"] = all_df["date"].dt.tz_localize("UTC")


    # filter
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    all_df = all_df[(all_df["date"] >= start) & (all_df["date"] <= end)].copy()

    out = out_dir / "ust_yc_raw.parquet"
    all_df.to_parquet(out, index=False)
    return out

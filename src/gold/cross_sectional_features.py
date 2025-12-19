from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.utils.paths import ensure_dir


@dataclass
class CSConfig:
    gold_dir: Path = Path("data/gold")
    in_file: str = "final_dataset.parquet"
    out_file: str = "final_dataset_cs.parquet"

    # Which columns to rank cross-sectionally each day
    rank_cols: tuple[str, ...] = (
        "ret_5",
        "ret_20",
        "vol_20",
        "close_sma_gap_20",
        "close_sma_gap_50",
    )

    # Rank direction: returns/gaps usually "higher is better" for momentum;
    # vol is often "lower is better" (but keep both if you want)
    lower_is_better: tuple[str, ...] = ("vol_20",)


def _to_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def run(cfg: CSConfig) -> Path:
    ensure_dir(cfg.gold_dir)

    inp = cfg.gold_dir / cfg.in_file
    df = pd.read_parquet(inp).copy()

    df["timestamp"] = _to_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp", "symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str)

    # Ensure rank cols exist (skip missing gracefully)
    cols = [c for c in cfg.rank_cols if c in df.columns]
    if not cols:
        raise RuntimeError(f"No rank_cols found in dataset. Tried: {cfg.rank_cols}")

    # Cross-sectional ranks per timestamp
    g = df.groupby("timestamp", group_keys=False)

    for c in cols:
        # rank in [0,1] where 1 = "best"
        r = g[c].rank(pct=True, method="average")
        if c in cfg.lower_is_better:
            r = 1.0 - r
        df[f"{c}_rank"] = r

        # z-score cross-sectionally (optional, often helpful)
        # z = (x - mean)/std per day
        mu = g[c].transform("mean")
        sd = g[c].transform("std").replace(0, pd.NA)
        df[f"{c}_z_cs"] = (df[c] - mu) / sd

    out = cfg.gold_dir / cfg.out_file
    df.to_parquet(out, index=False)
    print("Wrote", out)
    return out


if __name__ == "__main__":
    run(CSConfig())

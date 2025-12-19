from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.lake.calendar import to_utc
from src.lake.io import ParquetIO


@dataclass
class BuildDailyPanelConfig:
    prices_path: str = "data/silver/prices_daily.parquet"
    universe_membership_path: str = "data/silver/universe_membership.parquet"
    sector_map_path: str = "data/silver/sector_map.parquet"
    out_path: str = "data/gold/equities_daily_panel.parquet"

    horizon_days: int = 5  # target horizon


def _add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    # expects: timestamp, symbol, close
    g = df.groupby("symbol", group_keys=False)

    df["ret_1"] = g["close"].pct_change()
    df["ret_5"] = g["close"].pct_change(5)
    df["ret_20"] = g["close"].pct_change(20)

    df["vol_20"] = g["ret_1"].rolling(20).std().reset_index(level=0, drop=True)

    df["sma_20"] = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    df["sma_50"] = g["close"].rolling(50).mean().reset_index(level=0, drop=True)

    df["close_sma_gap_20"] = (df["close"] / df["sma_20"]) - 1.0
    df["close_sma_gap_50"] = (df["close"] / df["sma_50"]) - 1.0

    return df


def _add_cross_sectional_ranks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("timestamp", group_keys=False)
    for c in cols:
        if c not in df.columns:
            continue
        df[f"{c}_rank"] = g[c].rank(pct=True, method="average")
    return df


def run(cfg: BuildDailyPanelConfig) -> Path:
    io = ParquetIO()

    prices = io.read(cfg.prices_path).copy()
    # Required columns
    need = {"timestamp", "symbol", "close"}
    if not need.issubset(prices.columns):
        raise ValueError(f"prices_daily.parquet must contain {need}. Found: {list(prices.columns)}")

    prices["timestamp"] = to_utc(prices["timestamp"])
    prices["symbol"] = prices["symbol"].astype(str).str.upper().str.strip()
    prices = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    uni = io.read(cfg.universe_membership_path).copy()
    uni["timestamp"] = to_utc(uni["timestamp"])
    uni["symbol"] = uni["symbol"].astype(str).str.upper().str.strip()
    uni = uni[uni["in_universe"] == True].copy()

    sec = io.read(cfg.sector_map_path).copy()
    sec["symbol"] = sec["symbol"].astype(str).str.upper().str.strip()
    sec = sec.drop_duplicates(subset=["symbol"]).copy()

    # Join: prices ∩ universe, then add sector
    df = pd.merge(prices, uni[["timestamp", "symbol", "index_id", "weight"]], on=["timestamp", "symbol"], how="inner")
    df = pd.merge(df, sec[["symbol", "sector"]], on="symbol", how="left")

    # Features
    df = _add_time_series_features(df)

    # Sector-relative features (simple sector equal-weight return)
    if "sector" in df.columns:
        df["sector_ret_20"] = (
            df.groupby(["timestamp", "sector"])["ret_20"].transform("mean")
        )
        df["rel_ret_20_vs_sector"] = df["ret_20"] - df["sector_ret_20"]
    else:
        df["sector_ret_20"] = np.nan
        df["rel_ret_20_vs_sector"] = np.nan

    # Cross-sectional ranks (per day)
    rank_cols = ["ret_20", "vol_20", "close_sma_gap_20", "close_sma_gap_50", "rel_ret_20_vs_sector"]
    df = _add_cross_sectional_ranks(df, rank_cols)

    # Target: forward return / up-down label
    g = df.groupby("symbol", group_keys=False)
    fwd = g["close"].shift(-cfg.horizon_days) / df["close"] - 1.0
    df[f"fwd_ret_{cfg.horizon_days}"] = fwd
    df[f"target_up_{cfg.horizon_days}"] = (fwd > 0).astype(int)

    df = df.dropna(subset=["ret_1", "ret_20", f"fwd_ret_{cfg.horizon_days}"]).copy()
    out = io.write(df, cfg.out_path)
    print("Wrote", out, f"({len(df)} rows, {len(df.columns)} cols)")
    return out


if __name__ == "__main__":
    run(BuildDailyPanelConfig())

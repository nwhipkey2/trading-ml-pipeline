from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class BuildDailyPanelConfig:
    membership_path: str = "data/silver/universe_membership.parquet"
    prices_path: str = "data/silver/prices_daily.parquet"
    sector_map_path: str = "data/silver/sector_map.parquet"
    out_path: str = "data/gold/equities_daily_panel.parquet"

    vol_window: int = 20
    sma_fast: int = 20
    sma_slow: int = 50

    fwd_horizon_days: int = 5
    min_history: int = 60


def _as_utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce")


def _to_date(ts: pd.Series) -> pd.Series:
    # Normalize to midnight UTC so membership and prices align
    return _as_utc(ts).dt.floor("D")


def run(cfg: BuildDailyPanelConfig) -> None:
    mem = pd.read_parquet(Path(cfg.membership_path)).copy()
    prices = pd.read_parquet(Path(cfg.prices_path)).copy()
    sector = pd.read_parquet(Path(cfg.sector_map_path)).copy()

    mem["timestamp"] = _to_date(mem["timestamp"])
    prices["timestamp"] = _to_date(prices["timestamp"])

    mem["symbol"] = mem["symbol"].astype(str).str.upper()
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    sector["symbol"] = sector["symbol"].astype(str).str.upper()

    mem = mem[mem["in_universe"] == True].copy()

    overlap = set(mem["symbol"]).intersection(set(prices["symbol"]))
    mem = mem[mem["symbol"].isin(overlap)].copy()
    prices = prices[prices["symbol"].isin(overlap)].copy()

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols_out = [
        "timestamp","symbol","open","high","low","close","adj_close","volume",
        "index_id","weight","sector",
        "ret_1","ret_5","ret_20","vol_20","sma_20","sma_50",
        "close_sma_gap_20","close_sma_gap_50","sector_ret_20","rel_ret_20_vs_sector",
        "ret_20_rank","vol_20_rank","close_sma_gap_20_rank","close_sma_gap_50_rank",
        "rel_ret_20_vs_sector_rank","fwd_ret_5","target_up_5"
    ]

    if mem.empty or prices.empty:
        pd.DataFrame(columns=cols_out).to_parquet(out_path, index=False)
        print(f"Wrote {cfg.out_path} (0 rows, {len(cols_out)} cols) [empty after overlap filter]")
        return

    df = prices.merge(
        mem[["timestamp", "symbol", "index_id", "weight"]],
        on=["timestamp", "symbol"],
        how="inner",
        validate="many_to_one",
    )

    df = df.merge(sector[["symbol", "sector"]], on="symbol", how="left")
    df["sector"] = df["sector"].fillna("Unknown")

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    df["ret_1"] = df.groupby("symbol")["adj_close"].pct_change(1)
    df["ret_5"] = df.groupby("symbol")["adj_close"].pct_change(5)
    df["ret_20"] = df.groupby("symbol")["adj_close"].pct_change(20)

    df["vol_20"] = (
        df.groupby("symbol")["ret_1"]
        .rolling(cfg.vol_window)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["sma_20"] = (
        df.groupby("symbol")["adj_close"]
        .rolling(cfg.sma_fast)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["sma_50"] = (
        df.groupby("symbol")["adj_close"]
        .rolling(cfg.sma_slow)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["close_sma_gap_20"] = df["adj_close"] / df["sma_20"] - 1.0
    df["close_sma_gap_50"] = df["adj_close"] / df["sma_50"] - 1.0

    df["sector_ret_20"] = df.groupby(["timestamp", "sector"])["ret_20"].transform("mean")
    df["rel_ret_20_vs_sector"] = df["ret_20"] - df["sector_ret_20"]

    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    df["ret_20_rank"] = df.groupby("timestamp")["ret_20"].transform(_rank)
    df["vol_20_rank"] = df.groupby("timestamp")["vol_20"].transform(_rank)
    df["close_sma_gap_20_rank"] = df.groupby("timestamp")["close_sma_gap_20"].transform(_rank)
    df["close_sma_gap_50_rank"] = df.groupby("timestamp")["close_sma_gap_50"].transform(_rank)
    df["rel_ret_20_vs_sector_rank"] = df.groupby("timestamp")["rel_ret_20_vs_sector"].transform(_rank)

    df["fwd_ret_5"] = (
        df.groupby("symbol")["adj_close"]
        .pct_change(cfg.fwd_horizon_days)
        .shift(-cfg.fwd_horizon_days)
    )
    df["target_up_5"] = (df["fwd_ret_5"] > 0).astype("float")

    df["obs_idx"] = df.groupby("symbol").cumcount()
    df = df[df["obs_idx"] >= cfg.min_history].copy()
    df = df.drop(columns=["obs_idx"])

    required = [
        "ret_1","ret_5","ret_20","vol_20","sma_20","sma_50",
        "close_sma_gap_20","close_sma_gap_50","sector_ret_20","rel_ret_20_vs_sector",
        "ret_20_rank","vol_20_rank","close_sma_gap_20_rank","close_sma_gap_50_rank",
        "rel_ret_20_vs_sector_rank","fwd_ret_5","target_up_5"
    ]
    df = df.dropna(subset=required)

    # Keep columns in the expected order
    df = df[cols_out]

    df.to_parquet(out_path, index=False)
    print(f"Wrote {cfg.out_path} ({len(df)} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    run(BuildDailyPanelConfig())

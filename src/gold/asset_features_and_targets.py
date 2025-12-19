from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.paths import ensure_dir


@dataclass
class FeatureConfig:
    gold_dir: Path = Path("data/gold")
    in_file: str = "dataset_with_macro.parquet"
    out_file: str = "final_dataset.parquet"

    # Feature windows (trading days)
    ret_windows: tuple[int, ...] = (1, 5, 20)
    vol_windows: tuple[int, ...] = (5, 20)
    sma_windows: tuple[int, ...] = (5, 20, 50)

    # Optional: volume features (works for ETFs/stocks; safe to include)
    volchg_windows: tuple[int, ...] = (5, 20)

    # Targets (forward horizons)
    target_horizons: tuple[int, ...] = (5, 20)

    # Minimum history required for a row to be "valid"
    # (largest SMA + largest target horizon)
    min_history: int = 60

    # Keep only these base columns if you want (set None to keep everything)
    required_price_cols: tuple[str, ...] = ("close",)


def _to_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def _safe_pct_change(x: pd.Series, periods: int) -> pd.Series:
    # pct_change is fine; this wrapper keeps dtype clean
    return x.pct_change(periods=periods)


def _rolling_vol(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window=window, min_periods=window).std()


def _rolling_sma_gap(close: pd.Series, window: int) -> pd.Series:
    sma = close.rolling(window=window, min_periods=window).mean()
    return (close / sma) - 1.0


def _make_features_for_symbol(g: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    g: one symbol's rows, sorted by timestamp.
    Generates per-asset features only (no macro).
    """
    g = g.sort_values("timestamp").copy()

    # Core series
    close = g["close"].astype(float)

    # Returns
    g["ret_1"] = _safe_pct_change(close, 1)
    for w in cfg.ret_windows:
        g[f"ret_{w}"] = _safe_pct_change(close, w)

    # Volatility on daily returns (ret_1)
    for w in cfg.vol_windows:
        g[f"vol_{w}"] = _rolling_vol(g["ret_1"], w)

    # SMA gaps (dip/trend structure)
    for w in cfg.sma_windows:
        g[f"close_sma_gap_{w}"] = _rolling_sma_gap(close, w)

    # Volume change features if volume exists
    if "volume" in g.columns:
        vol = pd.to_numeric(g["volume"], errors="coerce")
        for w in cfg.volchg_windows:
            vma = vol.rolling(window=w, min_periods=w).mean()
            g[f"vol_sma_gap_{w}"] = (vol / vma) - 1.0

    # Targets: forward returns + direction labels
    for h in cfg.target_horizons:
        g[f"fwd_ret_{h}"] = close.pct_change(periods=h).shift(-h)
        g[f"target_up_{h}"] = (g[f"fwd_ret_{h}"] > 0).astype("Int64")

    return g


def run(cfg: FeatureConfig) -> Path:
    ensure_dir(cfg.gold_dir)

    in_path = cfg.gold_dir / cfg.in_file
    df = pd.read_parquet(in_path).copy()

    # Basic sanity + sort
    if "timestamp" not in df.columns or "symbol" not in df.columns:
        raise KeyError("Input dataset must contain columns: timestamp, symbol")

    df["timestamp"] = _to_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp", "symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str)

    # Ensure required price cols exist
    for c in cfg.required_price_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {in_path}")

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Build per-symbol features/targets
    out = (
        df.groupby("symbol", group_keys=False)
          .apply(lambda g: _make_features_for_symbol(g, cfg))
          .reset_index(drop=True)
    )

    # Drop rows with insufficient history and missing targets
    # (Largest SMA and largest target horizon imply the earliest valid row and last valid row)
    max_sma = max(cfg.sma_windows) if cfg.sma_windows else 0
    max_h = max(cfg.target_horizons) if cfg.target_horizons else 0

    # Require close + ret_1 + the largest sma_gap + at least one target
    required_cols = ["close", "ret_1"]
    if max_sma:
        required_cols.append(f"close_sma_gap_{max_sma}")
    if cfg.target_horizons:
        required_cols.append(f"fwd_ret_{max_h}")

    out = out.dropna(subset=required_cols).copy()

    # Optional: enforce a minimum rows per symbol (helps later with lots of tickers)
    counts = out.groupby("symbol")["timestamp"].count()
    keep_syms = counts[counts >= cfg.min_history].index
    out = out[out["symbol"].isin(keep_syms)].copy()

    out_path = cfg.gold_dir / cfg.out_file
    out.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    p = run(FeatureConfig())
    print("Wrote", p)

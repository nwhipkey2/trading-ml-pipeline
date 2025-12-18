from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, read_parquet, write_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("features")


@dataclass(frozen=True)
class FeatureConfig:
    silver_prices_path: Path
    gold_dir: Path
    windows: list[int]
    horizon_days: int


def add_features(df: pd.DataFrame, windows: list[int], horizon_days: int) -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).copy()

    # returns
    df["ret_1"] = df.groupby("symbol")["close"].pct_change(1)
    for w in windows:
        df[f"ret_{w}"] = df.groupby("symbol")["close"].pct_change(w)
        df[f"vol_{w}"] = df.groupby("symbol")["ret_1"].transform(lambda s: s.rolling(w).std())

        df[f"sma_{w}"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(w).mean())
        df[f"ema_{w}"] = df.groupby("symbol")["close"].transform(lambda s: s.ewm(span=w, adjust=False).mean())
        df[f"close_sma_gap_{w}"] = (df["close"] / df[f"sma_{w}"]) - 1.0

    # volume z-score (rolling)
    if "volume" in df.columns:
        df["vol_z_20"] = df.groupby("symbol")["volume"].transform(
            lambda s: (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-9)
        )

    # label: forward return horizon_days
    fwd = df.groupby("symbol")["close"].shift(-horizon_days)
    df["y_fwd_ret"] = (fwd / df["close"]) - 1.0
    df["y_up"] = (df["y_fwd_ret"] > 0).astype(int)

    # drop rows where features/label not available yet
    feat_cols = [c for c in df.columns if c.startswith(("ret_", "vol_", "sma_", "ema_", "close_sma_gap_", "vol_z_"))]
    df = df.dropna(subset=feat_cols + ["y_up"])

    return df.reset_index(drop=True)


def run(cfg: FeatureConfig) -> Path:
    ensure_dir(cfg.gold_dir)
    prices = read_parquet(cfg.silver_prices_path)
    feat = add_features(prices, cfg.windows, cfg.horizon_days)

    out = cfg.gold_dir / "dataset.parquet"
    write_parquet(feat, out)
    log.info("Wrote %s (%d rows, %d cols)", out.as_posix(), len(feat), feat.shape[1])
    return out


if __name__ == "__main__":
    import yaml

    y = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    run(
        FeatureConfig(
            silver_prices_path=Path(y["paths"]["silver"]) / "prices.parquet",
            gold_dir=Path(y["paths"]["gold"]),
            windows=list(y["features"]["windows"]),
            horizon_days=int(y["labels"]["horizon_days"]),
        )
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.utils.paths import ensure_dir


@dataclass
class JoinConfig:
    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")

    prices_file: str = "prices.parquet"
    macro_file: str = "macro_features.parquet"
    out_file: str = "dataset_with_macro.parquet"

    # forward-fill macro series to trading days
    ffill_macro: bool = True


def _to_utc(s: pd.Series) -> pd.Series:
    """
    Force a datetime series to UTC, whether it's naive or tz-aware.
    """
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def run(cfg: JoinConfig) -> Path:
    ensure_dir(cfg.gold_dir)

    # --- Load prices (panel) ---
    prices = pd.read_parquet(cfg.silver_dir / cfg.prices_file).copy()
    prices["timestamp"] = _to_utc(prices["timestamp"])
    prices = (
        prices
        .dropna(subset=["timestamp", "symbol"])
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )

    # --- Load macro features ---
    macro = pd.read_parquet(cfg.gold_dir / cfg.macro_file).copy()
    macro["date"] = _to_utc(macro["date"])
    macro = macro.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # --- Align macro to all trading days ---
    if cfg.ffill_macro:
        all_days = (
            prices[["timestamp"]]
            .drop_duplicates()
            .rename(columns={"timestamp": "date"})
            .sort_values("date")
            .reset_index(drop=True)
        )

        # merge on same tz-aware dtype
        macro = (
            pd.merge(all_days, macro, on="date", how="left")
            .sort_values("date")
            .ffill()
        )

    # --- Join macro onto prices ---
    macro = macro.rename(columns={"date": "timestamp"})
    out = pd.merge(prices, macro, on="timestamp", how="left")

    out_path = cfg.gold_dir / cfg.out_file
    out.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    p = run(JoinConfig())
    print("Wrote", p)

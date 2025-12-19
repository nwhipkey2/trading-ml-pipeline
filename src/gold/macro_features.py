from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.paths import ensure_dir


@dataclass
class MacroFeatureConfig:
    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")

    # Choose which curve nodes to use for level/slope features
    level_maturity_years: float = 10.0
    slope_short_years: float = 2.0
    slope_long_years: float = 10.0

    # FX return horizons (trading days)
    fx_ret_windows: tuple[int, ...] = (1, 5, 20)


def _nearest_maturity(df: pd.DataFrame, target_years: float) -> pd.DataFrame:
    """
    For each (date, region), pick the row with maturity_years closest to target_years.
    """
    df = df.dropna(subset=["maturity_years", "yield"]).copy()
    df["maturity_years"] = df["maturity_years"].astype(float)
    df["_dist"] = (df["maturity_years"] - target_years).abs()

    # choose nearest maturity per date/region
    df = (
        df.sort_values(["date", "region", "_dist"])
        .groupby(["date", "region"], as_index=False)
        .first()
        .drop(columns=["_dist"])
    )
    return df


def build_yield_curve_features(yc: pd.DataFrame, cfg: MacroFeatureConfig) -> pd.DataFrame:
    """
    Returns a date-indexed dataframe with columns like:
    yc_US_10y, yc_EU_10y, yc_US_slope_10y_2y, etc.
    """
    yc = yc.copy()
    yc["date"] = pd.to_datetime(yc["date"], utc=True, errors="coerce")
    yc = yc.dropna(subset=["date", "region"]).copy()
    yc["region"] = yc["region"].astype(str)

    # level at maturity (nearest node)
    lvl = _nearest_maturity(yc, cfg.level_maturity_years)
    lvl_w = lvl.pivot(index="date", columns="region", values="yield").sort_index()
    lvl_w.columns = [f"yc_{c}_{int(cfg.level_maturity_years)}y" for c in lvl_w.columns]

    # slope = long - short (nearest nodes)
    short = _nearest_maturity(yc, cfg.slope_short_years).pivot(index="date", columns="region", values="yield").sort_index()
    long = _nearest_maturity(yc, cfg.slope_long_years).pivot(index="date", columns="region", values="yield").sort_index()

    slope = (long - short)
    slope.columns = [f"yc_{c}_slope_{int(cfg.slope_long_years)}y_{int(cfg.slope_short_years)}y" for c in slope.columns]

    out = pd.concat([lvl_w, slope], axis=1).sort_index()

    # simple changes (levels and slopes)
    for col in out.columns:
        out[f"{col}_d1"] = out[col].diff(1)
        out[f"{col}_d5"] = out[col].diff(5)
        out[f"{col}_d20"] = out[col].diff(20)

    return out.reset_index()


def build_fx_features(fx: pd.DataFrame, cfg: MacroFeatureConfig) -> pd.DataFrame:
    """
    fx input: wide table with date + columns like EURUSD, USDJPY...
    Returns date-indexed features: fx_EURUSD_ret_5, etc.
    """
    fx = fx.copy()
    fx["date"] = pd.to_datetime(fx["date"], utc=True, errors="coerce")
    fx = fx.dropna(subset=["date"]).sort_values("date").copy()

    # set index for pct_change
    fx_i = fx.set_index("date")

    feats = {}
    for col in fx_i.columns:
        # Skip non-numeric columns if any
        if not np.issubdtype(fx_i[col].dtype, np.number):
            continue
        for w in cfg.fx_ret_windows:
            feats[f"fx_{col}_ret_{w}"] = fx_i[col].pct_change(w)

    out = pd.DataFrame(feats, index=fx_i.index).reset_index()
    return out


def run(cfg: MacroFeatureConfig) -> Path:
    ensure_dir(cfg.gold_dir)

    yc_path = cfg.silver_dir / "yield_curves.parquet"
    fx_path = cfg.silver_dir / "fx_majors.parquet"

    yc = pd.read_parquet(yc_path)
    fx = pd.read_parquet(fx_path)

    yc_feats = build_yield_curve_features(yc, cfg)
    fx_feats = build_fx_features(fx, cfg)

    # Merge macro features on date
    macro = pd.merge(yc_feats, fx_feats, on="date", how="outer").sort_values("date")

    out = cfg.gold_dir / "macro_features.parquet"
    macro.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    p = run(MacroFeatureConfig())
    print("Wrote", p)

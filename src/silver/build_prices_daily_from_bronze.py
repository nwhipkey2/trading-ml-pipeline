from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class BuildPricesConfig:
    bronze_dir: str = "data/bronze"
    out_path: str = "data/silver/prices_daily.parquet"
    # Keep it small at first; set to 0 to include all bronze symbols
    max_symbols: int = 0


def main(cfg: BuildPricesConfig = BuildPricesConfig()) -> None:
    bronze = Path(cfg.bronze_dir)
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sym_dirs = sorted([d for d in bronze.glob("symbol=*") if (d / "data.parquet").exists()])
    if not sym_dirs:
        raise RuntimeError("No bronze symbol parquet files found under data/bronze/symbol=*/data.parquet")

    if cfg.max_symbols and cfg.max_symbols > 0:
        sym_dirs = sym_dirs[: cfg.max_symbols]

    parts = []
    bad = 0

    for d in sym_dirs:
        sym = d.name.split("symbol=")[-1].strip().upper()
        fp = d / "data.parquet"
        try:
            df = pd.read_parquet(fp).copy()
            if df.empty:
                bad += 1
                continue

            # Ensure expected cols
            if "timestamp" not in df.columns:
                raise ValueError(f"{fp} missing timestamp")

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["symbol"] = sym

            # standardize cols
            keep = ["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
            for c in keep:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[keep].sort_values("timestamp")

            parts.append(df)
        except Exception:
            bad += 1

    if not parts:
        raise RuntimeError("No usable price data found in bronze files.")

    prices = pd.concat(parts, ignore_index=True)
    prices = prices.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])

    prices.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(prices):,} rows, symbols={prices['symbol'].nunique():,}, bad_files={bad})")


if __name__ == "__main__":
    main()

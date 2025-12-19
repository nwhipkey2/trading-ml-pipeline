from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.lake.calendar import to_utc
from src.lake.io import ParquetIO


@dataclass
class SectorMapConfig:
    sector_map_csv: str = "inputs/universe/sector_map.csv"
    out_path: str = "data/silver/sector_map.parquet"


def run(cfg: SectorMapConfig) -> Path:
    df = pd.read_csv(cfg.sector_map_csv)
    if "symbol" not in df.columns or "sector" not in df.columns:
        raise ValueError("sector_map.csv must have columns: symbol, sector (optional: industry, subindustry)")

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    # add a timestamp column for convenience (not required)
    df["timestamp"] = to_utc(pd.Series([pd.Timestamp.utcnow()] * len(df)))

    out = ParquetIO().write(df, cfg.out_path)
    print("Wrote", out)
    return out


if __name__ == "__main__":
    run(SectorMapConfig())

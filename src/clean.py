from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir, read_parquet, write_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean")


@dataclass(frozen=True)
class CleanConfig:
    bronze_dir: Path
    silver_dir: Path


def clean_symbol_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize timestamp column from yfinance quirks
    if "timestamp" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "timestamp"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "timestamp"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})
        else:
            raise KeyError(f"No timestamp-like column found. Columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # Normalize column names to lower snake_case
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Handle common yfinance variants
    rename_map = {
        "adj close": "adj_close",
        "adj_close": "adj_close",
    }
    df = df.rename(columns=rename_map)

    # Ensure we have a single 'close' column
    if "close" not in df.columns:
        candidates = [c for c in df.columns if c.startswith("close")]
        if len(candidates) == 1:
            df = df.rename(columns={candidates[0]: "close"})
        elif len(candidates) > 1:
            raise KeyError(
                f"Multiple close-like columns found: {candidates}. "
                f"Bronze schema is still multi-ticker."
            )
        else:
            raise KeyError(f"No close-like column found. Columns: {list(df.columns)}")

    df = df.dropna(subset=["timestamp", "symbol", "close"])

    df = df.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)



def run(cfg: CleanConfig) -> None:
    ensure_dir(cfg.silver_dir)
    bronze_parts = sorted(cfg.bronze_dir.glob("symbol=*/data.parquet"))
    if not bronze_parts:
        raise FileNotFoundError(f"No bronze files found under {cfg.bronze_dir}")

    all_rows = []
    for p in bronze_parts:
        df = read_parquet(p)
        all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = clean_symbol_df(df_all)

    out = cfg.silver_dir / "prices.parquet"
    write_parquet(df_all, out)
    log.info("Wrote %s (%d rows)", out.as_posix(), len(df_all))


if __name__ == "__main__":
    import yaml

    cfg_y = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    run(
        CleanConfig(
            bronze_dir=Path(cfg_y["paths"]["bronze"]),
            silver_dir=Path(cfg_y["paths"]["silver"]),
        )
    )

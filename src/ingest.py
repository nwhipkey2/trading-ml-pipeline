from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from src.utils.io import ensure_dir, write_parquet


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


@dataclass(frozen=True)
class IngestConfig:
    universe: list[str]
    start: str
    end: str
    interval: str
    bronze_dir: Path


def fetch_ohlcv(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import pandas as pd
    import yfinance as yf

    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # index -> column
    df = df.reset_index()

    # If yfinance gives MultiIndex columns, collapse to first level names
    # e.g. ('Close','SPY') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns.to_list()]

    # Normalize timestamp column name
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    elif "index" in df.columns:
        df = df.rename(columns={"index": "timestamp"})

    # Lowercase all columns
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Strip accidental ticker suffixes like close_spy, open_qqq, etc.
    suf = f"_{symbol.lower()}"
    df = df.rename(columns={c: c[: -len(suf)] for c in df.columns if c.endswith(suf)})

    # Normalize common names
    df = df.rename(columns={"adj close": "adj_close"})

    # Add symbol
    df["symbol"] = symbol

    # Keep canonical schema (only if present)
    keep = ["timestamp", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    return df



def run(cfg: IngestConfig) -> None:
    ensure_dir(cfg.bronze_dir)

    for sym in cfg.universe:
        log.info("Fetching %s %s→%s interval=%s", sym, cfg.start, cfg.end, cfg.interval)
        df = fetch_ohlcv(sym, cfg.start, cfg.end, cfg.interval)
        if df.empty:
            log.warning("No data for %s", sym)
            continue

        out = cfg.bronze_dir / f"symbol={sym}" / "data.parquet"
        write_parquet(df, out)
        log.info("Wrote %s (%d rows)", out.as_posix(), len(df))


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    run(
        IngestConfig(
            universe=cfg["universe"],
            start=cfg["date_range"]["start"],
            end=cfg["date_range"]["end"],
            interval=cfg.get("frequency", "1d"),
            bronze_dir=Path(cfg["paths"]["bronze"]),
        )
    )

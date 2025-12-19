from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.ingest.ecb_sdmx import fetch_ecb_csv, EcbQuery
from src.utils.paths import ensure_dir

@dataclass
class EcbFxMajorsConfig:
    bronze_dir: Path
    start: str
    end: str
    currencies_vs_eur: list[str]  # e.g. ["USD","JPY","GBP","AUD","NZD","CAD","CHF"]
    majors: list[str]             # e.g. ["EURUSD","GBPUSD","USDJPY",...]

def _exr_key(ccy: str) -> str:
    # ECB EXR: D.<CURRENCY>.EUR.SP00.A is "units of <CURRENCY> per 1 EUR" for spot average
    return f"D.{ccy}.EUR.SP00.A"

def run_ecb_fx_majors(cfg: EcbFxMajorsConfig) -> Path:
    out_dir = ensure_dir(cfg.bronze_dir / "source=ecb_exr")

    frames = []
    for ccy in cfg.currencies_vs_eur:
        df = fetch_ecb_csv(EcbQuery(flow="EXR", key=_exr_key(ccy), start=cfg.start, end=cfg.end))
        # ECB CSV typically has TIME_PERIOD + OBS_VALUE (plus some dimension columns)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    # Normalize to a wide table: date -> {CCY_per_EUR}
    # Try common ECB column names
    time_col = "TIME_PERIOD" if "TIME_PERIOD" in raw.columns else "TIME"
    val_col = "OBS_VALUE" if "OBS_VALUE" in raw.columns else "OBS_VALUE"

    # Currency code column in ECB CSV is often "CURRENCY"
    ccy_col = "CURRENCY" if "CURRENCY" in raw.columns else "CURRENCY"

    wide = (
        raw[[time_col, ccy_col, val_col]]
        .rename(columns={time_col: "date", ccy_col: "ccy", val_col: "value"})
        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce").dt.tz_localize("UTC"))
        .pivot_table(index="date", columns="ccy", values="value", aggfunc="last")
        .sort_index()
    )

    # Build majors vs USD using cross rates from the ECB table
    # Let X be "CCY per EUR". Then:
    # EURUSD = USD per EUR
    # GBPUSD = (USD/EUR) / (GBP/EUR)
    # USDJPY = (JPY/EUR) / (USD/EUR)
    majors = {}
    usd_per_eur = wide["USD"]

    for pair in cfg.majors:
        base = pair[:3]
        quote = pair[3:]

        if pair == "EURUSD":
            majors[pair] = usd_per_eur
        elif quote == "USD":
            # baseUSD -> (USD/EUR) / (base/EUR)
            majors[pair] = usd_per_eur / wide[base]
        elif base == "USD":
            # USDquote -> (quote/EUR) / (USD/EUR)
            majors[pair] = wide[quote] / usd_per_eur
        else:
            # basequote -> (quote/EUR) / (base/EUR)
            majors[pair] = wide[quote] / wide[base]

    out_df = pd.DataFrame(majors).reset_index().rename(columns={"index": "date"})
    out = out_dir / "fx_majors_raw.parquet"
    out_df.to_parquet(out, index=False)
    return out

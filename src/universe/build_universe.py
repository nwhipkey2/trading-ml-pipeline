from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.lake.calendar import to_utc, daily_range
from src.lake.io import ParquetIO


@dataclass
class UniverseConfig:
    index_id: str = "SP500"
    constituents_csv: str = "inputs/universe/constituents.csv"
    start: str = "2010-01-01"
    end: str = "2025-12-18"
    out_path: str = "data/silver/universe_membership.parquet"


def run(cfg: UniverseConfig) -> Path:
    c = pd.read_csv(cfg.constituents_csv)

    required = {"symbol", "start_date"}
    if not required.issubset(set(c.columns)):
        raise ValueError("constituents.csv must include columns: symbol, start_date, (optional end_date, weight)")

    c["symbol"] = c["symbol"].astype(str).str.upper().str.strip()
    c["start_date"] = pd.to_datetime(c["start_date"], errors="coerce")
    c["end_date"] = pd.to_datetime(c.get("end_date", pd.NaT), errors="coerce")

    if "weight" not in c.columns:
        c["weight"] = 1.0

    # Panel dates
    dates = daily_range(cfg.start, cfg.end)
    dates_df = pd.DataFrame({"timestamp": dates})

    # Build point-in-time membership panel (date x symbol)
    rows = []
    for _, r in c.iterrows():
        sym = r["symbol"]
        sd = r["start_date"]
        ed = r["end_date"]

        if pd.isna(sd):
            continue

        # inclusive membership: sd <= date <= ed (or ed is NaT)
        if pd.isna(ed):
            mask = (dates >= sd.tz_localize("UTC") if sd.tzinfo is None else dates >= sd.tz_convert("UTC"))
        else:
            sd_utc = sd.tz_localize("UTC") if sd.tzinfo is None else sd.tz_convert("UTC")
            ed_utc = ed.tz_localize("UTC") if ed.tzinfo is None else ed.tz_convert("UTC")
            mask = (dates >= sd_utc) & (dates <= ed_utc)

        member_dates = dates[mask]
        if len(member_dates) == 0:
            continue

        rows.append(
            pd.DataFrame(
                {
                    "timestamp": member_dates,
                    "symbol": sym,
                    "index_id": cfg.index_id,
                    "in_universe": True,
                    "weight": float(r.get("weight", 1.0)),
                }
            )
        )

    if not rows:
        raise RuntimeError("No universe rows produced. Check your constituents.csv dates.")

    panel = pd.concat(rows, ignore_index=True)
    panel["timestamp"] = to_utc(panel["timestamp"])
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    out = ParquetIO().write(panel, cfg.out_path)
    print("Wrote", out, f"({len(panel)} rows)")
    return out


if __name__ == "__main__":
    run(UniverseConfig())

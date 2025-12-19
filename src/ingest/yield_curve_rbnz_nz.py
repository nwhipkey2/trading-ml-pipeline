from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pandas as pd

from src.utils.http import get_bytes
from src.utils.paths import ensure_dir


@dataclass
class RbnzWholesaleConfig:
    bronze_dir: Path
    xlsx_urls: list[str]
    start: str
    end: str


def _read_rbnz_xlsx(content: bytes) -> pd.DataFrame:
    # RBNZ files are XLSX; first sheet usually contains the table
    df = pd.read_excel(BytesIO(content))
    # First column is the date
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize("UTC")
    return df


def run_rbnz_wholesale(cfg: RbnzWholesaleConfig) -> Path:
    out_dir = ensure_dir(cfg.bronze_dir / "source=rbnz_wholesale")

    frames = []
    for url in cfg.xlsx_urls:
        raw = get_bytes(url)
        df = _read_rbnz_xlsx(raw)
        df["source_url"] = url
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    df_all = df_all[(df_all["date"] >= start) & (df_all["date"] <= end)].copy()

    out = out_dir / "rbnz_wholesale_raw.parquet"
    df_all.to_parquet(out, index=False)
    return out

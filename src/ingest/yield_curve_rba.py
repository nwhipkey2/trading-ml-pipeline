from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pandas as pd

from src.utils.http import get_bytes
from src.utils.paths import ensure_dir


@dataclass
class RbaGovBondConfig:
    bronze_dir: Path
    csv_url: str
    start: str
    end: str


def _parse_rba_date(s: pd.Series) -> pd.Series:
    """
    RBA CSVs can contain:
    - ISO dates (YYYY-MM-DD)
    - "01-Jan-1990"
    - section title rows / blank rows (non-date)
    We parse robustly and return a datetime64[ns] series with NaT for non-dates.
    """
    s = s.astype(str).str.strip()

    # ISO first
    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    # Common RBA "01-Jan-1990"
    m = dt.isna()
    if m.any():
        dt.loc[m] = pd.to_datetime(s.loc[m], format="%d-%b-%Y", errors="coerce")

    # Fallback: dateutil (only for remaining)
    m2 = dt.isna()
    if m2.any():
        dt.loc[m2] = pd.to_datetime(s.loc[m2], errors="coerce")

    return dt


def run_rba_gov_bonds(cfg: RbaGovBondConfig) -> Path:
    out_dir = ensure_dir(cfg.bronze_dir / "source=rba_f2_1")

    raw = get_bytes(cfg.csv_url)
    df = pd.read_csv(BytesIO(raw))

    # First col is date-like (but the file also includes title rows)
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # Parse dates, drop non-date rows BEFORE comparisons
    df["date"] = _parse_rba_date(df["date"])
    df = df.dropna(subset=["date"]).copy()

    # Make timezone-aware UTC
    df["date"] = df["date"].dt.tz_localize("UTC")

    # Filter range
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    out = out_dir / "rba_gov_bonds_raw.parquet"
    df.to_parquet(out, index=False)
    return out


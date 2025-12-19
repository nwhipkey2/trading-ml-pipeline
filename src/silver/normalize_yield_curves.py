from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

from src.utils.paths import ensure_dir

@dataclass
class NormalizeYieldCurvesConfig:
    bronze_dir: Path
    silver_dir: Path

def _maturity_to_years(label: str) -> float | None:
    # Handles "SR_10Y", "SR_3M", Treasury "2 Yr", etc.
    label = label.strip()

    m = re.search(r"SR_(\d+)([MY])", label)
    if m:
        n = float(m.group(1))
        unit = m.group(2)
        return n if unit == "Y" else n / 12.0

    m = re.search(r"(\d+)\s*Mo", label, re.I)
    if m:
        return float(m.group(1)) / 12.0

    m = re.search(r"(\d+)\s*Yr", label, re.I)
    if m:
        return float(m.group(1))

    return None

def run_normalize_yield_curves(cfg: NormalizeYieldCurvesConfig) -> Path:
    out_dir = ensure_dir(Path(cfg.silver_dir))
    rows = []

    # US Treasury (wide)
    ust_path = cfg.bronze_dir / "source=us_treasury_yc" / "ust_yc_raw.parquet"
    if ust_path.exists():
        ust = pd.read_parquet(ust_path)
        for col in ust.columns:
            if col in ("date", "source_url"):
                continue
            yrs = _maturity_to_years(col)
            if yrs is None:
                continue
            tmp = ust[["date", col]].rename(columns={col: "yield"})
            tmp["region"] = "US"
            tmp["maturity_years"] = yrs
            tmp["source"] = "US_TREASURY"
            rows.append(tmp)

    # ECB YC (longish already)
    ecb_path = cfg.bronze_dir / "source=ecb_yc" / "yc_raw.parquet"
    if ecb_path.exists():
        ecb = pd.read_parquet(ecb_path)
        # ECB CSV usually has TIME_PERIOD + OBS_VALUE + "series_key"
        time_col = "TIME_PERIOD" if "TIME_PERIOD" in ecb.columns else "TIME"
        val_col = "OBS_VALUE" if "OBS_VALUE" in ecb.columns else "OBS_VALUE"
        tmp = ecb[[time_col, val_col, "series_key"]].rename(columns={time_col: "date", val_col: "yield"})
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.tz_localize("UTC")
        tmp["region"] = "EU"
        tmp["maturity_years"] = tmp["series_key"].apply(_maturity_to_years)
        tmp["source"] = "ECB_YC"
        rows.append(tmp[["date", "region", "maturity_years", "yield", "source"]])

    # RBA F2.1 (wide)
    rba_path = cfg.bronze_dir / "source=rba_f2_1" / "rba_gov_bonds_raw.parquet"
    if rba_path.exists():
        rba = pd.read_parquet(rba_path)
        for col in rba.columns:
            if col == "date":
                continue
            yrs = _maturity_to_years(col)  # works if columns include "2 year" etc; if not, you can map manually later
            if yrs is None:
                continue
            tmp = rba[["date", col]].rename(columns={col: "yield"})
            tmp["region"] = "AU"
            tmp["maturity_years"] = yrs
            tmp["source"] = "RBA_F2_1"
            rows.append(tmp)

    # Japan MOF (wide)
    jp_path = cfg.bronze_dir / "source=japan_mof_jgb" / "japan_mof_jgb_raw.parquet"
    if jp_path.exists():
        jp = pd.read_parquet(jp_path)
        for col in jp.columns:
            if col == "date":
                continue
            yrs = _maturity_to_years(col)
            if yrs is None:
                continue
            tmp = jp[["date", col]].rename(columns={col: "yield"})
            tmp["region"] = "JP"
            tmp["maturity_years"] = yrs
            tmp["source"] = "JAPAN_MOF"
            rows.append(tmp)

    # RBNZ wholesale (wide; you may need to adjust column mapping once you see headers)
    nz_path = cfg.bronze_dir / "source=rbnz_wholesale" / "rbnz_wholesale_raw.parquet"
    if nz_path.exists():
        nz = pd.read_parquet(nz_path)
        for col in nz.columns:
            if col == "date":
                continue
            yrs = _maturity_to_years(col)
            if yrs is None:
                continue
            tmp = nz[["date", col]].rename(columns={col: "yield"})
            tmp["region"] = "NZ"
            tmp["maturity_years"] = yrs
            tmp["source"] = "RBNZ_WHOLESALE"
            rows.append(tmp)

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["date","region","maturity_years","yield","source"]
    )
    out = out_dir / "yield_curves.parquet"
    out_df.to_parquet(out, index=False)
    return out

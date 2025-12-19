from __future__ import annotations

from pathlib import Path
import yaml

from src.utils.io import read_parquet  # only if you already have this; otherwise remove (not used here)

from src.ingest.yield_curve_us_treasury import run_us_treasury_curve, UsTreasuryCurveConfig
from src.ingest.yield_curve_ecb import run_ecb_yield_curve, EcbYieldCurveConfig
from src.ingest.yield_curve_rba import run_rba_gov_bonds, RbaGovBondConfig
from src.ingest.yield_curve_mof_jp import run_japan_mof, JapanMofConfig
from src.ingest.yield_curve_rbnz_nz import run_rbnz_wholesale, RbnzWholesaleConfig
from src.ingest.fx_ecb_majors import run_ecb_fx_majors, EcbFxMajorsConfig

from src.silver.normalize_yield_curves import run_normalize_yield_curves, NormalizeYieldCurvesConfig
from src.silver.normalize_fx import run_normalize_fx, NormalizeFxConfig


def main():
    cfg = yaml.safe_load(Path("config/macro.yaml").read_text(encoding="utf-8"))

    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    bronze = Path(cfg["paths"]["bronze"])
    silver = Path(cfg["paths"]["silver"])

    yc = cfg.get("yield_curves", {})

    # --- US Treasury Yield Curve ---
    us = yc.get("us_treasury", {})
    if us.get("enabled", False):
        run_us_treasury_curve(
            UsTreasuryCurveConfig(
                bronze_dir=bronze,
                archive_urls=us["archive_urls"],
                start=start,
                end=end,
            )
        )

    # --- ECB Euro Area Yield Curve (YC dataflow) ---
    ecb = yc.get("ecb_yc", {})
    if ecb.get("enabled", False):
        run_ecb_yield_curve(
            EcbYieldCurveConfig(
                bronze_dir=bronze,
                series_keys=ecb["series_keys"],
                start=start,
                end=end,
            )
        )

    # --- RBA Government Bond Yields (F2.1) ---
    rba = yc.get("rba_gov_bonds", {})
    if rba.get("enabled", False):
        run_rba_gov_bonds(
            RbaGovBondConfig(
                bronze_dir=bronze,
                csv_url=rba["csv_url"],
                start=start,
                end=end,
            )
        )

    # --- Japan MOF JGB curve ---
    jp = yc.get("japan_mof", {})
    if jp.get("enabled", False):
        run_japan_mof(
            JapanMofConfig(
                bronze_dir=bronze,
                csv_url=jp["csv_url"],
                start=start,
                end=end,
            )
        )

    # --- New Zealand RBNZ wholesale (optional; supports xlsx_urls or legacy data_url) ---
    nz = yc.get("rbnz_wholesale", {})
    if nz.get("enabled", False):
        if "xlsx_urls" in nz:
            xlsx_urls = nz["xlsx_urls"]
        elif "data_url" in nz:
            # legacy: treat it as one URL
            xlsx_urls = [nz["data_url"]]
        else:
            raise KeyError("yield_curves.rbnz_wholesale must contain xlsx_urls (preferred) or data_url")

        run_rbnz_wholesale(
            RbnzWholesaleConfig(
                bronze_dir=bronze,
                xlsx_urls=xlsx_urls,
                start=start,
                end=end,
            )
        )

    # --- FX majors derived from ECB EXR ---
    fx = cfg.get("fx", {}).get("ecb_majors", {})
    if fx.get("enabled", False):
        run_ecb_fx_majors(
            EcbFxMajorsConfig(
                bronze_dir=bronze,
                start=start,
                end=end,
                currencies_vs_eur=fx["currencies_vs_eur"],
                majors=fx["majors"],
            )
        )

    # --- Silver normalization outputs ---
    run_normalize_yield_curves(NormalizeYieldCurvesConfig(bronze_dir=bronze, silver_dir=silver))
    run_normalize_fx(NormalizeFxConfig(bronze_dir=bronze, silver_dir=silver))

    print("DONE: wrote", Path(silver) / "yield_curves.parquet", "and", Path(silver) / "fx_majors.parquet")


if __name__ == "__main__":
    main()

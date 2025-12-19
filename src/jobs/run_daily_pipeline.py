from __future__ import annotations

from pathlib import Path
import yaml

from src.universe.build_universe import run as run_universe, UniverseConfig
from src.universe.sector_map import run as run_sector, SectorMapConfig
from src.datasets.build_daily_panel import run as run_panel, BuildDailyPanelConfig


def main() -> None:
    ucfg = yaml.safe_load(Path("config/universe.yaml").read_text(encoding="utf-8"))
    scfg = yaml.safe_load(Path("config/equities_sleeve.yaml").read_text(encoding="utf-8"))

    # 1) Sector map
    run_sector(
        SectorMapConfig(
            sector_map_csv=ucfg["universe"]["sector_map_csv"],
            out_path=ucfg["outputs"]["silver_sector_path"],
        )
    )

    # 2) Universe membership panel (point-in-time)
    run_universe(
        UniverseConfig(
            index_id=ucfg["universe"]["index_id"],
            constituents_csv=ucfg["universe"]["constituents_csv"],
            start=ucfg["universe"]["start"],
            end=ucfg["universe"]["end"],
            out_path=ucfg["outputs"]["silver_universe_path"],
        )
    )

    # 3) Build gold daily panel (prices must already exist at data/silver/prices_daily.parquet)
    run_panel(
        BuildDailyPanelConfig(
            prices_path=scfg["data"]["prices_path"],
            universe_membership_path=scfg["data"]["universe_membership_path"],
            sector_map_path=scfg["data"]["sector_map_path"],
            out_path=scfg["data"]["panel_out_path"],
            horizon_days=scfg["model"]["horizon_days"],
        )
    )

    print("DONE: Phase 1 daily pipeline complete.")


if __name__ == "__main__":
    main()

from __future__ import annotations
from pathlib import Path
import yaml

from src.ingest import IngestConfig, run as run_ingest
from src.clean import CleanConfig, run as run_clean
from src.features import FeatureConfig, run as run_features
from src.train import TrainConfig, run as run_train

def load_config() -> dict:
    return yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))

def main() -> None:
    cfg = load_config()

    run_ingest(
        IngestConfig(
            universe=cfg["universe"],
            start=cfg["date_range"]["start"],
            end=cfg["date_range"]["end"],
            interval=cfg.get("frequency", "1d"),
            bronze_dir=Path(cfg["paths"]["bronze"]),
        )
    )

    run_clean(
        CleanConfig(
            bronze_dir=Path(cfg["paths"]["bronze"]),
            silver_dir=Path(cfg["paths"]["silver"]),
        )
    )

    dataset_path = run_features(
        FeatureConfig(
            silver_prices_path=Path(cfg["paths"]["silver"]) / "prices.parquet",
            gold_dir=Path(cfg["paths"]["gold"]),
            windows=list(cfg["features"]["windows"]),
            horizon_days=int(cfg["labels"]["horizon_days"]),
        )
    )

    metrics = run_train(
        TrainConfig(
            dataset_path=dataset_path,
            artifacts_dir=Path(cfg["paths"]["artifacts"]),
            test_frac=0.2,
        )
    )

    print("DONE. Metrics saved:", metrics)

if __name__ == "__main__":
    main()

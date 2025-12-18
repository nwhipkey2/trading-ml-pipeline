from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.io import ensure_dir, read_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: Path
    artifacts_dir: Path
    test_frac: float  # last X% of time as test


def run(cfg: TrainConfig) -> dict:
    ensure_dir(cfg.artifacts_dir / "models")
    ensure_dir(cfg.artifacts_dir / "metrics")

    df = read_parquet(cfg.dataset_path).sort_values(["timestamp", "symbol"]).copy()

    # time-based split: last test_frac portion by timestamp
    unique_ts = df["timestamp"].sort_values().unique()
    split_idx = int(len(unique_ts) * (1.0 - cfg.test_frac))
    split_ts = unique_ts[split_idx]

    train = df[df["timestamp"] < split_ts]
    test = df[df["timestamp"] >= split_ts]

    feature_cols = [c for c in df.columns if c.startswith(("ret_", "vol_", "close_sma_gap_", "vol_z_"))]
    X_train, y_train = train[feature_cols].to_numpy(), train["y_up"].to_numpy()
    X_test, y_test = test[feature_cols].to_numpy(), test["y_up"].to_numpy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )

    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    yhat = (p >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "split_timestamp": str(split_ts),
        "accuracy": float(accuracy_score(y_test, yhat)),
        "roc_auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else None,
        "features": feature_cols,
    }

    # Save metrics
    metrics_path = cfg.artifacts_dir / "metrics" / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved metrics to %s", metrics_path.as_posix())
    return metrics


if __name__ == "__main__":
    import yaml

    y = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    m = run(
        TrainConfig(
            dataset_path=Path(y["paths"]["gold"]) / "dataset.parquet",
            artifacts_dir=Path(y["paths"]["artifacts"]),
            test_frac=0.2,
        )
    )
    log.info("METRICS: %s", m)

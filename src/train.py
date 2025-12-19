from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io import ensure_dir, read_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


@dataclass(frozen=True)
class WalkForwardConfig:
    dataset_path: Path
    artifacts_dir: Path

    # Walk-forward parameters
    train_years: int = 3          # training window length
    test_months: int = 3          # test window length per fold
    step_months: int = 3          # how far to roll forward each fold

    min_train_rows: int = 500     # safety guard
    prob_threshold: float = 0.5   # classification threshold


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("ret_", "vol_", "close_sma_gap_", "vol_z_"))]


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], thr: float) -> dict:
    X_train = train[feature_cols].to_numpy()
    y_train = train["y_up"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["y_up"].to_numpy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )

    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    yhat = (p >= thr).astype(int)

    fold = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "accuracy": float(accuracy_score(y_test, yhat)),
        "roc_auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else None,
        "pos_rate_pred": float(yhat.mean()),
        "pos_rate_true": float(y_test.mean()),
    }
    return fold


def run_walk_forward(cfg: WalkForwardConfig) -> dict:
    ensure_dir(cfg.artifacts_dir / "metrics")

    df = read_parquet(cfg.dataset_path).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    feat_cols = _feature_cols(df)
    if not feat_cols:
        raise ValueError("No feature columns found. Did features.py run?")

    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    # Define rolling windows
    train_delta = pd.DateOffset(years=cfg.train_years)
    test_delta = pd.DateOffset(months=cfg.test_months)
    step_delta = pd.DateOffset(months=cfg.step_months)

    folds: list[dict] = []

    # First fold: train_end is (start + train_years)
    train_start = start_ts
    train_end = train_start + train_delta
    test_start = train_end
    test_end = test_start + test_delta

    fold_idx = 0
    while test_end <= end_ts:
        train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)]

        if len(train) < cfg.min_train_rows or len(test) == 0:
            log.info("Stopping: insufficient rows (train=%d test=%d)", len(train), len(test))
            break

        fold_metrics = _fit_predict(train, test, feat_cols, cfg.prob_threshold)
        fold_metrics.update(
            {
                "fold": fold_idx,
                "train_start": str(train_start),
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
            }
        )
        folds.append(fold_metrics)

        log.info(
            "Fold %d | acc=%.3f auc=%s | train=%d test=%d | %s→%s",
            fold_idx,
            fold_metrics["accuracy"],
            "NA" if fold_metrics["roc_auc"] is None else f"{fold_metrics['roc_auc']:.3f}",
            fold_metrics["n_train"],
            fold_metrics["n_test"],
            test_start.date(),
            test_end.date(),
        )

        # roll forward
        train_start = train_start + step_delta
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        fold_idx += 1

    folds_df = pd.DataFrame(folds)
    if folds_df.empty:
        raise RuntimeError("No folds were produced. Increase date range or reduce window sizes.")

    summary = {
        "dataset_path": str(cfg.dataset_path),
        "n_folds": int(len(folds_df)),
        "train_years": cfg.train_years,
        "test_months": cfg.test_months,
        "step_months": cfg.step_months,
        "prob_threshold": cfg.prob_threshold,
        "features": feat_cols,
        "accuracy_mean": float(folds_df["accuracy"].mean()),
        "accuracy_std": float(folds_df["accuracy"].std(ddof=0)),
        "roc_auc_mean": float(folds_df["roc_auc"].dropna().mean()) if folds_df["roc_auc"].notna().any() else None,
        "roc_auc_std": float(folds_df["roc_auc"].dropna().std(ddof=0)) if folds_df["roc_auc"].notna().any() else None,
    }

    # Save per-fold + summary
    out_folds = cfg.artifacts_dir / "metrics" / "walk_forward_folds.json"
    out_summary = cfg.artifacts_dir / "metrics" / "walk_forward_summary.json"
    out_folds.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Saved %s", out_folds.as_posix())
    log.info("Saved %s", out_summary.as_posix())

    return {"summary": summary, "folds": folds}


if __name__ == "__main__":
    import yaml

    y = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    run_walk_forward(
        WalkForwardConfig(
            dataset_path=Path(y["paths"]["gold"]) / "dataset.parquet",
            artifacts_dir=Path(y["paths"]["artifacts"]),
            train_years=3,
            test_months=3,
            step_months=3,
        )
    )


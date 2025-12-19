from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class TrainConfig:
    dataset_path: Path = Path("data/gold/final_dataset.parquet")
    artifacts_dir: Path = Path("artifacts")

    # Target
    target_col: str = "target_up_5"
    proba_threshold: float = 0.5

    # Time split (global, across all assets)
    split_date: str = "2024-01-01"  # everything before is train, on/after is test

    # Portfolio test
    top_n: int = 5
    horizon_days: int = 5  # must match target horizon for the simple backtest

    # Minimum rows per symbol in train/test to keep
    min_rows_per_symbol: int = 200

    # Optional: keep only these symbols (empty = all)
    symbols: tuple[str, ...] = ()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Select numeric feature columns while excluding obvious non-features and leakage columns.
    """
    exclude_prefixes = ("target_", "fwd_ret_")
    exclude_exact = {
        target_col,
        "timestamp",
        "date",
        "symbol",
        # raw OHLCV often ok, but start simple; you can add later
        "open", "high", "low", "adj_close", "volume",
    }

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = []
    for c in numeric_cols:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        feats.append(c)

    return sorted(feats)


def make_time_split(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_ts = pd.Timestamp(split_date, tz="UTC")
    train = df[df["timestamp"] < split_ts].copy()
    test = df[df["timestamp"] >= split_ts].copy()
    return train, test


def filter_symbols_by_rows(train: pd.DataFrame, test: pd.DataFrame, min_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_counts = train.groupby("symbol")["timestamp"].count()
    te_counts = test.groupby("symbol")["timestamp"].count()
    keep = tr_counts[tr_counts >= min_rows].index.intersection(te_counts[te_counts >= max(20, min_rows // 4)].index)
    train = train[train["symbol"].isin(keep)].copy()
    test = test[test["symbol"].isin(keep)].copy()
    return train, test


def topn_portfolio_backtest(
    test: pd.DataFrame,
    proba_col: str,
    fwd_ret_col: str,
    top_n: int,
) -> dict:
    """
    Very simple daily-rebalanced Top-N:
    For each date, buy top_n symbols by predicted probability.
    Realized return uses the already-computed forward return column (e.g. fwd_ret_5).
    """
    rows = []
    for ts, g in test.groupby("timestamp"):
        g = g.dropna(subset=[proba_col, fwd_ret_col]).copy()
        if len(g) == 0:
            continue
        g = g.sort_values(proba_col, ascending=False).head(top_n)
        port_ret = float(np.nanmean(g[fwd_ret_col].values)) if len(g) else np.nan
        rows.append((ts, port_ret))

    eq = pd.DataFrame(rows, columns=["timestamp", "port_fwd_ret"]).sort_values("timestamp")
    if eq.empty:
        return {"n_days": 0}

    # Equity curve: compound forward returns as if overlapping is allowed (this is a *signal quality* check)
    eq["equity"] = (1.0 + eq["port_fwd_ret"].fillna(0.0)).cumprod()

    # Metrics
    total_return = float(eq["equity"].iloc[-1] - 1.0)
    peak = eq["equity"].cummax()
    drawdown = (eq["equity"] / peak) - 1.0
    max_dd = float(drawdown.min())

    hit_rate = float((eq["port_fwd_ret"] > 0).mean())
    avg_ret = float(eq["port_fwd_ret"].mean())
    vol = float(eq["port_fwd_ret"].std(ddof=0))
    sharpe = float((avg_ret / vol) * np.sqrt(252)) if vol and np.isfinite(vol) and vol > 0 else float("nan")

    return {
        "n_days": int(eq.shape[0]),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "avg_fwd_ret": avg_ret,
        "vol_fwd_ret": vol,
        "sharpe_like": sharpe,
    }


def main(cfg: TrainConfig) -> None:
    ensure_dir(cfg.artifacts_dir / "models")
    ensure_dir(cfg.artifacts_dir / "metrics")
    ensure_dir(cfg.artifacts_dir / "preds")

    df = pd.read_parquet(cfg.dataset_path).copy()

    # Normalize timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", cfg.target_col]).copy()
    df["symbol"] = df["symbol"].astype(str)

    if cfg.symbols:
        df = df[df["symbol"].isin(cfg.symbols)].copy()

    # Split
    train, test = make_time_split(df, cfg.split_date)

    # Keep symbols with enough data in both sets
    train, test = filter_symbols_by_rows(train, test, cfg.min_rows_per_symbol)

    if train.empty or test.empty:
        raise RuntimeError(
            f"Train or test is empty after filtering. "
            f"train={len(train)} test={len(test)} split_date={cfg.split_date}"
        )

    # Features
    feature_cols = select_feature_columns(df, cfg.target_col)

    # Safety: drop rows missing any feature
    train = train.dropna(subset=feature_cols).copy()
    test = test.dropna(subset=feature_cols).copy()

    X_train = train[feature_cols].values
    y_train = train[cfg.target_col].astype(int).values

    X_test = test[feature_cols].values
    y_test = test[cfg.target_col].astype(int).values

    # Model: baseline logistic regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                max_iter=2000,
                n_jobs=None,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]
    )

    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= cfg.proba_threshold).astype(int)

    # Metrics
    acc = float(accuracy_score(y_test, pred_test))
    try:
        auc = float(roc_auc_score(y_test, proba_test))
    except Exception:
        auc = float("nan")

    # Save preds
    preds = test[["timestamp", "symbol"]].copy()
    preds["proba_up"] = proba_test
    preds["pred_up"] = pred_test
    # include realized fwd ret for quick analysis
    fwd_ret_col = f"fwd_ret_{cfg.horizon_days}"
    if fwd_ret_col in test.columns:
        preds[fwd_ret_col] = test[fwd_ret_col].values
    preds_path = cfg.artifacts_dir / "preds" / "panel_test_predictions.parquet"
    preds.to_parquet(preds_path, index=False)

    # Portfolio backtest (signal-quality check)
    port_metrics = {}
    if fwd_ret_col in test.columns:
        port_metrics = topn_portfolio_backtest(
            test=pd.concat([test[["timestamp", "symbol", fwd_ret_col]].reset_index(drop=True),
                            pd.Series(proba_test, name="proba_up")], axis=1),
            proba_col="proba_up",
            fwd_ret_col=fwd_ret_col,
            top_n=cfg.top_n,
        )

    out = {
        "dataset_path": str(cfg.dataset_path),
        "split_date": cfg.split_date,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_symbols_train": int(train["symbol"].nunique()),
        "n_symbols_test": int(test["symbol"].nunique()),
        "target_col": cfg.target_col,
        "features_n": int(len(feature_cols)),
        "features": feature_cols,
        "accuracy": acc,
        "roc_auc": auc,
        "proba_threshold": cfg.proba_threshold,
        "top_n": cfg.top_n,
        "horizon_days": cfg.horizon_days,
        "portfolio": port_metrics,
        "preds_path": str(preds_path),
    }

    # Save model + metrics
    model_path = cfg.artifacts_dir / "models" / "panel_logreg.pkl"
    joblib.dump(model, model_path)

    metrics_path = cfg.artifacts_dir / "metrics" / "panel_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2))

    print("DONE")
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print("Preds:", preds_path)
    print("Accuracy:", acc)
    print("ROC AUC:", auc)
    if port_metrics:
        print("Portfolio:", port_metrics)


if __name__ == "__main__":
    main(TrainConfig())

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
class WFConfig:
    dataset_path: Path = Path("data/gold/final_dataset_cs.parquet")
    artifacts_dir: Path = Path("artifacts")

    # Target horizon
    horizon_days: int = 5
    target_col: str = "target_up_5"

    # Walk-forward scheme
    train_years: int = 3
    test_months: int = 3
    step_months: int = 3

    # Signal/portfolio parameters (same as prod config default)
    top_n: int = 1
    hold_days: int = 5
    min_proba: float = 0.45
    cost_bps_per_side: float = 0.0005

    # Vol targeting
    target_vol_annual: float = 0.10
    vol_lookback_days: int = 20
    max_leverage: float = 3.0

    # Model
    proba_threshold: float = 0.5


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    exclude_prefixes = ("target_", "fwd_ret_")
    exclude_exact = {"timestamp", "symbol", target_col, "open", "high", "low", "adj_close", "volume"}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = []
    for c in numeric_cols:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        feats.append(c)
    return sorted(feats)


def _vol_target_leverage(realized_rets: list[float], lookback: int, target_vol_ann: float, max_lev: float) -> float:
    if len(realized_rets) < lookback:
        return 1.0
    w = np.array(realized_rets[-lookback:], dtype=float)
    dv = float(np.std(w, ddof=0))
    if not np.isfinite(dv) or dv <= 1e-12:
        return 1.0
    av = dv * np.sqrt(252)
    lev = target_vol_ann / av
    return float(np.clip(lev, 0.0, max_lev))


def backtest_fold_voltarget(
    df_test: pd.DataFrame,
    proba: np.ndarray,
    cfg: WFConfig,
) -> dict:
    """
    Non-overlapping Top-N hold with vol targeting, using df_test ret_1.
    Assumes df_test has columns: timestamp, symbol, ret_1
    """
    test = df_test[["timestamp", "symbol", "ret_1"]].copy()
    test["proba_up"] = proba
    test = test.dropna(subset=["ret_1", "proba_up"]).sort_values(["timestamp", "symbol"])

    dates = test["timestamp"].drop_duplicates().sort_values().tolist()
    day_map = {ts: g for ts, g in test.groupby("timestamp")}

    active: dict[str, int] = {}
    equity = 1.0
    realized: list[float] = []
    cost = float(cfg.cost_bps_per_side)

    rows = []

    for ts in dates:
        day = day_map.get(ts)
        if day is None or day.empty:
            continue

        held_syms = list(active.keys())
        if held_syms:
            dh = day[day["symbol"].isin(held_syms)]
            rets = dh.set_index("symbol")["ret_1"].to_dict()
            held_rets = [float(rets.get(s, 0.0)) for s in held_syms]
            gross_ret = float(np.mean(held_rets)) if held_rets else 0.0
            gross_exposure = 1.0
        else:
            gross_ret = 0.0
            gross_exposure = 0.0

        lev = _vol_target_leverage(realized, cfg.vol_lookback_days, cfg.target_vol_annual, cfg.max_leverage)
        if gross_exposure <= 0.0:
            lev = 0.0

        port_ret = lev * gross_ret
        equity *= (1.0 + port_ret)
        realized.append(port_ret)

        # decrement holds and exit
        exited = []
        for s in list(active.keys()):
            active[s] -= 1
            if active[s] <= 0:
                exited.append(s)
                del active[s]
        if exited:
            equity *= (1.0 - cost * (len(exited) / max(1, cfg.top_n)))

        # enter new
        slots = cfg.top_n - len(active)
        entered = []
        if slots > 0:
            cand = day[~day["symbol"].isin(active.keys())].copy()
            cand = cand[cand["proba_up"] >= cfg.min_proba].sort_values("proba_up", ascending=False).head(slots)
            for s in cand["symbol"].tolist():
                active[s] = int(cfg.hold_days)
                entered.append(s)
        if entered:
            equity *= (1.0 - cost * (len(entered) / max(1, cfg.top_n)))

        rows.append({"timestamp": ts, "port_ret": port_ret, "equity": equity, "leverage": lev})

    eq = pd.DataFrame(rows)
    if eq.empty:
        return {"n_days": 0}

    total_return = float(eq["equity"].iloc[-1] - 1.0)
    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak) - 1.0
    max_dd = float(dd.min())
    avg = float(eq["port_ret"].mean())
    vol = float(eq["port_ret"].std(ddof=0))
    sharpe = float((avg / vol) * np.sqrt(252)) if vol and np.isfinite(vol) and vol > 0 else float("nan")

    return {"n_days": int(len(eq)), "total_return": total_return, "max_drawdown": max_dd, "sharpe": sharpe}


def main(cfg: WFConfig) -> None:
    ensure_dir(cfg.artifacts_dir / "metrics")
    ensure_dir(cfg.artifacts_dir / "models")

    df = pd.read_parquet(cfg.dataset_path).copy()
    df["timestamp"] = _to_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp", "symbol", cfg.target_col, "ret_1"]).copy()
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    feats = select_feature_columns(df, cfg.target_col)

    # fold boundaries (calendar-based on timestamps)
    t0 = df["timestamp"].min()
    t1 = df["timestamp"].max()

    # Start after we have train_years history
    start_train_end = (t0 + pd.DateOffset(years=cfg.train_years)).normalize()
    fold_start = start_train_end

    folds = []
    while True:
        train_end = fold_start
        test_start = train_end
        test_end = (test_start + pd.DateOffset(months=cfg.test_months))

        if test_end > t1:
            break

        folds.append((train_end, test_start, test_end))
        fold_start = (fold_start + pd.DateOffset(months=cfg.step_months))

    if not folds:
        raise RuntimeError("No folds created. Not enough history for the chosen train/test windows.")

    fold_results = []

    for i, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train = df[df["timestamp"] < train_end].copy()
        test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)].copy()

        # drop NA features
        train = train.dropna(subset=feats).copy()
        test = test.dropna(subset=feats).copy()
        if train.empty or test.empty:
            continue

        X_train = train[feats].values
        y_train = train[cfg.target_col].astype(int).values
        X_test = test[feats].values
        y_test = test[cfg.target_col].astype(int).values

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= cfg.proba_threshold).astype(int)

        acc = float(accuracy_score(y_test, pred))
        try:
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = float("nan")

        bt = backtest_fold_voltarget(test, proba, cfg)

        fold_results.append(
            {
                "fold": i,
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "accuracy": acc,
                "roc_auc": auc,
                **{f"bt_{k}": v for k, v in bt.items()},
            }
        )

        # Save last fold model as a convenience
        if i == len(folds):
            joblib.dump(model, cfg.artifacts_dir / "models" / "walkforward_last_model.pkl")

    res = pd.DataFrame(fold_results)
    if res.empty:
        raise RuntimeError("All folds empty after filtering. Check your dataset coverage.")

    summary = {
        "dataset_path": str(cfg.dataset_path),
        "target_col": cfg.target_col,
        "train_years": cfg.train_years,
        "test_months": cfg.test_months,
        "step_months": cfg.step_months,
        "n_folds": int(res.shape[0]),
        "roc_auc_mean": float(res["roc_auc"].mean()),
        "roc_auc_std": float(res["roc_auc"].std(ddof=0)),
        "bt_sharpe_mean": float(res["bt_sharpe"].mean()),
        "bt_sharpe_std": float(res["bt_sharpe"].std(ddof=0)),
        "bt_total_return_mean": float(res["bt_total_return"].mean()),
        "bt_max_drawdown_mean": float(res["bt_max_drawdown"].mean()),
        "prod_like": {
            "top_n": cfg.top_n,
            "min_proba": cfg.min_proba,
            "hold_days": cfg.hold_days,
            "target_vol_annual": cfg.target_vol_annual,
            "max_leverage": cfg.max_leverage,
            "cost_bps_per_side": cfg.cost_bps_per_side,
        },
    }

    out = {"summary": summary, "folds": fold_results}
    out_path = cfg.artifacts_dir / "metrics" / "walk_forward_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("DONE")
    print("Metrics:", out_path)
    print("Summary:", summary)


if __name__ == "__main__":
    main(WFConfig())

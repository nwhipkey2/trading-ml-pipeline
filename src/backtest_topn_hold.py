from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    # Inputs
    dataset_path: Path = Path("data/gold/final_dataset.parquet")
    preds_path: Path = Path("artifacts/preds/panel_test_predictions.parquet")

    # Output
    out_dir: Path = Path("artifacts/backtests")
    equity_out: str = "topn_hold_equity.parquet"
    metrics_out: str = "topn_hold_metrics.json"

    # Strategy
    top_n: int = 5
    hold_days: int = 5
    min_proba: float = 0.0  # set e.g. 0.55 to be more selective
    allow_cash: bool = True  # if True, unfilled slots remain in cash (0 return)

    # Costs
    # charged per side (entry and exit), e.g. 5 bps = 0.0005
    cost_bps_per_side: float = 0.0005

    # Columns
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    proba_col: str = "proba_up"
    daily_ret_col: str = "ret_1"  # daily close-to-close returns


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def compute_metrics(eq: pd.DataFrame) -> dict:
    if eq.empty:
        return {"n_days": 0}

    equity = eq["equity"].astype(float)
    daily = eq["port_ret"].astype(float)

    total_return = float(equity.iloc[-1] - 1.0)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min())

    hit_rate = float((daily > 0).mean())
    avg = float(daily.mean())
    vol = float(daily.std(ddof=0))
    sharpe = float((avg / vol) * np.sqrt(252)) if vol and np.isfinite(vol) and vol > 0 else float("nan")

    return {
        "n_days": int(len(eq)),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "avg_daily_ret": avg,
        "vol_daily_ret": vol,
        "sharpe": sharpe,
        "final_equity": float(equity.iloc[-1]),
    }


def run(cfg: BacktestConfig) -> tuple[Path, Path]:
    ensure_dir(cfg.out_dir)

    # --- Load predictions (usually test set only) ---
    preds = pd.read_parquet(cfg.preds_path).copy()
    preds[cfg.timestamp_col] = _to_utc(preds[cfg.timestamp_col])
    preds = preds.dropna(subset=[cfg.timestamp_col, cfg.symbol_col, cfg.proba_col]).copy()
    preds[cfg.symbol_col] = preds[cfg.symbol_col].astype(str)
    preds = preds.sort_values([cfg.timestamp_col, cfg.symbol_col]).reset_index(drop=True)

    # --- Load dataset for daily returns ---
    df = pd.read_parquet(cfg.dataset_path).copy()
    df[cfg.timestamp_col] = _to_utc(df[cfg.timestamp_col])
    df = df.dropna(subset=[cfg.timestamp_col, cfg.symbol_col, cfg.daily_ret_col]).copy()
    df[cfg.symbol_col] = df[cfg.symbol_col].astype(str)
    df = df[[cfg.timestamp_col, cfg.symbol_col, cfg.daily_ret_col]].sort_values(
        [cfg.timestamp_col, cfg.symbol_col]
    )

    # Merge predictions with daily returns on (timestamp, symbol)
    m = pd.merge(
        preds[[cfg.timestamp_col, cfg.symbol_col, cfg.proba_col]],
        df,
        on=[cfg.timestamp_col, cfg.symbol_col],
        how="left",
    )
    m = m.dropna(subset=[cfg.daily_ret_col]).copy()
    m = m.sort_values([cfg.timestamp_col, cfg.symbol_col]).reset_index(drop=True)

    # Use only dates present after merge
    dates = m[cfg.timestamp_col].drop_duplicates().sort_values().tolist()

    # Active positions: dict[symbol] = remaining_days (int)
    active: dict[str, int] = {}

    # For costs: we charge on entry and exit
    cost = float(cfg.cost_bps_per_side)

    rows = []
    equity = 1.0

    for ts in dates:
        day = m[m[cfg.timestamp_col] == ts].copy()
        if day.empty:
            continue

        # --- 1) Apply returns for positions held overnight into this day ---
        # We assume ret_1 corresponds to close-to-close ending at this timestamp.
        # So positions "active" experience today's ret_1.
        held_syms = list(active.keys())
        if held_syms:
            day_held = day[day[cfg.symbol_col].isin(held_syms)]
            # If any held symbol missing that day, treat as 0 return (halted/no data)
            rets = day_held.set_index(cfg.symbol_col)[cfg.daily_ret_col].to_dict()
            held_returns = [float(rets.get(sym, 0.0)) for sym in held_syms]

            # Equal-weight among held positions; cash gets 0 if allow_cash and fewer than top_n held
            if cfg.allow_cash and len(held_syms) < cfg.top_n:
                # weights sum to len(held)/top_n
                port_ret = (sum(held_returns) / cfg.top_n)
            else:
                port_ret = float(np.mean(held_returns)) if held_returns else 0.0
        else:
            port_ret = 0.0

        # Update equity
        equity *= (1.0 + port_ret)

        # --- 2) Decrement holding timers, exit positions that expire AFTER today's close ---
        exited = []
        for sym in list(active.keys()):
            active[sym] -= 1
            if active[sym] <= 0:
                exited.append(sym)
                del active[sym]

        # Charge exit costs (at exit)
        if exited:
            # If allow_cash, cost scales by 1/top_n; otherwise by 1/len(held) (approx)
            if cfg.allow_cash:
                equity *= (1.0 - cost * (len(exited) / cfg.top_n))
            else:
                denom = max(1, len(held_syms))
                equity *= (1.0 - cost * (len(exited) / denom))

        # --- 3) Select new entries for tomorrow (after close) ---
        # Available slots:
        slots = cfg.top_n - len(active)
        entered = []

        if slots > 0:
            # Filter candidates: not already held, above min_proba
            cand = day[~day[cfg.symbol_col].isin(active.keys())].copy()
            if cfg.min_proba > 0:
                cand = cand[cand[cfg.proba_col] >= cfg.min_proba].copy()

            cand = cand.sort_values(cfg.proba_col, ascending=False).head(slots)
            for sym in cand[cfg.symbol_col].tolist():
                active[sym] = int(cfg.hold_days)
                entered.append(sym)

        # Charge entry costs (at entry)
        if entered:
            if cfg.allow_cash:
                equity *= (1.0 - cost * (len(entered) / cfg.top_n))
            else:
                denom = max(1, len(active))
                equity *= (1.0 - cost * (len(entered) / denom))

        rows.append(
            {
                "timestamp": ts,
                "port_ret": port_ret,
                "equity": equity,
                "n_active": len(active),
                "entered": entered,
                "exited": exited,
            }
        )

    eq = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    metrics = compute_metrics(eq)
    metrics.update(
        {
            "top_n": cfg.top_n,
            "hold_days": cfg.hold_days,
            "min_proba": cfg.min_proba,
            "allow_cash": cfg.allow_cash,
            "cost_bps_per_side": cfg.cost_bps_per_side,
            "dataset_path": str(cfg.dataset_path),
            "preds_path": str(cfg.preds_path),
        }
    )

    equity_path = cfg.out_dir / cfg.equity_out
    metrics_path = cfg.out_dir / cfg.metrics_out

    eq.to_parquet(equity_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("DONE")
    print("Equity:", equity_path)
    print("Metrics:", metrics_path)
    print("Summary:", metrics)

    return equity_path, metrics_path


if __name__ == "__main__":
    run(BacktestConfig())

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestVolTargetConfig:
    # Inputs
    dataset_path: Path = Path("data/gold/final_dataset.parquet")
    preds_path: Path = Path("artifacts/preds/panel_test_predictions.parquet")

    # Output
    out_dir: Path = Path("artifacts/backtests")
    equity_out: str = "voltarget_equity.parquet"
    metrics_out: str = "voltarget_metrics.json"

    # Strategy
    top_n: int = 3
    hold_days: int = 5
    min_proba: float = 0.55  # based on your sweep
    allow_cash: bool = True

    # Costs (per side)
    cost_bps_per_side: float = 0.0005  # 5 bps

    # Vol targeting
    target_vol_annual: float = 0.10     # 10% annualized vol target
    vol_lookback_days: int = 20         # rolling window for realized vol
    max_leverage: float = 3.0           # cap leverage
    min_vol_floor: float = 1e-6         # avoid divide by zero

    # Columns
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    proba_col: str = "proba_up"
    daily_ret_col: str = "ret_1"        # close-to-close daily return


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

    avg = float(daily.mean())
    vol = float(daily.std(ddof=0))
    sharpe = float((avg / vol) * np.sqrt(252)) if vol and np.isfinite(vol) and vol > 0 else float("nan")

    invested = eq["gross_exposure"] > 0
    invested_day_frac = float(invested.mean())
    hit_rate_invested = float((eq.loc[invested, "port_ret"] > 0).mean()) if invested.any() else float("nan")

    avg_gross_exposure = float(eq["gross_exposure"].mean()) if "gross_exposure" in eq.columns else float("nan")

    return {
        "n_days": int(len(eq)),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "avg_daily_ret": avg,
        "vol_daily_ret": vol,
        "sharpe": sharpe,
        "final_equity": float(equity.iloc[-1]),
        "avg_gross_exposure": avg_gross_exposure,
        "invested_day_frac": invested_day_frac,
        "hit_rate_invested": hit_rate_invested,
    }



def run(cfg: BacktestVolTargetConfig) -> tuple[Path, Path]:
    ensure_dir(cfg.out_dir)

    # --- Load predictions ---
    preds = pd.read_parquet(cfg.preds_path).copy()
    preds[cfg.timestamp_col] = _to_utc(preds[cfg.timestamp_col])
    preds = preds.dropna(subset=[cfg.timestamp_col, cfg.symbol_col, cfg.proba_col]).copy()
    preds[cfg.symbol_col] = preds[cfg.symbol_col].astype(str)
    preds = preds.sort_values([cfg.timestamp_col, cfg.symbol_col]).reset_index(drop=True)

    # --- Load daily returns from dataset ---
    df = pd.read_parquet(cfg.dataset_path).copy()
    df[cfg.timestamp_col] = _to_utc(df[cfg.timestamp_col])
    df = df.dropna(subset=[cfg.timestamp_col, cfg.symbol_col, cfg.daily_ret_col]).copy()
    df[cfg.symbol_col] = df[cfg.symbol_col].astype(str)
    df = df[[cfg.timestamp_col, cfg.symbol_col, cfg.daily_ret_col]].sort_values(
        [cfg.timestamp_col, cfg.symbol_col]
    )

    # Merge predictions with daily returns
    m = pd.merge(
        preds[[cfg.timestamp_col, cfg.symbol_col, cfg.proba_col]],
        df,
        on=[cfg.timestamp_col, cfg.symbol_col],
        how="left",
    ).dropna(subset=[cfg.daily_ret_col]).copy()

    m = m.sort_values([cfg.timestamp_col, cfg.symbol_col]).reset_index(drop=True)

    dates = m[cfg.timestamp_col].drop_duplicates().sort_values().tolist()

    active: dict[str, int] = {}
    cost = float(cfg.cost_bps_per_side)

    rows = []
    equity = 1.0

    # For vol targeting: keep history of realized (net) portfolio daily returns
    realized_port_rets: list[float] = []

    for ts in dates:
        day = m[m[cfg.timestamp_col] == ts].copy()
        if day.empty:
            continue

        # --- 1) Compute gross (unlevered) portfolio return from held positions ---
        held_syms = list(active.keys())
        if held_syms:
            day_held = day[day[cfg.symbol_col].isin(held_syms)]
            rets = day_held.set_index(cfg.symbol_col)[cfg.daily_ret_col].to_dict()
            held_returns = [float(rets.get(sym, 0.0)) for sym in held_syms]

            if cfg.allow_cash and len(held_syms) < cfg.top_n:
                gross_ret = (sum(held_returns) / cfg.top_n)
                gross_exposure = len(held_syms) / cfg.top_n
            else:
                gross_ret = float(np.mean(held_returns)) if held_returns else 0.0
                gross_exposure = 1.0 if held_syms else 0.0
        else:
            gross_ret = 0.0
            gross_exposure = 0.0

        # --- 2) Determine leverage via vol targeting ---
        # Estimate realized annualized vol from past realized returns
        if len(realized_port_rets) >= cfg.vol_lookback_days:
            window = np.array(realized_port_rets[-cfg.vol_lookback_days:], dtype=float)
            realized_daily_vol = float(np.std(window, ddof=0))
        else:
            realized_daily_vol = float("nan")

        if np.isfinite(realized_daily_vol) and realized_daily_vol > cfg.min_vol_floor:
            realized_annual_vol = realized_daily_vol * np.sqrt(252)
            lev = cfg.target_vol_annual / realized_annual_vol
        else:
            # Not enough history yet: use 1x
            lev = 1.0

        lev = float(np.clip(lev, 0.0, cfg.max_leverage))

        # If no positions, leverage effectively 0
        if gross_exposure <= 0.0:
            lev = 0.0

        # --- 3) Apply leveraged return to equity ---
        port_ret = lev * gross_ret
        equity *= (1.0 + port_ret)

        # record realized return for vol targeting (after leverage)
        realized_port_rets.append(port_ret)

        # --- 4) Decrement holds; exit expired positions after today ---
        exited = []
        for sym in list(active.keys()):
            active[sym] -= 1
            if active[sym] <= 0:
                exited.append(sym)
                del active[sym]

        # Apply exit costs proportional to position slots
        if exited:
            # costs scale with gross exposure slots, not leverage
            equity *= (1.0 - cost * (len(exited) / cfg.top_n))

        # --- 5) Enter new positions for next day ---
        slots = cfg.top_n - len(active)
        entered = []
        if slots > 0:
            cand = day[~day[cfg.symbol_col].isin(active.keys())].copy()
            if cfg.min_proba > 0:
                cand = cand[cand[cfg.proba_col] >= cfg.min_proba].copy()

            cand = cand.sort_values(cfg.proba_col, ascending=False).head(slots)
            for sym in cand[cfg.symbol_col].tolist():
                active[sym] = int(cfg.hold_days)
                entered.append(sym)

        if entered:
            equity *= (1.0 - cost * (len(entered) / cfg.top_n))

        rows.append(
            {
                "timestamp": ts,
                "gross_ret": gross_ret,
                "leverage": lev,
                "port_ret": port_ret,
                "equity": equity,
                "n_active": len(active),
                "gross_exposure": gross_exposure,
                "entered": entered,
                "exited": exited,
                "realized_daily_vol": realized_daily_vol,
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
            "target_vol_annual": cfg.target_vol_annual,
            "vol_lookback_days": cfg.vol_lookback_days,
            "max_leverage": cfg.max_leverage,
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
    run(BacktestVolTargetConfig())

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, read_parquet

def run_fixed_params_cross_asset(
    prices: pd.DataFrame,
    symbols: list[str],
    sma_window: int,
    z_entry: float,
    artifacts_dir: Path,
):
    from src.strategy_optimize import backtest_entry_then_hold, buy_and_hold_metrics, OptConfig
    from src.utils.io import ensure_dir
    import json
    import logging

    log = logging.getLogger("cross_asset")

    ensure_dir(artifacts_dir / "metrics")
    results = {}

    cfg_stub = OptConfig(
        prices_path=Path(""),
        artifacts_dir=artifacts_dir,
        symbols=[],
    )

    for sym in symbols:
        df = prices[prices["symbol"] == sym].copy()
        if len(df) < sma_window * 5:
            log.warning("Skipping %s (insufficient history)", sym)
            continue

        strat = backtest_entry_then_hold(
            df=df,
            n=sma_window,
            k=z_entry,
            cfg=cfg_stub,
            return_curve=True,
        )

        bh = buy_and_hold_metrics(df)

        results[sym] = {
            "strategy": {
                "total_return": strat["total_return"],
                "sharpe": strat["sharpe"],
                "max_drawdown": strat["max_drawdown"],
                "time_in_market": strat["time_in_market"],
            },
            "buy_hold": {
                "total_return": bh["total_return"],
                "sharpe": bh["sharpe"],
                "max_drawdown": bh["max_drawdown"],
            },
            "delta": {
                "total_return": strat["total_return"] - bh["total_return"],
                "sharpe": strat["sharpe"] - bh["sharpe"],
                "max_drawdown": strat["max_drawdown"] - bh["max_drawdown"],
            },
        }

    out = artifacts_dir / "metrics" / "cross_asset_fixed_params.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved %s", out)

    return results


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("opt")


@dataclass(frozen=True)
class OptConfig:
    prices_path: Path                 # data/silver/prices.parquet
    artifacts_dir: Path               # artifacts/
    symbols: list[str]                # ["SPY", "QQQ"]

    # Walk-forward parameters
    train_years: int = 3
    test_months: int = 3
    step_months: int = 3

    # Parameter grid (aggressively relaxed dips)
    sma_windows: tuple[int, ...] = (10, 20, 50, 100)
    z_entry: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)   # enter when z <= -k

    # Costs (conservative defaults)
    fee_bps: float = 1.0
    slippage_bps: float = 1.0


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _sharpe(daily_ret: pd.Series) -> float:
    r = daily_ret.dropna()
    if len(r) < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((mu / sd) * np.sqrt(252))


def buy_and_hold_metrics(df: pd.DataFrame) -> dict:
    df = df.sort_values("timestamp").copy()
    ret = df["close"].astype(float).pct_change(1).fillna(0.0)
    equity = (1.0 + ret).cumprod()
    return {
        "total_return": float(equity.iat[-1] - 1.0),
        "sharpe": _sharpe(ret),
        "max_drawdown": _max_drawdown(equity),
        "equity_curve": equity.tolist(),
        "timestamps": df["timestamp"].astype(str).tolist(),
    }


def backtest_entry_then_hold(
    df: pd.DataFrame,
    n: int,
    k: float,
    cfg: OptConfig,
    return_curve: bool = False,
) -> dict:
    """
    Buy & Hold with dip-optimized entry:
    - Start in cash
    - Compute z = (price - SMA_n) / STD_n
    - Enter on first z <= -k
    - Once entered, hold to the end
    - If no dip ever occurs, enter on first valid z day (prevents "never enter")
    """
    df = df.sort_values("timestamp").copy()
    px = df["close"].astype(float)

    sma = px.rolling(n).mean()
    std = px.rolling(n).std(ddof=0).replace(0, np.nan)
    z = (px - sma) / std
    df["z"] = z

    df["ret_1"] = px.pct_change(1).fillna(0.0)

    # Find first valid z index
    first_valid_pos = int(np.argmax(~np.isnan(z.to_numpy())))
    if np.isnan(z.iat[first_valid_pos]):
        # no valid z at all (too-short df)
        first_valid_pos = 0

    # Find first dip entry (z <= -k)
    entry_pos = None
    for i in range(first_valid_pos, len(df)):
        if np.isnan(df["z"].iat[i]):
            continue
        if df["z"].iat[i] <= -k:
            entry_pos = i
            break

    # If no dip ever occurs, enter at first valid z
    if entry_pos is None:
        entry_pos = first_valid_pos

    pos = np.zeros(len(df), dtype=int)
    pos[entry_pos:] = 1
    df["pos"] = pos

    # One-time entry cost
    cost = (cfg.fee_bps + cfg.slippage_bps) / 10000.0
    df["cost"] = 0.0
    df.iloc[entry_pos, df.columns.get_loc("cost")] = cost

    df["strategy_ret"] = df["pos"].shift(1).fillna(0) * df["ret_1"] - df["cost"]
    equity = (1.0 + df["strategy_ret"]).cumprod()

    out = {
        "n": int(n),
        "k": float(k),
        "entry_index": int(entry_pos),
        "entry_date": str(df["timestamp"].iat[entry_pos]),
        "trades": 1,
        "time_in_market": float(df["pos"].mean()),
        "total_return": float(equity.iat[-1] - 1.0),
        "sharpe": _sharpe(df["strategy_ret"]),
        "max_drawdown": _max_drawdown(equity),
    }

    if return_curve:
        out["equity_curve"] = equity.tolist()
        out["timestamps"] = df["timestamp"].astype(str).tolist()

    return out


def score_row(row: dict) -> float:
    """
    Drawdown-aware objective:
    - Prefer higher Sharpe
    - Slightly prefer higher total return
    - Penalize drawdown
    """
    dd_penalty = 1.0
    return row["sharpe"] + 0.25 * row["total_return"] - dd_penalty * abs(row["max_drawdown"])


def choose_best_params(train_df: pd.DataFrame, cfg: OptConfig) -> dict:
    best = None
    best_score = -1e18

    for n, k in itertools.product(cfg.sma_windows, cfg.z_entry):
        r = backtest_entry_then_hold(train_df, n, k, cfg, return_curve=False)
        s = score_row(r)
        if s > best_score:
            best_score = s
            best = r

    assert best is not None
    best["score"] = float(best_score)
    return best


def run(cfg: OptConfig) -> dict:
    ensure_dir(cfg.artifacts_dir / "metrics")

    prices = read_parquet(cfg.prices_path).copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["timestamp", "close", "symbol"]).copy()
    prices = prices[prices["symbol"].isin(cfg.symbols)].copy()
    prices = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    start_ts = prices["timestamp"].min()
    end_ts = prices["timestamp"].max()

    train_delta = pd.DateOffset(years=cfg.train_years)
    test_delta = pd.DateOffset(months=cfg.test_months)
    step_delta = pd.DateOffset(months=cfg.step_months)

    folds: list[dict] = []

    train_start = start_ts
    train_end = train_start + train_delta
    test_start = train_end
    test_end = test_start + test_delta

    fold_idx = 0
    while test_end <= end_ts:
        fold_out = {
            "fold": fold_idx,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "per_symbol": [],
        }

        for sym in cfg.symbols:
            df_sym = prices[prices["symbol"] == sym].sort_values("timestamp")

            train_df = df_sym[(df_sym["timestamp"] >= train_start) & (df_sym["timestamp"] < train_end)]
            test_df = df_sym[(df_sym["timestamp"] >= test_start) & (df_sym["timestamp"] < test_end)]

            # safety guards
            if len(train_df) < 200 or len(test_df) < 40:
                continue

            best = choose_best_params(train_df, cfg)

            test_res = backtest_entry_then_hold(
                test_df,
                best["n"],
                best["k"],
                cfg,
                return_curve=True,
            )

            bh = buy_and_hold_metrics(test_df)

            delta = {
                "total_return": float(test_res["total_return"] - bh["total_return"]),
                "sharpe": float(test_res["sharpe"] - bh["sharpe"]),
                "max_drawdown": float(test_res["max_drawdown"] - bh["max_drawdown"]),
            }

            fold_out["per_symbol"].append(
                {
                    "symbol": sym,
                    "chosen_params": {kk: best[kk] for kk in ["n", "k", "score"]},
                    "oos_strategy": {kk: test_res[kk] for kk in ["n", "k", "entry_date", "time_in_market", "total_return", "sharpe", "max_drawdown"]},
                    "oos_buy_hold": {kk: bh[kk] for kk in ["total_return", "sharpe", "max_drawdown"]},
                    "oos_delta": delta,
                    "curve": {
                        "timestamps": test_res["timestamps"],
                        "equity_strategy": test_res["equity_curve"],
                        "equity_buy_hold": bh["equity_curve"],
                    },
                }
            )

        folds.append(fold_out)
        log.info("Fold %d complete (%s→%s)", fold_idx, test_start.date(), test_end.date())

        train_start = train_start + step_delta
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        fold_idx += 1

    out_path = cfg.artifacts_dir / "metrics" / "dip_entry_hold_walkforward.json"
    out_path.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    log.info("Saved %s", out_path.as_posix())

    return {"n_folds": len(folds), "output": str(out_path)}


if __name__ == "__main__":
    import yaml

    y = yaml.safe_load(Path("config/pipeline.yaml").read_text())

    prices = read_parquet(Path(y["paths"]["silver"]) / "prices.parquet")

    FIXED_SMA = 50
    FIXED_Z = 0.5

    run_fixed_params_cross_asset(
        prices=prices,
        symbols=["SPY", "QQQ", "DIA", "XAU", "XAG"],
        sma_window=FIXED_SMA,
        z_entry=FIXED_Z,
        artifacts_dir=Path(y["paths"]["artifacts"]),
    )


def run_fixed_params_cross_asset(
    prices: pd.DataFrame,
    symbols: list[str],
    sma_window: int,
    z_entry: float,
    artifacts_dir: Path,
):
    ensure_dir(artifacts_dir / "metrics")

    results = {}

    for sym in symbols:
        df = prices[prices["symbol"] == sym].copy()
        if len(df) < sma_window * 5:
            log.warning("Skipping %s (insufficient history)", sym)
            continue

        strat = backtest_entry_then_hold(
            df,
            n=sma_window,
            k=z_entry,
            cfg=OptConfig(
                prices_path=Path(""),
                artifacts_dir=artifacts_dir,
                symbols=[],
            ),
            return_curve=True,
        )

        bh = buy_and_hold_metrics(df)

        results[sym] = {
            "strategy": {
                "total_return": strat["total_return"],
                "sharpe": strat["sharpe"],
                "max_drawdown": strat["max_drawdown"],
                "time_in_market": strat["time_in_market"],
            },
            "buy_hold": {
                "total_return": bh["total_return"],
                "sharpe": bh["sharpe"],
                "max_drawdown": bh["max_drawdown"],
            },
            "delta": {
                "total_return": strat["total_return"] - bh["total_return"],
                "sharpe": strat["sharpe"] - bh["sharpe"],
                "max_drawdown": strat["max_drawdown"] - bh["max_drawdown"],
            },
        }

    out = artifacts_dir / "metrics" / "cross_asset_fixed_params.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved %s", out)

    return results

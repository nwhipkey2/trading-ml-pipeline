from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest_topn_hold import BacktestConfig, run as run_backtest


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    # Sweep settings
    top_n_values = [1, 3, 5, 10]
    min_proba_values = [0.50, 0.55, 0.60, 0.65, 0.70]

    # Base config (inherits your current defaults)
    base = BacktestConfig(
        top_n=5,
        hold_days=5,
        min_proba=0.0,
        allow_cash=True,
        cost_bps_per_side=0.0005,
        # keep default paths:
        # dataset_path=data/gold/final_dataset.parquet
        # preds_path=artifacts/preds/panel_test_predictions.parquet
        # out_dir=artifacts/backtests
    )

    out_dir = ensure_dir(base.out_dir)
    results = []

    for n in top_n_values:
        for p in min_proba_values:
            cfg = BacktestConfig(**{**asdict(base), "top_n": int(n), "min_proba": float(p)})

            # Use unique output names per run
            tag = f"top{n}_p{int(p*100):02d}_h{cfg.hold_days}"
            cfg.equity_out = f"equity_{tag}.parquet"
            cfg.metrics_out = f"metrics_{tag}.json"

            try:
                equity_path, metrics_path = run_backtest(cfg)

                # load metrics json
                m = json.loads(Path(metrics_path).read_text())
                m["tag"] = tag
                m["equity_path"] = str(equity_path)
                m["metrics_path"] = str(metrics_path)

                results.append(m)

            except Exception as e:
                results.append(
                    {
                        "tag": tag,
                        "top_n": n,
                        "hold_days": cfg.hold_days,
                        "min_proba": p,
                        "error": repr(e),
                    }
                )

    df = pd.DataFrame(results)

    # Split successes vs failures
    ok = df[df.get("error").isna()] if "error" in df.columns else df.copy()
    bad = df[~df.get("error").isna()] if "error" in df.columns else df.iloc[0:0].copy()

    # Rank by sharpe first, then total_return
    if not ok.empty:
        ok["sharpe"] = pd.to_numeric(ok["sharpe"], errors="coerce")
        ok["total_return"] = pd.to_numeric(ok["total_return"], errors="coerce")
        ok["max_drawdown"] = pd.to_numeric(ok["max_drawdown"], errors="coerce")
        ok = ok.sort_values(["sharpe", "total_return"], ascending=False)

    # Save sweep table
    sweep_path = out_dir / "sweep_results.parquet"
    ok.to_parquet(sweep_path, index=False)

    # Also save CSV for easy viewing
    ok.to_csv(out_dir / "sweep_results.csv", index=False)

    print("\n=== SWEEP COMPLETE ===")
    print("Saved:", sweep_path)
    if not ok.empty:
        print("\nTop 10 by Sharpe:")
        cols = ["tag", "top_n", "min_proba", "total_return", "max_drawdown", "sharpe", "hit_rate", "n_days"]
        print(ok[cols].head(10).to_string(index=False))
    else:
        print("No successful runs.")

    if not bad.empty:
        print("\nFailures (first 10):")
        print(bad[["tag", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

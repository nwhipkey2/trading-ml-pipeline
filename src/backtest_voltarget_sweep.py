from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.backtest_topn_hold_voltarget import BacktestVolTargetConfig, run as run_voltarget


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    # Sweep grid
    top_n_values = [1, 3, 5, 10]
    min_proba_values = [0.45, 0.50, 0.55, 0.60, 0.65]
    target_vol_values = [0.08, 0.10, 0.12]
    max_lev_values = [2.0, 3.0, 4.0]

    base = BacktestVolTargetConfig(
        top_n=5,
        hold_days=5,
        min_proba=0.55,
        allow_cash=True,
        cost_bps_per_side=0.0005,
        target_vol_annual=0.10,
        vol_lookback_days=20,
        max_leverage=3.0,
        # uses default paths:
        # dataset_path=data/gold/final_dataset.parquet
        # preds_path=artifacts/preds/panel_test_predictions.parquet
        # out_dir=artifacts/backtests
    )

    out_dir = ensure_dir(base.out_dir)
    results = []

    for n in top_n_values:
        for p in min_proba_values:
            for tv in target_vol_values:
                for ml in max_lev_values:
                    cfg = BacktestVolTargetConfig(
                        **{
                            **asdict(base),
                            "top_n": int(n),
                            "min_proba": float(p),
                            "target_vol_annual": float(tv),
                            "max_leverage": float(ml),
                        }
                    )

                    tag = f"vt_top{n}_p{int(p*100):02d}_tv{int(tv*100):02d}_ml{int(ml*10):02d}_h{cfg.hold_days}"
                    cfg.equity_out = f"{tag}.parquet"
                    cfg.metrics_out = f"{tag}.json"

                    try:
                        equity_path, metrics_path = run_voltarget(cfg)
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
                                "min_proba": p,
                                "target_vol_annual": tv,
                                "max_leverage": ml,
                                "hold_days": cfg.hold_days,
                                "error": repr(e),
                            }
                        )

    df = pd.DataFrame(results)

    ok = df[df.get("error").isna()] if "error" in df.columns else df.copy()
    bad = df[~df.get("error").isna()] if "error" in df.columns else df.iloc[0:0].copy()

    # Rank by Sharpe, then total return, then drawdown (less negative is better)
    if not ok.empty:
        for c in ["sharpe", "total_return", "max_drawdown", "avg_gross_exposure"]:
            if c in ok.columns:
                ok[c] = pd.to_numeric(ok[c], errors="coerce")

        ok = ok.sort_values(["sharpe", "total_return", "max_drawdown"], ascending=[False, False, False])

    # Save
    out_parquet = out_dir / "voltarget_sweep_results.parquet"
    out_csv = out_dir / "voltarget_sweep_results.csv"
    ok.to_parquet(out_parquet, index=False)
    ok.to_csv(out_csv, index=False)

    print("\n=== VOLTARGET SWEEP COMPLETE ===")
    print("Saved:", out_parquet)
    print("Saved:", out_csv)

    if not ok.empty:
        cols = [
            "tag",
            "top_n",
            "min_proba",
            "target_vol_annual",
            "max_leverage",
            "total_return",
            "max_drawdown",
            "sharpe",
            "avg_gross_exposure",
            "n_days",
        ]
        cols = [c for c in cols if c in ok.columns]
        print("\nTop 15 by Sharpe:")
        print(ok[cols].head(15).to_string(index=False))
    else:
        print("No successful runs.")

    if not bad.empty:
        print("\nFailures (first 10):")
        print(bad[["tag", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

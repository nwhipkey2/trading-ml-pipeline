from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_optimizer(path: str = "artifacts/metrics/dip_entry_hold_walkforward.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_full_oos(folds: list[dict], symbol: str) -> None:
    # Collect and stitch all fold curves in chronological order
    rows = []

    for f in sorted(folds, key=lambda x: x["fold"]):
        for entry in f["per_symbol"]:
            if entry["symbol"] != symbol:
                continue

            ts = pd.to_datetime(entry["curve"]["timestamps"], utc=True)
            eq_s = pd.Series(entry["curve"]["equity_strategy"], index=ts)
            eq_b = pd.Series(entry["curve"]["equity_buy_hold"], index=ts)

            rows.append((eq_s, eq_b))

    if not rows:
        raise ValueError(f"No curves found for symbol={symbol}")

    stitched_s = []
    stitched_b = []
    last_s = 1.0
    last_b = 1.0

    for eq_s, eq_b in rows:
        eq_s2 = eq_s / eq_s.iloc[0] * last_s
        eq_b2 = eq_b / eq_b.iloc[0] * last_b

        stitched_s.append(eq_s2)
        stitched_b.append(eq_b2)

        last_s = float(eq_s2.iloc[-1])
        last_b = float(eq_b2.iloc[-1])

    full_s = pd.concat(stitched_s).sort_index()
    full_b = pd.concat(stitched_b).sort_index()

    plt.figure(figsize=(14, 8), dpi=120)
    plt.plot(full_s.index, full_s.values, label="Strategy (stitched OOS)", linewidth=2)
    plt.plot(full_b.index, full_b.values, label="Buy & Hold (stitched OOS)", linewidth=2, linestyle="--")
    plt.title(f"Full Walk-Forward OOS Equity | symbol={symbol}", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Equity", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    folds = load_optimizer()
    plot_full_oos(folds, symbol="SPY")


if __name__ == "__main__":
    main()

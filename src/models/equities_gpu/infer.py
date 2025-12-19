from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.lake.calendar import to_utc
from src.lake.io import ParquetIO


@dataclass
class InferConfig:
    panel_path: str = "data/gold/equities_daily_panel.parquet"
    model_path: str = "artifacts/models/equities_gpu/mlp.pt"
    out_signals_path: str = "artifacts/signals/equities_gpu/signals.parquet"

    top_k: int = 50
    min_score: float = 0.0
    side: str = "long"


def main(cfg: InferConfig) -> None:
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError("PyTorch not installed.") from e

    ckpt = torch.load(cfg.model_path, map_location="cpu", weights_only=False)
    feats = ckpt["features"]
    mu = ckpt["mu"]
    sd = ckpt["sd"]

    class MLP(nn.Module):
        def __init__(self, d_in: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(len(feats))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    df = ParquetIO().read(cfg.panel_path).copy()
    df["timestamp"] = to_utc(df["timestamp"])
    df["symbol"] = df["symbol"].astype(str)

    df = df.dropna(subset=feats).copy()

    X = df[feats].to_numpy(dtype=np.float32)
    X = (X - mu) / sd

    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy()
    proba = np.exp(logits[:, 1]) / (np.exp(logits[:, 0]) + np.exp(logits[:, 1]) + 1e-12)

    df["score"] = proba

    # Pick top_k per day
    signals = (
        df.sort_values(["timestamp", "score"], ascending=[True, False])
          .groupby("timestamp", group_keys=False)
          .head(cfg.top_k)
          .copy()
    )
    signals = signals[signals["score"] >= cfg.min_score].copy()

    signals["side"] = cfg.side
    signals["target_weight"] = 1.0 / cfg.top_k  # allocator can override
    signals = signals[["timestamp", "symbol", "score", "side", "target_weight"]].sort_values(["timestamp", "score"], ascending=[True, False])

    out = Path(cfg.out_signals_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(out, index=False)

    print("DONE")
    print("Signals:", out)


if __name__ == "__main__":
    main(InferConfig())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.lake.calendar import to_utc
from src.lake.io import ParquetIO


@dataclass
class TrainConfig:
    panel_path: str = "data/gold/equities_daily_panel.parquet"
    out_model_path: str = "artifacts/models/equities_gpu/mlp.pt"
    out_metrics_path: str = "artifacts/metrics/equities_gpu_train.json"

    target_col: str = "target_up_5"
    feature_cols: list[str] | None = None

    train_start: str = "2012-01-01"
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-18"

    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 4096
    hidden: int = 128
    seed: int = 7


def _default_features(df: pd.DataFrame, target_col: str) -> list[str]:
    # numeric, exclude obvious non-features
    exclude = {"timestamp", "symbol", "index_id", "sector", target_col}
    feats = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("fwd_ret_") or c.startswith("target_up_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return sorted(feats)


def main(cfg: TrainConfig) -> None:
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError("PyTorch not installed. Later we’ll enable GPU training by installing torch in Docker.") from e

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    io = ParquetIO()
    df = io.read(cfg.panel_path).copy()
    df["timestamp"] = to_utc(df["timestamp"])
    df["symbol"] = df["symbol"].astype(str)

    # date slicing
    train = df[(df["timestamp"] >= pd.Timestamp(cfg.train_start, tz="UTC")) & (df["timestamp"] <= pd.Timestamp(cfg.train_end, tz="UTC"))].copy()
    test = df[(df["timestamp"] >= pd.Timestamp(cfg.test_start, tz="UTC")) & (df["timestamp"] <= pd.Timestamp(cfg.test_end, tz="UTC"))].copy()

    if cfg.feature_cols is None:
        cfg.feature_cols = _default_features(df, cfg.target_col)

    feats = [c for c in cfg.feature_cols if c in df.columns]
    if not feats:
        raise RuntimeError("No feature columns found to train on.")

    train = train.dropna(subset=feats + [cfg.target_col]).copy()
    test = test.dropna(subset=feats + [cfg.target_col]).copy()

    Xtr = train[feats].to_numpy(dtype=np.float32)
    ytr = train[cfg.target_col].to_numpy(dtype=np.int64)
    Xte = test[feats].to_numpy(dtype=np.float32)
    yte = test[cfg.target_col].to_numpy(dtype=np.int64)

    # simple normalization
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MLP(nn.Module):
        def __init__(self, d_in: int, hidden: int):
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

    model = MLP(Xtr.shape[1], cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # train loop
    idx = np.arange(len(Xtr))
    for ep in range(cfg.epochs):
        np.random.shuffle(idx)
        model.train()
        total = 0.0
        for i in range(0, len(idx), cfg.batch_size):
            b = idx[i : i + cfg.batch_size]
            xb = torch.from_numpy(Xtr[b]).to(device)
            yb = torch.from_numpy(ytr[b]).to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(b)
        print(f"epoch {ep+1}/{cfg.epochs} loss={total/len(idx):.4f}")

    # eval
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
    proba = np.exp(logits[:, 1]) / (np.exp(logits[:, 0]) + np.exp(logits[:, 1]) + 1e-12)
    pred = (proba >= 0.5).astype(int)
    acc = float((pred == yte).mean())

    # save
    Path(cfg.out_model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mu": mu.astype(np.float32),
            "sd": sd.astype(np.float32),
            "features": feats,
        },
        cfg.out_model_path,
    )

    Path(cfg.out_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_metrics_path).write_text(json.dumps({"accuracy": acc, "n_test": int(len(yte)), "device": str(device), "features": feats}, indent=2))

    print("DONE")
    print("Model:", cfg.out_model_path)
    print("Metrics:", cfg.out_metrics_path)


if __name__ == "__main__":
    main(TrainConfig())

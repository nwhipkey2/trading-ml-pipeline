# stock-ml-pipeline

A portable data pipeline for market data -> features -> datasets -> baseline ML training.

## Quick start (Windows)
1) Create venv
2) Install deps
3) Run pipeline

## Repo layout
- data/bronze: raw immutable ingests
- data/silver: cleaned standardized tables
- data/gold: features + labels
- artifacts/: models + metrics outputs

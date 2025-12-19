<<<<<<< HEAD
## Overview
This repository implements a macro-aware, cross-asset ML pipeline for
portfolio allocation and research.

The system:
- ingests multi-asset market and macro data
- engineers time-series and cross-sectional features
- trains panel models with walk-forward validation
- evaluates strategies using realistic, volatility-targeted backtests
- is fully containerized for reproducibility and distributed execution

This is a research system, not trading advice.
=======
# Trading ML Pipeline

End-to-end quantitative research pipeline for:
- Cross-asset macro data ingestion
- Point-in-time equity universe construction (S&P 500)
- Feature engineering (price, sector, macro)
- Machine learning stock selection (GPU-ready)
- Signal generation and portfolio construction

## Architecture
- **Bronze**: raw market & macro data
- **Silver**: cleaned, normalized datasets
- **Gold**: model-ready panels
- **Artifacts**: models, metrics, backtests (not committed)

## Phase 1 (Completed)
- Bootstrap universe
- Sector mapping
- Daily equity panel
- ML model training & inference
- Signal generation

## Next Phases
- S&P 500 point-in-time universe (Finnhub)
- Intraday + tick data
- Options implied volatility
- GPU-distributed training
- Multi-sleeve portfolio allocator

## Run (Windows / PowerShell)
```powershell
python -m src.jobs.run_daily_pipeline
python -m src.models.equities_gpu.train
python -m src.models.equities_gpu.infer

>>>>>>> f9247c4 (Phase 1: equity universe, panel builder, ML stock selection pipeline)

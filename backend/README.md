# Backend (FastAPI)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY=your_key
export MODE=stub  # or prod once real model artifacts are wired
uvicorn app.main:app --reload --port 8000
```

## Endpoints
- `GET /health`
- `POST /pipeline/run-daily`
- `POST /pipeline/run-hourly`
- `GET /signals/latest?limit=...&tier=basic|pro|elite`
- `GET /signals/{signal_id}`
- `POST /elite/auto/enable`
- `POST /elite/trade/execute`
- `POST /kill`
- `GET /metrics`

## Notes
- Daily pipeline runs heavy models once/day; hourly refinement only confirms entries for existing signals.
- SMC features (BOS/CHoCH/MSS/liquidity sweeps/order blocks/FVG/premium-discount) feed StockFormer/TFT/LightGBM-veto inputs.
- Data cached to Parquet + DuckDB. pk-stock-screener used only for candidate selection.
- Model registry lives in `app/services/model_registry.py`. `MODE=stub` uses deterministic heuristics; `MODE=prod` requires artifact paths (`kronos_artifact_path`, `stockformer_artifact_path`, `tft_artifact_path`, `lightgbm_artifact_path`) and will refuse to run without them (no heuristics in prod).

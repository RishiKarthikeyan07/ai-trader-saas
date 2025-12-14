# AI Swing Trading SaaS (NSE) – Cutting Edge SOTA Model Trading BOT

Production-grade scaffold for a hedge-fund-style AI swing trading platform targeting NSE. Daily heavy ML (Kronos/StockFormer/TFT/CatBoost), Smart Money Concepts validation, hourly 1H/4H entry refinement only, Elite BUY-only auto execution with PPO-based exits (hooks ready).

## Repo layout
- `backend/` FastAPI service (pipelines, feature/SMC engine, DuckDB/Parquet cache, tier-gated endpoints)
- `frontend/` Next.js 13 app router UI (dashboard, signal detail, admin triggers, elite automation controls)
- `notebooks/` Colab-ready training (dataset + Kronos embeddings, StockFormer, TFT + CatBoost veto)
- `cache/` + `data/` created at runtime for Parquet/DuckDB

## Backend quickstart
```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export MODE=stub  # switch to prod only when real model artifacts are configured
uvicorn app.main:app --reload --port 8000
```
Key env vars (no secrets committed): `ALPHAVANTAGE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `REDIS_URL`, `DUCKDB_PATH`, `DATA_CACHE_DIR`, `APP_ENV`.

### Pipelines
- **Daily** (`POST /pipeline/run-daily`):
  1) pk-stock-screener filters NSE universe (fallback list)
  2) Multi-timeframe builder (M/W/D/4H/1H) with Parquet/DuckDB cache
  3) TA + SMC feature engine (BOS/CHoCH/MSS/liquidity sweeps/order blocks/FVG/premium-discount)
  4) Kronos/StockFormer/TFT/CatBoost fusion -> BUY/SELL/HOLD signal with entry/SL/TP/SMC flags
  5) Persist to DuckDB table `signals`
- **Hourly** (`POST /pipeline/run-hourly`): only refines existing daily BUY/SELL into `READY_TO_ENTER` vs `WAIT` using 1H/4H alignment + SMC micro confirms. Never flips direction.

### Endpoints
- `GET /health`
- `POST /pipeline/run-daily`
- `POST /pipeline/run-hourly`
- `GET /signals/latest?limit=&tier=basic|pro|elite`
- `GET /signals/{id}`
- `POST /elite/auto/enable`
- `POST /elite/trade/execute` (BUY-only, mock adapter)
- `POST /kill`, `GET /kill`
- `GET /metrics` (Prometheus text)

### Tests
```bash
cd backend
python3.12 -m pytest -q
```

## Frontend quickstart
```bash
cd frontend
npm install
npm run dev
```
Set `NEXT_PUBLIC_API_BASE` to FastAPI URL and Supabase anon creds if using hosted auth. Pages: dashboard (`/`), signal detail (`/signals/[id]`), admin (`/admin`), elite auto (`/elite`).

## Training (Colab)
- `notebooks/01_build_dataset_and_kronos.ipynb`: yfinance dataset + TF-align/SMC/TA + 512d Kronos embeddings -> `training_data/v1/dataset.parquet`
- `notebooks/02_train_stockformer.ipynb`: trains StockFormer, saves to `artifacts/v1/stockformer/`
- `notebooks/03_train_tft_and_veto.ipynb`: trains TFT + CatBoost veto, saves to `artifacts/v1/tft/` and `artifacts/v1/veto/`

## Notes
- Heavy AI models run **daily only**; hourly layer is entry refinement only.
- pk-stock-screener is strictly for universe reduction; SMC features are validation/feature inputs, not execution logic.
- Auto-trading is BUY-only for Elite tier; PPO exits act on open positions only.
- Data cached locally to Parquet + DuckDB; bhavcopy ingestion supported via CSV upload hook.
- Model registry (`backend/app/services/model_registry.py`) enforces mode guardrails: `MODE=stub` uses deterministic stubs, `MODE=prod` requires artifact paths for Kronos/StockFormer/TFT/CatBoost and will fail if missing (no heuristics in prod).
- Kronos download helper: `PYTHONPATH=.. python backend/scripts/download_kronos.py --dest artifacts/kronos` (uses Hugging Face repo `NeoQuasar/Kronos-base`; set `HF_TOKEN` if gated). Point `kronos_artifact_path` to the download.
- Signals are generated centrally and tier-gated at fetch; no per-user inference.

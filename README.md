# AI Swing Trading SaaS (NSE) — Production-Grade, SOTA Stack

Institutional-style swing trading platform for NSE with daily heavy ML (Kronos/StockFormer/TFT/CatBoost), Smart Money Concepts validation, hourly 1H/4H entry refinement (no intraday scalping), and Elite BUY-only auto execution with PPO exits (hooked). Built to scale centrally to 5,000+ users; no per-user inference.

## Architecture
- **Model stack (prod):** Kronos 512d embeddings → StockFormer (direction + returns) → TFT (vol/bands) → CatBoost veto. FinRL PPO reserved for exits only.
- **Pipelines:** Daily heavy inference; hourly light entry refinement. PKScreener only for universe reduction; SMC for feature/validation.
- **Data:** AlphaVantage primary, yfinance fallback, bhavcopy upload. Cached to Parquet + DuckDB; aggregates to Postgres (Supabase).
- **Backend:** FastAPI (`backend/`), model registry with MODE=stub|prod guardrails.
- **Frontend:** Next.js 13 app router (`frontend/`), tier-gated dashboards and admin/elite panels.

## Repo layout
- `backend/` FastAPI app, pipelines, feature/SMC engine, model registry/loaders, DuckDB/Parquet cache
- `frontend/` Next.js UI (dashboard, signal detail, admin triggers, elite automation controls)
- `notebooks/` Colab training kit (dataset + Kronos embeddings, StockFormer, TFT + CatBoost)
- `artifacts/` Model artifacts (expected layout: `artifacts/v1/...`)
- `training_data/` Local cache for training exports (ignored in git)
- `cache/`, `data/` Runtime stores (ignored)

## Backend quickstart
```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export MODE=stub  # switch to prod after artifacts are in place
uvicorn app.main:app --reload --port 8000
```
Key env vars (no secrets committed): `ALPHAVANTAGE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `REDIS_URL`, `DUCKDB_PATH`, `DATA_CACHE_DIR`, `APP_ENV`, `HF_TOKEN` (if needed for Kronos).

## Pipelines (FastAPI endpoints)
- **Daily** `POST /pipeline/run-daily`
  1) PKScreener filters NSE universe
  2) Multi-timeframe builder (M/W/D/4H/1H) with Parquet/DuckDB cache
  3) TA + SMC features (BOS/CHoCH/MSS/liquidity sweeps/order blocks/FVG/premium-discount)
  4) Model fusion (Kronos → StockFormer → TFT → CatBoost) → BUY/SELL/HOLD with entry/SL/TP/SMC flags
  5) Persist to DuckDB `signals`
- **Hourly** `POST /pipeline/run-hourly`
  - Only refines existing daily signals to `READY_TO_ENTER` vs `WAIT` using 1H/4H alignment + SMC micro confirms. Never flips direction.

### API surface
- `GET /health`
- `POST /pipeline/run-daily`
- `POST /pipeline/run-hourly`
- `GET /signals/latest?limit=&tier=basic|pro|elite`
- `GET /signals/{id}`
- `POST /elite/auto/enable`
- `POST /elite/trade/execute` (BUY-only, mock adapter)
- `POST /kill`, `GET /kill`
- `GET /metrics` (Prometheus text)

## Training (Colab, GPU T4)
- `notebooks/01_build_dataset_and_kronos.ipynb`: yfinance daily data → normalize + TF-align + SMC + TA + 512d Kronos embeddings → `training_data/v1/dataset.parquet`
- `notebooks/02_train_stockformer.ipynb`: trains StockFormer (context_dim=29, Kronos=512) → `artifacts/v1/stockformer/weights.pt` + `config.json`
- `notebooks/03_train_tft_and_veto.ipynb`: trains TFT (returns/vol/bands) and CatBoost veto → `artifacts/v1/tft/`, `artifacts/v1/veto/`
Set `REPO_URL=https://github.com/RishiKarthikeyan07/ai-trader-saas` in Colab before running.

### Artifact layout (v1)
```
artifacts/v1/
  kronos/              # downloaded HF snapshot (NeoQuasar/Kronos-base)
  stockformer/
    weights.pt
    config.json
  tft/
    weights.pt
    config.json
  veto/
    catboost.cbm
    config.json
  manifest.json        # model versions, feature schema, normalization hash, data range
```
Prod mode requires all artifact paths; otherwise it fails fast (no heuristics).

## Frontend quickstart
```bash
cd frontend
npm install
npm run dev
```
Set `NEXT_PUBLIC_API_BASE` to FastAPI URL and Supabase anon creds if using hosted auth. Pages: dashboard (`/`), signal detail (`/signals/[id]`), admin (`/admin`), elite auto (`/elite`).

## Testing
```bash
cd backend
python -m pytest -q
```

## Guardrails (locked principles)
- PKScreener only for universe reduction.
- SMC as feature/validation; not execution logic.
- Heavy models run daily only; hourly refines entries only.
- PPO manages exits only; never selects entries/assets.
- Auto execution BUY-only, Elite tier only.

## Data + normalization (parity)
- Normalization locked: prices ÷ last close; volume log1p then z-score over 120 bars.
- Shared preprocessors: `backend/app/ml/preprocess/normalize.py` (ohlcv, TF-align, SMC vec, TA vec, veto vec).
- Use the same module in training (Colab) and inference (backend) to avoid schema drift.

## Model IO (v1)
- **ModelInput:** `ohlcv_120` (120×5 float32 normalized), `kronos_emb` (512 via loader), `context` (29 = TF-align 5 + SMC 12 + TA 12)
- **StockFormer:** returns `[3,5,10]`, up_probs `[3,5,10]`
- **TFT:** returns `[3,5,10]`, `vol_10d`, `upper_10d`, `lower_10d`
- **CatBoost veto:** features from SF/TFT + SMC/TF/TA summaries; thresholds block=0.65, boost=0.35

## Ops checklist for prod
- Download Kronos: `PYTHONPATH=.. python backend/scripts/download_kronos.py --dest artifacts/v1/kronos`
- Place trained artifacts under `artifacts/v1/` (or point env paths).
- Set `MODE=prod` and corresponding `*_artifact_path` env vars.
- Run `POST /pipeline/run-daily`; verify `signals` table populated; then enable hourly.
- For Elite auto: ensure kill switch and position limits configured; PPO exits to be added after signal stability.

# Project Overview – AI Swing Trading SaaS (NSE)

This document walks through the entire system A→Z: problem framing, data, feature engine, models, pipelines, SaaS architecture, ops, and guardrails.

## 1) Mission & Constraints
- Generate 2–10 day swing signals on NSE equities.
- Heavy models run **daily**; hourly logic only refines entries for existing daily signals.
- PKScreener is **universe reduction only**.
- SMC is a **feature/validation layer**, not execution logic.
- Auto-execution: **BUY-only** and **Elite-only**. SELL is informational/manual.
- FinRL PPO is **exits-only**; never selects assets or entries.

## 2) Data Sources & Storage
- Primary: AlphaVantage OHLCV (API key).
- Fallback: yfinance; bhavcopy CSV upload supported.
- Caching: Parquet + DuckDB for speed and reproducibility.
- Aggregates/serving: Postgres (Supabase) with row-level security; Redis for cache.

## 3) Timeframes (mandatory)
- Monthly, Weekly, Daily, 4H, 1H.
- Daily drives heavy inference; 1H/4H refine entries only.

## 4) Feature Engine
- TA: RSI, EMA fast/slow, ATR, volatility, OBV, supertrend, EMA gaps, returns, volume log/z.
- SMC: BOS, CHoCH, MSS, liquidity sweeps (buy/sell), fair value gaps (up/down), order block distances (bull/bear), premium/discount, swing highs/lows; combined `smc_score`.
- TF alignment vector: monthly_bias, weekly_bias, daily_bias, h4_align, h1_align.
- Normalization (locked): prices ÷ last close; volume log1p then z-score over 120 bars.
- Shared preprocessing module: `backend/app/ml/preprocess/normalize.py` is used in both training and inference to guarantee parity.

## 5) Model Stack (V1)
1) **Kronos** (pretrained candle encoder) → 512-d embeddings; batched; cached per (symbol, asof).
2) **StockFormer** (core alpha): Transformer encoder; outputs returns [3,5,10] + up_probs [3,5,10]; inputs: normalized OHLCV (120×5), Kronos emb (512), context vec (29).
3) **TFT** (risk/range): returns [3,5,10], vol_10d, upper_10d, lower_10d; same inputs as StockFormer.
4) **CatBoost veto**: features from StockFormer/TFT + SMC/TF/TA summaries; thresholds: block >0.65, boost <0.35.
5) **FinRL PPO**: exits-only (HOLD, partial exits, trail); integrated later once signals are stable.

## 6) Model IO Contracts (locked)
- ModelInput: `ohlcv_120` (120×5 float32 normalized), `tf_align` (5), `smc_vec` (12), `ta_vec` (12), context = concat (29), raw_features for explanations.
- StockFormer: returns `[3,5,10]`, up_probs `[3,5,10]`.
- TFT: returns `[3,5,10]`, vol_10d, upper_10d, lower_10d.
- CatBoost veto: veto probability using fused vector from build_veto_vec.

## 7) Artifact Layout & Versioning
```
artifacts/v1/
  kronos/              # HF snapshot NeoQuasar/Kronos-base
  stockformer/         # weights.pt, config.json
  tft/                 # weights.pt, config.json
  veto/                # catboost.cbm, config.json
  manifest.json        # model versions, feature schema, normalization hash, data range
```
Prod mode requires valid paths for all artifacts; otherwise it fails fast.

## 8) Pipelines
- **Daily** (heavy): PKScreener → multi-timeframe cache → TA+SMC → build ModelInput → Kronos → StockFormer → TFT → CatBoost veto → fusion → BUY/SELL/HOLD with entry/SL/TP/SMC flags → store signals.
- **Hourly** (light): reads existing daily signals; uses 1H/4H alignment + SMC micro-confirmations to output READY_TO_ENTER vs WAIT; never flips BUY↔SELL.

## 9) Backend (FastAPI)
- Key endpoints: `/health`, `/pipeline/run-daily`, `/pipeline/run-hourly`, `/signals/latest`, `/signals/{id}`, `/elite/auto/enable`, `/elite/trade/execute` (BUY-only), `/kill`, `/metrics`.
- Model registry enforces MODE guardrails: `MODE=stub` uses deterministic stubs; `MODE=prod` requires artifacts (no heuristics in prod).
- Data persistence: DuckDB for signals cache; Postgres/Supabase for multi-tenant serving; Redis for cache; audit logs planned.

## 10) Frontend (Next.js 13)
- Pages: dashboard (tier-gated signals), signal detail (TF alignment, SMC explanation, levels), admin panel (run pipeline/health/model versions), elite panel (auto enable, positions, PnL, kill switch).
- Env: `NEXT_PUBLIC_API_BASE` for backend; Supabase anon for auth if hosted.

## 11) Training (Colab T4 workflow)
- `notebooks/01_build_dataset_and_kronos.ipynb`: yfinance → normalize → TF/SMC/TA → Kronos embeddings → `training_data/v1/dataset.parquet`.
- `notebooks/02_train_stockformer.ipynb`: trains StockFormer with context_dim=29, Kronos 512 → saves to `artifacts/v1/stockformer/`.
- `notebooks/03_train_tft_and_veto.ipynb`: trains TFT + CatBoost veto → saves to `artifacts/v1/tft/` and `artifacts/v1/veto/`.
- Set `REPO_URL=https://github.com/RishiKarthikeyan07/ai-trader-saas` before running in Colab.

## 12) Guardrails & Risk Controls
- Max open positions, daily loss limits, position sizing caps (to be configured).
- Slippage + fees model in execution adapter.
- Kill switch endpoint `/kill` with admin/elite scope.
- Prod refuses to run without artifacts; no heuristics leak into prod.
- Signals generated centrally, not per-user; tier gating at fetch.

## 13) Ops Runbook (prod)
1) Download Kronos: `PYTHONPATH=.. python backend/scripts/download_kronos.py --dest artifacts/v1/kronos` (set HF_TOKEN if needed).
2) Train in Colab with the provided notebooks; copy `artifacts/v1` to prod (or S3) and set env paths.
3) Set `MODE=prod` and run `POST /pipeline/run-daily`; verify signals persisted.
4) Enable hourly refinement; monitor `/metrics`.
5) For Elite auto, ensure kill switch, sizing limits, and PPO (exits) are configured before going live.

## 14) Current Status
- Backend/front-end scaffold complete; feature/SMC engine, model registry, and stub/prod guardrails implemented.
- Colab notebooks authored for dataset + training; ready to produce v1 artifacts.
- README covers quickstarts; this document captures full A→Z architecture and ops.

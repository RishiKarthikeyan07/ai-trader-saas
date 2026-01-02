# 📋 Migration Checklist

## ✅ Completed

### Phase 1: Implementation
- [x] Created `UnifiedFeaturePipeline` class
- [x] Created `DeepModelInference` class
- [x] Created `ai_ranker_unified.py`
- [x] Created new training notebook `01_build_dataset_UNIFIED.ipynb`
- [x] Created comprehensive documentation

### Phase 2: Cleanup
- [x] Archived old `features.py`
- [x] Archived old `ai_ranker.py`
- [x] Archived old notebooks
- [x] Removed nested clone junk
- [x] Added deprecation notices
- [x] Verified all files

---

## ⏳ TODO (Your Action Items)

### Phase 3: Model Retraining ⚠️ REQUIRED

- [ ] **1. Build New Dataset**
  ```bash
  cd /Users/rishi/Downloads/AI_TRADER
  jupyter notebook notebooks/01_build_dataset_UNIFIED.ipynb
  # Run all cells to generate: training_data/v1/dataset.parquet
  ```
  **Expected output:** 50K-100K samples with unified features

- [ ] **2. Train StockFormer**
  ```bash
  jupyter notebook notebooks/02_train_stockformer.ipynb
  # Update to load dataset.parquet from step 1
  # Save model to: apps/api/app/ml/models/stockformer.pt
  ```

- [ ] **3. Train TFT**
  ```bash
  jupyter notebook notebooks/03_train_tft.ipynb
  # Update to load dataset.parquet from step 1
  # Save model to: apps/api/app/ml/models/tft.pt
  ```

- [ ] **4. Train Veto Classifier**
  ```bash
  jupyter notebook notebooks/04_train_lightgbm_veto.ipynb
  # Needs predictions from StockFormer + TFT
  # Save model to: apps/api/app/ml/models/veto.txt
  ```

### Phase 4: API Integration

- [ ] **5. Update API Routes**

  Find your signal ranking endpoint (e.g., `apps/api/app/routes/signals.py`) and update:

  **Before:**
  ```python
  from app.services.ai_ranker import rank_signals_stub

  @router.post("/rank")
  async def rank_signals_endpoint(candidates: List[Dict]):
      signals = await rank_signals_stub(candidates)
      return signals
  ```

  **After:**
  ```python
  from app.services.ai_ranker_unified import rank_signals

  @router.post("/rank")
  async def rank_signals_endpoint(candidates: List[Dict]):
      signals = await rank_signals(candidates, top_k=20, min_confidence=0.6)
      return signals
  ```

- [ ] **6. Set Environment Variables**

  Add to `.env` file:
  ```bash
  # Model paths
  STOCKFORMER_PATH=apps/api/app/ml/models/stockformer.pt
  TFT_PATH=apps/api/app/ml/models/tft.pt
  VETO_PATH=apps/api/app/ml/models/veto.txt

  # API keys (optional)
  ALPHAVANTAGE_API_KEY=your_key_here

  # Data config
  DATA_START=2015-01-01
  TICKER_FILE=config/nifty100_yfinance.txt
  ```

- [ ] **7. Update Model Loading on Startup**

  In `apps/api/app/main.py` add:
  ```python
  from app.services.ai_ranker_unified import load_models

  @app.on_event("startup")
  async def startup_event():
      await load_models()  # Load models once at startup
      logger.info("Models loaded successfully")
  ```

### Phase 5: Testing

- [ ] **8. Test Feature Pipeline**
  ```bash
  cd apps/api
  python -m app.ml.unified_features
  # Expected: "✓ All tests passed!"
  ```

- [ ] **9. Test Deep Inference**
  ```bash
  python -m app.ml.deep_inference
  # Expected: "✓ DeepModelInference initialized"
  ```

- [ ] **10. Test API Endpoint**
  ```bash
  # Start server
  uvicorn app.main:app --reload

  # Test with curl
  curl -X POST http://localhost:8000/api/v1/signals/rank \
    -H "Content-Type: application/json" \
    -d '{
      "candidates": [
        {"symbol": "RELIANCE.NS", "signal_type": "momentum_breakout"}
      ]
    }'
  ```
  **Expected:** JSON response with ranked signals

- [ ] **11. Test with Real PKScreener Data**
  ```bash
  # Run PKScreener to get candidates
  # Pass candidates to your API
  # Verify signals are returned with:
  #   - Scores (1-10)
  #   - Confidence (0-100%)
  #   - Entry/SL/TP levels
  #   - Setup tags
  ```

### Phase 6: Deployment

- [ ] **12. Update Requirements**
  ```bash
  # Make sure requirements.txt has:
  pip freeze | grep -E "(torch|transformers|scikit-learn|lightgbm)" >> requirements.txt
  ```

- [ ] **13. Database Migration** (if needed)
  ```sql
  -- Update model_versions table
  INSERT INTO model_versions (
    model_type,
    version,
    file_path,
    status
  ) VALUES
    ('stockformer', 'v1_unified', 'models/stockformer.pt', 'active'),
    ('tft', 'v1_unified', 'models/tft.pt', 'active'),
    ('veto', 'v1_unified', 'models/veto.txt', 'active');
  ```

- [ ] **14. Deploy to Production**
  ```bash
  # Build Docker image
  docker build -t ai-trader-api .

  # Run container
  docker run -p 8000:8000 \
    -e STOCKFORMER_PATH=/app/models/stockformer.pt \
    -e TFT_PATH=/app/models/tft.pt \
    -e VETO_PATH=/app/models/veto.txt \
    ai-trader-api
  ```

- [ ] **15. Monitor Performance**
  - Track inference time (should be <500ms per symbol)
  - Monitor memory usage (Kronos uses ~2GB on GPU)
  - Log prediction confidence scores
  - Track veto rejection rate

### Phase 7: Validation

- [ ] **16. Verify Feature Consistency**
  ```python
  # Run this test to verify training/inference match
  from app.ml.unified_features import UnifiedFeaturePipeline
  from app.ml.deep_inference import DeepModelInference
  import pandas as pd
  import numpy as np

  # Load sample data
  df = pd.read_csv('sample_data.csv')

  # Compute with training pipeline
  train_pipeline = UnifiedFeaturePipeline(device='cpu')
  train_features = train_pipeline.compute_features(df)

  # Compute with inference pipeline
  inference = DeepModelInference(device='cpu')
  infer_features = inference.feature_pipeline.compute_features(df)

  # Verify identical
  assert np.allclose(train_features['ohlcv_norm'], infer_features['ohlcv_norm'])
  assert np.allclose(train_features['kronos_emb'], infer_features['kronos_emb'])
  assert np.allclose(train_features['context_vec'], infer_features['context_vec'])

  print("✅ Features are identical between training and inference!")
  ```

- [ ] **17. Compare Old vs New Predictions**
  - Run same stocks through old and new system
  - Verify new system produces valid predictions
  - Compare prediction quality (optional)

- [ ] **18. Production Smoke Test**
  - Get real PKScreener candidates
  - Run through full pipeline
  - Verify signals are actionable
  - Check all fields populated correctly

---

## 📊 Progress Tracker

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Implementation | ✅ Complete | 100% |
| Phase 2: Cleanup | ✅ Complete | 100% |
| Phase 3: Model Retraining | ⏳ Pending | 0% |
| Phase 4: API Integration | ⏳ Pending | 0% |
| Phase 5: Testing | ⏳ Pending | 0% |
| Phase 6: Deployment | ⏳ Pending | 0% |
| Phase 7: Validation | ⏳ Pending | 0% |

---

## 🎯 Critical Path

**Must complete in order:**

1. Phase 3 (Model Retraining) - **BLOCKS everything**
2. Phase 4 (API Integration) - **BLOCKS deployment**
3. Phase 5 (Testing) - **BLOCKS deployment**
4. Phase 6 (Deployment)
5. Phase 7 (Validation)

---

## ⏱️ Estimated Time

| Phase | Time | Notes |
|-------|------|-------|
| Phase 3: Model Retraining | 4-8 hours | Depends on GPU availability |
| Phase 4: API Integration | 1 hour | Straightforward |
| Phase 5: Testing | 2 hours | Including fixes |
| Phase 6: Deployment | 1 hour | If Docker setup ready |
| Phase 7: Validation | 1 hour | Quick verification |
| **Total** | **9-13 hours** | Can be split across days |

---

## 🆘 Troubleshooting

### Issue: "Models not loading"
- Check `STOCKFORMER_PATH`, `TFT_PATH`, `VETO_PATH` are set
- Verify model files exist
- Check file permissions

### Issue: "Feature shape mismatch"
- Make sure you retrained models with new dataset
- Verify using `01_build_dataset_UNIFIED.ipynb`

### Issue: "Slow inference"
- Enable GPU if available
- Consider disabling Kronos: `enable_kronos=False`
- Cache Kronos embeddings for repeated symbols

### Issue: "PKScreener integration not working"
- Check `fetch_symbol_data()` in `ai_ranker_unified.py`
- Verify market data service is working
- Test with manual symbol first

---

## 📞 Need Help?

- **Documentation:** `docs/UNIFIED_FEATURE_PIPELINE.md`
- **Code Examples:** See `__main__` blocks in each module
- **Troubleshooting:** `docs/UNIFIED_FEATURE_PIPELINE.md` (section 🐛)

---

## ✅ Final Checklist

Before going to production:

- [ ] All models retrained with unified features
- [ ] API endpoints updated
- [ ] Environment variables set
- [ ] All tests passing
- [ ] Smoke test with real data successful
- [ ] Feature consistency verified
- [ ] Performance acceptable (<500ms/symbol)
- [ ] Documentation reviewed
- [ ] Team trained on new system

---

**Good luck with the migration! You're 50% done already.** 🚀

The hard part (implementation) is complete. Now just train models and deploy!

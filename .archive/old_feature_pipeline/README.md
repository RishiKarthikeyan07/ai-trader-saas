# Old Feature Pipeline - Archived

This directory contains deprecated files from the old feature pipeline implementation.

## 📅 Archived On
January 2, 2026

## 🚫 Why Archived

These files were replaced by the **Unified Feature Pipeline** to fix critical issues:

1. **Feature Mismatch** - Training and inference used different features
2. **Incomplete Features** - Missing SMC, MTF, Kronos embeddings
3. **No Normalization** - Inference didn't normalize inputs
4. **Duplicate Code** - Multiple versions of same logic

## 📦 Archived Files

### ML Modules
- `ml/features.py` - Old incomplete feature engineering (only 16 features, no SMC/MTF/Kronos)

### Services
- `services/ai_ranker.py` - Old stub AI ranker (mock scores, no real models)

### Notebooks
- `notebooks/01_build_dataset_and_kronos.ipynb` - Original broken notebook
- `notebooks/01_build_dataset_and_kronos_FIXED.ipynb` - Fixed but still uses old approach

## ✅ Replaced By

| Old File | New File | Status |
|----------|----------|--------|
| `ml/features.py` | `ml/unified_features.py` | ✅ Active |
| `services/ai_ranker.py` | `services/ai_ranker_unified.py` | ✅ Active |
| `notebooks/01_build_dataset_and_kronos.ipynb` | `notebooks/01_build_dataset_UNIFIED.ipynb` | ✅ Active |

## 📚 Documentation

See the new unified pipeline documentation:
- `docs/UNIFIED_FEATURE_PIPELINE.md` - Complete guide
- `docs/IMPLEMENTATION_SUMMARY.md` - Quick reference
- `docs/CLEANUP_PLAN.md` - Cleanup details

## 🔄 Restore Instructions

If you need to restore these files (not recommended):

```bash
# From repo root
cp .archive/old_feature_pipeline/ml/features.py apps/api/app/ml/
cp .archive/old_feature_pipeline/services/ai_ranker.py apps/api/app/services/
cp .archive/old_feature_pipeline/notebooks/*.ipynb notebooks/
```

## ⚠️ Warning

These files should NOT be used in production. They have known issues:

- ❌ Feature mismatch between training/inference
- ❌ Incomplete feature sets
- ❌ No proper normalization
- ❌ Stub implementations instead of real models

Use the unified pipeline instead!

---

**For questions, see:** `docs/UNIFIED_FEATURE_PIPELINE.md`

# Cleanup Plan - Old Feature Pipeline Files

## 📋 Files to Archive/Remove

This document lists deprecated files from the old feature pipeline implementation.
These have been replaced by the **Unified Feature Pipeline**.

---

## 🗑️ Files to Archive

### 1. **Old Feature Engineering** (DEPRECATED)

**File:** `apps/api/app/ml/features.py`
- **Status:** DEPRECATED - Use `unified_features.py` instead
- **Reason:** Incomplete features, missing SMC, MTF, Kronos
- **Action:** Move to `.archive/old_feature_pipeline/`

### 2. **Old Inference Module** (PARTIALLY DEPRECATED)

**File:** `apps/api/app/ml/inference.py`
- **Status:** Keep for sklearn/LightGBM models, but not for deep learning
- **Reason:** Doesn't use unified pipeline
- **Action:** Keep but document that `deep_inference.py` should be used for StockFormer/TFT

### 3. **Old AI Ranker** (DEPRECATED)

**File:** `apps/api/app/services/ai_ranker.py`
- **Status:** DEPRECATED - Use `ai_ranker_unified.py` instead
- **Reason:** Uses stub implementation, no real models
- **Action:** Move to `.archive/old_feature_pipeline/`

### 4. **Old Notebooks** (DEPRECATED)

**Files:**
- `notebooks/01_build_dataset_and_kronos.ipynb` (original broken version)
- `notebooks/01_build_dataset_and_kronos_FIXED.ipynb` (fixed but old approach)

**Status:** DEPRECATED - Use `01_build_dataset_UNIFIED.ipynb` instead
- **Reason:** Don't use unified pipeline
- **Action:** Move to `.archive/old_feature_pipeline/`

### 5. **Cloned Repo in Notebooks** (CLEANUP)

**Directory:** `notebooks/AI_TRADER/`
- **Status:** JUNK - Created by accident during notebook execution
- **Reason:** Nested clone of the repo
- **Action:** DELETE completely (safe to remove)

---

## ✅ Files to KEEP

### Core Pipeline (NEW)
- ✅ `apps/api/app/ml/unified_features.py` - **MAIN PIPELINE**
- ✅ `apps/api/app/ml/deep_inference.py` - **Deep model inference**
- ✅ `apps/api/app/services/ai_ranker_unified.py` - **Production AI ranker**

### Supporting Modules (KEEP)
- ✅ `apps/api/app/services/feature_engine.py` - TA/SMC computation (used by unified pipeline)
- ✅ `apps/api/app/services/kronos_loader.py` - Kronos embeddings
- ✅ `apps/api/app/ml/preprocess/normalize.py` - Normalization functions

### Notebooks (KEEP)
- ✅ `notebooks/01_build_dataset_UNIFIED.ipynb` - **NEW training notebook**
- ✅ `notebooks/02_train_stockformer.ipynb` - StockFormer training
- ✅ `notebooks/03_train_tft.ipynb` - TFT training
- ✅ `notebooks/04_train_lightgbm_veto.ipynb` - Veto model training

### Documentation (KEEP)
- ✅ `docs/UNIFIED_FEATURE_PIPELINE.md` - Complete guide
- ✅ `docs/IMPLEMENTATION_SUMMARY.md` - Quick reference
- ✅ `docs/CLEANUP_PLAN.md` - This file

### Legacy Inference (KEEP for now)
- ⚠️ `apps/api/app/ml/inference.py` - Keep for sklearn/LightGBM models

---

## 🔧 Cleanup Commands

### Step 1: Archive Deprecated Files

```bash
# Create archive directory
mkdir -p .archive/old_feature_pipeline/{ml,services,notebooks}

# Archive old ML files
mv apps/api/app/ml/features.py .archive/old_feature_pipeline/ml/
# Keep inference.py but document it's for sklearn models only

# Archive old services
mv apps/api/app/services/ai_ranker.py .archive/old_feature_pipeline/services/

# Archive old notebooks
mv notebooks/01_build_dataset_and_kronos.ipynb .archive/old_feature_pipeline/notebooks/
mv notebooks/01_build_dataset_and_kronos_FIXED.ipynb .archive/old_feature_pipeline/notebooks/
```

### Step 2: Remove Junk

```bash
# Remove accidental nested clone
rm -rf notebooks/AI_TRADER/
```

### Step 3: Add Deprecation Notices

For files we keep but want to mark as deprecated:

**File:** `apps/api/app/ml/inference.py`
Add at top:
```python
"""
DEPRECATED for deep learning models. Use deep_inference.py instead.
This module is only kept for sklearn/LightGBM model compatibility.
"""
```

---

## 📝 Updated Import Statements

### Old (DEPRECATED):

```python
# ❌ DON'T USE
from app.ml.features import FeatureEngine
from app.services.ai_ranker import rank_signals_stub
```

### New (RECOMMENDED):

```python
# ✅ USE THIS
from app.ml.unified_features import UnifiedFeaturePipeline
from app.ml.deep_inference import DeepModelInference
from app.services.ai_ranker_unified import rank_signals
```

---

## 🎯 File Structure After Cleanup

```
AI_TRADER/
├── .archive/
│   └── old_feature_pipeline/          ← Archived old files
│       ├── ml/
│       │   └── features.py
│       ├── services/
│       │   └── ai_ranker.py
│       └── notebooks/
│           ├── 01_build_dataset_and_kronos.ipynb
│           └── 01_build_dataset_and_kronos_FIXED.ipynb
│
├── apps/api/app/
│   ├── ml/
│   │   ├── unified_features.py        ← MAIN PIPELINE ✅
│   │   ├── deep_inference.py          ← Deep models ✅
│   │   ├── inference.py               ← Keep for sklearn ⚠️
│   │   ├── preprocess/
│   │   │   └── normalize.py           ← Keep ✅
│   │   ├── stockformer/
│   │   │   └── model.py               ← Keep ✅
│   │   └── tft/
│   │       └── model.py               ← Keep ✅
│   │
│   └── services/
│       ├── feature_engine.py          ← Keep (used by unified) ✅
│       ├── kronos_loader.py           ← Keep ✅
│       └── ai_ranker_unified.py       ← Production ranker ✅
│
├── notebooks/
│   ├── 01_build_dataset_UNIFIED.ipynb ← NEW training ✅
│   ├── 02_train_stockformer.ipynb     ← Keep ✅
│   ├── 03_train_tft.ipynb             ← Keep ✅
│   └── 04_train_lightgbm_veto.ipynb   ← Keep ✅
│
└── docs/
    ├── UNIFIED_FEATURE_PIPELINE.md    ← Keep ✅
    ├── IMPLEMENTATION_SUMMARY.md      ← Keep ✅
    └── CLEANUP_PLAN.md                ← This file ✅
```

---

## ✅ Verification Checklist

After cleanup, verify:

- [ ] Old `features.py` archived
- [ ] Old `ai_ranker.py` archived
- [ ] Old notebooks archived
- [ ] Nested `notebooks/AI_TRADER/` deleted
- [ ] New unified pipeline files present
- [ ] Documentation updated
- [ ] No broken imports in active code

---

## 🚨 Important Notes

1. **Don't delete `inference.py`** - Still needed for sklearn/LightGBM models
2. **Don't delete `feature_engine.py`** - Used by unified pipeline
3. **Keep training notebooks 02-04** - Still valid for model training
4. **Archive, don't delete** - Old files moved to `.archive/` for reference

---

## 📞 Questions?

If you need to restore archived files:
```bash
# Restore from archive
cp .archive/old_feature_pipeline/ml/features.py apps/api/app/ml/
```

But you shouldn't need to - the unified pipeline is complete!

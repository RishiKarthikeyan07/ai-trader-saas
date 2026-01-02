# Unified Feature Pipeline Documentation

## 🎯 Overview

The **Unified Feature Pipeline** ensures that training and real-time inference use **IDENTICAL** features. This eliminates the #1 cause of ML model failures in production: training/serving skew.

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   UNIFIED FEATURE PIPELINE                   │
│                 (Single Source of Truth)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
      ┌─────▼──────┐         ┌──────▼──────┐
      │  TRAINING  │         │  INFERENCE  │
      │  (Notebooks)│         │  (API)      │
      └─────┬──────┘         └──────┬──────┘
            │                       │
      ┌─────▼──────┐         ┌──────▼──────┐
      │  Dataset   │         │  Real-time  │
      │  .parquet  │         │  Predictions│
      └────────────┘         └─────────────┘
```

## ✅ What Problem Does This Solve?

### Before (BROKEN):

**Training:**
```python
# In notebook: 01_build_dataset_and_kronos.ipynb
features = compute_ta_features(df)  # 28 indicators
features = compute_smc_features(df) # 12 SMC features
context = build_context_vec(...)     # 29D context
kronos_emb = kronos.embed(ohlcv)     # 512D embeddings
```

**Inference:**
```python
# In API: apps/api/app/ml/features.py
features = compute_basic_ta(df)  # Only 16 indicators
# NO SMC features
# NO Kronos embeddings
# NO normalization
# ❌ SHAPE MISMATCH - MODELS WILL FAIL
```

### After (FIXED):

**Both Training AND Inference:**
```python
from app.ml.unified_features import UnifiedFeaturePipeline

pipeline = UnifiedFeaturePipeline(device='cpu')
features = pipeline.compute_features(df)

# Returns:
# - ohlcv_norm: (120, 5) normalized to [-1, 1]
# - kronos_emb: (512,) time series embeddings
# - context_vec: (29,) [5 MTF + 12 SMC + 12 TA]

# ✅ IDENTICAL FEATURES EVERYWHERE
```

## 🔧 Components

### 1. UnifiedFeaturePipeline

Location: `apps/api/app/ml/unified_features.py`

**Main class that computes all features:**

```python
class UnifiedFeaturePipeline:
    def __init__(self, device='cpu', enable_kronos=True):
        # Load Kronos model if enabled

    def compute_features(self, df, lookback=120):
        # Returns dict with:
        # - ohlcv_norm: (120, 5)
        # - kronos_emb: (512,)
        # - context_vec: (29,)
        # - alignment_vec: (5,)
        # - smc_vec: (12,)
        # - ta_vec: (12,)
```

### 2. Feature Functions

#### Technical Analysis (TA)
Location: `apps/api/app/services/feature_engine.py`

**28 indicators:**
- RSI (14)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2σ)
- ATR (14)
- ADX (14)
- OBV
- VWAP
- EMAs (9, 21, 50)
- Volume ratio

#### Smart Money Concepts (SMC)
Location: `apps/api/app/services/feature_engine.py`

**12 features:**
- Order Blocks (bullish/bearish count)
- Fair Value Gaps (FVG count)
- Liquidity levels (distance to highs/lows)
- Break of Structure (BOS)
- Change of Character (CHOCH)

#### Multi-Timeframe Alignment (MTF)
Computed in: `UnifiedFeaturePipeline.compute_alignment()`

**5 timeframes:**
- Monthly bias
- Weekly bias
- Daily bias
- 4-hour alignment
- 1-hour alignment

Each is +1 (bullish) or -1 (bearish) based on EMA fast vs slow.

#### Normalization
Location: `apps/api/app/ml/preprocess/normalize.py`

- OHLC: Min-max scaling to [-1, 1]
- Volume: Log scaling then min-max to [-1, 1]

#### Kronos Embeddings
Location: `apps/api/app/services/kronos_loader.py`

- Uses Amazon Chronos foundation model
- Embeds 120-day OHLCV sequences
- Outputs 512D embedding vector

### 3. Context Vector Breakdown

**29 dimensions total:**

```
Context Vector (29D):
├─ MTF Alignment (5D):
│  ├─ [0] monthly_bias: +1/-1
│  ├─ [1] weekly_bias: +1/-1
│  ├─ [2] daily_bias: +1/-1
│  ├─ [3] h4_align: +1/-1
│  └─ [4] h1_align: +1/-1
├─ SMC Features (12D):
│  ├─ [5] num_bullish_ob
│  ├─ [6] num_bearish_ob
│  ├─ [7] num_bullish_fvg
│  ├─ [8] num_bearish_fvg
│  ├─ [9] nearest_ob_distance
│  ├─ [10] nearest_fvg_distance
│  ├─ [11] liquidity_high_distance
│  ├─ [12] liquidity_low_distance
│  ├─ [13] bos_bullish
│  ├─ [14] bos_bearish
│  ├─ [15] choch_bullish
│  └─ [16] choch_bearish
└─ TA Features (12D):
   ├─ [17] rsi (normalized 0-1)
   ├─ [18] macd
   ├─ [19] macd_signal
   ├─ [20] bb_position (0-1)
   ├─ [21] atr_normalized
   ├─ [22] adx (normalized 0-1)
   ├─ [23] obv_trend
   ├─ [24] vwap_distance
   ├─ [25] ema_9
   ├─ [26] ema_21
   ├─ [27] ema_50
   └─ [28] volume_ratio
```

## 📝 Usage Examples

### Training (Notebook)

```python
from app.ml.unified_features import UnifiedFeaturePipeline
import pandas as pd

# Initialize pipeline
pipeline = UnifiedFeaturePipeline(device='cuda', enable_kronos=True)

# Load your OHLCV data
df = pd.read_csv('RELIANCE.NS.csv')  # Must have at least 120 rows

# Compute features for single window
features = pipeline.compute_features(df, lookback=120)

# Access results
ohlcv_norm = features['ohlcv_norm']  # (120, 5)
kronos_emb = features['kronos_emb']  # (512,)
context_vec = features['context_vec'] # (29,)

# For training, compute features for ALL windows
batch_features = pipeline.compute_batch_features(df, lookback=120, batch_size=64)

# Returns:
# - ohlcv_norm: (n_windows, 120, 5)
# - kronos_emb: (n_windows, 512)
# - context_vec: (n_windows, 29)
```

### Inference (API)

```python
from app.ml.deep_inference import DeepModelInference

# Initialize (loads models + unified pipeline)
inference = DeepModelInference(
    stockformer_path='models/stockformer.pt',
    tft_path='models/tft.pt',
    device='cpu',
    enable_kronos=True
)

# Fetch data for a symbol
df = get_market_data('RELIANCE.NS', days=150)

# Get prediction (uses unified pipeline internally)
pred = inference.predict_symbol(df, symbol='RELIANCE.NS')

# Returns:
# {
#   'stockformer': {'returns': [...], 'up_prob': [...]},
#   'tft': {'returns': [...], 'volatility_upper': [...], 'volatility_lower': [...]},
#   'consensus': {'score': 0.85, 'direction': 'bullish', 'confidence': 0.92}
# }
```

### PKScreener Integration

```python
from app.services.ai_ranker_unified import rank_signals

# PKScreener gives you candidates
candidates = [
    {'symbol': 'RELIANCE.NS', 'signal_type': 'momentum_breakout'},
    {'symbol': 'TCS.NS', 'signal_type': 'pullback_continuation'},
    # ... more candidates
]

# Rank using unified pipeline
signals = await rank_signals(candidates, top_k=20, min_confidence=0.6)

# Returns ranked signals with:
# - AI score (1-10)
# - Confidence (0-100%)
# - Entry/SL/TP levels
# - Setup tags
```

## 🎓 Model Input Requirements

### StockFormer

```python
def forward(self, x_price, x_kron, x_ctx):
    """
    Args:
        x_price: (batch, 120, 5) - OHLCV normalized
        x_kron: (batch, 512) - Kronos embeddings
        x_ctx: (batch, 29) - Context vector

    Returns:
        {
            'ret': (batch, 3),      # Returns for 3/5/10 days
            'up_prob': (batch, 3),  # Probability of up move
            'up_logits': (batch, 3) # Raw logits
        }
    """
```

### TFT (Temporal Fusion Transformer)

```python
def forward(self, x_price, x_kron, x_ctx):
    """
    Args:
        x_price: (batch, 120, 5)
        x_kron: (batch, 512)
        x_ctx: (batch, 29)

    Returns:
        {
            'ret': (batch, 3),         # Expected returns
            'vol_upper': (batch, 3),   # Upper volatility bound
            'vol_lower': (batch, 3)    # Lower volatility bound
        }
    """
```

### Veto Classifier (LightGBM)

```python
# Takes predictions from both models + raw features
features = build_veto_vec(
    sf_out={'prob': [...], 'ret': [...]},
    tft_out={'ret': [...], 'vol_upper': [...], 'vol_lower': [...]},
    smc_vec=smc_vec,  # (12,)
    tf_align=tf_align,  # (5,)
    ta_vec=ta_vec,  # (12,)
    raw_features={}
)

# Returns: (1, 38) feature vector for veto decision
```

## 🧪 Testing Feature Consistency

```python
# Test that training and inference produce same features

# Training
from app.ml.unified_features import UnifiedFeaturePipeline
pipeline_train = UnifiedFeaturePipeline(device='cpu')
features_train = pipeline_train.compute_features(df)

# Inference
from app.ml.deep_inference import DeepModelInference
inference = DeepModelInference(device='cpu')
features_infer = inference.feature_pipeline.compute_features(df)

# Verify they're identical
assert np.allclose(features_train['ohlcv_norm'], features_infer['ohlcv_norm'])
assert np.allclose(features_train['kronos_emb'], features_infer['kronos_emb'])
assert np.allclose(features_train['context_vec'], features_infer['context_vec'])

print("✓ Features are consistent!")
```

## 📂 File Structure

```
apps/api/app/
├── ml/
│   ├── unified_features.py      ← MAIN: Unified pipeline
│   ├── deep_inference.py         ← Uses unified pipeline for inference
│   ├── preprocess/
│   │   └── normalize.py          ← Normalization functions
│   ├── stockformer/
│   │   └── model.py              ← StockFormer architecture
│   └── tft/
│       └── model.py              ← TFT architecture
├── services/
│   ├── feature_engine.py         ← TA/SMC computation (used by unified)
│   ├── kronos_loader.py          ← Kronos embeddings (used by unified)
│   └── ai_ranker_unified.py      ← API endpoint using unified pipeline
└── notebooks/
    └── 01_build_dataset_UNIFIED.ipynb  ← Training using unified pipeline
```

## 🚀 Migration Guide

### Step 1: Replace Old Feature Code

**Before:**
```python
# OLD - in notebook
from app.services.feature_engine import compute_ta_features, compute_smc_features
# ... lots of manual feature engineering
```

**After:**
```python
# NEW - unified
from app.ml.unified_features import UnifiedFeaturePipeline
pipeline = UnifiedFeaturePipeline()
features = pipeline.compute_features(df)
```

### Step 2: Update Training Notebooks

Use the new notebook: `01_build_dataset_UNIFIED.ipynb`

### Step 3: Update Inference Code

**Before:**
```python
# OLD - manual feature engineering
from app.ml.features import FeatureEngine
engine = FeatureEngine()
features = engine.engineer_features(df)  # WRONG FEATURES
```

**After:**
```python
# NEW - unified pipeline
from app.ml.deep_inference import DeepModelInference
inference = DeepModelInference(stockformer_path='...', tft_path='...')
pred = inference.predict_symbol(df, symbol='RELIANCE.NS')
```

### Step 4: Update AI Ranker

**Before:**
```python
# OLD - stub implementation
from app.services.ai_ranker import rank_signals_stub
signals = await rank_signals_stub(candidates)
```

**After:**
```python
# NEW - real inference with unified features
from app.services.ai_ranker_unified import rank_signals
signals = await rank_signals(candidates)
```

## ⚙️ Configuration

### Environment Variables

```bash
# Model paths (optional, will auto-detect in ml/models/)
export STOCKFORMER_PATH=/path/to/stockformer.pt
export TFT_PATH=/path/to/tft.pt
export VETO_PATH=/path/to/veto.txt

# AlphaVantage API key (optional, fallback to yfinance)
export ALPHAVANTAGE_API_KEY=your_key_here

# Data config (for training)
export DATA_START=2015-01-01
export DATA_END=  # Leave empty for today
export TICKER_FILE=config/nifty100_yfinance.txt
```

### Runtime Settings

```python
# Disable Kronos for faster inference (at cost of accuracy)
pipeline = UnifiedFeaturePipeline(enable_kronos=False)

# Use CPU instead of GPU
pipeline = UnifiedFeaturePipeline(device='cpu')

# Change lookback window
pipeline = UnifiedFeaturePipeline(lookback=60)  # 60 days instead of 120
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'app'"

**Solution:** Make sure `apps/api` is in your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/path/to/AI_TRADER/apps/api')))
```

### Issue: "Kronos model not loading"

**Solution:** Install transformers:

```bash
pip install transformers torch
```

Or disable Kronos:

```python
pipeline = UnifiedFeaturePipeline(enable_kronos=False)
```

### Issue: "Shape mismatch when loading model checkpoint"

**Solution:** Your model was trained with old features. Retrain using `01_build_dataset_UNIFIED.ipynb`.

### Issue: "Features don't match between training and inference"

**Solution:** Make sure BOTH use `UnifiedFeaturePipeline`:

```python
# Training
pipeline = UnifiedFeaturePipeline()

# Inference
inference = DeepModelInference()  # Uses UnifiedFeaturePipeline internally
```

## 📊 Performance

### Feature Computation Time

| Operation | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| TA features | ~50ms | ~50ms |
| SMC features | ~100ms | ~100ms |
| MTF alignment | ~150ms | ~150ms |
| OHLCV normalization | ~5ms | ~5ms |
| Kronos embedding | ~200ms | ~50ms |
| **Total per symbol** | **~500ms** | **~350ms** |

### Batch Processing

For 100 symbols:
- Sequential: ~50 seconds (CPU)
- Batch (GPU): ~10 seconds

## ✅ Benefits

1. **No Training/Serving Skew** - Same features everywhere
2. **Easy to Add Features** - Add once, works in training + inference
3. **Maintainable** - Single source of truth
4. **Testable** - Easy to verify consistency
5. **PKScreener Compatible** - Works with existing pipeline

## 📚 Next Steps

1. **Retrain Models:** Use `01_build_dataset_UNIFIED.ipynb` to build new dataset
2. **Update API:** Replace `ai_ranker.py` with `ai_ranker_unified.py`
3. **Test:** Verify predictions match expected shape
4. **Deploy:** Models now work in production!

## 🤝 Contributing

When adding new features:

1. Add computation to `feature_engine.py` (TA/SMC) or `unified_features.py` (MTF/normalization)
2. Update `build_*_vec()` functions in `normalize.py` to include new features
3. Update context vector dimension (currently 29)
4. Retrain all models with new features
5. Test consistency between training and inference

---

**Questions?** Check the code documentation or create an issue on GitHub.

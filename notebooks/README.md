# Training Notebooks

This directory contains Jupyter notebooks for training AI models for the AI Trader SaaS.

---

## 📓 Available Notebooks

### 1. [01_build_dataset_and_kronos.ipynb](01_build_dataset_and_kronos.ipynb) ⭐ **START HERE**

**Main dataset building and model training notebook**

**What it does:**
- Fetches historical market data from AlphaVantage/yfinance
- Engineers technical features (RSI, MACD, Bollinger Bands, etc.)
- Builds training dataset for swing trading signals
- Trains the Kronos model (LightGBM baseline)
- Evaluates model performance
- Exports model artifacts

**Required environment variables:**
```bash
ALPHAVANTAGE_API_KEY=your_key_here  # Get free key at alphavantage.co
```

**To run:**
```bash
cd notebooks
jupyter notebook 01_build_dataset_and_kronos.ipynb
```

**Outputs:**
- `artifacts/kronos_v1.pkl` - Trained model
- `artifacts/scaler.pkl` - Feature scaler (if used)
- `datasets/training_data.parquet` - Processed dataset

---

### 2. [02_train_stockformer.ipynb](02_train_stockformer.ipynb)

**Advanced: StockFormer transformer model**

**What it does:**
- Uses dataset from notebook 01
- Trains StockFormer (transformer architecture for time series)
- Captures temporal dependencies in stock movements
- More sophisticated than traditional ML models

**When to use:**
- After you have a working baseline (Kronos/LightGBM)
- When you want to capture complex temporal patterns
- For potentially higher accuracy (but slower inference)

**Outputs:**
- `artifacts/stockformer_v1.pkl` - Trained StockFormer model

---

### 3. [03_train_tft.ipynb](03_train_tft.ipynb)

**Advanced: Temporal Fusion Transformer (TFT)**

**What it does:**
- Uses dataset from notebook 01
- Trains TFT model (Google's time series transformer)
- Handles multiple time scales and attention mechanisms
- Provides interpretable predictions

**When to use:**
- For multi-horizon forecasting
- When you need interpretability
- For capturing multi-scale temporal patterns

**Outputs:**
- `artifacts/tft_v1.pkl` - Trained TFT model

---

### 4. [04_train_lightgbm_veto.ipynb](04_train_lightgbm_veto.ipynb)

**Advanced: LightGBM Veto Classifier**

**What it does:**
- Uses dataset from notebook 01
- Trains a "veto" classifier to filter out bad signals
- Works in ensemble with other models
- Reduces false positives

**When to use:**
- As a second-stage filter after main model
- To improve signal quality
- In ensemble with Kronos or other models

**Outputs:**
- `artifacts/lightgbm_veto_v1.pkl` - Trained veto model

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# From project root
cd apps/api
pip install -r requirements.txt

# Additional notebook dependencies
pip install jupyter ipykernel matplotlib seaborn
```

### Step 2: Set Up API Keys

```bash
# Get free AlphaVantage API key
# Visit: https://www.alphavantage.co/support/#api-key

# Set environment variable
export ALPHAVANTAGE_API_KEY="your_key_here"
```

### Step 3: Run the Notebook

```bash
jupyter notebook notebooks/01_build_dataset_and_kronos.ipynb
```

### Step 4: Deploy Model

After training, copy the model to the API:

```bash
# Copy trained model
cp artifacts/kronos_v1.pkl apps/api/app/ml/models/

# Copy scaler (if you created one)
cp artifacts/scaler.pkl apps/api/app/ml/models/

# Restart backend to load new model
```

---

## 📊 Dataset Building

The notebook fetches data for NIFTY 500 stocks:

**Data sources:**
- **AlphaVantage** (primary, requires API key)
- **yfinance** (fallback, free but rate-limited)

**Time period:**
- Default: 5-9 years of daily data
- Configurable in notebook

**Features computed:**
- Price-based: OHLC, returns, volatility
- Technical: RSI, MACD, Bollinger Bands, ATR, ADX
- Volume: OBV, volume ratios
- Trend: EMAs, momentum indicators

**Target variable:**
- Future return (5-10 day forward)
- Signal labels (BUY/WAIT/SKIP)
- Risk grades (low/medium/high)

---

## 🧠 Model Training

**Supported algorithms:**
- **LightGBM** (recommended, fast and accurate)
- **XGBoost** (alternative gradient boosting)
- **Random Forest** (baseline)
- **Neural Networks** (for advanced users)

**Training process:**
1. Load and clean dataset
2. Split train/validation/test (70/15/15)
3. Feature scaling/normalization
4. Model training with hyperparameter tuning
5. Evaluation on test set
6. Model export to pickle/joblib

**Evaluation metrics:**
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- Sharpe ratio (for return predictions)
- Win rate, avg R-multiple

---

## 🔧 Customization

### Change Ticker List

Edit the notebook cell:

```python
# Option 1: Use NIFTY 100 (faster)
ticker_list = 'nifty100'

# Option 2: Use NIFTY 500 (more data)
ticker_list = 'nifty500'

# Option 3: Custom list
tickers = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
```

### Adjust Lookback Period

```python
# Fetch more historical data (better for training)
lookback_years = 9

# Or less (faster iteration)
lookback_years = 3
```

### Tune Hyperparameters

```python
# LightGBM example
params = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

---

## 📈 Model Performance Targets

**Minimum viable model:**
- Accuracy: > 55% (better than random)
- Win rate: > 50%
- Sharpe ratio: > 1.0

**Good model:**
- Accuracy: > 60%
- Win rate: > 55%
- Sharpe ratio: > 1.5

**Excellent model:**
- Accuracy: > 65%
- Win rate: > 60%
- Sharpe ratio: > 2.0

**Remember:** Even 60% accuracy can be very profitable with proper risk management!

---

## 🔄 Retraining Schedule

For production:

- **Weekly**: Quick retrain on recent data
- **Monthly**: Full retrain with extended lookback
- **Quarterly**: Architecture review and hyperparameter tuning
- **Yearly**: Dataset rebuild from scratch

---

## 💾 Artifact Management

**Recommended structure:**

```
artifacts/
├── v1/
│   ├── kronos_v1.pkl
│   ├── scaler_v1.pkl
│   └── metadata.json
├── v2/
│   ├── kronos_v2.pkl
│   ├── scaler_v2.pkl
│   └── metadata.json
└── current/  # Symlink to active version
    ├── model.pkl -> ../v2/kronos_v2.pkl
    └── scaler.pkl -> ../v2/scaler_v2.pkl
```

**Metadata JSON:**

```json
{
  "version": "v2",
  "trained_at": "2025-12-26T18:00:00Z",
  "data_period": "2015-2024",
  "algorithm": "LightGBM",
  "metrics": {
    "accuracy": 0.63,
    "win_rate": 0.58,
    "sharpe": 1.7
  },
  "hyperparams": {
    "num_leaves": 31,
    "learning_rate": 0.05
  }
}
```

---

## 🐛 Troubleshooting

### AlphaVantage rate limits

```python
# Free tier: 5 API calls/minute, 500 calls/day
# Solution: Add delays between requests
time.sleep(12)  # 12s = 5 calls/minute
```

### Out of memory

```python
# Process in batches
batch_size = 50
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    process_batch(batch)
```

### Missing features

```python
# Handle NaN values
df = df.fillna(method='ffill')  # Forward fill
df = df.fillna(0)  # Or fill with 0
df = df.dropna()  # Or drop rows
```

---

## 📚 Resources

**Learning:**
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

**Data:**
- [AlphaVantage](https://www.alphavantage.co/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [NSE India](https://www.nseindia.com/)

**PKScreener:**
- [PKScreener GitHub](https://github.com/pkjmesra/pkscreener)

---

## ✅ Integration Checklist

After training a new model:

- [ ] Model achieves target metrics (>55% accuracy)
- [ ] Model artifact saved (.pkl or .joblib)
- [ ] Scaler saved (if used)
- [ ] Metadata documented
- [ ] Model copied to `apps/api/app/ml/models/`
- [ ] Backend restarted
- [ ] Test pipeline run executed
- [ ] Signals generated and verified
- [ ] Performance tracking enabled
- [ ] Model version logged in database

---

**Happy training! 🚀**

For deployment instructions, see [MODEL_INTEGRATION_GUIDE.md](../MODEL_INTEGRATION_GUIDE.md)

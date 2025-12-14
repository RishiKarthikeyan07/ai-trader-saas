import pandas as pd

from app.services.feature_engine import compute_ta_features, compute_smc_features


def _sample_df():
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    data = {
        "open": [100 + i * 0.5 for i in range(50)],
        "high": [101 + i * 0.5 for i in range(50)],
        "low": [99 + i * 0.5 for i in range(50)],
        "close": [100 + i * 0.5 for i in range(50)],
        "volume": [100000 + i * 100 for i in range(50)],
    }
    return pd.DataFrame(data, index=dates)


def test_ta_features_shape():
    df = compute_ta_features(_sample_df())
    assert not df.empty
    for col in ["ema_fast", "ema_slow", "rsi", "atr"]:
        assert col in df.columns


def test_smc_features_score_range():
    df = compute_ta_features(_sample_df())
    smc_df = compute_smc_features(df)
    assert "smc_score" in smc_df.columns
    assert smc_df["smc_score"].between(0, 1).all()

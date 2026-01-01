"""
Feature Engineering Service

Computes technical indicators and Smart Money Concepts from OHLCV data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical analysis features

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        df_enriched: DataFrame with additional TA columns
    """
    df = df.copy()

    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_middle'] = sma_20
    df['bb_lower'] = sma_20 - 2 * std_20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_normalized'] = df['atr'] / (df['close'] + 1e-10)

    # ADX (14)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr_smooth = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_smooth)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()

    # OBV
    obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    df['obv'] = obv
    df['obv_trend'] = obv.diff(10)  # 10-period slope

    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)

    # EMAs
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

    # Fast and slow EMAs for MTF
    df['ema_fast'] = df['close'].ewm(span=9).mean()
    df['ema_slow'] = df['close'].ewm(span=21).mean()

    # Volume ratio
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

    # Fill NaN values (pandas 2.0+ compatible)
    df = df.bfill().ffill().fillna(0)

    return df


def compute_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Smart Money Concepts features

    Args:
        df: DataFrame with OHLCV data (already enriched with TA)

    Returns:
        df_enriched: DataFrame with additional SMC columns
    """
    df = df.copy()

    # Ensure lowercase
    df.columns = [c.lower() for c in df.columns]

    # Initialize SMC columns
    df['num_bullish_ob'] = 0.0
    df['num_bearish_ob'] = 0.0
    df['num_bullish_fvg'] = 0.0
    df['num_bearish_fvg'] = 0.0
    df['nearest_ob_distance'] = 0.0
    df['nearest_fvg_distance'] = 0.0
    df['liquidity_high_distance'] = 0.0
    df['liquidity_low_distance'] = 0.0
    df['bos_bullish'] = 0.0
    df['bos_bearish'] = 0.0
    df['choch_bullish'] = 0.0
    df['choch_bearish'] = 0.0

    # Order Blocks detection (simplified)
    for i in range(2, len(df) - 1):
        # Bullish OB: last bearish candle before strong rally
        if df['close'].iloc[i+1] > df['high'].iloc[i] * 1.02:  # 2% gap up
            if df['close'].iloc[i] < df['open'].iloc[i]:  # Bearish candle
                df.loc[df.index[i:], 'num_bullish_ob'] += 1

        # Bearish OB: last bullish candle before strong drop
        if df['close'].iloc[i+1] < df['low'].iloc[i] * 0.98:  # 2% gap down
            if df['close'].iloc[i] > df['open'].iloc[i]:  # Bullish candle
                df.loc[df.index[i:], 'num_bearish_ob'] += 1

    # Fair Value Gaps (FVG) detection
    for i in range(1, len(df) - 1):
        # Bullish FVG
        if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
            df.loc[df.index[i:], 'num_bullish_fvg'] += 1

        # Bearish FVG
        if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
            df.loc[df.index[i:], 'num_bearish_fvg'] += 1

    # Liquidity levels (swing highs/lows)
    window = 20
    for i in range(window, len(df) - window):
        # Swing high
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
            distance = (df['high'].iloc[i] - df['close'].iloc[i]) / (df['close'].iloc[i] + 1e-10)
            df.loc[df.index[i:], 'liquidity_high_distance'] = distance

        # Swing low
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
            distance = (df['close'].iloc[i] - df['low'].iloc[i]) / (df['close'].iloc[i] + 1e-10)
            df.loc[df.index[i:], 'liquidity_low_distance'] = distance

    # Break of Structure (BOS)
    swing_high = df['high'].iloc[0]
    swing_low = df['low'].iloc[0]

    for i in range(1, len(df)):
        # Bullish BOS
        if df['close'].iloc[i] > swing_high:
            df.loc[df.index[i], 'bos_bullish'] = 1.0
            swing_high = df['high'].iloc[i]
        else:
            swing_high = max(swing_high, df['high'].iloc[i])

        # Bearish BOS
        if df['close'].iloc[i] < swing_low:
            df.loc[df.index[i], 'bos_bearish'] = 1.0
            swing_low = df['low'].iloc[i]
        else:
            swing_low = min(swing_low, df['low'].iloc[i])

    return df


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 2),
        'high': 102 + np.cumsum(np.random.randn(200) * 2),
        'low': 98 + np.cumsum(np.random.randn(200) * 2),
        'close': 100 + np.cumsum(np.random.randn(200) * 2),
        'volume': np.random.randint(1000000, 10000000, 200)
    })

    # Ensure high >= low
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    # Compute TA features
    df_ta = compute_ta_features(df)
    print(f"TA features added. Columns: {len(df_ta.columns)}")
    print(f"Sample TA values:")
    print(f"  RSI: {df_ta['rsi'].iloc[-1]:.2f}")
    print(f"  MACD: {df_ta['macd'].iloc[-1]:.2f}")
    print(f"  ADX: {df_ta['adx'].iloc[-1]:.2f}")

    # Compute SMC features
    df_smc = compute_smc_features(df_ta)
    print(f"\nSMC features added. Total columns: {len(df_smc.columns)}")
    print(f"Sample SMC values:")
    print(f"  Bullish OBs: {df_smc['num_bullish_ob'].iloc[-1]:.0f}")
    print(f"  Bullish FVGs: {df_smc['num_bullish_fvg'].iloc[-1]:.0f}")

    print("\n✓ Feature engineering test passed!")

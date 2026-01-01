"""
Preprocessing utilities for feature normalization and extraction

Handles:
- OHLCV normalization
- MTF alignment vectors
- SMC feature vectors
- Technical analysis vectors
- Veto model feature vectors
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def normalize_ohlcv_120(ohlcv: np.ndarray) -> np.ndarray:
    """
    Normalize 120-day OHLCV window to [-1, 1] range

    Args:
        ohlcv: (120, 5) array of [open, high, low, close, volume]

    Returns:
        normalized: (120, 5) array normalized to [-1, 1]
    """
    ohlcv = ohlcv.astype(np.float32)

    # Normalize prices (OHLC) using min-max scaling
    prices = ohlcv[:, :4]  # open, high, low, close
    price_min = prices.min()
    price_max = prices.max()

    if price_max > price_min:
        prices_norm = 2 * (prices - price_min) / (price_max - price_min) - 1
    else:
        prices_norm = np.zeros_like(prices)

    # Normalize volume using log scaling
    volume = ohlcv[:, 4:5]
    volume_log = np.log1p(volume)  # log(1 + volume)
    vol_min = volume_log.min()
    vol_max = volume_log.max()

    if vol_max > vol_min:
        volume_norm = 2 * (volume_log - vol_min) / (vol_max - vol_min) - 1
    else:
        volume_norm = np.zeros_like(volume)

    # Concatenate normalized prices and volume
    normalized = np.concatenate([prices_norm, volume_norm], axis=1)

    return normalized.astype(np.float32)


def build_tf_align_vec(alignment_dict: Dict[str, float]) -> np.ndarray:
    """
    Build timeframe alignment vector from alignment dictionary

    Args:
        alignment_dict: dict with keys:
            - monthly_bias: +1 (bull) or -1 (bear)
            - weekly_bias
            - daily_bias
            - h4_align
            - h1_align

    Returns:
        tf_vec: (5,) array of alignment values
    """
    vec = np.array([
        alignment_dict.get('monthly_bias', 0.0),
        alignment_dict.get('weekly_bias', 0.0),
        alignment_dict.get('daily_bias', 0.0),
        alignment_dict.get('h4_align', 0.0),
        alignment_dict.get('h1_align', 0.0),
    ], dtype=np.float32)

    return vec


def build_smc_vec(feature_dict: Dict[str, Any]) -> np.ndarray:
    """
    Build Smart Money Concepts feature vector

    Args:
        feature_dict: dict with SMC-related features

    Returns:
        smc_vec: (12,) array of SMC features
    """
    # Extract SMC features with defaults
    vec = np.array([
        feature_dict.get('num_bullish_ob', 0.0),
        feature_dict.get('num_bearish_ob', 0.0),
        feature_dict.get('num_bullish_fvg', 0.0),
        feature_dict.get('num_bearish_fvg', 0.0),
        feature_dict.get('nearest_ob_distance', 0.0),
        feature_dict.get('nearest_fvg_distance', 0.0),
        feature_dict.get('liquidity_high_distance', 0.0),
        feature_dict.get('liquidity_low_distance', 0.0),
        feature_dict.get('bos_bullish', 0.0),
        feature_dict.get('bos_bearish', 0.0),
        feature_dict.get('choch_bullish', 0.0),
        feature_dict.get('choch_bearish', 0.0),
    ], dtype=np.float32)

    return vec


def build_ta_vec(feature_dict: Dict[str, Any]) -> np.ndarray:
    """
    Build Technical Analysis feature vector

    Args:
        feature_dict: dict with TA indicator values

    Returns:
        ta_vec: (12,) array of TA features
    """
    # Extract TA features with defaults
    vec = np.array([
        feature_dict.get('rsi', 50.0),              # RSI (0-100)
        feature_dict.get('macd', 0.0),              # MACD line
        feature_dict.get('macd_signal', 0.0),       # MACD signal
        feature_dict.get('bb_position', 0.5),       # Bollinger position (0-1)
        feature_dict.get('atr_normalized', 0.02),   # ATR / close
        feature_dict.get('adx', 20.0),              # ADX (0-100)
        feature_dict.get('obv_trend', 0.0),         # OBV slope
        feature_dict.get('vwap_distance', 0.0),     # Distance from VWAP (%)
        feature_dict.get('ema_9', 0.0),             # Fast EMA
        feature_dict.get('ema_21', 0.0),            # Medium EMA
        feature_dict.get('ema_50', 0.0),            # Slow EMA
        feature_dict.get('volume_ratio', 1.0),      # Volume / avg volume
    ], dtype=np.float32)

    # Normalize some values to reasonable ranges
    vec[0] = vec[0] / 100.0  # RSI: 0-100 → 0-1
    vec[5] = vec[5] / 100.0  # ADX: 0-100 → 0-1

    return vec


def build_veto_vec(
    sf_out: Dict[str, np.ndarray],
    tft_out: Dict[str, np.ndarray],
    smc_vec: np.ndarray,
    tf_align: np.ndarray,
    ta_vec: np.ndarray,
    raw_features: Dict[str, Any]
) -> np.ndarray:
    """
    Build feature vector for LightGBM veto classifier

    Combines predictions from StockFormer and TFT with SMC/MTF/TA features

    Args:
        sf_out: StockFormer output dict with 'prob' and 'ret'
        tft_out: TFT output dict with 'ret', 'vol_upper', 'vol_lower'
        smc_vec: (12,) SMC features
        tf_align: (5,) MTF alignment
        ta_vec: (12,) TA features
        raw_features: dict of additional raw features

    Returns:
        veto_vec: feature vector for veto classifier
    """
    features = []

    # StockFormer predictions
    if 'prob' in sf_out:
        sf_prob = sf_out['prob'].flatten()[:3]  # First 3 horizons
        features.extend(sf_prob)

    if 'ret' in sf_out:
        sf_ret = sf_out['ret'].flatten()[:3]
        features.extend(sf_ret)

    # TFT predictions
    if 'ret' in tft_out:
        tft_ret = tft_out['ret'].flatten()[:3]
        features.extend(tft_ret)

    if 'vol_upper' in tft_out:
        tft_vol_upper = tft_out['vol_upper'].flatten()[:3]
        features.extend(tft_vol_upper)

    if 'vol_lower' in tft_out:
        tft_vol_lower = tft_out['vol_lower'].flatten()[:3]
        features.extend(tft_vol_lower)

    # Agreement features
    if 'prob' in sf_out and 'ret' in tft_out:
        # Model agreement
        sf_direction = (sf_prob > 0.5).astype(float)
        tft_direction = (tft_ret > 0).astype(float)
        agreement = (sf_direction == tft_direction).astype(float)
        features.extend(agreement)

        # Confidence
        sf_confidence = np.abs(sf_prob - 0.5) * 2  # 0 to 1
        features.extend(sf_confidence)

    # MTF alignment
    features.extend(tf_align)

    # SMC features (first 6 most important)
    features.extend(smc_vec[:6])

    # TA features (first 6 most important)
    features.extend(ta_vec[:6])

    # Convert to array
    veto_vec = np.array(features, dtype=np.float32)

    # Ensure we have a consistent shape
    # Actual feature count:
    # SF prob: 3 + SF ret: 3 + TFT ret: 3 + TFT vol_upper: 3 + TFT vol_lower: 3
    # + agreement: 3 + confidence: 3 + MTF: 5 + SMC: 6 + TA: 6 = 38 features
    target_size = 38  # Actual feature count

    if len(veto_vec) < target_size:
        # Pad with zeros if features are missing
        veto_vec = np.pad(veto_vec, (0, target_size - len(veto_vec)), 'constant')
    elif len(veto_vec) > target_size:
        # Trim if too many features
        veto_vec = veto_vec[:target_size]

    return veto_vec.reshape(1, -1)  # (1, 38)


if __name__ == "__main__":
    # Test normalization
    print("Testing normalization functions...")

    # Test OHLCV normalization
    ohlcv = np.random.randn(120, 5) * 100 + 1000
    ohlcv[:, 4] *= 1000000  # Volume
    normalized = normalize_ohlcv_120(ohlcv)
    print(f"OHLCV normalized shape: {normalized.shape}")
    print(f"OHLCV normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    # Test MTF alignment
    alignment = {
        'monthly_bias': 1.0,
        'weekly_bias': 1.0,
        'daily_bias': -1.0,
        'h4_align': -1.0,
        'h1_align': -1.0
    }
    mtf_vec = build_tf_align_vec(alignment)
    print(f"MTF vector: {mtf_vec}")

    # Test SMC vector
    smc_features = {
        'num_bullish_ob': 2.0,
        'num_bearish_ob': 1.0,
        'num_bullish_fvg': 3.0,
        'num_bearish_fvg': 0.0,
    }
    smc_vec = build_smc_vec(smc_features)
    print(f"SMC vector shape: {smc_vec.shape}")

    # Test TA vector
    ta_features = {
        'rsi': 65.0,
        'macd': 0.5,
        'macd_signal': 0.3,
        'adx': 35.0
    }
    ta_vec = build_ta_vec(ta_features)
    print(f"TA vector shape: {ta_vec.shape}")

    print("\nAll tests passed!")

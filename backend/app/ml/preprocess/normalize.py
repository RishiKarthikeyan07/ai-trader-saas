from __future__ import annotations

import numpy as np


def normalize_ohlcv_120(ohlcv: np.ndarray) -> np.ndarray:
    """Normalize OHLCV window (120,5) -> float32.

    Prices divided by last close; volume log1p + z-score.
    """
    if ohlcv.shape[0] < 120 or ohlcv.shape[1] != 5:
        raise ValueError("Expected ohlcv shape (>=120,5)")
    window = ohlcv[-120:].astype(np.float32)
    prices = window[:, :4]
    last_close = prices[-1, 3] if prices[-1, 3] != 0 else 1.0
    prices_norm = prices / last_close
    volume = window[:, 4]
    v_log = np.log1p(volume)
    v_norm = (v_log - v_log.mean()) / (v_log.std() + 1e-6)
    vol_col = v_norm.reshape(-1, 1)
    return np.concatenate([prices_norm, vol_col], axis=1).astype(np.float32)


def build_tf_align_vec(alignment: dict) -> np.ndarray:
    order = ["monthly_bias", "weekly_bias", "daily_bias", "h4_align", "h1_align"]
    return np.array([alignment.get(k, 0.0) for k in order], dtype=np.float32)


def build_smc_vec(features: dict) -> np.ndarray:
    keys = [
        "smc_score",
        "bos",
        "choch",
        "mss",
        "liquidity_sweep_buy",
        "liquidity_sweep_sell",
        "fvg_up",
        "fvg_down",
        "bearish_ob_dist",
        "bullish_ob_dist",
        "premium_discount",
        "swing_high",
    ]
    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)


def build_ta_vec(features: dict) -> np.ndarray:
    close = features.get("close", 1.0) or 1.0
    atr = features.get("atr", 0.0) or 0.0
    atr_pct = atr / close if close else 0.0
    ema_fast = features.get("ema_fast", 0.0)
    ema_slow = features.get("ema_slow", 0.0)
    supertrend = features.get("supertrend", 0.0)
    ema_gap_pct = (ema_fast - ema_slow) / close if close else 0.0
    volume = features.get("volume", 0.0)
    log_vol = np.log1p(volume)
    close_over_open = (close / (features.get("open", close) or close)) if features.get("open") else 1.0
    high_low = ((features.get("high", close) - features.get("low", close)) / close) if close else 0.0
    ta_values = [
        features.get("rsi", 0.0),
        atr_pct,
        features.get("volatility", 0.0),
        features.get("returns", 0.0),
        ema_fast / close if close else 0.0,
        ema_slow / close if close else 0.0,
        ema_gap_pct,
        supertrend / close if close else 0.0,
        features.get("ema_trend", 0.0),
        log_vol,
        close_over_open,
        high_low,
    ]
    return np.array(ta_values, dtype=np.float32)


def build_veto_vec(sf_out: dict, tft_out: dict, smc_vec: np.ndarray, tf_align: np.ndarray, ta_vec: np.ndarray, raw_features: dict) -> np.ndarray:
    sf_prob = sf_out["prob"][0] if isinstance(sf_out.get("prob"), np.ndarray) else sf_out.get("prob", [0, 0, 0])
    sf_ret = sf_out["ret"][0] if isinstance(sf_out.get("ret"), np.ndarray) else sf_out.get("ret", [0, 0, 0])
    prob_3, prob_5, prob_10 = sf_prob[:3]
    ret_3, ret_5, ret_10 = sf_ret[:3]
    vol_10 = tft_out.get("vol_10d", [[0]])[0][0] if isinstance(tft_out, dict) else 0
    upper_10 = tft_out.get("upper_10d", [[0]])[0][0] if isinstance(tft_out, dict) else 0
    lower_10 = tft_out.get("lower_10d", [[0]])[0][0] if isinstance(tft_out, dict) else 0
    upper_dist = upper_10
    lower_dist = lower_10
    smc_score = raw_features.get("smc_score", 0)
    smc_flags = [raw_features.get(k, 0) for k in ["bos", "choch", "mss", "liquidity_sweep_buy", "liquidity_sweep_sell"]]
    tf_list = tf_align.tolist()
    ta_list = ta_vec.tolist()
    vec = np.array(
        [prob_3, prob_5, prob_10, ret_3, ret_5, ret_10, vol_10, upper_dist, lower_dist, smc_score, *smc_flags, *tf_list, *ta_list],
        dtype=np.float32,
    )
    return vec.reshape(1, -1)

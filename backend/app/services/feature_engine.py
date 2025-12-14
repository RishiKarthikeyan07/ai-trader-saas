from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from app.services.data_ingestion import fetch_ohlcv
from app.core.config import Settings


@dataclass
class TimeframeSet:
    monthly: pd.DataFrame
    weekly: pd.DataFrame
    daily: pd.DataFrame
    h4: pd.DataFrame
    h1: pd.DataFrame


TA_WINDOW = 14


def build_timeframes(symbol: str, settings: Settings) -> TimeframeSet:
    daily = fetch_ohlcv(symbol, "1d", lookback_days=400, settings=settings)
    hourly = fetch_ohlcv(symbol, "1h", lookback_days=90, settings=settings)

    if hourly.empty:
        hourly = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    h4 = hourly.resample("4H").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    weekly = daily.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    monthly = daily.resample("M").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    return TimeframeSet(
        monthly=monthly,
        weekly=weekly,
        daily=daily,
        h4=h4,
        h1=hourly,
    )


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = TA_WINDOW) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(window=period).mean()
    roll_down = pd.Series(loss).rolling(window=period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.index = series.index
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["returns"] = out["close"].pct_change()
    out["ema_fast"] = _ema(out["close"], span=9)
    out["ema_slow"] = _ema(out["close"], span=21)
    out["rsi"] = _rsi(out["close"], period=14)
    out["atr"] = _atr(out)
    out["volatility"] = out["returns"].rolling(window=20).std()
    out["obv"] = (np.sign(out["close"].diff().fillna(0)) * out["volume"]).cumsum()
    out["ema_trend"] = (out["ema_fast"] > out["ema_slow"]).astype(int)
    out["supertrend"] = out["close"] - out["atr"] * 1.5
    return out.dropna()


def _swing_points(df: pd.DataFrame, lookback: int = 3) -> Tuple[pd.Series, pd.Series]:
    highs = df["high"].rolling(window=lookback, center=True).max()
    lows = df["low"].rolling(window=lookback, center=True).min()
    swing_high = (df["high"] >= highs) & (df["high"] > df["high"].shift(1))
    swing_low = (df["low"] <= lows) & (df["low"] < df["low"].shift(1))
    return swing_high.fillna(False), swing_low.fillna(False)


def compute_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    swing_high, swing_low = _swing_points(out, lookback=5)
    out["swing_high"] = swing_high.astype(int)
    out["swing_low"] = swing_low.astype(int)

    # Break of structure (BOS) and Change of Character (CHoCH)
    out["prev_high"] = out["high"].where(swing_high).ffill()
    out["prev_low"] = out["low"].where(swing_low).ffill()
    out["bos"] = ((out["close"] > out["prev_high"].shift()) & swing_high).astype(int)
    out["choch"] = ((out["close"] < out["prev_low"].shift()) & swing_low).astype(int)

    # Market structure shift (MSS) approximated via EMA flips
    out["mss"] = ((out["ema_fast"] < out["ema_slow"]) & (out["ema_fast"].shift() > out["ema_slow"].shift())).astype(int)

    # Liquidity sweeps: wick beyond previous extremes then close inside
    prev_high = out["high"].shift(1)
    prev_low = out["low"].shift(1)
    out["liquidity_sweep_buy"] = ((out["high"] > prev_high) & (out["close"] < prev_high)).astype(int)
    out["liquidity_sweep_sell"] = ((out["low"] < prev_low) & (out["close"] > prev_low)).astype(int)

    # Fair value gap detection over 3-candle windows
    out["fvg_up"] = ((out["low"] > out["high"].shift(2))).astype(int)
    out["fvg_down"] = ((out["high"] < out["low"].shift(2))).astype(int)

    # Order block distance: distance to last opposite candle before break
    bearish_blocks = out[(out["close"] < out["open"]) & out["bos"]]
    bullish_blocks = out[(out["close"] > out["open"]) & out["choch"]]
    out["bearish_ob_dist"] = (out["close"] - bearish_blocks["high"].ffill()).abs()
    out["bullish_ob_dist"] = (out["close"] - bullish_blocks["low"].ffill()).abs()

    # Premium/discount using mid-range of last 20 bars
    mid_range = (out["high"].rolling(20).max() + out["low"].rolling(20).min()) / 2
    out["premium_discount"] = (out["close"] - mid_range) / (mid_range + 1e-9)

    # SMC score aggregates signals
    smc_components = [
        out["bos"],
        out["choch"],
        out["mss"],
        out["liquidity_sweep_buy"],
        out["liquidity_sweep_sell"],
        out["fvg_up"],
        out["fvg_down"],
    ]
    out["smc_score"] = np.clip(pd.concat(smc_components, axis=1).sum(axis=1) / 7.0, 0, 1)
    return out.dropna()


def tf_alignment(tfset: TimeframeSet) -> Dict[str, float]:
    def bias(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        latest = df.iloc[-1]
        return 1.0 if latest.get("ema_fast", 0) > latest.get("ema_slow", 0) else -1.0

    monthly = compute_ta_features(tfset.monthly)
    weekly = compute_ta_features(tfset.weekly)
    daily = compute_ta_features(tfset.daily)
    h4 = compute_ta_features(tfset.h4)
    h1 = compute_ta_features(tfset.h1)
    return {
        "monthly_bias": bias(monthly),
        "weekly_bias": bias(weekly),
        "daily_bias": bias(daily),
        "h4_align": bias(h4),
        "h1_align": bias(h1),
    }


def build_feature_set(symbol: str, settings: Settings) -> Dict[str, pd.DataFrame]:
    tfset = build_timeframes(symbol, settings=settings)
    features = {}
    for key, df in {
        "monthly": tfset.monthly,
        "weekly": tfset.weekly,
        "daily": tfset.daily,
        "h4": tfset.h4,
        "h1": tfset.h1,
    }.items():
        enriched = compute_ta_features(df)
        enriched = compute_smc_features(enriched)
        features[key] = enriched
    return features

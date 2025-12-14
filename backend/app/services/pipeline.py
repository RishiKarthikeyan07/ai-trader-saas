from __future__ import annotations

import uuid
from typing import Dict, List
import numpy as np
import pandas as pd

from app.core.config import Settings
from app.models.signal import Signal
from app.services.data_ingestion import run_pk_stock_screener
from app.services.feature_engine import build_feature_set
from app.services.store import save_signals, fetch_latest, update_ready_state
from app.services.model_registry import ModelRegistry, ModelOutput, ModelInput
from app.ml.preprocess.normalize import (
    normalize_ohlcv_120,
    build_tf_align_vec,
    build_smc_vec,
    build_ta_vec,
)

DEFAULT_CANDIDATES = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "LT", "SBIN", "KOTAKBANK"]


def _candidate_symbols(settings: Settings) -> List[str]:
    screener = run_pk_stock_screener(universe=settings.default_universe, top_n=200)
    if screener:
        return [row.get("symbol") or row.get("Symbol") for row in screener if row.get("symbol") or row.get("Symbol")]
    return DEFAULT_CANDIDATES


def _alignment_from_features(feature_set: Dict[str, object]) -> Dict[str, float]:
    alignment = {}
    for key in ["monthly", "weekly", "daily", "h4", "h1"]:
        df = feature_set.get(key)
        if df is None or getattr(df, "empty", True):
            continue
        latest = df.iloc[-1]
        bias = 1.0 if latest.get("ema_fast", 0) > latest.get("ema_slow", 0) else -1.0
        alignment_key = {
            "monthly": "monthly_bias",
            "weekly": "weekly_bias",
            "daily": "daily_bias",
            "h4": "h4_align",
            "h1": "h1_align",
        }[key]
        alignment[alignment_key] = bias
    return alignment


def _build_signal(symbol: str, features: Dict[str, float], alignment: Dict[str, float], model_output: ModelOutput) -> Signal:
    direction_prob = model_output.direction_prob
    signal_type = "BUY" if direction_prob > 0.55 else "SELL" if direction_prob < 0.45 else "HOLD"

    atr = features.get("atr", 0.0) or 0.0
    close = features.get("close") or features.get("close", 0.0)
    entry_low = close - atr * 0.5
    entry_high = close + atr * 0.2
    stop_loss = close - atr * 1.2
    target_1 = model_output.upper_band if model_output.upper_band is not None else close + atr * 1.2
    target_2 = model_output.upper_band if model_output.upper_band is not None else close + atr * 2.0

    smc_flags = {
        "bos": features.get("bos", 0),
        "choch": features.get("choch", 0),
        "mss": features.get("mss", 0),
        "liquidity_sweep_buy": features.get("liquidity_sweep_buy", 0),
        "liquidity_sweep_sell": features.get("liquidity_sweep_sell", 0),
        "fvg_up": features.get("fvg_up", 0),
        "fvg_down": features.get("fvg_down", 0),
    }
    confidence = float(np.clip(direction_prob * model_output.reliability, 0, 1))
    return Signal(
        id=str(uuid.uuid4()),
        symbol=symbol,
        signal_type=signal_type,
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        confidence=confidence,
        expected_return=model_output.expected_return,
        expected_volatility=model_output.expected_volatility,
        tf_alignment=alignment,
        smc_score=float(features.get("smc_score", 0)),
        smc_flags=smc_flags,
        model_versions=model_output.model_versions,
    )


def run_daily_pipeline(settings: Settings) -> Dict[str, int]:
    symbols = _candidate_symbols(settings)
    registry = ModelRegistry(settings)
    registry.ensure_ready()
    generated: List[Signal] = []
    for symbol in symbols:
        feature_set = build_feature_set(symbol, settings=settings)
        daily = feature_set.get("daily")
        if daily is None or daily.empty or len(daily) < 120:
            continue
        latest_row = daily.iloc[-1]
        latest = latest_row.to_dict()
        alignment_dict = _alignment_from_features(feature_set)
        try:
            ohlcv_norm = normalize_ohlcv_120(daily[["open", "high", "low", "close", "volume"]].to_numpy())
        except ValueError:
            continue
        model_input = ModelInput(
            symbol=symbol,
            asof=latest_row.name,
            ohlcv_120=ohlcv_norm,
            tf_align=build_tf_align_vec(alignment_dict),
            smc_vec=build_smc_vec(latest),
            ta_vec=build_ta_vec(latest),
            raw_features=latest,
        )
        model_output = registry.infer(model_input)
        sig = _build_signal(symbol, latest, alignment_dict, model_output)
        generated.append(sig)

    # Only keep top N by confidence respecting tier caps
    generated = sorted(generated, key=lambda s: s.confidence, reverse=True)[: settings.max_daily_signals]
    save_signals(generated, settings=settings)
    return {"generated": len(generated), "universe": len(symbols)}


def run_hourly_refinement(settings: Settings) -> Dict[str, int]:
    signals = fetch_latest(limit=settings.max_hourly_signals, settings=settings)
    updated = 0
    for sig in signals:
        if sig.signal_type not in {"BUY", "SELL"}:
            continue
        # use lighter 1H/4H confirmation: alignment must agree with daily bias
        feature_set = build_feature_set(sig.symbol, settings=settings)
        h1 = feature_set.get("h1")
        h4 = feature_set.get("h4")
        if h1 is None or h1.empty or h4 is None or h4.empty:
            continue
        h1_latest = h1.iloc[-1]
        h4_latest = h4.iloc[-1]
        trend_ok = (h1_latest["ema_trend"] > 0 and h4_latest["ema_trend"] > 0) if sig.signal_type == "BUY" else (h1_latest["ema_trend"] < 0 and h4_latest["ema_trend"] < 0)
        smc_ok = h1_latest.get("smc_score", 0) > 0.3 and h4_latest.get("smc_score", 0) > 0.3
        ready_state = "READY_TO_ENTER" if trend_ok and smc_ok else "WAIT"
        update_ready_state(sig.id, ready_state, settings=settings)
        updated += 1
    return {"checked": len(signals), "updated": updated}

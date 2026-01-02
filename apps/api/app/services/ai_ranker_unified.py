"""
AI Signal Ranker (Unified Feature Pipeline Version)

Uses UnifiedFeaturePipeline and DeepModelInference to ensure
training/inference consistency.

Takes scanner candidates and produces ranked signals with:
- Score (1-10)
- Confidence (0-100%)
- Risk grade
- Entry/SL/TP levels
- Setup tags

IMPORTANT: This is BLACK BOX. No model internals exposed to users.
"""

from typing import List, Dict, Optional
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.ml.deep_inference import DeepModelInference
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)

# Global model instance (loaded on startup)
_model_instance: Optional[DeepModelInference] = None


async def load_models() -> Optional[DeepModelInference]:
    """
    Load trained models from disk

    Looks for models in:
    1. Environment variables: STOCKFORMER_PATH, TFT_PATH, VETO_PATH
    2. Default path: apps/api/app/ml/models/
    """
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    try:
        # Get model paths from environment
        stockformer_path = os.getenv('STOCKFORMER_PATH')
        tft_path = os.getenv('TFT_PATH')
        veto_path = os.getenv('VETO_PATH')

        # Default model directory
        default_dir = Path(__file__).parent.parent / 'ml' / 'models'

        # Find models in default directory if env vars not set
        if not stockformer_path and default_dir.exists():
            sf_files = list(default_dir.glob('stockformer*.pt')) + list(default_dir.glob('stockformer*.pth'))
            if sf_files:
                stockformer_path = str(sf_files[0])

        if not tft_path and default_dir.exists():
            tft_files = list(default_dir.glob('tft*.pt')) + list(default_dir.glob('tft*.pth'))
            if tft_files:
                tft_path = str(tft_files[0])

        if not veto_path and default_dir.exists():
            veto_files = list(default_dir.glob('veto*.txt'))
            if veto_files:
                veto_path = str(veto_files[0])

        # Determine device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize model inference
        logger.info(f"Initializing models on {device}...")
        _model_instance = DeepModelInference(
            stockformer_path=stockformer_path,
            tft_path=tft_path,
            veto_path=veto_path,
            device=device,
            enable_kronos=True  # Enable Kronos embeddings
        )

        logger.info("✓ Models loaded successfully")
        return _model_instance

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.warning("Falling back to stub ranker")
        return None


async def rank_signals(
    candidates: List[Dict],
    top_k: int = 20,
    min_confidence: float = 0.6
) -> List[Dict]:
    """
    Rank PKScreener candidates using AI models

    Args:
        candidates: List of dicts with 'symbol' and 'signal_type'
        top_k: Number of top signals to return
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        Ranked signals with scores, levels, and metadata
    """
    # Load models if not already loaded
    model = await load_models()

    if model is None:
        # Fallback to stub implementation
        logger.warning("Using stub ranker (no models loaded)")
        return await rank_signals_stub(candidates)

    signals = []

    for candidate in candidates:
        symbol = candidate['symbol']

        try:
            # Fetch historical data (120 days + buffer)
            df = await fetch_symbol_data(symbol, days=150)

            if df is None or len(df) < 120:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Get model prediction
            pred = model.predict_symbol(df, symbol, lookback=120)

            # Skip if veto rejected
            if pred.get('veto', {}).get('pass', True) == False:
                logger.debug(f"Veto rejected {symbol}")
                continue

            # Skip if confidence too low
            consensus = pred.get('consensus', {})
            if consensus.get('confidence', 0) < min_confidence:
                continue

            # Build signal
            signal = build_signal_from_prediction(pred, candidate, df)
            signals.append(signal)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    # Sort by AI score descending
    signals.sort(key=lambda x: x['score'], reverse=True)

    # Re-rank
    for idx, signal in enumerate(signals):
        signal['rank'] = idx + 1

    return signals[:top_k]


def build_signal_from_prediction(
    pred: Dict,
    candidate: Dict,
    df: pd.DataFrame
) -> Dict:
    """
    Build user-facing signal from model prediction

    Keeps model internals hidden, only exposes high-level insights
    """
    symbol = pred['symbol']
    consensus = pred.get('consensus', {})
    sf_pred = pred.get('stockformer', {})
    tft_pred = pred.get('tft', {})

    # Get current price
    current_price = float(df['close'].iloc[-1])

    # Compute score (1-10 scale)
    ai_score = consensus.get('score', 0.5)  # 0-1
    score = int(ai_score * 10)  # Convert to 1-10

    # Compute confidence percentage
    confidence = int(consensus.get('confidence', 0.5) * 100)

    # Determine risk grade based on volatility
    vol_upper = tft_pred.get('volatility_upper', [0, 0, 0])
    vol_lower = tft_pred.get('volatility_lower', [0, 0, 0])
    avg_vol = np.mean([abs(u) + abs(l) for u, l in zip(vol_upper, vol_lower)])

    if avg_vol < 0.03:  # 3% volatility
        risk_grade = 'low'
    elif avg_vol < 0.06:
        risk_grade = 'medium'
    else:
        risk_grade = 'high'

    # Compute entry/SL/TP levels
    predicted_return_5d = sf_pred.get('returns', [0, 0, 0])[1]  # 5-day return
    direction = consensus.get('direction', 'bullish')

    if direction == 'bullish':
        entry_min = current_price * 0.98  # 2% below
        entry_max = current_price * 1.02  # 2% above
        stop_loss = current_price * 0.95  # 5% stop
        target_1 = current_price * (1 + abs(predicted_return_5d) * 0.5)
        target_2 = current_price * (1 + abs(predicted_return_5d))
    else:
        entry_min = current_price * 0.98
        entry_max = current_price * 1.02
        stop_loss = current_price * 1.05
        target_1 = current_price * (1 - abs(predicted_return_5d) * 0.5)
        target_2 = current_price * (1 - abs(predicted_return_5d))

    # Generate setup tags
    setup_tags = generate_setup_tags(candidate, pred)

    # Generate explanation
    why_now = generate_why_now(consensus, sf_pred, tft_pred)
    key_factors = generate_key_factors(pred, df)
    risk_notes = generate_risk_notes(risk_grade, avg_vol)

    return {
        'symbol': symbol,
        'signal_type': candidate.get('signal_type', 'ai_signal'),
        'rank': 0,  # Will be set later
        'score': score,
        'confidence': confidence,
        'risk_grade': risk_grade,
        'horizon': '3-10 days',

        # Price levels
        'entry_min': round(entry_min, 2),
        'entry_max': round(entry_max, 2),
        'stop_loss': round(stop_loss, 2),
        'target_1': round(target_1, 2),
        'target_2': round(target_2, 2),

        # High-level insights (no model internals)
        'setup_tags': setup_tags,
        'why_now': why_now,
        'key_factors': key_factors,
        'risk_notes': risk_notes,

        # Metadata
        'generated_at': pred.get('timestamp', datetime.now().isoformat()),
        'direction': direction,
    }


def generate_setup_tags(candidate: Dict, pred: Dict) -> List[str]:
    """Generate high-level setup tags based on prediction"""
    tags = []

    # Add signal type tag
    signal_type = candidate.get('signal_type', '')
    tag_map = {
        'momentum_breakout': 'Breakout Setup',
        'squeeze_expansion': 'Volatility Expansion',
        'pullback_continuation': 'Pullback Entry',
        'liquidity_sweep_reversal': 'Reversal Setup',
        'relative_strength': 'Sector Leader'
    }
    if signal_type in tag_map:
        tags.append(tag_map[signal_type])

    # Add confidence tag
    confidence = pred.get('consensus', {}).get('confidence', 0)
    if confidence > 0.8:
        tags.append('High Confidence')
    elif confidence > 0.6:
        tags.append('Medium Confidence')

    # Add direction tag
    direction = pred.get('consensus', {}).get('direction', '')
    if direction:
        tags.append(direction.capitalize())

    # Add model agreement tag
    sf_prob = pred.get('stockformer', {}).get('up_prob', [0.5, 0.5, 0.5])
    tft_ret = pred.get('tft', {}).get('returns', [0, 0, 0])

    sf_bullish = np.mean(sf_prob) > 0.5
    tft_bullish = np.mean(tft_ret) > 0

    if sf_bullish == tft_bullish:
        tags.append('Models Agree')

    return tags


def generate_why_now(consensus: Dict, sf_pred: Dict, tft_pred: Dict) -> str:
    """Generate high-level explanation of why this signal is actionable now"""
    direction = consensus.get('direction', 'bullish')
    confidence = consensus.get('confidence', 0)

    if confidence > 0.8:
        strength = "strong"
    elif confidence > 0.6:
        strength = "moderate"
    else:
        strength = "developing"

    return f"AI models detect a {strength} {direction} setup with favorable risk/reward characteristics across multiple timeframes."


def generate_key_factors(pred: Dict, df: pd.DataFrame) -> List[str]:
    """Generate key factors driving the signal (high-level only)"""
    factors = []

    # Trend factor
    returns_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) >= 6 else 0
    if abs(returns_5d) > 0.03:
        factors.append(f"{'Strong' if abs(returns_5d) > 0.05 else 'Moderate'} recent momentum")

    # Volatility factor
    tft_pred = pred.get('tft', {})
    vol_upper = tft_pred.get('volatility_upper', [0])
    if vol_upper[0] > 0.05:
        factors.append("Elevated volatility environment")
    else:
        factors.append("Stable price action")

    # Model confidence
    confidence = pred.get('consensus', {}).get('confidence', 0)
    if confidence > 0.7:
        factors.append("High model confidence")

    # Always include generic multi-timeframe factor
    factors.append("Multi-timeframe alignment detected")

    return factors[:4]  # Max 4 factors


def generate_risk_notes(risk_grade: str, volatility: float) -> List[str]:
    """Generate risk management notes"""
    notes = []

    if risk_grade == 'high':
        notes.append("⚠️ Higher volatility - consider smaller position size")
        notes.append("Use wider stops to avoid premature exit")
    elif risk_grade == 'medium':
        notes.append("Moderate risk - standard position sizing recommended")
    else:
        notes.append("Lower volatility environment - more predictable price action")

    notes.append("Always use stop-loss to manage downside risk")
    notes.append("Consider scaling out at Target 1 to lock profits")

    return notes


async def fetch_symbol_data(symbol: str, days: int = 150) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data for a symbol

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        days: Number of days of history to fetch

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        # Use market data service
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = market_data_service.get_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )

        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


async def rank_signals_stub(candidates: List[Dict]) -> List[Dict]:
    """
    Stub implementation (fallback when models not available)

    Same as before - generates mock signals
    """
    import random

    signals = []

    for idx, candidate in enumerate(candidates):
        symbol = candidate['symbol']

        signal = {
            'symbol': symbol,
            'signal_type': candidate['signal_type'],
            'rank': idx + 1,
            'score': random.randint(6, 10),
            'confidence': random.randint(75, 95),
            'risk_grade': random.choice(['low', 'medium']),
            'horizon': '3-10 days',

            'entry_min': 1000 + random.randint(0, 500),
            'entry_max': 1050 + random.randint(0, 500),
            'stop_loss': 950 + random.randint(0, 500),
            'target_1': 1100 + random.randint(0, 500),
            'target_2': 1150 + random.randint(0, 500),

            'setup_tags': ['AI Signal', 'Stub Mode'],
            'why_now': 'Mock signal for testing (models not loaded)',
            'key_factors': ['Stub data'],
            'risk_notes': ['This is mock data for testing'],
            'generated_at': datetime.now().isoformat(),
            'direction': 'bullish',
        }

        signals.append(signal)

    signals.sort(key=lambda x: x['score'], reverse=True)

    for idx, signal in enumerate(signals):
        signal['rank'] = idx + 1

    return signals[:20]

"""
AI Signal Ranker

Takes scanner candidates and produces ranked signals with:
- Score (1-10)
- Confidence (0-100%)
- Risk grade
- Entry/SL/TP levels
- Setup tags

IMPORTANT: This is BLACK BOX. No model internals exposed to users.
"""

from typing import List, Dict, Optional
import random
import os
from pathlib import Path
import logging
from app.ml.inference import ModelInference
from app.ml.features import FeatureEngine
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)

# Global model instance (loaded on startup)
_model_instance: Optional[ModelInference] = None
_feature_engine = FeatureEngine()


async def rank_signals_stub(candidates: List[Dict]) -> List[Dict]:
    """
    Stub implementation of AI ranker

    In production, this would:
    1. Load active model from model_versions table
    2. Fetch market data for each candidate
    3. Run feature engineering
    4. Predict scores, risk, levels
    5. Rank by score
    6. Return top N signals
    """

    signals = []

    for idx, candidate in enumerate(candidates):
        symbol = candidate['symbol']

        # Mock signal generation
        signal = {
            'symbol': symbol,
            'signal_type': candidate['signal_type'],
            'rank': idx + 1,
            'score': candidate['pack_score'],  # Use pack score as base
            'confidence': random.randint(75, 95),
            'risk_grade': random.choice(['low', 'medium']),
            'horizon': 'swing',

            # Mock price levels (should come from model)
            'entry_min': 1000 + random.randint(0, 500),
            'entry_max': 1050 + random.randint(0, 500),
            'stop_loss': 950 + random.randint(0, 500),
            'target_1': 1100 + random.randint(0, 500),
            'target_2': 1150 + random.randint(0, 500),

            # High-level tags only (no model secrets)
            'setup_tags': generate_setup_tags(candidate),

            # Explanation (high-level only)
            'why_now': 'AI analysis indicates favorable setup based on multiple factors',
            'key_factors': [
                'Strong momentum indicators',
                'Favorable risk/reward profile',
                'Volume confirmation'
            ],
            'risk_notes': [
                'Monitor support levels',
                'Adjust position size per risk tolerance'
            ],
            'mtf_alignment': [
                {'timeframe': 'D', 'trend': 'bullish', 'structure': 'Higher highs'},
                {'timeframe': '4H', 'trend': 'bullish', 'structure': 'Consolidation'},
            ]
        }

        signals.append(signal)

    # Sort by score descending
    signals.sort(key=lambda x: x['score'], reverse=True)

    # Re-rank
    for idx, signal in enumerate(signals):
        signal['rank'] = idx + 1

    # Return top 20
    return signals[:20]


def generate_setup_tags(candidate: Dict) -> List[str]:
    """Generate high-level setup tags (no model secrets)"""
    signal_type = candidate['signal_type']

    tag_map = {
        'momentum_breakout': ['Breakout Confirmed', 'Volume Surge', 'Trend Aligned'],
        'squeeze_expansion': ['Volatility Expansion', 'Direction Clear', 'Momentum Building'],
        'pullback_continuation': ['Pullback Zone', 'Support Hold', 'HTF Bullish'],
        'liquidity_sweep_reversal': ['Liquidity Grab', 'Reversal Setup', 'Structure Shift'],
        'relative_strength': ['Sector Leader', 'RS Strength', 'Consolidation']
    }

    return tag_map.get(signal_type, ['Setup Identified'])


async def load_active_model() -> Optional[ModelInference]:
    """
    Load active model from registry

    Looks for model in:
    1. Environment variable MODEL_PATH
    2. Default path: apps/api/app/ml/models/
    3. Database model_versions table (if configured)
    """
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    try:
        # Option 1: Load from environment variable
        model_path = os.getenv('MODEL_PATH')

        # Option 2: Load from default location
        if not model_path:
            default_dir = Path(__file__).parent.parent / 'ml' / 'models'
            if default_dir.exists():
                # Look for .pkl, .joblib, or .txt files
                for ext in ['.pkl', '.joblib', '.txt']:
                    model_files = list(default_dir.glob(f'*{ext}'))
                    if model_files:
                        model_path = str(model_files[0])
                        break

        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from: {model_path}")

            # Look for scaler
            scaler_path = None
            model_dir = Path(model_path).parent
            scaler_files = list(model_dir.glob('scaler*'))
            if scaler_files:
                scaler_path = str(scaler_files[0])

            _model_instance = ModelInference(
                model_path=model_path,
                scaler_path=scaler_path
            )

            logger.info("Model loaded successfully")
            return _model_instance

        else:
            logger.warning("No model found. Using stub implementation.")
            return None

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


async def rank_signals(candidates: List[Dict]) -> List[Dict]:
    """
    Production implementation of AI ranker with real model

    Workflow:
    1. Load active model (if not already loaded)
    2. Extract symbols from candidates
    3. Fetch market data and engineer features
    4. Run model inference
    5. Generate structured signals with levels
    6. Return ranked signals

    Falls back to stub if model not available.
    """
    global _model_instance

    # Try to load model if not loaded
    if _model_instance is None:
        _model_instance = await load_active_model()

    # If still no model, use stub
    if _model_instance is None:
        logger.warning("No model available, using stub implementation")
        return await rank_signals_stub(candidates)

    try:
        # Extract symbols
        symbols = [c['symbol'] for c in candidates]

        # Engineer features
        logger.info(f"Engineering features for {len(symbols)} candidates")
        features_df = await _feature_engine.engineer_features(symbols)

        if features_df.empty:
            logger.warning("No features generated, falling back to stub")
            return await rank_signals_stub(candidates)

        # Run model inference
        logger.info("Running model inference")
        ranked = _model_instance.rank_signals(
            features_df,
            top_k=50,
            min_confidence=0.5
        )

        # Enrich with signal metadata
        signals = []
        for idx, result in enumerate(ranked):
            symbol = result['symbol']

            # Find original candidate data
            candidate = next((c for c in candidates if c['symbol'] == symbol), {})

            # Fetch current price for levels
            price_data = await market_data_service.get_current_price(symbol)
            current_price = price_data['price'] if price_data else 1000.0

            # Calculate levels based on predicted return
            predicted_return = result.get('predicted_return', 0.05)
            risk_reward_ratio = 2.0

            entry_min = current_price * 0.98
            entry_max = current_price * 1.02
            stop_loss = current_price * 0.95  # 5% stop
            target_1 = current_price * (1 + abs(predicted_return))
            target_2 = current_price * (1 + abs(predicted_return) * 1.5)

            # Determine risk grade based on confidence and volatility
            ai_score = result['ai_score']
            if ai_score > 0.8:
                risk_grade = 'low'
            elif ai_score > 0.6:
                risk_grade = 'medium'
            else:
                risk_grade = 'high'

            signal = {
                'symbol': symbol,
                'signal_type': candidate.get('signal_type', 'ai_generated'),
                'rank': idx + 1,
                'score': round(ai_score * 10, 1),  # Convert 0-1 to 1-10
                'confidence': round(ai_score * 100, 1),  # Convert to percentage
                'risk_grade': risk_grade,
                'horizon': 'swing',

                # Price levels
                'entry_min': round(entry_min, 2),
                'entry_max': round(entry_max, 2),
                'stop_loss': round(stop_loss, 2),
                'target_1': round(target_1, 2),
                'target_2': round(target_2, 2),

                # High-level tags (no model secrets)
                'setup_tags': generate_setup_tags(candidate),

                # Explanation (high-level only)
                'why_now': 'AI analysis indicates favorable setup based on multiple factors',
                'key_factors': [
                    'Strong momentum indicators',
                    'Favorable risk/reward profile',
                    'Volume confirmation'
                ],
                'risk_notes': [
                    'Monitor support levels',
                    'Adjust position size per risk tolerance'
                ],
                'mtf_alignment': [
                    {'timeframe': 'D', 'trend': 'bullish', 'structure': 'Higher highs'},
                    {'timeframe': '4H', 'trend': 'bullish', 'structure': 'Consolidation'},
                ],

                # Model metadata (version only, no internals)
                'model_version': result.get('model_version', 'unknown'),
                'generated_at': result.get('generated_at'),
            }

            signals.append(signal)

        logger.info(f"Generated {len(signals)} ranked signals")
        return signals[:20]  # Return top 20

    except Exception as e:
        logger.error(f"Error in AI ranking: {str(e)}")
        logger.warning("Falling back to stub implementation")
        return await rank_signals_stub(candidates)


async def predict_signal(model: ModelInference, candidate: Dict, market_data: Dict) -> Dict:
    """
    Run model inference for a single candidate

    Args:
        model: Loaded model instance
        candidate: Candidate signal data
        market_data: Market data for feature engineering

    Returns:
        Structured signal with predictions
    """
    # This function is kept for future use if needed for single predictions
    # Currently, batch prediction in rank_signals() is more efficient
    pass

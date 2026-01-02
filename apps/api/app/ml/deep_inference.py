"""
Deep Learning Model Inference for StockFormer and TFT

Uses the unified feature pipeline to ensure training/inference consistency.
Handles loading PyTorch models and running predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from app.ml.unified_features import UnifiedFeaturePipeline

logger = logging.getLogger(__name__)


class DeepModelInference:
    """
    Inference for deep learning models (StockFormer, TFT)

    Uses UnifiedFeaturePipeline to ensure features match training
    """

    def __init__(
        self,
        stockformer_path: Optional[str] = None,
        tft_path: Optional[str] = None,
        veto_path: Optional[str] = None,
        device: str = 'cpu',
        enable_kronos: bool = True
    ):
        """
        Initialize deep model inference

        Args:
            stockformer_path: Path to StockFormer checkpoint (.pt/.pth)
            tft_path: Path to TFT checkpoint (.pt/.pth)
            veto_path: Path to LightGBM veto model (.txt)
            device: 'cpu' or 'cuda'
            enable_kronos: Whether to use Kronos embeddings
        """
        self.device = device
        self.stockformer = None
        self.tft = None
        self.veto = None

        # Initialize unified feature pipeline
        logger.info(f"Initializing unified feature pipeline on {device}...")
        self.feature_pipeline = UnifiedFeaturePipeline(
            device=device,
            enable_kronos=enable_kronos
        )
        logger.info("✓ Feature pipeline ready")

        # Load models
        if stockformer_path:
            self.load_stockformer(stockformer_path)
        if tft_path:
            self.load_tft(tft_path)
        if veto_path:
            self.load_veto(veto_path)

    def load_stockformer(self, model_path: str) -> None:
        """Load StockFormer model from checkpoint"""
        try:
            from app.ml.stockformer.model import StockFormer

            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract hyperparams from checkpoint
            if 'hyperparams' in checkpoint:
                hp = checkpoint['hyperparams']
            else:
                # Default hyperparams
                hp = {
                    'd_model': 256,
                    'n_heads': 8,
                    'n_layers': 4,
                    'dim_feedforward': 1024,
                    'dropout': 0.1
                }

            # Initialize model
            self.stockformer = StockFormer(
                price_dim=5,  # OHLCV
                kron_dim=512,
                context_dim=29,
                d_model=hp.get('d_model', 256),
                n_heads=hp.get('n_heads', 8),
                n_layers=hp.get('n_layers', 4),
                dim_feedforward=hp.get('dim_feedforward', 1024),
                dropout=hp.get('dropout', 0.1),
                n_horizons=3  # 3, 5, 10 days
            ).to(self.device)

            # Load weights
            self.stockformer.load_state_dict(checkpoint['model_state_dict'])
            self.stockformer.eval()

            logger.info(f"✓ Loaded StockFormer from {model_path}")

        except Exception as e:
            logger.error(f"Error loading StockFormer: {e}")
            raise

    def load_tft(self, model_path: str) -> None:
        """Load TFT model from checkpoint"""
        try:
            from app.ml.tft.model import TemporalFusionTransformer

            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract hyperparams
            if 'hyperparams' in checkpoint:
                hp = checkpoint['hyperparams']
            else:
                hp = {
                    'd_model': 256,
                    'n_heads': 8,
                    'n_layers': 4,
                    'dropout': 0.1
                }

            # Initialize model
            self.tft = TemporalFusionTransformer(
                price_dim=5,
                kron_dim=512,
                context_dim=29,
                d_model=hp.get('d_model', 256),
                n_heads=hp.get('n_heads', 8),
                n_layers=hp.get('n_layers', 4),
                dropout=hp.get('dropout', 0.1),
                n_horizons=3
            ).to(self.device)

            # Load weights
            self.tft.load_state_dict(checkpoint['model_state_dict'])
            self.tft.eval()

            logger.info(f"✓ Loaded TFT from {model_path}")

        except Exception as e:
            logger.error(f"Error loading TFT: {e}")
            raise

    def load_veto(self, model_path: str) -> None:
        """Load LightGBM veto model"""
        try:
            import lightgbm as lgb
            self.veto = lgb.Booster(model_file=model_path)
            logger.info(f"✓ Loaded veto model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading veto model: {e}")
            raise

    def predict_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 120
    ) -> Dict[str, Any]:
        """
        Run predictions for a single symbol

        Args:
            df: DataFrame with OHLCV data (must have >= lookback rows)
            symbol: Stock symbol
            lookback: Number of days to use (default: 120)

        Returns:
            dict with predictions from all models
        """
        try:
            # Compute features using unified pipeline
            features = self.feature_pipeline.compute_features(df, lookback=lookback)

            # Prepare tensors
            x_price = torch.tensor(
                features['ohlcv_norm'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # (1, 120, 5)

            x_kron = torch.tensor(
                features['kronos_emb'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # (1, 512)

            x_ctx = torch.tensor(
                features['context_vec'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # (1, 29)

            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
            }

            # StockFormer prediction
            if self.stockformer is not None:
                with torch.no_grad():
                    sf_out = self.stockformer(x_price, x_kron, x_ctx)

                result['stockformer'] = {
                    'returns': sf_out['ret'].cpu().numpy()[0].tolist(),  # [3d, 5d, 10d]
                    'up_prob': sf_out['up_prob'].cpu().numpy()[0].tolist(),
                    'direction': (sf_out['up_prob'].cpu().numpy()[0] > 0.5).astype(int).tolist()
                }

            # TFT prediction
            if self.tft is not None:
                with torch.no_grad():
                    tft_out = self.tft(x_price, x_kron, x_ctx)

                result['tft'] = {
                    'returns': tft_out['ret'].cpu().numpy()[0].tolist(),
                    'volatility_upper': tft_out['vol_upper'].cpu().numpy()[0].tolist(),
                    'volatility_lower': tft_out['vol_lower'].cpu().numpy()[0].tolist()
                }

            # Veto model (if loaded)
            if self.veto is not None and self.stockformer is not None and self.tft is not None:
                from app.ml.preprocess.normalize import build_veto_vec

                veto_features = build_veto_vec(
                    sf_out={
                        'prob': sf_out['up_prob'].cpu().numpy(),
                        'ret': sf_out['ret'].cpu().numpy()
                    },
                    tft_out={
                        'ret': tft_out['ret'].cpu().numpy(),
                        'vol_upper': tft_out['vol_upper'].cpu().numpy(),
                        'vol_lower': tft_out['vol_lower'].cpu().numpy()
                    },
                    smc_vec=features['smc_vec'],
                    tf_align=features['alignment_vec'],
                    ta_vec=features['ta_vec'],
                    raw_features={}
                )

                veto_pred = self.veto.predict(veto_features)[0]
                result['veto'] = {
                    'pass': bool(veto_pred > 0.5),
                    'confidence': float(veto_pred)
                }

            # Compute consensus score
            if 'stockformer' in result and 'tft' in result:
                sf_bullish = np.mean(np.array(result['stockformer']['up_prob']) > 0.5)
                tft_bullish = np.mean(np.array(result['tft']['returns']) > 0)

                result['consensus'] = {
                    'score': (sf_bullish + tft_bullish) / 2,
                    'direction': 'bullish' if (sf_bullish + tft_bullish) > 1 else 'bearish',
                    'confidence': abs(sf_bullish - 0.5) * 2  # 0 to 1
                }

            return result

        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            raise

    def rank_signals(
        self,
        symbols_df: pd.DataFrame,
        top_k: int = 50,
        min_confidence: float = 0.5,
        use_veto: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rank multiple symbols based on model predictions

        Args:
            symbols_df: DataFrame with 'symbol' column and OHLCV data for each
                       OR dict of {symbol: dataframe}
            top_k: Number of top signals to return
            min_confidence: Minimum consensus confidence (0-1)
            use_veto: Whether to filter using veto model

        Returns:
            List of ranked signals sorted by consensus score
        """
        results = []

        # Handle dict input
        if isinstance(symbols_df, dict):
            symbols_data = symbols_df
        else:
            # Group by symbol
            symbols_data = {
                symbol: group for symbol, group in symbols_df.groupby('symbol')
            }

        for symbol, df in symbols_data.items():
            try:
                # Skip if not enough data
                if len(df) < 120:
                    logger.warning(f"Skipping {symbol}: only {len(df)} days of data")
                    continue

                # Get prediction
                pred = self.predict_symbol(df, symbol)

                # Apply veto filter
                if use_veto and 'veto' in pred:
                    if not pred['veto']['pass']:
                        logger.debug(f"Veto rejected {symbol}")
                        continue

                # Apply confidence filter
                if 'consensus' in pred:
                    if pred['consensus']['confidence'] < min_confidence:
                        continue

                    results.append({
                        'symbol': symbol,
                        'ai_score': pred['consensus']['score'],
                        'confidence': pred['consensus']['confidence'],
                        'direction': pred['consensus']['direction'],
                        'predicted_return_3d': pred.get('stockformer', {}).get('returns', [None])[0],
                        'predicted_return_5d': pred.get('stockformer', {}).get('returns', [None, None])[1] if len(pred.get('stockformer', {}).get('returns', [])) > 1 else None,
                        'predicted_return_10d': pred.get('stockformer', {}).get('returns', [None, None, None])[2] if len(pred.get('stockformer', {}).get('returns', [])) > 2 else None,
                        'veto_pass': pred.get('veto', {}).get('pass', True),
                        'generated_at': pred['timestamp'],
                        'full_prediction': pred
                    })

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        # Sort by consensus score
        results.sort(key=lambda x: x['ai_score'], reverse=True)

        return results[:top_k]


if __name__ == "__main__":
    """Test deep model inference"""
    print("Testing DeepModelInference...")

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

    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    df.set_index('date', inplace=True)

    # Initialize inference (without loading models, just test feature pipeline)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference = DeepModelInference(device=device, enable_kronos=True)

    print(f"\n✓ DeepModelInference initialized on {device}")
    print("✓ Unified feature pipeline ready")
    print("\nNote: To run actual predictions, load trained models:")
    print("  inference.load_stockformer('path/to/stockformer.pt')")
    print("  inference.load_tft('path/to/tft.pt')")
    print("  inference.load_veto('path/to/veto.txt')")

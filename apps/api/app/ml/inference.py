"""
Model Inference Module for AI Trader

⚠️ DEPRECATED for Deep Learning models (StockFormer, TFT)
Use deep_inference.py instead for PyTorch models.

This module is kept ONLY for sklearn/LightGBM model compatibility.
For production inference with StockFormer/TFT, use:
    from app.ml.deep_inference import DeepModelInference

Handles loading trained models and running predictions.
"""

import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Handles model loading and inference for signal ranking

    Supports multiple model formats:
    - Scikit-learn models (pickle/joblib)
    - LightGBM/XGBoost models
    - Custom ensemble models
    """

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Initialize model inference

        Args:
            model_path: Path to trained model file
            scaler_path: Path to feature scaler (optional)
        """
        self.model = None
        self.scaler = None
        self.model_version = None
        self.model_metadata = {}

        if model_path:
            self.load_model(model_path, scaler_path)

    def load_model(self, model_path: str, scaler_path: Optional[str] = None) -> None:
        """
        Load trained model from disk

        Args:
            model_path: Path to model file (.pkl, .joblib, .txt for LightGBM)
            scaler_path: Path to scaler file (optional)
        """
        try:
            model_file = Path(model_path)

            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model based on file extension
            if model_file.suffix in ['.pkl', '.pickle']:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded pickle model from {model_path}")

            elif model_file.suffix == '.joblib':
                self.model = joblib.load(model_path)
                logger.info(f"Loaded joblib model from {model_path}")

            elif model_file.suffix == '.txt':
                # LightGBM text format
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=model_path)
                logger.info(f"Loaded LightGBM model from {model_path}")

            else:
                raise ValueError(f"Unsupported model format: {model_file.suffix}")

            # Load scaler if provided
            if scaler_path:
                scaler_file = Path(scaler_path)
                if scaler_file.exists():
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info(f"Loaded scaler from {scaler_path}")

            # Extract model metadata
            self.model_version = model_file.stem
            self.model_metadata = {
                'model_path': str(model_path),
                'loaded_at': datetime.now().isoformat(),
                'model_type': type(self.model).__name__,
            }

            logger.info(f"Model loaded successfully: {self.model_metadata}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference

        Args:
            features: Feature array (n_samples, n_features)

        Returns:
            Tuple of (predictions, probabilities/confidence scores)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Apply scaling if scaler exists
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                # Classifier with probability output
                predictions = self.model.predict(features_scaled)
                probabilities = self.model.predict_proba(features_scaled)[:, 1]
            elif hasattr(self.model, 'predict'):
                # Regressor or LightGBM
                predictions = self.model.predict(features_scaled)
                probabilities = predictions  # Use predictions as confidence
            else:
                raise ValueError("Model does not support predict() method")

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def rank_signals(
        self,
        features_df: pd.DataFrame,
        top_k: int = 50,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate signals based on model predictions

        Args:
            features_df: DataFrame with engineered features and 'symbol' column
            top_k: Number of top signals to return
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            List of ranked signals with metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Extract symbols for later
            symbols = features_df['symbol'].values

            # Prepare features for prediction (exclude non-feature columns)
            feature_cols = [col for col in features_df.columns
                          if col not in ['symbol', 'timestamp']]
            X = features_df[feature_cols].fillna(0).values

            # Get predictions
            predictions, confidences = self.predict(X)

            # Create ranked results
            results = []
            for i, symbol in enumerate(symbols):
                score = float(confidences[i])
                prediction = float(predictions[i])

                # Skip low confidence signals
                if score < min_confidence:
                    continue

                results.append({
                    'symbol': symbol,
                    'ai_score': score,
                    'predicted_return': prediction,
                    'model_version': self.model_version,
                    'generated_at': datetime.now().isoformat(),
                })

            # Sort by score descending
            results.sort(key=lambda x: x['ai_score'], reverse=True)

            # Return top K
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error ranking signals: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from model (if available)

        Returns:
            Dictionary of feature names and importance scores
        """
        if self.model is None:
            return None

        try:
            if hasattr(self.model, 'feature_importances_'):
                # Scikit-learn style models
                importances = self.model.feature_importances_
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                return dict(zip(feature_names, importances))

            elif hasattr(self.model, 'feature_importance'):
                # LightGBM/XGBoost
                importances = self.model.feature_importance()
                feature_names = self.model.feature_name()
                return dict(zip(feature_names, importances))

            else:
                logger.warning("Model does not support feature importance")
                return None

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None


class EnsembleInference:
    """
    Ensemble model inference combining multiple models

    Useful for combining different strategies or model types
    """

    def __init__(self, model_configs: List[Dict[str, str]]):
        """
        Initialize ensemble with multiple models

        Args:
            model_configs: List of dicts with 'model_path' and optional 'scaler_path', 'weight'
        """
        self.models = []
        self.weights = []

        for config in model_configs:
            model = ModelInference(
                model_path=config['model_path'],
                scaler_path=config.get('scaler_path')
            )
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        logger.info(f"Initialized ensemble with {len(self.models)} models")

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble prediction using weighted average

        Args:
            features: Feature array

        Returns:
            Weighted average predictions and confidences
        """
        all_predictions = []
        all_confidences = []

        for model in self.models:
            preds, confs = model.predict(features)
            all_predictions.append(preds)
            all_confidences.append(confs)

        # Weighted average
        weighted_preds = np.average(all_predictions, axis=0, weights=self.weights)
        weighted_confs = np.average(all_confidences, axis=0, weights=self.weights)

        return weighted_preds, weighted_confs

    def rank_signals(
        self,
        features_df: pd.DataFrame,
        top_k: int = 50,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Use first model's rank_signals with ensemble predictions"""
        # Use the ensemble predict method
        symbols = features_df['symbol'].values
        feature_cols = [col for col in features_df.columns
                      if col not in ['symbol', 'timestamp']]
        X = features_df[feature_cols].fillna(0).values

        predictions, confidences = self.predict(X)

        results = []
        for i, symbol in enumerate(symbols):
            score = float(confidences[i])
            if score >= min_confidence:
                results.append({
                    'symbol': symbol,
                    'ai_score': score,
                    'predicted_return': float(predictions[i]),
                    'model_version': 'ensemble',
                    'generated_at': datetime.now().isoformat(),
                })

        results.sort(key=lambda x: x['ai_score'], reverse=True)
        return results[:top_k]

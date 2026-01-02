"""
Unified Feature Pipeline for Training and Inference

This module ensures that training and real-time inference use IDENTICAL features.
No more feature mismatch between training and production!

Features computed:
- OHLCV normalization: (120, 5) normalized to [-1, 1]
- Kronos embeddings: (512,) time series foundation model embeddings
- Context vector: (29,) = 5 MTF + 12 SMC + 12 TA

Usage:
    # Training
    pipeline = UnifiedFeaturePipeline(device='cuda')
    features = pipeline.compute_features(df, lookback=120)

    # Inference
    pipeline = UnifiedFeaturePipeline(device='cpu')
    features = pipeline.compute_features(df, lookback=120)
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional
from pathlib import Path

from app.services.feature_engine import compute_ta_features, compute_smc_features
from app.services.kronos_loader import load_kronos_hf
from app.ml.preprocess.normalize import (
    normalize_ohlcv_120,
    build_tf_align_vec,
    build_smc_vec,
    build_ta_vec,
)


class UnifiedFeaturePipeline:
    """
    Unified feature pipeline that ensures training and inference consistency

    This pipeline computes:
    1. OHLCV normalized to [-1, 1] range
    2. Kronos (Chronos) time series embeddings
    3. Multi-timeframe alignment (5D)
    4. Smart Money Concepts features (12D)
    5. Technical Analysis features (12D)

    Final output:
    - ohlcv_norm: (batch, 120, 5)
    - kronos_emb: (batch, 512)
    - context_vec: (batch, 29)
    """

    def __init__(
        self,
        device: str = 'cpu',
        max_context: int = 512,
        lookback: int = 120,
        enable_kronos: bool = True
    ):
        """
        Initialize the unified feature pipeline

        Args:
            device: 'cpu' or 'cuda'
            max_context: Maximum context length for Kronos
            lookback: Number of historical days to use (default: 120)
            enable_kronos: Whether to compute Kronos embeddings (set False for faster inference)
        """
        self.device = device
        self.max_context = max_context
        self.lookback = lookback
        self.enable_kronos = enable_kronos

        # Load Kronos model if enabled
        self.kronos = None
        if enable_kronos:
            print(f"Loading Kronos foundation model on {device}...")
            self.kronos = load_kronos_hf(device=device, max_context=max_context)
            print("✓ Kronos model loaded")

    def _prep_window(self, window: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare window for feature computation"""
        if window.empty:
            return None
        df = window.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None
        return df

    def compute_alignment(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        Compute multi-timeframe alignment

        Returns dict with:
        - monthly_bias: +1.0 (bull) or -1.0 (bear)
        - weekly_bias
        - daily_bias
        - h4_align
        - h1_align
        """
        base = self._prep_window(window)
        if base is None:
            return {
                'monthly_bias': 0.0,
                'weekly_bias': 0.0,
                'daily_bias': 0.0,
                'h4_align': 0.0,
                'h1_align': 0.0
            }

        core = base[['open', 'high', 'low', 'close', 'volume']]

        # Resample to different timeframes
        wk = core.resample('W-FRI').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        mo = core.resample('ME').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        h4 = core.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        h1 = core.copy()  # Daily data, sparse for 1H

        def bias(df: pd.DataFrame) -> float:
            """Compute bias: +1 if EMA fast > slow, -1 otherwise"""
            enriched = compute_ta_features(df)
            if enriched.empty:
                return 0.0
            latest = enriched.iloc[-1]
            ema_fast = latest.get('ema_fast', 0)
            ema_slow = latest.get('ema_slow', 0)
            return 1.0 if ema_fast > ema_slow else -1.0

        return {
            'monthly_bias': bias(mo),
            'weekly_bias': bias(wk),
            'daily_bias': bias(core),
            'h4_align': bias(h4),
            'h1_align': bias(h1),
        }

    def compute_feature_dict(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        Compute full feature dictionary with TA and SMC features

        Returns dict with all computed features
        """
        base = self._prep_window(window)
        if base is None:
            return {}

        # Compute TA features
        enriched = compute_ta_features(base)

        # Compute SMC features
        enriched = compute_smc_features(enriched)

        if enriched.empty:
            return {}

        # Get latest row as dict
        latest = enriched.iloc[-1].to_dict()

        # Add raw OHLCV
        latest.update({
            'open': float(base.iloc[-1]['open']),
            'high': float(base.iloc[-1]['high']),
            'low': float(base.iloc[-1]['low']),
            'close': float(base.iloc[-1]['close']),
            'volume': float(base.iloc[-1]['volume']),
        })

        return latest

    def kronos_embed(self, batch_norm: np.ndarray) -> np.ndarray:
        """
        Compute Kronos embeddings for normalized OHLCV

        Args:
            batch_norm: (batch, 120, 5) normalized OHLCV

        Returns:
            embeddings: (batch, 512) Kronos embeddings
        """
        if self.kronos is None:
            # Return zeros if Kronos is disabled
            batch_size = batch_norm.shape[0]
            return np.zeros((batch_size, 512), dtype=np.float32)

        # Convert to tensor
        x = torch.tensor(batch_norm, dtype=torch.float32, device=self.device)

        # Pad amount channel if needed (Kronos expects 6 features)
        if x.shape[-1] == 5:
            amt = torch.zeros(x.shape[0], x.shape[1], 1, device=self.device)
            x = torch.cat([x, amt], dim=-1)

        # Get embeddings
        with torch.no_grad():
            z = self.kronos.tokenizer.embed(x)
            if isinstance(z, tuple):
                z = z[0]

        # Pool to (batch, 512)
        emb = z.mean(dim=1).detach().cpu().numpy().astype(np.float32)

        # Ensure shape is exactly (batch, 512)
        if emb.shape[1] < 512:
            pad = np.zeros((emb.shape[0], 512 - emb.shape[1]), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=1)
        elif emb.shape[1] > 512:
            emb = emb[:, :512]

        return emb

    def compute_features(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for a single window

        Args:
            df: DataFrame with OHLCV data (must have at least `lookback` rows)
            lookback: Number of days to use (default: self.lookback = 120)

        Returns:
            dict with:
                - ohlcv_norm: (120, 5) normalized OHLCV
                - kronos_emb: (512,) embeddings
                - context_vec: (29,) context [5 MTF + 12 SMC + 12 TA]
                - alignment: (5,) MTF alignment vector
                - smc_vec: (12,) SMC feature vector
                - ta_vec: (12,) TA feature vector
        """
        if lookback is None:
            lookback = self.lookback

        # Ensure we have enough data
        if len(df) < lookback:
            raise ValueError(f"Need at least {lookback} rows, got {len(df)}")

        # Take last `lookback` rows
        window = df.iloc[-lookback:].copy()

        # 1. Normalize OHLCV
        ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
        ohlcv_norm = normalize_ohlcv_120(ohlcv)

        # 2. Compute alignment
        alignment_dict = self.compute_alignment(window)
        alignment_vec = build_tf_align_vec(alignment_dict)

        # 3. Compute TA and SMC features
        feat_dict = self.compute_feature_dict(window)
        smc_vec = build_smc_vec(feat_dict)
        ta_vec = build_ta_vec(feat_dict)

        # 4. Build context vector
        context_vec = np.concatenate([alignment_vec, smc_vec, ta_vec]).astype(np.float32)

        # 5. Compute Kronos embeddings
        kronos_emb = self.kronos_embed(np.expand_dims(ohlcv_norm, axis=0))[0]

        return {
            'ohlcv_norm': ohlcv_norm,  # (120, 5)
            'kronos_emb': kronos_emb,   # (512,)
            'context_vec': context_vec, # (29,)
            'alignment_vec': alignment_vec,  # (5,)
            'smc_vec': smc_vec,         # (12,)
            'ta_vec': ta_vec,           # (12,)
        }

    def compute_batch_features(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
        batch_size: int = 64
    ) -> Dict[str, np.ndarray]:
        """
        Compute features for multiple windows in a DataFrame

        This is useful for training where you want to extract all possible
        windows from a long time series.

        Args:
            df: DataFrame with OHLCV data
            lookback: Window size (default: self.lookback = 120)
            batch_size: Batch size for Kronos embedding (for efficiency)

        Returns:
            dict with batched features:
                - ohlcv_norm: (n_windows, 120, 5)
                - kronos_emb: (n_windows, 512)
                - context_vec: (n_windows, 29)
        """
        if lookback is None:
            lookback = self.lookback

        if len(df) < lookback:
            raise ValueError(f"Need at least {lookback} rows, got {len(df)}")

        # Prepare batches
        batch_ohlcv = []
        batch_context = []

        # Slide window through dataframe
        for i in range(lookback - 1, len(df)):
            window = df.iloc[i - lookback + 1 : i + 1]

            # Normalize OHLCV
            ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
            if ohlcv.shape[0] != lookback:
                continue
            ohlcv_norm = normalize_ohlcv_120(ohlcv)

            # Compute alignment
            alignment = build_tf_align_vec(self.compute_alignment(window))

            # Compute features
            feat_dict = self.compute_feature_dict(window)
            smc_vec = build_smc_vec(feat_dict)
            ta_vec = build_ta_vec(feat_dict)

            # Context vector
            context = np.concatenate([alignment, smc_vec, ta_vec]).astype(np.float32)

            batch_ohlcv.append(ohlcv_norm)
            batch_context.append(context)

        # Convert to arrays
        ohlcv_array = np.stack(batch_ohlcv, axis=0)  # (n, 120, 5)
        context_array = np.stack(batch_context, axis=0)  # (n, 29)

        # Compute Kronos embeddings in batches
        kronos_embs = []
        for i in range(0, len(ohlcv_array), batch_size):
            batch = ohlcv_array[i:i+batch_size]
            emb = self.kronos_embed(batch)
            kronos_embs.append(emb)

        kronos_array = np.concatenate(kronos_embs, axis=0)  # (n, 512)

        return {
            'ohlcv_norm': ohlcv_array,
            'kronos_emb': kronos_array,
            'context_vec': context_array,
        }


if __name__ == "__main__":
    """Test the unified feature pipeline"""
    print("Testing Unified Feature Pipeline...")

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
    df.set_index('date', inplace=True)

    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = UnifiedFeaturePipeline(device=device, enable_kronos=True)

    # Test single window
    print("\n=== Testing Single Window ===")
    features = pipeline.compute_features(df, lookback=120)

    print(f"OHLCV normalized shape: {features['ohlcv_norm'].shape}")  # (120, 5)
    print(f"Kronos embedding shape: {features['kronos_emb'].shape}")  # (512,)
    print(f"Context vector shape: {features['context_vec'].shape}")    # (29,)
    print(f"  - Alignment: {features['alignment_vec'].shape}")         # (5,)
    print(f"  - SMC: {features['smc_vec'].shape}")                     # (12,)
    print(f"  - TA: {features['ta_vec'].shape}")                       # (12,)

    # Test batch computation
    print("\n=== Testing Batch Computation ===")
    batch_features = pipeline.compute_batch_features(df, lookback=120, batch_size=16)

    n_windows = len(df) - 120 + 1
    print(f"Expected windows: {n_windows}")
    print(f"OHLCV batch shape: {batch_features['ohlcv_norm'].shape}")
    print(f"Kronos batch shape: {batch_features['kronos_emb'].shape}")
    print(f"Context batch shape: {batch_features['context_vec'].shape}")

    # Verify shapes
    assert batch_features['ohlcv_norm'].shape == (n_windows, 120, 5)
    assert batch_features['kronos_emb'].shape == (n_windows, 512)
    assert batch_features['context_vec'].shape == (n_windows, 29)

    print("\n✓ All tests passed!")
    print("\n=== Feature Summary ===")
    print(f"Context vector breakdown (29D):")
    print(f"  - MTF Alignment: 5D (monthly, weekly, daily, 4H, 1H bias)")
    print(f"  - SMC Features: 12D (OB, FVG, liquidity, BOS, CHOCH)")
    print(f"  - TA Features: 12D (RSI, MACD, BB, ATR, ADX, OBV, VWAP, EMAs)")
    print(f"\nThis pipeline ensures training and inference use IDENTICAL features!")

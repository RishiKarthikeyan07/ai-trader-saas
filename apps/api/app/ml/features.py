"""
Feature Engineering Pipeline for AI Trader

Handles feature extraction from raw market data for model inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import yfinance as yf


class FeatureEngine:
    """
    Feature engineering for swing trading signals

    Extracts technical indicators and market features from raw price data.
    """

    def __init__(self):
        self.required_features = [
            'symbol',
            'close',
            'volume',
            'rsi_14',
            'macd',
            'macd_signal',
            'bb_upper',
            'bb_lower',
            'atr_14',
            'adx_14',
            'obv',
            'vwap',
            'ema_9',
            'ema_21',
            'ema_50',
            'volume_sma_20',
        ]

    async def engineer_features(self, tickers: List[str], lookback_days: int = 60) -> pd.DataFrame:
        """
        Fetch market data and compute technical indicators

        Args:
            tickers: List of stock symbols (NSE format)
            lookback_days: Number of days of historical data to fetch

        Returns:
            DataFrame with computed features for each ticker
        """
        features_list = []

        for ticker in tickers:
            try:
                # Convert NSE ticker to Yahoo Finance format (e.g., RELIANCE.NS)
                yf_ticker = f"{ticker}.NS"

                # Fetch historical data
                df = await self._fetch_price_data(yf_ticker, lookback_days)

                if df is None or len(df) < 50:
                    continue

                # Compute technical indicators
                df = self._add_technical_indicators(df)

                # Get latest row as features
                latest = df.iloc[-1].to_dict()
                latest['symbol'] = ticker
                latest['timestamp'] = datetime.now().isoformat()

                features_list.append(latest)

            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue

        return pd.DataFrame(features_list)

    async def _fetch_price_data(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Fetch historical price data from yfinance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                return None

            # Rename columns to lowercase
            df.columns = df.columns.str.lower()

            return df

        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            return None

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators"""

        # RSI (14-period)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)

        # MACD (12, 26, 9)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands (20-period, 2 std)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']

        # ATR (14-period)
        df['tr'] = self._calculate_true_range(df)
        df['atr_14'] = df['tr'].rolling(window=14).mean()

        # ADX (14-period)
        df['adx_14'] = self._calculate_adx(df, 14)

        # OBV (On-Balance Volume)
        df['obv'] = self._calculate_obv(df)

        # VWAP (Volume-Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # EMAs
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price momentum
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range for ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._calculate_true_range(df)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

    def prepare_for_inference(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature DataFrame for model inference

        Args:
            df: DataFrame with computed features

        Returns:
            Numpy array ready for model prediction
        """
        # Select only required features (excluding symbol, timestamp)
        feature_cols = [col for col in self.required_features if col not in ['symbol', 'timestamp']]

        # Handle missing values
        df_clean = df[feature_cols].fillna(0)

        return df_clean.values

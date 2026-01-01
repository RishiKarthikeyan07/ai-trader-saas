"""
Market Data Service for Real-Time Price Fetching

Provides live and historical market data for signal confirmation and monitoring.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Fetches real-time and historical market data from various sources

    Primary source: yfinance (Yahoo Finance)
    Fallback: Can be extended to Alpha Vantage, NSE API, etc.
    """

    def __init__(self, cache_ttl_seconds: int = 60):
        """
        Initialize market data service

        Args:
            cache_ttl_seconds: Time-to-live for cached price data (default 60s)
        """
        self.cache_ttl = cache_ttl_seconds
        self._price_cache = {}
        self._cache_timestamps = {}

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price for a symbol

        Args:
            symbol: Stock symbol (NSE format, e.g., "RELIANCE")

        Returns:
            Dict with current price, change, volume, etc.
        """
        try:
            # Convert to Yahoo Finance format
            yf_symbol = f"{symbol}.NS"

            # Check cache first
            cached = self._get_from_cache(yf_symbol)
            if cached:
                return cached

            # Fetch from yfinance
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            # Get latest quote
            hist = ticker.history(period='1d')
            if hist.empty:
                logger.warning(f"No price data for {symbol}")
                return None

            latest = hist.iloc[-1]

            price_data = {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'change_pct': self._calculate_change_pct(latest['Open'], latest['Close']),
                'timestamp': datetime.now().isoformat(),
                'market_cap': info.get('marketCap'),
                'previous_close': info.get('previousClose'),
            }

            # Cache the result
            self._cache_price(yf_symbol, price_data)

            return price_data

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return None

    async def get_batch_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get current prices for multiple symbols efficiently

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to price data
        """
        results = {}

        # Fetch in parallel with rate limiting
        tasks = []
        for symbol in symbols:
            task = self.get_current_price(symbol)
            tasks.append(task)

            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.1)

        prices = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, price_data in zip(symbols, prices):
            if isinstance(price_data, Exception):
                logger.error(f"Error fetching {symbol}: {price_data}")
                results[symbol] = None
            else:
                results[symbol] = price_data

        return results

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data

        Args:
            symbol: Stock symbol
            start_date: Start date for historical data
            end_date: End date (default: today)

        Returns:
            DataFrame with historical price data
        """
        try:
            yf_symbol = f"{symbol}.NS"
            end_date = end_date or datetime.now()

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            # Standardize column names
            df.columns = df.columns.str.lower()
            df['symbol'] = symbol

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    async def check_price_levels(
        self,
        symbol: str,
        target_price: float,
        stop_loss: float,
        entry_price: float
    ) -> Dict[str, Any]:
        """
        Check if current price has hit target or stop loss

        Args:
            symbol: Stock symbol
            target_price: Target price level
            stop_loss: Stop loss level
            entry_price: Entry price for reference

        Returns:
            Dict with hit status and current price
        """
        try:
            price_data = await self.get_current_price(symbol)

            if not price_data:
                return {
                    'symbol': symbol,
                    'current_price': None,
                    'target_hit': False,
                    'stop_hit': False,
                    'error': 'Failed to fetch price'
                }

            current_price = price_data['price']

            # Calculate returns
            entry_return = ((current_price - entry_price) / entry_price) * 100

            result = {
                'symbol': symbol,
                'current_price': current_price,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'current_return_pct': round(entry_return, 2),
                'target_hit': current_price >= target_price,
                'stop_hit': current_price <= stop_loss,
                'timestamp': price_data['timestamp'],
            }

            return result

        except Exception as e:
            logger.error(f"Error checking price levels for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'target_hit': False,
                'stop_hit': False
            }

    async def check_confirmation_criteria(
        self,
        symbol: str,
        signal_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check if signal meets confirmation criteria for WAIT → READY transition

        Criteria:
        1. Volume surge (> 1.5x average)
        2. Price within 2% of entry
        3. No major resistance breach
        4. Momentum maintained

        Args:
            symbol: Stock symbol
            signal_data: Original signal data with entry price, etc.

        Returns:
            Dict with confirmation flags
        """
        try:
            # Get current price and recent history
            current = await self.get_current_price(symbol)
            if not current:
                return {'confirmed': False, 'reason': 'No price data'}

            # Get 5-day history for volume check
            hist = await self.get_historical_data(
                symbol,
                start_date=datetime.now() - timedelta(days=5)
            )

            if hist is None or len(hist) < 3:
                return {'confirmed': False, 'reason': 'Insufficient history'}

            # Criterion 1: Volume surge
            avg_volume = hist['volume'].mean()
            volume_ratio = current['volume'] / avg_volume
            volume_surge = volume_ratio > 1.5

            # Criterion 2: Price within 2% of entry
            entry_price = signal_data.get('entry_price', current['price'])
            price_diff_pct = abs((current['price'] - entry_price) / entry_price) * 100
            price_within_range = price_diff_pct < 2.0

            # Criterion 3: Momentum maintained (close > open today)
            momentum_positive = current['price'] >= current['open']

            # Criterion 4: Not hitting stop loss
            stop_loss = signal_data.get('stop_loss', entry_price * 0.95)
            not_stopped = current['price'] > stop_loss

            # Overall confirmation
            confirmed = volume_surge and price_within_range and momentum_positive and not_stopped

            return {
                'confirmed': confirmed,
                'volume_surge': volume_surge,
                'volume_ratio': round(volume_ratio, 2),
                'price_within_range': price_within_range,
                'price_diff_pct': round(price_diff_pct, 2),
                'momentum_positive': momentum_positive,
                'not_stopped': not_stopped,
                'current_price': current['price'],
                'timestamp': current['timestamp'],
            }

        except Exception as e:
            logger.error(f"Error checking confirmation for {symbol}: {str(e)}")
            return {'confirmed': False, 'reason': str(e)}

    def _get_from_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price data from cache if not expired"""
        if symbol in self._price_cache:
            cache_time = self._cache_timestamps.get(symbol)
            if cache_time and (datetime.now() - cache_time).seconds < self.cache_ttl:
                return self._price_cache[symbol]
        return None

    def _cache_price(self, symbol: str, price_data: Dict[str, Any]) -> None:
        """Cache price data"""
        self._price_cache[symbol] = price_data
        self._cache_timestamps[symbol] = datetime.now()

    def _calculate_change_pct(self, open_price: float, close_price: float) -> float:
        """Calculate percentage change"""
        if open_price == 0:
            return 0.0
        return round(((close_price - open_price) / open_price) * 100, 2)

    def clear_cache(self) -> None:
        """Clear price cache"""
        self._price_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Price cache cleared")


# Global singleton instance
market_data_service = MarketDataService(cache_ttl_seconds=60)

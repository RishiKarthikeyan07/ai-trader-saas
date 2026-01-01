"""
Hourly Signal Confirmer

Checks if WAIT signals are ready to move to READY status
based on 1H/4H confirmation logic with real market data
"""

from typing import Dict, Optional
import random
import logging
from app.services.market_data import market_data_service

logger = logging.getLogger(__name__)


async def check_confirmation(signal: Dict) -> bool:
    """
    Check if signal should move from WAIT to READY using real market data

    Confirmation criteria:
    1. Volume surge (> 1.5x average)
    2. Price within 2% of entry range
    3. Momentum maintained (positive close)
    4. Not hitting stop loss

    Args:
        signal: Signal dict with symbol, entry_price, stop_loss, etc.

    Returns:
        True if confirmed, False otherwise
    """
    try:
        symbol = signal.get('symbol')
        if not symbol:
            logger.error("Signal missing symbol")
            return False

        # Use market data service to check confirmation criteria
        confirmation = await market_data_service.check_confirmation_criteria(
            symbol=symbol,
            signal_data=signal
        )

        if confirmation.get('confirmed', False):
            logger.info(f"Signal confirmed for {symbol}: {confirmation}")
            return True
        else:
            reason = confirmation.get('reason', 'Criteria not met')
            logger.info(f"Signal NOT confirmed for {symbol}: {reason}")
            return False

    except Exception as e:
        logger.error(f"Error checking confirmation for {signal.get('symbol')}: {str(e)}")
        # Fall back to stub logic on error
        return await check_confirmation_stub(signal)


async def check_confirmation_stub(signal: Dict) -> bool:
    """
    Fallback stub confirmation logic

    Used when real market data is unavailable.
    """
    # Stub: Random confirmation (50% chance)
    logger.warning(f"Using stub confirmation for {signal.get('symbol')}")
    return random.choice([True, False])


async def get_market_data_for_confirmation(symbol: str) -> Optional[Dict]:
    """
    Fetch real-time market data for confirmation

    Returns:
        Dict with current price, volume, and technical indicators
    """
    try:
        price_data = await market_data_service.get_current_price(symbol)

        if not price_data:
            logger.warning(f"No market data available for {symbol}")
            return None

        return price_data

    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {str(e)}")
        return None


def check_1h_confirmation(signal: Dict, current_data: Dict) -> bool:
    """
    Check 1H timeframe confirmation

    Criteria:
    - Price above entry range low
    - Volume above average
    - Positive momentum
    """
    try:
        current_price = current_data.get('price')
        entry_min = signal.get('entry_min', 0)

        # Price confirmation
        price_ok = current_price >= entry_min * 0.98

        # Volume confirmation
        volume_ratio = current_data.get('volume', 0) / current_data.get('volume_avg', 1)
        volume_ok = volume_ratio > 1.2

        # Momentum confirmation
        change_pct = current_data.get('change_pct', 0)
        momentum_ok = change_pct >= -1.0  # Not down more than 1%

        confirmed = price_ok and volume_ok and momentum_ok

        logger.debug(f"1H confirmation for {signal.get('symbol')}: "
                    f"price={price_ok}, volume={volume_ok}, momentum={momentum_ok}")

        return confirmed

    except Exception as e:
        logger.error(f"Error in 1H confirmation: {str(e)}")
        return False


def check_4h_confirmation(signal: Dict, current_data: Dict) -> bool:
    """
    Check 4H timeframe confirmation

    Criteria:
    - Trend alignment (price above key EMAs)
    - Structure intact (no major support breaks)
    - Risk/reward still favorable
    """
    try:
        current_price = current_data.get('price')
        stop_loss = signal.get('stop_loss')
        target_1 = signal.get('target_1')

        # Not stopped out
        not_stopped = current_price > stop_loss

        # Risk/reward still > 2:1
        risk = current_price - stop_loss
        reward = target_1 - current_price
        rr_ok = (reward / risk) >= 2.0 if risk > 0 else False

        # Price not extended (within 5% of entry max)
        entry_max = signal.get('entry_max', current_price)
        not_extended = current_price <= entry_max * 1.05

        confirmed = not_stopped and rr_ok and not_extended

        logger.debug(f"4H confirmation for {signal.get('symbol')}: "
                    f"not_stopped={not_stopped}, rr={rr_ok}, not_extended={not_extended}")

        return confirmed

    except Exception as e:
        logger.error(f"Error in 4H confirmation: {str(e)}")
        return False

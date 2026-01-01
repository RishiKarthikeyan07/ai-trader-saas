"""
Base Broker Connector Interface
All broker connectors must implement this interface
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class BrokerConnector(ABC):
    """
    Abstract base class for all broker connectors
    Provides unified interface for multi-broker support
    """

    def __init__(self, user_id: str, connection_data: Dict[str, Any]):
        self.user_id = user_id
        self.connection_data = connection_data
        self.broker_type = self._get_broker_type()

    @abstractmethod
    def _get_broker_type(self) -> str:
        """Return broker type identifier"""
        pass

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    @abstractmethod
    async def connect(self, auth_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Establish connection with broker
        Returns: {access_token, refresh_token, expires_at, ...}
        """
        pass

    @abstractmethod
    async def refresh(self) -> Dict[str, Any]:
        """
        Refresh authentication tokens
        Returns: {access_token, refresh_token, expires_at, ...}
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if connection is healthy
        Returns: True if healthy, False otherwise
        """
        pass

    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================

    @abstractmethod
    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place order with broker
        Args:
            order: {
                symbol: str,
                exchange: str (NSE/BSE),
                side: str (BUY/SELL),
                order_type: str (MARKET/LIMIT/SL/SL-M),
                product: str (MIS/CNC/NRML),
                quantity: int,
                price: float (optional),
                trigger_price: float (optional),
                tag: str (optional)
            }
        Returns: {
            broker_order_id: str,
            status: str,
            message: str,
            timestamp: str
        }
        """
        pass

    @abstractmethod
    async def modify_order(
        self, broker_order_id: str, changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Modify existing order
        Args:
            broker_order_id: Broker's order ID
            changes: Fields to modify (quantity, price, trigger_price)
        Returns: {broker_order_id, status, message}
        """
        pass

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> Dict[str, Any]:
        """
        Cancel order
        Returns: {broker_order_id, status, message}
        """
        pass

    @abstractmethod
    async def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for the day
        Returns: List of orders with status, fills, etc.
        """
        pass

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> Dict[str, Any]:
        """
        Get status of specific order
        Returns: {broker_order_id, status, filled_quantity, average_price, ...}
        """
        pass

    # ========================================================================
    # PORTFOLIO MANAGEMENT
    # ========================================================================

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        Returns: List of {symbol, quantity, average_price, pnl, ...}
        """
        pass

    @abstractmethod
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get long-term holdings
        Returns: List of {symbol, quantity, average_price, ...}
        """
        pass

    @abstractmethod
    async def get_funds(self) -> Dict[str, float]:
        """
        Get available funds and margins
        Returns: {
            available_cash: float,
            used_margin: float,
            available_margin: float,
            total_collateral: float
        }
        """
        pass

    # ========================================================================
    # MARKET DATA (Optional)
    # ========================================================================

    async def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict[str, Any]]:
        """
        Get real-time quote
        Returns: {ltp, open, high, low, volume, ...}
        """
        return None  # Optional - implement if broker provides

    # ========================================================================
    # INSTRUMENT MAPPING
    # ========================================================================

    @abstractmethod
    def get_broker_symbol(self, canonical_symbol: str, exchange: str = "NSE") -> str:
        """
        Convert canonical symbol to broker-specific format
        Example: "RELIANCE" -> broker's format
        """
        pass

    @abstractmethod
    def get_canonical_symbol(self, broker_symbol: str) -> str:
        """
        Convert broker-specific symbol to canonical format
        Example: broker's format -> "RELIANCE"
        """
        pass

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _normalize_order_status(self, broker_status: str) -> str:
        """
        Normalize broker-specific status to standard format
        Returns: PENDING | OPEN | FILLED | PARTIALLY_FILLED | CANCELLED | REJECTED
        """
        status_map = {
            # Common mappings (override in subclass if needed)
            "OPEN": "OPEN",
            "COMPLETE": "FILLED",
            "CANCELLED": "CANCELLED",
            "REJECTED": "REJECTED",
            "PENDING": "PENDING",
            "TRIGGER PENDING": "PENDING",
        }
        return status_map.get(broker_status.upper(), "PENDING")

    def _get_access_token(self) -> str:
        """Get access token from connection data"""
        return self.connection_data.get("access_token", "")

    def _update_connection_data(self, new_data: Dict[str, Any]) -> None:
        """Update connection data (call after refresh)"""
        self.connection_data.update(new_data)

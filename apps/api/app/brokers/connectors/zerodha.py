"""
Zerodha Kite Connector - Full Implementation
Official Kite Connect API integration
"""
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

from ..base import BrokerConnector


class ZerodhaConnector(BrokerConnector):
    """
    Zerodha Kite Connect API Connector
    Docs: https://kite.trade/docs/connect/v3/
    """

    BASE_URL = "https://api.kite.trade"
    LOGIN_URL = "https://kite.zerodha.com/connect/login"

    def _get_broker_type(self) -> str:
        return "zerodha"

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    async def connect(self, auth_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect to Zerodha using request_token from OAuth flow
        Args:
            auth_payload: {
                request_token: str (from OAuth redirect),
                api_key: str,
                api_secret: str
            }
        Returns:
            {access_token, user_id, ...}
        """
        request_token = auth_payload.get("request_token")
        api_key = auth_payload.get("api_key")
        api_secret = auth_payload.get("api_secret")

        if not all([request_token, api_key, api_secret]):
            raise ValueError("Missing required auth parameters")

        # Generate checksum
        checksum = hashlib.sha256(
            f"{api_key}{request_token}{api_secret}".encode()
        ).hexdigest()

        # Exchange request_token for access_token
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/session/token",
                data={
                    "api_key": api_key,
                    "request_token": request_token,
                    "checksum": checksum,
                },
            )

            if response.status_code != 200:
                raise Exception(f"Zerodha auth failed: {response.text}")

            data = response.json()["data"]

            return {
                "access_token": data["access_token"],
                "user_id": data["user_id"],
                "user_name": data["user_name"],
                "email": data["email"],
                "api_key": api_key,
                "api_secret": api_secret,
                "connected_at": datetime.utcnow().isoformat(),
            }

    async def refresh(self) -> Dict[str, Any]:
        """
        Zerodha access tokens don't expire within a day
        But we can validate and return existing token
        """
        is_valid = await self.health_check()
        if not is_valid:
            raise Exception("Access token invalid. User must re-authenticate.")

        return self.connection_data

    async def disconnect(self) -> None:
        """Invalidate session"""
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{self.BASE_URL}/session/token",
                params={"api_key": self.connection_data["api_key"]},
                headers=self._get_headers(),
            )

    async def health_check(self) -> bool:
        """Check if connection is healthy"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/user/profile",
                    headers=self._get_headers(),
                )
                return response.status_code == 200
        except:
            return False

    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================

    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place order on Zerodha
        Args:
            order: {
                symbol: str (canonical),
                exchange: str,
                side: str (BUY/SELL),
                order_type: str (MARKET/LIMIT/SL/SL-M),
                product: str (MIS/CNC/NRML),
                quantity: int,
                price: float (optional),
                trigger_price: float (optional),
                tag: str (optional)
            }
        """
        # Convert to Zerodha format
        tradingsymbol = self.get_broker_symbol(order["symbol"], order["exchange"])

        payload = {
            "tradingsymbol": tradingsymbol,
            "exchange": order["exchange"],
            "transaction_type": order["side"],
            "order_type": order["order_type"],
            "product": order["product"],
            "quantity": order["quantity"],
            "validity": "DAY",
        }

        if order.get("price"):
            payload["price"] = order["price"]

        if order.get("trigger_price"):
            payload["trigger_price"] = order["trigger_price"]

        if order.get("tag"):
            payload["tag"] = order["tag"]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/orders/{order['order_type'].lower()}",
                data=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                error_data = response.json()
                return {
                    "broker_order_id": None,
                    "status": "REJECTED",
                    "message": error_data.get("message", "Order placement failed"),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            data = response.json()["data"]

            return {
                "broker_order_id": data["order_id"],
                "status": "PENDING",
                "message": "Order placed successfully",
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def modify_order(
        self, broker_order_id: str, changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modify order"""
        payload = {}

        if "quantity" in changes:
            payload["quantity"] = changes["quantity"]
        if "price" in changes:
            payload["price"] = changes["price"]
        if "trigger_price" in changes:
            payload["trigger_price"] = changes["trigger_price"]

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.BASE_URL}/orders/{broker_order_id}",
                data=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                return {
                    "broker_order_id": broker_order_id,
                    "status": "REJECTED",
                    "message": "Modification failed",
                }

            return {
                "broker_order_id": broker_order_id,
                "status": "MODIFIED",
                "message": "Order modified successfully",
            }

    async def cancel_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.BASE_URL}/orders/{broker_order_id}",
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                return {
                    "broker_order_id": broker_order_id,
                    "status": "FAILED",
                    "message": "Cancellation failed",
                }

            return {
                "broker_order_id": broker_order_id,
                "status": "CANCELLED",
                "message": "Order cancelled successfully",
            }

    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders for the day"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/orders",
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                return []

            orders = response.json()["data"]

            # Normalize to standard format
            normalized = []
            for order in orders:
                normalized.append({
                    "broker_order_id": order["order_id"],
                    "symbol": self.get_canonical_symbol(order["tradingsymbol"]),
                    "exchange": order["exchange"],
                    "side": order["transaction_type"],
                    "order_type": order["order_type"],
                    "product": order["product"],
                    "quantity": order["quantity"],
                    "price": order.get("price", 0),
                    "trigger_price": order.get("trigger_price", 0),
                    "status": self._normalize_order_status(order["status"]),
                    "filled_quantity": order.get("filled_quantity", 0),
                    "average_price": order.get("average_price", 0),
                    "placed_at": order.get("order_timestamp"),
                })

            return normalized

    async def get_order_status(self, broker_order_id: str) -> Dict[str, Any]:
        """Get order status"""
        orders = await self.get_orders()
        for order in orders:
            if order["broker_order_id"] == broker_order_id:
                return order

        raise ValueError(f"Order {broker_order_id} not found")

    # ========================================================================
    # PORTFOLIO MANAGEMENT
    # ========================================================================

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/portfolio/positions",
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                return []

            data = response.json()["data"]
            positions = data.get("net", [])

            # Normalize
            normalized = []
            for pos in positions:
                if pos["quantity"] == 0:
                    continue

                normalized.append({
                    "symbol": self.get_canonical_symbol(pos["tradingsymbol"]),
                    "exchange": pos["exchange"],
                    "product": pos["product"],
                    "quantity": pos["quantity"],
                    "average_price": pos["average_price"],
                    "last_price": pos["last_price"],
                    "pnl": pos["pnl"],
                    "pnl_percent": (pos["pnl"] / (pos["average_price"] * abs(pos["quantity"]))) * 100
                    if pos["average_price"] > 0 else 0,
                })

            return normalized

    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/portfolio/holdings",
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                return []

            holdings = response.json()["data"]

            normalized = []
            for holding in holdings:
                normalized.append({
                    "symbol": self.get_canonical_symbol(holding["tradingsymbol"]),
                    "exchange": holding["exchange"],
                    "quantity": holding["quantity"],
                    "average_price": holding["average_price"],
                    "last_price": holding["last_price"],
                    "pnl": holding["pnl"],
                })

            return normalized

    async def get_funds(self) -> Dict[str, float]:
        """Get available funds"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/user/margins",
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                return {
                    "available_cash": 0,
                    "used_margin": 0,
                    "available_margin": 0,
                    "total_collateral": 0,
                }

            data = response.json()["data"]
            equity = data.get("equity", {})

            return {
                "available_cash": equity.get("available", {}).get("cash", 0),
                "used_margin": equity.get("utilised", {}).get("debits", 0),
                "available_margin": equity.get("available", {}).get("live_balance", 0),
                "total_collateral": equity.get("available", {}).get("collateral", 0),
            }

    # ========================================================================
    # MARKET DATA
    # ========================================================================

    async def get_quote(self, symbol: str, exchange: str = "NSE") -> Optional[Dict[str, Any]]:
        """Get real-time quote"""
        tradingsymbol = self.get_broker_symbol(symbol, exchange)
        instrument_key = f"{exchange}:{tradingsymbol}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/quote",
                params={"i": instrument_key},
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                return None

            data = response.json()["data"].get(instrument_key, {})

            return {
                "ltp": data.get("last_price", 0),
                "open": data.get("ohlc", {}).get("open", 0),
                "high": data.get("ohlc", {}).get("high", 0),
                "low": data.get("ohlc", {}).get("low", 0),
                "close": data.get("ohlc", {}).get("close", 0),
                "volume": data.get("volume", 0),
            }

    # ========================================================================
    # INSTRUMENT MAPPING
    # ========================================================================

    def get_broker_symbol(self, canonical_symbol: str, exchange: str = "NSE") -> str:
        """
        Convert canonical symbol to Zerodha format
        For NSE equities, usually same as canonical
        """
        # For now, direct mapping (can enhance with instrument DB lookup)
        return canonical_symbol

    def get_canonical_symbol(self, broker_symbol: str) -> str:
        """Convert Zerodha symbol to canonical"""
        return broker_symbol

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth"""
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self.connection_data['api_key']}:{self._get_access_token()}",
        }

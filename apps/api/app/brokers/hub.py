"""
BrokerHub - Central broker management system
Handles multi-broker connections and operations
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from supabase import create_client, Client
from cryptography.fernet import Fernet
import os

from .base import BrokerConnector
from .connectors.zerodha import ZerodhaConnector
from .connectors.upstox import UpstoxConnector
from .connectors.angel_one import AngelOneConnector
from .connectors.dhan import DhanConnector
from .connectors.fyers import FyersConnector
from .connectors.icici_breeze import ICICIBreezeConnector
from .connectors.kotak_neo import KotakNeoConnector
from .connectors.fivepaisa import FivePaisaConnector


class BrokerHub:
    """
    Central hub for managing multiple broker connections
    Provides unified interface for all broker operations
    """

    # Broker connector registry
    CONNECTORS = {
        "zerodha": ZerodhaConnector,
        "upstox": UpstoxConnector,
        "angel_one": AngelOneConnector,
        "dhan": DhanConnector,
        "fyers": FyersConnector,
        "icici_breeze": ICICIBreezeConnector,
        "kotak_neo": KotakNeoConnector,
        "fivepaisa": FivePaisaConnector,
    }

    def __init__(self, supabase_url: str, supabase_key: str, encryption_key: Optional[str] = None):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.encryption_key = encryption_key or os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    async def connect_broker(
        self, user_id: str, broker_type: str, auth_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Connect user to broker
        Stores encrypted credentials in database
        """
        if broker_type not in self.CONNECTORS:
            raise ValueError(f"Unsupported broker: {broker_type}")

        # Create connector instance
        connector_class = self.CONNECTORS[broker_type]
        connector = connector_class(user_id, {})

        # Authenticate with broker
        connection_data = await connector.connect(auth_payload)

        # Encrypt and store credentials
        encrypted_tokens = self._encrypt_data(connection_data)

        # Upsert to database
        result = self.supabase.table("broker_connections").upsert({
            "user_id": user_id,
            "broker_type": broker_type,
            "encrypted_tokens": encrypted_tokens,
            "status": "connected",
            "last_refresh": datetime.utcnow().isoformat(),
        }).execute()

        return {
            "user_id": user_id,
            "broker_type": broker_type,
            "status": "connected",
            "message": f"Successfully connected to {broker_type}"
        }

    async def disconnect_broker(self, user_id: str, broker_type: str) -> Dict[str, Any]:
        """Disconnect user from broker"""
        connector = await self._get_connector(user_id, broker_type)
        await connector.disconnect()

        # Update database
        self.supabase.table("broker_connections").update({
            "status": "disconnected"
        }).eq("user_id", user_id).eq("broker_type", broker_type).execute()

        return {"message": f"Disconnected from {broker_type}"}

    async def refresh_connection(self, user_id: str, broker_type: str) -> Dict[str, Any]:
        """Refresh broker connection tokens"""
        connector = await self._get_connector(user_id, broker_type)
        new_tokens = await connector.refresh()

        # Update encrypted tokens
        encrypted_tokens = self._encrypt_data(new_tokens)

        self.supabase.table("broker_connections").update({
            "encrypted_tokens": encrypted_tokens,
            "last_refresh": datetime.utcnow().isoformat(),
            "status": "connected"
        }).eq("user_id", user_id).eq("broker_type", broker_type).execute()

        return {"message": "Connection refreshed successfully"}

    async def get_connection_status(self, user_id: str, broker_type: str) -> Dict[str, Any]:
        """Check broker connection health"""
        try:
            connector = await self._get_connector(user_id, broker_type)
            is_healthy = await connector.health_check()

            status = "connected" if is_healthy else "error"

            # Update database
            self.supabase.table("broker_connections").update({
                "status": status
            }).eq("user_id", user_id).eq("broker_type", broker_type).execute()

            return {
                "broker_type": broker_type,
                "status": status,
                "healthy": is_healthy
            }
        except Exception as e:
            return {
                "broker_type": broker_type,
                "status": "error",
                "healthy": False,
                "error": str(e)
            }

    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================

    async def place_order(
        self, user_id: str, broker_type: str, order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Place order through specified broker"""
        connector = await self._get_connector(user_id, broker_type)
        result = await connector.place_order(order)
        return result

    async def modify_order(
        self, user_id: str, broker_type: str, broker_order_id: str, changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modify existing order"""
        connector = await self._get_connector(user_id, broker_type)
        result = await connector.modify_order(broker_order_id, changes)
        return result

    async def cancel_order(
        self, user_id: str, broker_type: str, broker_order_id: str
    ) -> Dict[str, Any]:
        """Cancel order"""
        connector = await self._get_connector(user_id, broker_type)
        result = await connector.cancel_order(broker_order_id)
        return result

    async def get_orders(self, user_id: str, broker_type: str) -> List[Dict[str, Any]]:
        """Get all orders for user"""
        connector = await self._get_connector(user_id, broker_type)
        orders = await connector.get_orders()
        return orders

    # ========================================================================
    # PORTFOLIO MANAGEMENT
    # ========================================================================

    async def get_positions(self, user_id: str, broker_type: str) -> List[Dict[str, Any]]:
        """Get current positions"""
        connector = await self._get_connector(user_id, broker_type)
        positions = await connector.get_positions()
        return positions

    async def get_holdings(self, user_id: str, broker_type: str) -> List[Dict[str, Any]]:
        """Get holdings"""
        connector = await self._get_connector(user_id, broker_type)
        holdings = await connector.get_holdings()
        return holdings

    async def get_funds(self, user_id: str, broker_type: str) -> Dict[str, float]:
        """Get available funds"""
        connector = await self._get_connector(user_id, broker_type)
        funds = await connector.get_funds()
        return funds

    # ========================================================================
    # MARKET DATA
    # ========================================================================

    async def get_quote(
        self, user_id: str, broker_type: str, symbol: str, exchange: str = "NSE"
    ) -> Optional[Dict[str, Any]]:
        """Get real-time quote"""
        connector = await self._get_connector(user_id, broker_type)
        quote = await connector.get_quote(symbol, exchange)
        return quote

    # ========================================================================
    # HELPERS
    # ========================================================================

    async def _get_connector(self, user_id: str, broker_type: str) -> BrokerConnector:
        """Get connector instance for user's broker connection"""
        # Fetch connection from database
        result = self.supabase.table("broker_connections").select("*").eq(
            "user_id", user_id
        ).eq("broker_type", broker_type).execute()

        if not result.data:
            raise ValueError(f"No {broker_type} connection found for user {user_id}")

        connection = result.data[0]

        # Decrypt tokens
        connection_data = self._decrypt_data(connection["encrypted_tokens"])

        # Create connector instance
        connector_class = self.CONNECTORS[broker_type]
        connector = connector_class(user_id, connection_data)

        return connector

    def _encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data"""
        json_data = json.dumps(data)
        encrypted = self.cipher.encrypt(json_data.encode())
        return encrypted.decode()

    def _decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt sensitive data"""
        decrypted = self.cipher.decrypt(encrypted_data.encode())
        return json.loads(decrypted.decode())

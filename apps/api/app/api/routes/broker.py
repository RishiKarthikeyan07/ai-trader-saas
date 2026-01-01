"""
Broker connection management routes
Handles multi-broker connections and account management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from pydantic import BaseModel
from app.core.security import get_current_user
from app.core.supabase import supabase
from app.brokers.hub import BrokerHub
from app.core.config import settings

router = APIRouter(prefix="/broker", tags=["broker"])


# Initialize BrokerHub
broker_hub = BrokerHub(settings.SUPABASE_URL, settings.SUPABASE_KEY)


# ========================================================================
# REQUEST/RESPONSE MODELS
# ========================================================================

class BrokerConnectRequest(BaseModel):
    """Request to connect to broker"""
    broker_type: str
    auth_payload: Dict[str, Any]


class BrokerDisconnectRequest(BaseModel):
    """Request to disconnect from broker"""
    broker_type: str


class BrokerStatusResponse(BaseModel):
    """Broker connection status response"""
    broker_type: str
    status: str
    healthy: bool
    last_refresh: str = None
    error: str = None


class BrokerListResponse(BaseModel):
    """Supported broker information"""
    broker_type: str
    name: str
    description: str
    supported_features: List[str]


# ========================================================================
# CONNECTION MANAGEMENT ENDPOINTS
# ========================================================================

@router.post("/connect")
async def connect_broker(
    request: BrokerConnectRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Connect user to broker

    Stores encrypted credentials in Supabase and establishes connection.
    Supports: Zerodha, Upstox, Angel One, Dhan, Fyers, ICICI Breeze, Kotak Neo, 5Paisa
    """
    try:
        result = await broker_hub.connect_broker(
            user_id=user_id,
            broker_type=request.broker_type,
            auth_payload=request.auth_payload
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect broker: {str(e)}"
        )


@router.post("/disconnect")
async def disconnect_broker(
    request: BrokerDisconnectRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Disconnect from broker

    Removes broker connection while keeping historical data in database
    """
    try:
        result = await broker_hub.disconnect_broker(
            user_id=user_id,
            broker_type=request.broker_type
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disconnect broker: {str(e)}"
        )


@router.post("/refresh")
async def refresh_connection(
    request: BrokerDisconnectRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Refresh broker connection tokens

    Refreshes authentication tokens and updates connection status
    """
    try:
        result = await broker_hub.refresh_connection(
            user_id=user_id,
            broker_type=request.broker_type
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh connection: {str(e)}"
        )


@router.get("/status", response_model=BrokerStatusResponse)
async def get_broker_status(
    user_id: str = Depends(get_current_user)
):
    """
    Get connection status for user's connected broker

    Returns health check status and connection details
    """
    try:
        # Get user's broker connection from database
        response = supabase.table("broker_connections") \
            .select("*") \
            .eq("user_id", user_id) \
            .execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No broker connection found for user"
            )

        connection = response.data[0]
        broker_type = connection.get("broker_type")

        # Get fresh status from broker
        status_result = await broker_hub.get_connection_status(
            user_id=user_id,
            broker_type=broker_type
        )

        return {
            "broker_type": status_result.get("broker_type"),
            "status": status_result.get("status"),
            "healthy": status_result.get("healthy"),
            "last_refresh": connection.get("last_refresh"),
            "error": status_result.get("error")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch broker status: {str(e)}"
        )


@router.get("/list", response_model=List[BrokerListResponse])
async def list_supported_brokers(
    user_id: str = Depends(get_current_user)
):
    """
    List all supported brokers

    Returns information about all available broker integrations
    """
    broker_info = [
        {
            "broker_type": "zerodha",
            "name": "Zerodha",
            "description": "India's largest broker by retail segments",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        },
        {
            "broker_type": "upstox",
            "name": "Upstox",
            "description": "Leading fintech brokerage platform",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        },
        {
            "broker_type": "angel_one",
            "name": "Angel One",
            "description": "Full-service stock broker and distributor",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        },
        {
            "broker_type": "dhan",
            "name": "Dhan",
            "description": "Modern stock broker with advanced tools",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        },
        {
            "broker_type": "fyers",
            "name": "Fyers",
            "description": "Fintech-first broking platform",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        },
        {
            "broker_type": "icici_breeze",
            "name": "ICICI Breeze",
            "description": "ICICI Bank's broking platform",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        },
        {
            "broker_type": "kotak_neo",
            "name": "Kotak Neo",
            "description": "Kotak Securities' modern trading platform",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        },
        {
            "broker_type": "fivepaisa",
            "name": "5Paisa",
            "description": "Discount broker with competitive pricing",
            "supported_features": ["orders", "positions", "holdings", "funds", "quotes"]
        }
    ]

    return broker_info

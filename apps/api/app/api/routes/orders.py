"""
Order blotter management routes
Handles order tracking, cancellations, and fill tracking
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from app.core.security import get_current_user
from app.core.supabase import supabase
from app.brokers.hub import BrokerHub
from app.core.config import settings

router = APIRouter(prefix="/orders", tags=["orders"])


# Initialize BrokerHub
broker_hub = BrokerHub(settings.SUPABASE_URL, settings.SUPABASE_KEY)


# ========================================================================
# REQUEST/RESPONSE MODELS
# ========================================================================

class OrderResponse(BaseModel):
    """Order details"""
    id: str
    broker_type: str
    broker_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    price: Optional[float]
    filled_quantity: int
    average_price: Optional[float]
    status: str
    created_at: datetime
    updated_at: datetime
    filled_at: Optional[datetime] = None


class FillResponse(BaseModel):
    """Trade fill details"""
    id: str
    broker_type: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    filled_at: datetime


class OrderFilterParams(BaseModel):
    """Order filter parameters"""
    status: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 50


# ========================================================================
# ORDER RETRIEVAL ENDPOINTS
# ========================================================================

@router.get("", response_model=List[OrderResponse])
async def get_orders(
    user_id: str = Depends(get_current_user),
    status: Optional[str] = Query(None, description="Filter by order status"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    side: Optional[str] = Query(None, description="Filter by side (buy/sell)"),
    start_date: Optional[datetime] = Query(None, description="Filter orders from this date"),
    end_date: Optional[datetime] = Query(None, description="Filter orders until this date"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results to return")
):
    """
    Get all orders with optional filters

    Supports filtering by status, symbol, date range, and side
    Status values: pending, completed, rejected, cancelled
    """
    try:
        # Build query
        query = supabase.table("orders") \
            .select("*") \
            .eq("user_id", user_id)

        # Apply filters
        if status:
            query = query.eq("status", status)
        if symbol:
            query = query.eq("symbol", symbol)
        if side:
            query = query.eq("side", side)
        if start_date:
            query = query.gte("created_at", start_date.isoformat())
        if end_date:
            query = query.lte("created_at", end_date.isoformat())

        response = query.order("created_at", desc=True) \
            .limit(limit) \
            .execute()

        return response.data if response.data else []

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch orders: {str(e)}"
        )


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order_detail(
    order_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get order detail

    Returns complete order information including fills and status
    """
    try:
        response = supabase.table("orders") \
            .select("*") \
            .eq("id", order_id) \
            .eq("user_id", user_id) \
            .single() \
            .execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Order not found"
            )

        return response.data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch order: {str(e)}"
        )


# ========================================================================
# FILLS ENDPOINTS
# ========================================================================

@router.get("/fills", response_model=List[FillResponse])
async def get_all_fills(
    user_id: str = Depends(get_current_user),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    start_date: Optional[datetime] = Query(None, description="Filter fills from this date"),
    end_date: Optional[datetime] = Query(None, description="Filter fills until this date"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results to return")
):
    """
    Get all fills (executed trades)

    Returns all fill records with optional filtering by symbol and date range
    """
    try:
        # Build query
        query = supabase.table("fills") \
            .select("*") \
            .eq("user_id", user_id)

        # Apply filters
        if symbol:
            query = query.eq("symbol", symbol)
        if start_date:
            query = query.gte("filled_at", start_date.isoformat())
        if end_date:
            query = query.lte("filled_at", end_date.isoformat())

        response = query.order("filled_at", desc=True) \
            .limit(limit) \
            .execute()

        return response.data if response.data else []

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch fills: {str(e)}"
        )


# ========================================================================
# ORDER MANAGEMENT ENDPOINTS
# ========================================================================

@router.post("/cancel/{order_id}")
async def cancel_order(
    order_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Cancel an open order (manual override)

    Sends cancellation request to broker for pending orders
    Returns error if order cannot be cancelled
    """
    try:
        # Fetch order from database
        response = supabase.table("orders") \
            .select("*") \
            .eq("id", order_id) \
            .eq("user_id", user_id) \
            .single() \
            .execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Order not found"
            )

        order = response.data

        # Check if order can be cancelled
        if order.get("status") not in ["pending", "open", "partially_filled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel order with status: {order.get('status')}"
            )

        broker_type = order.get("broker_type")
        broker_order_id = order.get("broker_order_id")

        # Call broker to cancel
        await broker_hub.cancel_order(
            user_id=user_id,
            broker_type=broker_type,
            broker_order_id=broker_order_id
        )

        # Update order status in database
        supabase.table("orders") \
            .update({
                "status": "cancelled",
                "updated_at": datetime.utcnow().isoformat()
            }) \
            .eq("id", order_id) \
            .execute()

        # Log cancellation event
        from app.core.supabase import log_audit_event
        await log_audit_event(
            user_id=user_id,
            event_type="ORDER_CANCELLED",
            details={
                "order_id": order_id,
                "symbol": order.get("symbol"),
                "broker_type": broker_type
            }
        )

        return {
            "status": "cancelled",
            "message": f"Order {order_id} has been cancelled",
            "order_id": order_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}"
        )

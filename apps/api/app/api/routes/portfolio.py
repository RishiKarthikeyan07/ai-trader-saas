"""
Portfolio and positions management routes
Handles positions, holdings, funds, and portfolio summaries
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from app.core.security import get_current_user
from app.core.supabase import supabase
from app.brokers.hub import BrokerHub
from app.core.config import settings

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


# Initialize BrokerHub
broker_hub = BrokerHub(settings.SUPABASE_URL, settings.SUPABASE_KEY)


# ========================================================================
# REQUEST/RESPONSE MODELS
# ========================================================================

class PositionResponse(BaseModel):
    """Position details"""
    id: str
    broker_type: str
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    exchange: str
    status: str = "open"


class HoldingResponse(BaseModel):
    """Holdings information"""
    id: str
    broker_type: str
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    total_value: float
    pnl: float
    pnl_percentage: float


class FundsResponse(BaseModel):
    """Available funds information"""
    broker_type: str
    total_balance: float
    available_balance: float
    used_balance: float
    margin_available: float
    margin_used: float
    portfolio_value: float


class PortfolioSummaryResponse(BaseModel):
    """Portfolio summary information"""
    broker_type: str
    total_value: float
    total_invested: float
    total_pnl: float
    total_pnl_percentage: float
    positions_count: int
    holdings_count: int
    available_funds: float
    margin_utilization: float
    last_updated: datetime


# ========================================================================
# POSITIONS ENDPOINTS
# ========================================================================

@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    user_id: str = Depends(get_current_user)
):
    """
    Get all open positions

    Returns current open positions across all connected brokers
    """
    try:
        # Get user's broker connection
        response = supabase.table("broker_connections") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("status", "connected") \
            .execute()

        if not response.data:
            return []

        all_positions = []

        for connection in response.data:
            broker_type = connection.get("broker_type")
            try:
                positions = await broker_hub.get_positions(user_id, broker_type)
                for position in positions:
                    position['broker_type'] = broker_type
                    all_positions.append(position)
            except Exception as e:
                # Log error but continue with other brokers
                continue

        return all_positions

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch positions: {str(e)}"
        )


@router.get("/positions/{position_id}", response_model=PositionResponse)
async def get_position_detail(
    position_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get position detail

    Returns detailed information for a specific position
    """
    try:
        # Fetch position from database
        response = supabase.table("positions") \
            .select("*") \
            .eq("id", position_id) \
            .eq("user_id", user_id) \
            .single() \
            .execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Position not found"
            )

        return response.data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch position: {str(e)}"
        )


# ========================================================================
# HOLDINGS ENDPOINTS
# ========================================================================

@router.get("/holdings", response_model=List[HoldingResponse])
async def get_holdings(
    user_id: str = Depends(get_current_user)
):
    """
    Get all holdings

    Returns all long-term holdings across all connected brokers
    """
    try:
        # Get user's broker connection
        response = supabase.table("broker_connections") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("status", "connected") \
            .execute()

        if not response.data:
            return []

        all_holdings = []

        for connection in response.data:
            broker_type = connection.get("broker_type")
            try:
                holdings = await broker_hub.get_holdings(user_id, broker_type)
                for holding in holdings:
                    holding['broker_type'] = broker_type
                    all_holdings.append(holding)
            except Exception as e:
                # Log error but continue with other brokers
                continue

        return all_holdings

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch holdings: {str(e)}"
        )


# ========================================================================
# FUNDS ENDPOINTS
# ========================================================================

@router.get("/funds", response_model=FundsResponse)
async def get_available_funds(
    user_id: str = Depends(get_current_user)
):
    """
    Get available funds

    Returns account balance and available margin across brokers
    """
    try:
        # Get user's broker connection
        response = supabase.table("broker_connections") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("status", "connected") \
            .single() \
            .execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No connected broker found"
            )

        connection = response.data
        broker_type = connection.get("broker_type")

        funds = await broker_hub.get_funds(user_id, broker_type)

        return {
            "broker_type": broker_type,
            **funds
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch funds: {str(e)}"
        )


# ========================================================================
# PORTFOLIO SUMMARY ENDPOINT
# ========================================================================

@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(
    user_id: str = Depends(get_current_user)
):
    """
    Get portfolio summary

    Returns aggregated portfolio metrics including total value, P&L, and positions
    """
    try:
        # Get user's broker connection
        response = supabase.table("broker_connections") \
            .select("*") \
            .eq("user_id", user_id) \
            .eq("status", "connected") \
            .single() \
            .execute()

        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No connected broker found"
            )

        connection = response.data
        broker_type = connection.get("broker_type")

        # Fetch positions and holdings
        positions = await broker_hub.get_positions(user_id, broker_type)
        holdings = await broker_hub.get_holdings(user_id, broker_type)
        funds = await broker_hub.get_funds(user_id, broker_type)

        # Calculate aggregates
        total_pnl = sum(p.get('pnl', 0) for p in positions) + \
                    sum(h.get('pnl', 0) for h in holdings)
        total_invested = sum(h.get('total_value', 0) / (1 + h.get('pnl_percentage', 0) / 100)
                           for h in holdings if h.get('pnl_percentage', 0) != -100)
        total_value = funds.get('portfolio_value', 0)
        total_pnl_percentage = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        margin_utilization = 0
        if funds.get('margin_available', 0) > 0:
            margin_utilization = (funds.get('margin_used', 0) /
                                funds.get('margin_available', 1)) * 100

        return {
            "broker_type": broker_type,
            "total_value": total_value,
            "total_invested": total_invested,
            "total_pnl": total_pnl,
            "total_pnl_percentage": total_pnl_percentage,
            "positions_count": len(positions),
            "holdings_count": len(holdings),
            "available_funds": funds.get('available_balance', 0),
            "margin_utilization": margin_utilization,
            "last_updated": datetime.utcnow()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch portfolio summary: {str(e)}"
        )

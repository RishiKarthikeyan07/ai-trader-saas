"""
Risk management routes
Handles risk limits, exposure monitoring, and risk metrics
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.core.security import get_current_user
from app.core.supabase import supabase, log_audit_event
from app.brokers.hub import BrokerHub
from app.core.config import settings

router = APIRouter(prefix="/risk", tags=["risk"])


# Initialize BrokerHub
broker_hub = BrokerHub(settings.SUPABASE_URL, settings.SUPABASE_KEY)


# ========================================================================
# REQUEST/RESPONSE MODELS
# ========================================================================

class RiskLimits(BaseModel):
    """Risk limit configuration"""
    max_position_size: float = Field(
        100000,
        description="Maximum amount per position"
    )
    max_daily_loss: float = Field(
        10000,
        description="Maximum daily loss allowed"
    )
    max_leverage: float = Field(
        2.0,
        description="Maximum allowed leverage"
    )
    max_open_positions: int = Field(
        5,
        description="Maximum number of concurrent open positions"
    )
    stop_loss_percentage: float = Field(
        2.0,
        description="Default stop loss percentage"
    )
    max_correlation: float = Field(
        0.7,
        description="Maximum correlation between positions"
    )
    halt_on_daily_loss: bool = Field(
        True,
        description="Halt trading when daily loss limit reached"
    )


class UpdateRiskLimitsRequest(BaseModel):
    """Request to update risk limits"""
    max_position_size: Optional[float] = None
    max_daily_loss: Optional[float] = None
    max_leverage: Optional[float] = None
    max_open_positions: Optional[int] = None
    stop_loss_percentage: Optional[float] = None
    max_correlation: Optional[float] = None
    halt_on_daily_loss: Optional[bool] = None


class ExposureBreakdown(BaseModel):
    """Exposure information"""
    symbol: str
    sector: str
    quantity: int
    value: float
    percentage_of_portfolio: float
    correlation_with_portfolio: float


class ExposureResponse(BaseModel):
    """Current exposure breakdown"""
    total_exposure: float
    cash_exposure: float
    equity_exposure: float
    derivative_exposure: float
    sector_exposure: Dict[str, float]
    top_exposures: list
    diversification_index: float


class RiskMetricsResponse(BaseModel):
    """Risk metrics"""
    value_at_risk: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    volatility: float
    correlation_with_market: float
    var_95: float
    var_99: float
    conditional_var: float


# ========================================================================
# RISK LIMITS ENDPOINTS
# ========================================================================

@router.get("/limits", response_model=RiskLimits)
async def get_risk_limits(
    user_id: str = Depends(get_current_user)
):
    """
    Get user's risk limits

    Returns current risk limit configuration for the user
    """
    try:
        response = supabase.table("user_risk_limits") \
            .select("*") \
            .eq("user_id", user_id) \
            .single() \
            .execute()

        if not response.data:
            # Return default limits if not configured
            return RiskLimits()

        limits_data = response.data
        return RiskLimits(
            max_position_size=limits_data.get("max_position_size", 100000),
            max_daily_loss=limits_data.get("max_daily_loss", 10000),
            max_leverage=limits_data.get("max_leverage", 2.0),
            max_open_positions=limits_data.get("max_open_positions", 5),
            stop_loss_percentage=limits_data.get("stop_loss_percentage", 2.0),
            max_correlation=limits_data.get("max_correlation", 0.7),
            halt_on_daily_loss=limits_data.get("halt_on_daily_loss", True)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch risk limits: {str(e)}"
        )


@router.post("/limits", response_model=RiskLimits)
async def update_risk_limits(
    request: UpdateRiskLimitsRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Update risk limits

    Updates user's risk limit configuration
    Validates that limits are reasonable before saving
    """
    try:
        # Validate limits
        if request.max_position_size is not None and request.max_position_size <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_position_size must be positive"
            )

        if request.max_daily_loss is not None and request.max_daily_loss <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_daily_loss must be positive"
            )

        if request.max_leverage is not None and (request.max_leverage <= 0 or request.max_leverage > 10):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_leverage must be between 0 and 10"
            )

        if request.max_open_positions is not None and request.max_open_positions <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_open_positions must be positive"
            )

        if request.stop_loss_percentage is not None and \
           (request.stop_loss_percentage <= 0 or request.stop_loss_percentage > 100):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="stop_loss_percentage must be between 0 and 100"
            )

        if request.max_correlation is not None and \
           (request.max_correlation < 0 or request.max_correlation > 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_correlation must be between 0 and 1"
            )

        # Get existing limits
        existing = supabase.table("user_risk_limits") \
            .select("*") \
            .eq("user_id", user_id) \
            .execute()

        # Prepare update data
        update_data = {
            "user_id": user_id,
            "updated_at": datetime.utcnow().isoformat()
        }

        if request.max_position_size is not None:
            update_data["max_position_size"] = request.max_position_size
        if request.max_daily_loss is not None:
            update_data["max_daily_loss"] = request.max_daily_loss
        if request.max_leverage is not None:
            update_data["max_leverage"] = request.max_leverage
        if request.max_open_positions is not None:
            update_data["max_open_positions"] = request.max_open_positions
        if request.stop_loss_percentage is not None:
            update_data["stop_loss_percentage"] = request.stop_loss_percentage
        if request.max_correlation is not None:
            update_data["max_correlation"] = request.max_correlation
        if request.halt_on_daily_loss is not None:
            update_data["halt_on_daily_loss"] = request.halt_on_daily_loss

        # Upsert limits
        supabase.table("user_risk_limits") \
            .upsert(update_data) \
            .execute()

        # Log audit event
        await log_audit_event(
            user_id=user_id,
            event_type="RISK_LIMITS_UPDATED",
            details=update_data
        )

        # Return updated limits
        return await get_risk_limits(user_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update risk limits: {str(e)}"
        )


# ========================================================================
# EXPOSURE ENDPOINTS
# ========================================================================

@router.get("/exposure", response_model=ExposureResponse)
async def get_current_exposure(
    user_id: str = Depends(get_current_user)
):
    """
    Get current exposure breakdown

    Returns portfolio exposure across symbols, sectors, and asset classes
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

        # Calculate total exposure
        total_equity_exposure = 0
        top_exposures = []
        sector_exposure = {}

        all_holdings = positions + holdings
        total_portfolio_value = funds.get('portfolio_value', 1)

        for holding in all_holdings:
            value = holding.get('current_price', 0) * holding.get('quantity', 0)
            total_equity_exposure += value
            percentage = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

            top_exposures.append({
                "symbol": holding.get('symbol'),
                "value": value,
                "percentage": percentage,
                "quantity": holding.get('quantity')
            })

            # Aggregate sector exposure (simplified)
            sector = holding.get('sector', 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value

        # Sort top exposures
        top_exposures.sort(key=lambda x: x['value'], reverse=True)
        top_exposures = top_exposures[:10]

        # Calculate diversification index (simplified)
        max_exposure = max([e['percentage'] for e in top_exposures], default=0)
        diversification_index = 1 - (max_exposure / 100) if max_exposure > 0 else 1

        cash_exposure = funds.get('available_balance', 0)
        derivative_exposure = 0  # Placeholder

        return ExposureResponse(
            total_exposure=total_equity_exposure,
            cash_exposure=cash_exposure,
            equity_exposure=total_equity_exposure,
            derivative_exposure=derivative_exposure,
            sector_exposure=sector_exposure,
            top_exposures=top_exposures,
            diversification_index=diversification_index
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch exposure: {str(e)}"
        )


# ========================================================================
# RISK METRICS ENDPOINTS
# ========================================================================

@router.get("/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(
    user_id: str = Depends(get_current_user)
):
    """
    Get risk metrics

    Returns portfolio risk metrics including VAR, Sharpe ratio, max drawdown, etc.
    Calculated based on historical returns data
    """
    try:
        # Fetch user's portfolio metrics from database
        response = supabase.table("portfolio_metrics") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("calculated_at", desc=True) \
            .limit(1) \
            .execute()

        if response.data:
            metrics = response.data[0]
            return RiskMetricsResponse(
                value_at_risk=metrics.get("value_at_risk", 0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
                max_drawdown=metrics.get("max_drawdown", 0),
                beta=metrics.get("beta", 0),
                volatility=metrics.get("volatility", 0),
                correlation_with_market=metrics.get("correlation_with_market", 0),
                var_95=metrics.get("var_95", 0),
                var_99=metrics.get("var_99", 0),
                conditional_var=metrics.get("conditional_var", 0)
            )

        # Return default metrics if none calculated yet
        return RiskMetricsResponse(
            value_at_risk=0,
            sharpe_ratio=0,
            max_drawdown=0,
            beta=0,
            volatility=0,
            correlation_with_market=0,
            var_95=0,
            var_99=0,
            conditional_var=0
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch risk metrics: {str(e)}"
        )

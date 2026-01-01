from fastapi import APIRouter, Depends, HTTPException
from typing import List
from pydantic import BaseModel
from app.models.schemas import PositionResponse, OrderResponse
from app.core.security import get_current_user, require_tier
from app.core.supabase import supabase, log_audit_event
from app.core.config import settings

router = APIRouter()


class AutoTradingToggle(BaseModel):
    enabled: bool


class PaperExecuteRequest(BaseModel):
    signal_id: str


@router.post("/auto/enable")
async def enable_auto_trading(
    request: AutoTradingToggle,
    user_id: str = Depends(require_tier("elite"))
):
    """Enable/disable auto trading (Elite only)"""
    # Update user profile
    supabase.table('profiles') \
        .update({'auto_trading_enabled': request.enabled}) \
        .eq('user_id', user_id) \
        .execute()

    # Log audit event
    await log_audit_event(user_id, 'auto_trading_toggled', {
        'enabled': request.enabled
    })

    return {"status": "ok", "enabled": request.enabled}


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(user_id: str = Depends(require_tier("elite"))):
    """Get user positions (Elite only)"""
    response = supabase.table('positions') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('opened_at', desc=True) \
        .execute()

    return response.data


@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(user_id: str = Depends(require_tier("elite"))):
    """Get user orders (Elite only)"""
    response = supabase.table('orders') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('created_at', desc=True) \
        .execute()

    return response.data


@router.post("/paper/execute")
async def execute_paper_trade(
    request: PaperExecuteRequest,
    user_id: str = Depends(require_tier("elite"))
):
    """Execute paper trade (Elite only, BUY only)"""
    # Get signal
    signal_response = supabase.table('signals') \
        .select('*') \
        .eq('id', request.signal_id) \
        .single() \
        .execute()

    if not signal_response.data:
        raise HTTPException(status_code=404, detail="Signal not found")

    signal = signal_response.data

    # Check if signal is READY
    if signal['status'] != 'ready':
        raise HTTPException(
            status_code=400,
            detail="Signal must be in READY status to execute"
        )

    # Create paper order (BUY only)
    order_data = {
        'user_id': user_id,
        'signal_id': request.signal_id,
        'symbol': signal['symbol'],
        'side': 'buy',
        'type': 'market',
        'quantity': 1,  # Placeholder - should be calculated based on risk
        'status': 'filled',
        'is_paper': True
    }

    order_response = supabase.table('orders').insert(order_data).execute()
    order = order_response.data[0]

    # Create position
    position_data = {
        'user_id': user_id,
        'signal_id': request.signal_id,
        'symbol': signal['symbol'],
        'entry_price': (signal['entry_min'] + signal['entry_max']) / 2,
        'quantity': 1,
        'stop_loss': signal['stop_loss'],
        'target_1': signal['target_1'],
        'target_2': signal['target_2'],
        'status': 'open',
        'is_paper': True
    }

    position_response = supabase.table('positions').insert(position_data).execute()

    # Log audit event
    await log_audit_event(user_id, 'paper_trade_executed', {
        'signal_id': request.signal_id,
        'symbol': signal['symbol'],
        'order_id': order['id']
    })

    return {"status": "executed", "order": order, "position": position_response.data[0]}


@router.post("/kill")
async def kill_switch(user_id: str = Depends(require_tier("elite"))):
    """Emergency kill switch - close all positions and disable auto trading"""
    # Disable auto trading
    supabase.table('profiles') \
        .update({'auto_trading_enabled': False}) \
        .eq('user_id', user_id) \
        .execute()

    # Close all open positions (paper only for now)
    supabase.table('positions') \
        .update({'status': 'closed'}) \
        .eq('user_id', user_id) \
        .eq('status', 'open') \
        .execute()

    # Log audit event
    await log_audit_event(user_id, 'kill_switch_activated', {})

    return {"status": "killed", "message": "All positions closed, auto trading disabled"}

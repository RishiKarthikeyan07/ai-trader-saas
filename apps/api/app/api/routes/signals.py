from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
from app.models.schemas import SignalResponse, SignalExplanationResponse
from app.core.security import get_current_user
from app.core.supabase import supabase, can_access_signal

router = APIRouter()


@router.get("/latest", response_model=List[SignalResponse])
async def get_latest_signals(
    limit: int = Query(10, ge=1, le=50),
    user_id: str = Depends(get_current_user)
):
    """Get latest signals (tier-gated)"""
    # Get user tier to determine access
    from app.core.supabase import get_user_tier
    tier = await get_user_tier(user_id)

    # Apply tier limits
    tier_limits = {'basic': 3, 'pro': 10, 'elite': limit}
    actual_limit = min(limit, tier_limits.get(tier, 3))

    # Fetch signals from Supabase
    response = supabase.table('signals') \
        .select('*') \
        .order('rank') \
        .limit(actual_limit) \
        .execute()

    return response.data


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal(
    signal_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get signal by ID"""
    response = supabase.table('signals') \
        .select('*') \
        .eq('id', signal_id) \
        .single() \
        .execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Signal not found")

    # Check tier access
    if not await can_access_signal(user_id, response.data['rank']):
        raise HTTPException(
            status_code=403,
            detail="Upgrade your plan to access this signal"
        )

    return response.data


@router.get("/{signal_id}/explanation", response_model=SignalExplanationResponse)
async def get_signal_explanation(
    signal_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get signal explanation (high-level only, no model secrets)"""
    response = supabase.table('signal_explanations') \
        .select('*') \
        .eq('signal_id', signal_id) \
        .single() \
        .execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Explanation not found")

    return response.data

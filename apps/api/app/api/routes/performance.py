from fastapi import APIRouter, Depends, Query
from typing import List, Dict
from app.core.security import get_current_user
from app.services.performance import (
    get_performance_heatmap,
    get_performance_summary,
    get_top_k_comparison
)

router = APIRouter()


@router.get("/heatmap")
async def get_heatmap(
    days: int = Query(30, ge=7, le=90),
    user_id: str = Depends(get_current_user)
) -> List[Dict]:
    """
    Get performance heatmap data for last N days

    Returns daily win rates and avg R for each K bucket and horizon.
    Used to render the heatmap visualization.
    """
    return await get_performance_heatmap(days)


@router.get("/summary")
async def get_summary(
    days: int = Query(30, ge=7, le=90),
    user_id: str = Depends(get_current_user)
) -> Dict:
    """
    Get aggregated performance summary

    Returns overall metrics for each K bucket and horizon.
    Used for summary cards and statistics.
    """
    return await get_performance_summary(days)


@router.get("/comparison")
async def get_comparison(
    user_id: str = Depends(get_current_user)
) -> Dict:
    """
    Get Top-K comparison for upsell messaging

    Shows Basic users (K3) what Pro (K10) and Elite (KALL) performance looks like.
    Used to drive upgrades.
    """
    return await get_top_k_comparison()

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import Settings, get_settings
from app.models.signal import Signal
from app.services.store import fetch_latest, fetch_by_id

router = APIRouter()


@router.get("/signals/latest", response_model=list[Signal])
async def latest_signals(
    limit: int = Query(10, gt=0, le=100),
    tier: str = Query("basic", regex="^(basic|pro|elite)$"),
    settings: Settings = Depends(get_settings),
):
    tier_caps = {
        "basic": settings.tier_basic_limit,
        "pro": settings.tier_pro_limit,
        "elite": settings.tier_elite_limit,
    }
    cap = tier_caps.get(tier, settings.tier_basic_limit)
    signals = fetch_latest(limit=min(limit, cap), settings=settings)
    return signals


@router.get("/signals/{signal_id}", response_model=Signal)
async def signal_detail(signal_id: str, settings: Settings = Depends(get_settings)):
    sig = fetch_by_id(signal_id, settings=settings)
    if not sig:
        raise HTTPException(status_code=404, detail="Signal not found")
    return sig

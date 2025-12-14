from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.state import get_state, set_elite_auto
from app.services.store import fetch_by_id

router = APIRouter()


class EliteAutoToggle(BaseModel):
    enabled: bool


class EliteTradeRequest(BaseModel):
    signal_id: str
    quantity: float = Field(gt=0)


@router.post("/elite/auto/enable")
async def elite_auto(toggle: EliteAutoToggle, settings: Settings = Depends(get_settings)):
    state = set_elite_auto(toggle.enabled)
    return {"elite_auto_enabled": state.elite_auto_enabled}


@router.post("/elite/trade/execute")
async def elite_trade(req: EliteTradeRequest, settings: Settings = Depends(get_settings)):
    signal = fetch_by_id(req.signal_id, settings=settings)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    if signal.signal_type != "BUY":
        raise HTTPException(status_code=400, detail="Auto execution allowed for BUY only")
    if not get_state().elite_auto_enabled:
        raise HTTPException(status_code=403, detail="Elite auto trading disabled")

    # Mock broker adapter placeholder
    order = {
        "symbol": signal.symbol,
        "side": "BUY",
        "quantity": req.quantity,
        "status": "filled",
        "price": signal.entry_zone_high,
        "timestamp": signal.updated_at,
    }
    return {"order": order}

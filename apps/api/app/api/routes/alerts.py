from fastapi import APIRouter, Depends, HTTPException
from typing import List
from pydantic import BaseModel
from app.models.schemas import AlertResponse
from app.core.security import get_current_user
from app.core.supabase import supabase

router = APIRouter()


class AlertSubscribeRequest(BaseModel):
    signal_id: str


@router.get("", response_model=List[AlertResponse])
async def get_alerts(user_id: str = Depends(get_current_user)):
    """Get user alerts"""
    response = supabase.table('alerts') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('created_at', desc=True) \
        .limit(50) \
        .execute()

    return response.data


@router.post("/subscribe")
async def subscribe_to_alert(
    request: AlertSubscribeRequest,
    user_id: str = Depends(get_current_user)
):
    """Subscribe to signal alerts"""
    # Create alert subscription (will be triggered by hourly confirmer)
    # For now, just return success
    return {"status": "subscribed", "signal_id": request.signal_id}


@router.post("/{alert_id}/mark-read")
async def mark_alert_read(
    alert_id: str,
    user_id: str = Depends(get_current_user)
):
    """Mark alert as read"""
    supabase.table('alerts') \
        .update({'read': True}) \
        .eq('id', alert_id) \
        .eq('user_id', user_id) \
        .execute()

    return {"status": "ok"}

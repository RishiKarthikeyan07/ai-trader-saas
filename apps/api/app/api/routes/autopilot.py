"""
AutoPilot Routes
Control AutoPilot bot, view status, manage risk controls
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from datetime import datetime

from ...core.deps import get_current_user, get_supabase
from supabase import Client

router = APIRouter(prefix="/autopilot", tags=["autopilot"])


@router.get("/status")
async def get_autopilot_status(
    current_user: Dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """Get AutoPilot status for current user"""
    user_id = current_user["id"]

    # Get profile
    profile_result = supabase.table("profiles").select("*").eq("user_id", user_id).execute()

    if not profile_result.data:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = profile_result.data[0]

    # Get risk limits
    risk_result = supabase.table("risk_limits").select("*").eq("user_id", user_id).execute()

    risk_limits = risk_result.data[0] if risk_result.data else None

    # Get open positions count
    positions_result = (
        supabase.table("positions")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("status", "OPEN")
        .execute()
    )

    positions_count = positions_result.count or 0

    # Get today's P&L
    today = datetime.utcnow().date().isoformat()
    perf_result = (
        supabase.table("performance_daily")
        .select("net_pnl, drawdown")
        .eq("user_id", user_id)
        .eq("date", today)
        .execute()
    )

    daily_pnl = perf_result.data[0]["net_pnl"] if perf_result.data else 0
    drawdown = perf_result.data[0]["drawdown"] if perf_result.data else 0

    # Calculate exposure
    positions_data = (
        supabase.table("positions")
        .select("quantity, current_price")
        .eq("user_id", user_id)
        .eq("status", "OPEN")
        .execute()
    )

    current_exposure = sum(
        (p["quantity"] * (p["current_price"] or 0)) for p in positions_data.data
    )

    # Determine AutoPilot status
    autopilot_status = "ON" if profile["autopilot_enabled"] else "PAUSED"

    # Check if in PROTECT mode (risk limits breached)
    if risk_limits:
        max_daily_loss = risk_limits["max_daily_loss"]
        max_exposure = risk_limits["max_exposure"]

        if daily_pnl < -max_daily_loss:
            autopilot_status = "PROTECT"
        elif current_exposure > max_exposure:
            autopilot_status = "PROTECT"

    return {
        "status": autopilot_status,
        "is_active_subscriber": profile["is_active_subscriber"],
        "autopilot_enabled": profile["autopilot_enabled"],
        "positions_count": positions_count,
        "max_positions": risk_limits["max_positions"] if risk_limits else 5,
        "current_exposure": current_exposure,
        "max_exposure": risk_limits["max_exposure"] if risk_limits else 100000,
        "daily_pnl": daily_pnl,
        "daily_loss_remaining": (risk_limits["max_daily_loss"] + daily_pnl)
        if risk_limits
        else 10000,
        "drawdown": drawdown,
    }


@router.post("/toggle")
async def toggle_autopilot(
    enabled: bool,
    current_user: Dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """Enable/disable AutoPilot"""
    user_id = current_user["id"]

    # Check subscription
    profile = supabase.table("profiles").select("is_active_subscriber").eq("user_id", user_id).execute()

    if not profile.data or not profile.data[0]["is_active_subscriber"]:
        raise HTTPException(
            status_code=403, detail="Active subscription required to use AutoPilot"
        )

    # Update profile
    supabase.table("profiles").update({"autopilot_enabled": enabled}).eq(
        "user_id", user_id
    ).execute()

    # Log audit
    supabase.table("audit_logs").insert({
        "user_id": user_id,
        "action": "AUTOPILOT_ENABLED" if enabled else "AUTOPILOT_DISABLED",
        "payload": {"enabled": enabled, "timestamp": datetime.utcnow().isoformat()},
    }).execute()

    # Send notification
    supabase.table("notifications").insert({
        "user_id": user_id,
        "type": "AUTOPILOT_PAUSED" if not enabled else "ORDER_PLACED",
        "title": f"AutoPilot {'Enabled' if enabled else 'Paused'}",
        "message": f"AutoPilot has been {'activated' if enabled else 'paused'}",
        "severity": "INFO",
    }).execute()

    return {"message": f"AutoPilot {'enabled' if enabled else 'disabled'}", "enabled": enabled}


@router.post("/kill-switch")
async def activate_kill_switch(
    current_user: Dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """Emergency kill switch - immediately pause AutoPilot and cancel all pending orders"""
    user_id = current_user["id"]

    # Disable AutoPilot
    supabase.table("profiles").update({"autopilot_enabled": False}).eq(
        "user_id", user_id
    ).execute()

    # Cancel all PENDING orders
    pending_orders = (
        supabase.table("orders")
        .select("id, broker_order_id, broker_type")
        .eq("user_id", user_id)
        .in_("status", ["PENDING", "OPEN"])
        .execute()
    )

    cancelled_count = len(pending_orders.data) if pending_orders.data else 0

    # Mark orders as cancelled (actual broker cancellation would happen via BrokerHub)
    for order in pending_orders.data:
        supabase.table("orders").update({"status": "CANCELLED"}).eq("id", order["id"]).execute()

    # Log audit
    supabase.table("audit_logs").insert({
        "user_id": user_id,
        "action": "KILL_SWITCH_ACTIVATED",
        "payload": {
            "timestamp": datetime.utcnow().isoformat(),
            "orders_cancelled": cancelled_count,
        },
    }).execute()

    # Send critical notification
    supabase.table("notifications").insert({
        "user_id": user_id,
        "type": "KILL_SWITCH_ACTIVATED",
        "title": "🚨 Kill Switch Activated",
        "message": f"AutoPilot paused. {cancelled_count} pending orders cancelled.",
        "severity": "ERROR",
    }).execute()

    return {
        "message": "Kill switch activated",
        "autopilot_enabled": False,
        "orders_cancelled": cancelled_count,
    }


@router.get("/today-actions")
async def get_today_actions(
    current_user: Dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """Get today's AutoPilot actions summary"""
    user_id = current_user["id"]
    today = datetime.utcnow().date().isoformat()

    # Orders placed today
    orders_result = (
        supabase.table("orders")
        .select("status, side", count="exact")
        .eq("user_id", user_id)
        .gte("created_at", f"{today}T00:00:00")
        .execute()
    )

    orders = orders_result.data if orders_result.data else []

    # Positions opened today
    positions_result = (
        supabase.table("positions")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("status", "OPEN")
        .gte("entry_timestamp", f"{today}T00:00:00")
        .execute()
    )

    # Positions closed today
    closed_positions_result = (
        supabase.table("positions")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("status", "CLOSED")
        .gte("exit_timestamp", f"{today}T00:00:00")
        .execute()
    )

    return {
        "date": today,
        "orders_placed": orders_result.count or 0,
        "buy_orders": len([o for o in orders if o["side"] == "BUY"]),
        "sell_orders": len([o for o in orders if o["side"] == "SELL"]),
        "positions_opened": positions_result.count or 0,
        "positions_closed": closed_positions_result.count or 0,
        "orders_filled": len([o for o in orders if o["status"] == "FILLED"]),
        "orders_pending": len([o for o in orders if o["status"] in ["PENDING", "OPEN"]]),
        "orders_rejected": len([o for o in orders if o["status"] == "REJECTED"]),
    }

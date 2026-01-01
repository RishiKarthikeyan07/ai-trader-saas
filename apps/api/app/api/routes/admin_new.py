"""
Admin Routes - Refactored for AutoPilot
Pipeline control, user management, system monitoring
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ...core.deps import get_current_user, get_supabase, require_admin
from supabase import Client

router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# PIPELINE MANAGEMENT
# ============================================================================

@router.post("/pipeline/run-daily")
async def trigger_daily_pipeline(
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """
    Trigger daily brain pipeline manually
    Generates trade_intentions for today
    """
    # Create pipeline run record
    run = supabase.table("pipeline_runs").insert({
        "type": "daily_brain",
        "status": "RUNNING",
        "started_at": datetime.utcnow().isoformat(),
        "metadata": {"triggered_by": current_user["id"], "manual": True},
    }).execute()

    run_id = run.data[0]["id"]

    # In production, this would trigger the worker via Redis queue
    # For now, return the run_id

    return {
        "message": "Daily pipeline triggered",
        "run_id": run_id,
        "status": "RUNNING",
    }


@router.post("/pipeline/run-executor")
async def trigger_executor_pipeline(
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """
    Trigger executor pipeline manually
    Executes pending trade entries/exits
    """
    run = supabase.table("pipeline_runs").insert({
        "type": "executor",
        "status": "RUNNING",
        "started_at": datetime.utcnow().isoformat(),
        "metadata": {"triggered_by": current_user["id"], "manual": True},
    }).execute()

    run_id = run.data[0]["id"]

    return {
        "message": "Executor pipeline triggered",
        "run_id": run_id,
        "status": "RUNNING",
    }


@router.get("/pipeline/runs")
async def get_pipeline_runs(
    type: Optional[str] = None,
    limit: int = 50,
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """Get pipeline run history"""
    query = supabase.table("pipeline_runs").select("*").order("started_at", desc=True).limit(limit)

    if type:
        query = query.eq("type", type)

    result = query.execute()

    return {"runs": result.data, "count": len(result.data)}


# ============================================================================
# USER MANAGEMENT
# ============================================================================

@router.get("/users")
async def get_all_users(
    limit: int = 100,
    offset: int = 0,
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """Get all users with subscription and AutoPilot status"""
    result = (
        supabase.table("profiles")
        .select("*, subscriptions(*)")
        .range(offset, offset + limit - 1)
        .execute()
    )

    return {"users": result.data, "count": len(result.data)}


@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: str,
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """Get detailed user info"""
    # Profile
    profile = supabase.table("profiles").select("*").eq("user_id", user_id).execute()

    if not profile.data:
        raise HTTPException(status_code=404, detail="User not found")

    # Subscription
    subscription = supabase.table("subscriptions").select("*").eq("user_id", user_id).execute()

    # Positions count
    positions = (
        supabase.table("positions")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("status", "OPEN")
        .execute()
    )

    # Orders count (today)
    today = datetime.utcnow().date().isoformat()
    orders = (
        supabase.table("orders")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .gte("created_at", f"{today}T00:00:00")
        .execute()
    )

    # Broker connections
    brokers = supabase.table("broker_connections").select("broker_type, status").eq("user_id", user_id).execute()

    return {
        "profile": profile.data[0],
        "subscription": subscription.data[0] if subscription.data else None,
        "positions_count": positions.count or 0,
        "orders_today": orders.count or 0,
        "broker_connections": brokers.data,
    }


@router.post("/users/{user_id}/toggle-autopilot")
async def admin_toggle_user_autopilot(
    user_id: str,
    enabled: bool,
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """Admin override to enable/disable user's AutoPilot"""
    supabase.table("profiles").update({"autopilot_enabled": enabled}).eq("user_id", user_id).execute()

    # Log
    supabase.table("audit_logs").insert({
        "user_id": user_id,
        "action": "AUTOPILOT_ENABLED" if enabled else "AUTOPILOT_DISABLED",
        "payload": {
            "admin_override": True,
            "admin_user_id": current_user["id"],
            "timestamp": datetime.utcnow().isoformat(),
        },
    }).execute()

    return {"message": f"AutoPilot {'enabled' if enabled else 'disabled'} for user {user_id}"}


# ============================================================================
# KILL SWITCH
# ============================================================================

@router.post("/kill/global")
async def global_kill_switch(
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """
    GLOBAL KILL SWITCH - Disable AutoPilot for ALL users
    Emergency use only
    """
    # Disable all users' AutoPilot
    supabase.table("profiles").update({"autopilot_enabled": False}).execute()

    # Cancel all PENDING orders
    pending_orders = (
        supabase.table("orders")
        .select("id", count="exact")
        .in_("status", ["PENDING", "OPEN"])
        .execute()
    )

    cancelled_count = pending_orders.count or 0

    # Mark all as cancelled
    supabase.table("orders").update({"status": "CANCELLED"}).in_(
        "status", ["PENDING", "OPEN"]
    ).execute()

    # Log
    supabase.table("audit_logs").insert({
        "action": "KILL_SWITCH_ACTIVATED",
        "payload": {
            "global": True,
            "admin_user_id": current_user["id"],
            "orders_cancelled": cancelled_count,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }).execute()

    # Notify all users
    users = supabase.table("profiles").select("user_id").eq("is_active_subscriber", True).execute()

    for user in users.data:
        supabase.table("notifications").insert({
            "user_id": user["user_id"],
            "type": "KILL_SWITCH_ACTIVATED",
            "title": "🚨 Global Kill Switch Activated",
            "message": "AutoPilot has been disabled system-wide by admin. All pending orders cancelled.",
            "severity": "ERROR",
        }).execute()

    return {
        "message": "Global kill switch activated",
        "users_affected": len(users.data),
        "orders_cancelled": cancelled_count,
    }


# ============================================================================
# MONITORING & LOGS
# ============================================================================

@router.get("/logs")
async def get_audit_logs(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """Get audit logs"""
    query = supabase.table("audit_logs").select("*").order("created_at", desc=True).limit(limit)

    if user_id:
        query = query.eq("user_id", user_id)

    if action:
        query = query.eq("action", action)

    result = query.execute()

    return {"logs": result.data, "count": len(result.data)}


@router.get("/metrics")
async def get_system_metrics(
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """Get system-wide metrics"""
    # Total users
    total_users = supabase.table("profiles").select("id", count="exact").execute()

    # Active subscribers
    active_subs = (
        supabase.table("profiles")
        .select("id", count="exact")
        .eq("is_active_subscriber", True)
        .execute()
    )

    # AutoPilot ON users
    autopilot_on = (
        supabase.table("profiles")
        .select("id", count="exact")
        .eq("autopilot_enabled", True)
        .execute()
    )

    # Open positions (system-wide)
    open_positions = (
        supabase.table("positions")
        .select("id", count="exact")
        .eq("status", "OPEN")
        .execute()
    )

    # Orders today
    today = datetime.utcnow().date().isoformat()
    orders_today = (
        supabase.table("orders")
        .select("id", count="exact")
        .gte("created_at", f"{today}T00:00:00")
        .execute()
    )

    # Broker connections status
    broker_health = supabase.table("broker_connections").select("status", count="exact").execute()

    connected_brokers = len([b for b in broker_health.data if b["status"] == "connected"])

    # Last pipeline runs
    last_daily = (
        supabase.table("pipeline_runs")
        .select("*")
        .eq("type", "daily_brain")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )

    last_executor = (
        supabase.table("pipeline_runs")
        .select("*")
        .eq("type", "executor")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )

    return {
        "users": {
            "total": total_users.count or 0,
            "active_subscribers": active_subs.count or 0,
            "autopilot_enabled": autopilot_on.count or 0,
        },
        "trading": {
            "open_positions": open_positions.count or 0,
            "orders_today": orders_today.count or 0,
        },
        "brokers": {
            "total_connections": len(broker_health.data),
            "connected": connected_brokers,
        },
        "pipelines": {
            "last_daily_run": last_daily.data[0] if last_daily.data else None,
            "last_executor_run": last_executor.data[0] if last_executor.data else None,
        },
    }


# ============================================================================
# TRADE INTENTIONS (INTERNAL VIEW)
# ============================================================================

@router.get("/trade-intentions")
async def get_trade_intentions(
    date: Optional[str] = None,
    limit: int = 50,
    current_user: Dict = Depends(require_admin),
    supabase: Client = Depends(get_supabase),
):
    """
    View trade_intentions (ADMIN ONLY - NOT exposed to users)
    This is internal system data
    """
    query = supabase.table("trade_intentions").select("*").order("confidence", desc=True).limit(limit)

    if date:
        query = query.eq("date", date)
    else:
        # Today by default
        today = datetime.utcnow().date().isoformat()
        query = query.eq("date", today)

    result = query.execute()

    return {"intentions": result.data, "count": len(result.data)}

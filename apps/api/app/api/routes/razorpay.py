"""
Razorpay Billing Routes
Handle subscription creation and webhooks
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import Dict, Any
import razorpay
import hmac
import hashlib
from datetime import datetime, timedelta

from ...core.deps import get_current_user, get_supabase
from ...core.config import settings
from supabase import Client

router = APIRouter(prefix="/billing/razorpay", tags=["billing"])

# Initialize Razorpay client
razorpay_client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))


@router.post("/create-subscription")
async def create_subscription(
    plan_period: str,  # 'monthly' or 'yearly'
    current_user: Dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Create Razorpay subscription
    Returns subscription_id and payment link
    """
    user_id = current_user["id"]
    email = current_user.get("email")

    if plan_period not in ["monthly", "yearly"]:
        raise HTTPException(status_code=400, detail="Invalid plan period")

    # Plan IDs (you'll create these in Razorpay Dashboard)
    plan_ids = {
        "monthly": "plan_autopilot_monthly",  # Replace with actual Razorpay plan ID
        "yearly": "plan_autopilot_yearly",  # Replace with actual Razorpay plan ID
    }

    try:
        # Create or get customer
        customer_result = (
            supabase.table("subscriptions")
            .select("razorpay_customer_id")
            .eq("user_id", user_id)
            .execute()
        )

        if customer_result.data and customer_result.data[0].get("razorpay_customer_id"):
            customer_id = customer_result.data[0]["razorpay_customer_id"]
        else:
            # Create new customer
            customer = razorpay_client.customer.create({
                "name": current_user.get("full_name", "User"),
                "email": email,
                "fail_existing": 0,
            })
            customer_id = customer["id"]

        # Create subscription
        subscription = razorpay_client.subscription.create({
            "plan_id": plan_ids[plan_period],
            "customer_id": customer_id,
            "total_count": 12 if plan_period == "monthly" else 1,  # 12 months or 1 year
            "quantity": 1,
            "notify_info": {
                "notify_email": email,
            },
            "notes": {
                "user_id": user_id,
                "plan_period": plan_period,
            },
        })

        # Store subscription in database
        current_period_start = datetime.utcnow()
        current_period_end = (
            current_period_start + timedelta(days=30)
            if plan_period == "monthly"
            else current_period_start + timedelta(days=365)
        )

        supabase.table("subscriptions").upsert({
            "user_id": user_id,
            "razorpay_subscription_id": subscription["id"],
            "razorpay_customer_id": customer_id,
            "plan_period": plan_period,
            "status": "active",
            "current_period_start": current_period_start.isoformat(),
            "current_period_end": current_period_end.isoformat(),
        }).execute()

        # Update profile
        supabase.table("profiles").update({"is_active_subscriber": True}).eq(
            "user_id", user_id
        ).execute()

        return {
            "subscription_id": subscription["id"],
            "status": subscription["status"],
            "plan_period": plan_period,
            "short_url": subscription.get("short_url"),  # Payment link
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Razorpay error: {str(e)}")


@router.post("/webhook")
async def razorpay_webhook(
    request: Request,
    supabase: Client = Depends(get_supabase),
):
    """
    Handle Razorpay webhook events
    IMPORTANT: Verify signature before processing
    """
    # Get raw body for signature verification
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature")

    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature")

    # Verify signature
    expected_signature = hmac.new(
        settings.RAZORPAY_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    if signature != expected_signature:
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    payload = await request.json()
    event = payload.get("event")

    if event == "subscription.activated":
        subscription_entity = payload["payload"]["subscription"]["entity"]
        subscription_id = subscription_entity["id"]
        user_id = subscription_entity["notes"].get("user_id")

        if user_id:
            supabase.table("subscriptions").update({"status": "active"}).eq(
                "razorpay_subscription_id", subscription_id
            ).execute()

            supabase.table("profiles").update({"is_active_subscriber": True}).eq(
                "user_id", user_id
            ).execute()

            # Send notification
            supabase.table("notifications").insert({
                "user_id": user_id,
                "type": "ORDER_PLACED",
                "title": "Subscription Activated",
                "message": "Your AutoPilot subscription is now active!",
                "severity": "INFO",
            }).execute()

    elif event == "subscription.cancelled":
        subscription_entity = payload["payload"]["subscription"]["entity"]
        subscription_id = subscription_entity["id"]

        supabase.table("subscriptions").update({"status": "cancelled"}).eq(
            "razorpay_subscription_id", subscription_id
        ).execute()

        # Get user_id
        sub_result = (
            supabase.table("subscriptions")
            .select("user_id")
            .eq("razorpay_subscription_id", subscription_id)
            .execute()
        )

        if sub_result.data:
            user_id = sub_result.data[0]["user_id"]

            # Disable AutoPilot
            supabase.table("profiles").update({
                "is_active_subscriber": False,
                "autopilot_enabled": False,
            }).eq("user_id", user_id).execute()

            # Send notification
            supabase.table("notifications").insert({
                "user_id": user_id,
                "type": "AUTOPILOT_PAUSED",
                "title": "Subscription Cancelled",
                "message": "Your subscription has been cancelled. AutoPilot has been disabled.",
                "severity": "WARNING",
            }).execute()

    elif event == "subscription.charged":
        # Payment successful
        subscription_entity = payload["payload"]["subscription"]["entity"]
        subscription_id = subscription_entity["id"]

        # Update current_period_end
        current_end = datetime.fromtimestamp(subscription_entity["current_end"])

        supabase.table("subscriptions").update({
            "current_period_end": current_end.isoformat()
        }).eq("razorpay_subscription_id", subscription_id).execute()

    elif event == "subscription.pending":
        # Payment pending (retry in progress)
        subscription_entity = payload["payload"]["subscription"]["entity"]
        subscription_id = subscription_entity["id"]

        supabase.table("subscriptions").update({"status": "paused"}).eq(
            "razorpay_subscription_id", subscription_id
        ).execute()

    # Log webhook event
    supabase.table("audit_logs").insert({
        "action": "AUTOPILOT_ENABLED",
        "payload": {
            "event": event,
            "subscription_id": payload.get("payload", {})
            .get("subscription", {})
            .get("entity", {})
            .get("id"),
            "timestamp": datetime.utcnow().isoformat(),
        },
    }).execute()

    return {"status": "success"}


@router.get("/status")
async def get_billing_status(
    current_user: Dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """Get user's subscription status"""
    user_id = current_user["id"]

    result = supabase.table("subscriptions").select("*").eq("user_id", user_id).execute()

    if not result.data:
        return {"has_subscription": False, "status": None}

    subscription = result.data[0]

    return {
        "has_subscription": True,
        "status": subscription["status"],
        "plan_period": subscription["plan_period"],
        "current_period_start": subscription["current_period_start"],
        "current_period_end": subscription["current_period_end"],
        "cancel_at_period_end": subscription["cancel_at_period_end"],
    }


@router.post("/cancel")
async def cancel_subscription(
    current_user: Dict = Depends(get_current_user),
    supabase: Client = Depends(get_supabase),
):
    """Cancel subscription (at period end)"""
    user_id = current_user["id"]

    # Get subscription
    result = supabase.table("subscriptions").select("*").eq("user_id", user_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="No active subscription")

    subscription = result.data[0]
    razorpay_subscription_id = subscription["razorpay_subscription_id"]

    try:
        # Cancel on Razorpay (at period end)
        razorpay_client.subscription.cancel(razorpay_subscription_id, {
            "cancel_at_cycle_end": 1
        })

        # Update database
        supabase.table("subscriptions").update({"cancel_at_period_end": True}).eq(
            "user_id", user_id
        ).execute()

        return {"message": "Subscription will be cancelled at period end"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

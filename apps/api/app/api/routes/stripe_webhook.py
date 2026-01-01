from fastapi import APIRouter, Request, HTTPException
from app.core.config import settings
from app.core.supabase import update_user_tier, log_audit_event
import stripe

router = APIRouter()
stripe.api_key = settings.STRIPE_SECRET_KEY


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle subscription events
    if event['type'] == 'customer.subscription.created':
        await handle_subscription_created(event['data']['object'])
    elif event['type'] == 'customer.subscription.updated':
        await handle_subscription_updated(event['data']['object'])
    elif event['type'] == 'customer.subscription.deleted':
        await handle_subscription_deleted(event['data']['object'])

    return {"status": "success"}


async def handle_subscription_created(subscription):
    """Handle new subscription"""
    user_id = subscription['metadata'].get('user_id')
    tier = get_tier_from_price_id(subscription['items']['data'][0]['price']['id'])

    if user_id and tier:
        await update_user_tier(user_id, tier)
        await log_audit_event(user_id, 'subscription_created', {
            'tier': tier,
            'subscription_id': subscription['id']
        })


async def handle_subscription_updated(subscription):
    """Handle subscription update"""
    user_id = subscription['metadata'].get('user_id')
    tier = get_tier_from_price_id(subscription['items']['data'][0]['price']['id'])

    if user_id and tier:
        await update_user_tier(user_id, tier)
        await log_audit_event(user_id, 'subscription_updated', {
            'tier': tier,
            'subscription_id': subscription['id']
        })


async def handle_subscription_deleted(subscription):
    """Handle subscription cancellation"""
    user_id = subscription['metadata'].get('user_id')

    if user_id:
        await update_user_tier(user_id, 'basic')
        await log_audit_event(user_id, 'subscription_cancelled', {
            'subscription_id': subscription['id']
        })


def get_tier_from_price_id(price_id: str) -> str:
    """Map Stripe price ID to tier"""
    # These should be configured in environment variables
    price_tier_map = {
        'price_pro_monthly': 'pro',
        'price_pro_yearly': 'pro',
        'price_elite_monthly': 'elite',
        'price_elite_yearly': 'elite',
    }
    return price_tier_map.get(price_id, 'basic')

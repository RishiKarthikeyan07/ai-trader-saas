from supabase import create_client, Client
from app.core.config import settings

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


async def get_user_tier(user_id: str) -> str:
    """Get user tier from Supabase"""
    response = supabase.table('profiles').select('tier').eq('user_id', user_id).single().execute()
    return response.data.get('tier', 'basic') if response.data else 'basic'


async def can_access_signal(user_id: str, signal_rank: int) -> bool:
    """Check if user can access signal based on tier"""
    tier = await get_user_tier(user_id)

    tier_limits = {
        'basic': 3,
        'pro': 10,
        'elite': float('inf')
    }

    return signal_rank <= tier_limits.get(tier, 3)


async def get_user_settings(user_id: str):
    """Get user settings"""
    response = supabase.table('profiles').select('*').eq('user_id', user_id).single().execute()
    return response.data


async def update_user_tier(user_id: str, tier: str):
    """Update user tier"""
    supabase.table('profiles').update({'tier': tier}).eq('user_id', user_id).execute()


async def log_audit_event(user_id: str, event_type: str, details: dict):
    """Log audit event"""
    supabase.table('audit_logs').insert({
        'user_id': user_id,
        'event_type': event_type,
        'details': details
    }).execute()

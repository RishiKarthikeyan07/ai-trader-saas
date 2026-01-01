from fastapi import APIRouter, Depends
from app.models.schemas import UserProfileResponse
from app.core.security import get_current_user
from app.core.supabase import supabase

router = APIRouter()


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(user_id: str = Depends(get_current_user)):
    """Get current user profile"""
    response = supabase.table('profiles') \
        .select('*') \
        .eq('user_id', user_id) \
        .single() \
        .execute()

    return response.data

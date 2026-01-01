"""
FastAPI Dependencies
Authentication, database clients, etc.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict
from supabase import create_client, Client
import jwt

from .config import settings

# Security
security = HTTPBearer()


def get_supabase() -> Client:
    """Get Supabase client"""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase),
) -> Dict:
    """
    Get current authenticated user from JWT token
    Token should be Supabase auth token
    """
    token = credentials.credentials

    try:
        # Verify token with Supabase
        user = supabase.auth.get_user(token)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )

        # Get profile
        profile = supabase.table("profiles").select("*").eq("user_id", user.user.id).execute()

        if not profile.data:
            # Create profile if doesn't exist
            profile_data = {
                "user_id": user.user.id,
                "email": user.user.email,
                "is_active_subscriber": False,
                "autopilot_enabled": False,
                "is_admin": False,
            }
            supabase.table("profiles").insert(profile_data).execute()
            return {**user.user.model_dump(), **profile_data}

        return {**user.user.model_dump(), **profile.data[0]}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
        )


async def require_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Require admin role"""
    if not current_user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def require_subscription(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Require active subscription"""
    if not current_user.get("is_active_subscriber"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Active subscription required",
        )
    return current_user

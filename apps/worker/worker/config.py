"""
Worker Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Environment
    ENVIRONMENT: str = "development"

    # Market Hours (IST)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MINUTE: int = 30

    # Pipeline Settings
    DAILY_PIPELINE_HOUR: int = 7  # 7 AM IST
    EXECUTOR_INTERVAL_MINUTES: int = 15
    POSITION_MONITOR_INTERVAL_MINUTES: int = 5

    # Data Sources
    ALPHAVANTAGE_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# Supabase client factory
def get_supabase_client() -> Client:
    """
    Create and return a Supabase client instance.
    Uses service role key for full database access.
    """
    try:
        client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        raise

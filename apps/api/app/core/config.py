from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "AI Trader API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_ANON_KEY: str

    # Razorpay (replaces Stripe)
    RAZORPAY_KEY_ID: str
    RAZORPAY_KEY_SECRET: str
    RAZORPAY_WEBHOOK_SECRET: str

    # Encryption
    ENCRYPTION_KEY: str  # For encrypting broker tokens

    # SendGrid
    SENDGRID_API_KEY: str
    FROM_EMAIL: str

    # Redis
    REDIS_URL: Optional[str] = None

    # Feature Flags
    ENABLE_LIVE_TRADING: bool = False
    ENABLE_AUTO_TRADING: bool = True

    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None

    # Sentry
    SENTRY_DSN: Optional[str] = None

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

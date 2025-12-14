from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    env: str = Field("development")
    mode: str = Field("stub")  # stub | prod
    api_prefix: str = "/api"
    alphavantage_api_key: str | None = Field(default=None)
    supabase_url: str | None = Field(default=None)
    supabase_key: str | None = Field(default=None)
    redis_url: str | None = Field(default=None)
    duckdb_path: Path = Field(default=Path("cache/market.duckdb"))
    data_cache_dir: Path = Field(default=Path("cache"))
    kronos_artifact_path: Path | None = Field(default=None)
    stockformer_artifact_path: Path | None = Field(default=None)
    tft_artifact_path: Path | None = Field(default=None)
    catboost_artifact_path: Path | None = Field(default=None)
    default_universe: str = "NIFTY500"
    max_daily_signals: int = 10
    max_hourly_signals: int = 50
    tier_basic_limit: int = 3
    tier_pro_limit: int = 10
    tier_elite_limit: int = 50
    max_open_positions: int = 8
    daily_loss_limit_pct: float = 0.03
    position_size_cap_pct: float = 0.08
    timezone: str = "Asia/Kolkata"

    model_config = SettingsConfigDict(case_sensitive=False, env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.data_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "vclone-api"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/v1"
    database_url: str = "sqlite:///./vclone.db"
    signed_url_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

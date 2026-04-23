from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "vclone-api"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/v1"
    database_url: str = "sqlite:///./vclone.db"
    signed_url_ttl_seconds: int = 3600
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_device: str = "auto"
    xtts_default_language: str = "en"
    xtts_temperature: float = 0.7
    xtts_enable_text_splitting: bool = True
    xtts_final_temperature: float = 0.55
    xtts_preview_temperature: float = 0.75
    engine_prefer_premium_final: bool = True
    premium_engine_enabled: bool = False
    premium_engine_name: str = "premium_final_placeholder"
    fail_on_derived_final_master: bool = True
    delivery_default_format: str = "flac"
    delivery_default_sample_rate_hz: int = 48000
    delivery_default_channels: int = 2
    delivery_target_lufs: float = -16.0
    delivery_true_peak_db: float = -1.5
    delivery_target_lra: float = 7.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

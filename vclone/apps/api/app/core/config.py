from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "vclone-api"  # Human-readable API/service name used in responses and docs.
    app_version: str = "0.1.0"  # Current API version string.
    api_v1_prefix: str = "/v1"  # Base prefix for versioned HTTP routes.
    database_url: str = "sqlite:///./vclone.db"  # SQLAlchemy connection string for the app database.
    signed_url_ttl_seconds: int = 3600  # Download URL lifetime for generated assets in seconds.

    upload_chunk_size_bytes: int = 8 * 1024 * 1024  # Browser resumable-upload chunk size.
    upload_max_size_bytes: int = 10 * 1024 * 1024 * 1024  # Maximum accepted upload size for local MVP storage.

    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"  # Coqui TTS model identifier used for XTTS inference.
    xtts_device: str = "auto"  # Inference device: auto/cpu/cuda.
    xtts_default_language: str = "en"  # Default XTTS language code when locale is not provided.
    xtts_temperature: float = 0.7  # General XTTS sampling temperature for balanced generation.
    xtts_enable_text_splitting: bool = True  # Allow XTTS internal sentence splitting during synthesis.
    xtts_final_temperature: float = 0.55  # Lower-temperature target intended for more stable final renders.
    xtts_preview_temperature: float = 0.75  # Higher-temperature target intended for faster/more flexible previews.

    asr_model_size: str = "small"  # faster-whisper model size for transcription/back-check.
    asr_device: str = "auto"  # ASR inference device: auto/cpu/cuda.
    asr_compute_type: str = "auto"  # faster-whisper compute type: auto/int8/float16/etc.
    asr_beam_size: int = 5  # Beam size for transcription decoding quality.

    engine_prefer_premium_final: bool = True  # Prefer the premium final engine path when enabled.
    premium_engine_enabled: bool = False  # Master flag to allow premium final engine selection.
    premium_engine_name: str = "premium_final_placeholder"  # Friendly label/config slot for the premium final engine.
    fail_on_derived_final_master: bool = True  # Reject final renders that are only packaged/up-sampled/dual-mono derivatives.

    delivery_default_format: str = "flac"  # Default download format for final delivery.
    delivery_default_sample_rate_hz: int = 48000  # Default requested output sample rate.
    delivery_default_channels: int = 2  # Default requested output channel count.
    delivery_target_lufs: float = -16.0  # Loudness normalization target in LUFS.
    delivery_true_peak_db: float = -1.5  # True-peak ceiling in dBTP.
    delivery_target_lra: float = 7.0  # Loudness range target used during mastering.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

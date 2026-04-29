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

    primary_tts_engine: str = "voxcpm2"  # Preferred free local engine for best clone quality.
    engine_allow_xtts_fallback: bool = False  # Do not silently fall back to XTTS because it produced poor clone quality.
    synthesis_max_chunk_chars: int = 140  # Shorter chunks reduce voice drift for clone engines.

    voxcpm_model_name: str = "openbmb/VoxCPM2"  # VoxCPM2 HF/model identifier.
    voxcpm_device: str = "cpu"  # auto/cpu/cuda/mps for VoxCPM2 when supported by runtime.
    voxcpm_allow_cpu: bool = True  # Allow CPU fallback; Mac M-series users may accept slow primary-engine inference.
    voxcpm_load_denoiser: bool = False  # Upstream quickstart default; faster and less processing.
    voxcpm_cfg_value: float = 2.0  # Upstream recommended CFG value.
    voxcpm_inference_timesteps: int = 10  # Upstream quickstart default.
    voxcpm_optimize: bool = False  # Stability first; avoid VoxCPM warm-up/compile path unless explicitly enabled.
    voxcpm_enable_ultimate: bool = False  # Ultimate/Hi-Fi mode is quarantined until a prompt smoke test proves it does not leak.
    synthesis_chunk_timeout_seconds: int = 900  # User-facing timeout guidance for long model calls.
    synthesis_single_chunk_timeout_seconds: int = 900  # Timeout applied to one isolated generated chunk.
    synthesis_resume_existing_chunks: bool = True  # Reuse already-rendered chunk WAVs when retrying a failed/partial job.
    synthesis_allow_partial_output: bool = True  # Master/stitch completed chunks even if later chunks timeout.
    synthesis_long_text_chunk_threshold: int = 8  # Use long-form settings when text splits into at least this many chunks.
    synthesis_long_text_chunk_chars: int = 240  # Larger chunks reduce CPU VoxCPM orchestration overhead for long text.
    synthesis_long_text_candidate_limit: int = 1  # Long CPU VoxCPM jobs lock one reference candidate to avoid bakeoff blowups.
    synthesis_heartbeat_interval_seconds: int = 5  # Progress heartbeat while isolated synthesis worker is running.
    synthesis_stale_seconds: int = 180  # Mark running jobs failed when no heartbeat/update arrives in this window.

    chatterbox_device: str = "auto"  # auto/cpu/cuda for Chatterbox.
    chatterbox_variant: str = "original"  # original is the stable pip API; turbo is used only when explicitly installed/supported.
    chatterbox_language_id: str = "en"  # Used by multilingual Chatterbox.

    voice_dataset_min_segment_seconds: float = 2.0  # Hard minimum for clone dataset clips.
    voice_dataset_max_segment_seconds: float = 20.0  # Hard maximum for stable clone dataset clips.
    voice_prompt_target_seconds: int = 20  # Exact prompt pack size for VoxCPM2 ultimate cloning.
    voice_prompt_candidate_count: int = 16  # Number of prompt-bank candidates to keep from enrollment.
    voice_prompt_min_seconds: float = 5.0  # Minimum safe prompt/reference duration for VoxCPM candidate audio.
    voice_prompt_max_seconds: float = 30.0  # Maximum safe prompt/reference duration for VoxCPM candidate audio.
    voice_prompt_min_non_silent_seconds: float = 3.0  # Minimum non-silent speech required in a prompt candidate.
    voice_prompt_duration_tolerance_ratio: float = 0.15  # Reject candidates whose actual duration drifts too far from expected extraction duration.
    voice_dataset_validate_with_asr: bool = True  # Validate curated segments against ASR when possible.
    voice_dataset_max_segment_wer: float = 0.35  # Reject segments whose ASR diverges too much from cleaned transcript.
    voice_dataset_hard_reject_with_asr: bool = False  # ASR on short accented segments can be noisy; score by default rather than hard reject.
    voice_dataset_hard_reject_min_confidence: float = 0.75  # Only hard-reject ASR mismatch when the ASR itself is confident.

    synthesis_preview_candidate_limit: int = 3  # Number of prompt/strategy candidates to try for preview selection.
    synthesis_final_candidate_limit: int = 2  # Number of prompt/strategy candidates to try for final selection.
    synthesis_candidate_max_wer: float = 0.45  # Threshold beyond which a candidate is considered poor text fidelity.
    synthesis_similarity_hard_gate: bool = False  # Speaker embeddings rank candidates, but should not alone fail synthesis.
    synthesis_enable_chatterbox_bakeoff: bool = False  # Keep preview stable; only compare Chatterbox when explicitly enabled.
    synthesis_pause_sentence_ms: int = 650  # Insert a natural sentence pause between rendered chunks.
    synthesis_pause_clause_ms: int = 250  # Insert a shorter pause after clause-level boundaries.
    synthesis_pause_default_ms: int = 150  # Default inter-chunk pause when text gives no better hint.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

from typing import Any

from pydantic import BaseModel, Field


class SynthesisRequest(BaseModel):
    voice_profile_id: str
    text: str = Field(..., min_length=1)
    mode: str = "preview"
    format: str = "wav"
    sample_rate_hz: int = 24000
    channels: int = 1
    locale: str = "en-IN"
    require_native_master: bool = False


class SynthesisJobResponse(BaseModel):
    job_id: str
    status: str
    message: str | None = None


class SynthesisPreviewResponse(BaseModel):
    job_id: str
    status: str
    preview_text: str
    chunks: list[str]
    request: dict[str, Any]


class SynthesizedAssetInfo(BaseModel):
    format: str
    sample_rate_hz: int
    channels: int
    duration_ms: int
    local_path: str
    checksum_sha256: str


class DownloadUrlResponse(BaseModel):
    url: str
    expires_in_seconds: int
    asset: SynthesizedAssetInfo
    delivery_report: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    asr_backcheck: dict[str, Any] = Field(default_factory=dict)
    engine_selection: dict[str, Any] = Field(default_factory=dict)
    engine_registry: dict[str, Any] = Field(default_factory=dict)

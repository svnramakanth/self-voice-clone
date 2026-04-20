from pydantic import BaseModel, Field


class SynthesisRequest(BaseModel):
    voice_profile_id: str
    text: str = Field(..., min_length=1)
    mode: str = "preview"
    format: str = "wav"
    sample_rate_hz: int = 24000
    locale: str = "en-IN"


class SynthesisJobResponse(BaseModel):
    job_id: str
    status: str


class SynthesisPreviewResponse(BaseModel):
    job_id: str
    status: str
    preview_text: str
    chunks: list[str]


class DownloadUrlResponse(BaseModel):
    url: str
    expires_in_seconds: int

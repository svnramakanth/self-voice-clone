from pydantic import BaseModel


class CreateVoiceProfileRequest(BaseModel):
    mode: str = "mvp_embedding"
    engine_preference: str = "auto"
    allow_adaptation: bool = True


class VoiceProfileResponse(BaseModel):
    voice_profile_id: str
    job_id: str


class SimpleVoiceProfileResponse(BaseModel):
    voice_profile_id: str
    status: str
    name: str


class VoiceProfileDetail(BaseModel):
    id: str
    name: str
    enrollment_id: str
    status: str
    engine_family: str
    base_model_version: str
    readiness_report: dict


class VoiceProfileListResponse(BaseModel):
    items: list[VoiceProfileDetail]

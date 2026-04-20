import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas.voice_profile import (
    CreateVoiceProfileRequest,
    SimpleVoiceProfileResponse,
    VoiceProfileDetail,
    VoiceProfileListResponse,
    VoiceProfileResponse,
)
from app.services.voice_profiles import VoiceProfileService


router = APIRouter()


@router.post("", response_model=SimpleVoiceProfileResponse)
async def create_simple_voice_profile(
    name: str = Form("My Voice"),
    transcript_text: str = Form(""),
    audio_file: UploadFile = File(...),
    transcript_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db_session),
) -> SimpleVoiceProfileResponse:
    if not transcript_text.strip() and transcript_file is None:
        raise HTTPException(status_code=400, detail="Provide transcript text or an SRT/TXT transcript file")

    service = VoiceProfileService(db)
    service.ensure_schema()
    try:
        profile = service.create_simple_profile(
            name=name,
            transcript_text=transcript_text,
            audio_filename=audio_file.filename or "sample-audio.bin",
            audio_bytes=await audio_file.read(),
            transcript_filename=transcript_file.filename if transcript_file else None,
            transcript_bytes=await transcript_file.read() if transcript_file else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SimpleVoiceProfileResponse(voice_profile_id=profile.id, status=profile.status, name=profile.name)


@router.post("/create/from-enrollment/{enrollment_id}", response_model=VoiceProfileResponse)
def create_voice_profile(
    enrollment_id: str, payload: CreateVoiceProfileRequest, db: Session = Depends(get_db_session)
) -> VoiceProfileResponse:
    try:
        profile, job_id = VoiceProfileService(db).create_profile(enrollment_id, **payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return VoiceProfileResponse(voice_profile_id=profile.id, job_id=job_id)


@router.get("", response_model=VoiceProfileListResponse)
def list_voice_profiles(db: Session = Depends(get_db_session)) -> VoiceProfileListResponse:
    service = VoiceProfileService(db)
    service.ensure_schema()
    profiles = service.list_profiles()
    items = [
        VoiceProfileDetail(
            id=profile.id,
            name=profile.name,
            enrollment_id=profile.enrollment_id,
            status=profile.status,
            engine_family=profile.engine_family,
            base_model_version=profile.base_model_version,
            readiness_report=json.loads(profile.readiness_report_json or "{}"),
        )
        for profile in profiles
    ]
    return VoiceProfileListResponse(items=items)


@router.get("/{voice_profile_id}", response_model=VoiceProfileDetail)
def get_voice_profile(voice_profile_id: str, db: Session = Depends(get_db_session)) -> VoiceProfileDetail:
    service = VoiceProfileService(db)
    service.ensure_schema()
    profile = service.get_profile(voice_profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    return VoiceProfileDetail(
        id=profile.id,
        name=profile.name,
        enrollment_id=profile.enrollment_id,
        status=profile.status,
        engine_family=profile.engine_family,
        base_model_version=profile.base_model_version,
        readiness_report=json.loads(profile.readiness_report_json or "{}"),
    )

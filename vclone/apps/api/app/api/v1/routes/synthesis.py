from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas.synthesis import DownloadUrlResponse, SynthesisJobResponse, SynthesisPreviewResponse, SynthesisRequest
from app.services.synthesis import SynthesisService


router = APIRouter()


@router.post("", response_model=SynthesisJobResponse)
def submit_synthesis(payload: SynthesisRequest, db: Session = Depends(get_db_session)) -> SynthesisJobResponse:
    try:
        job = SynthesisService(db).create_job(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SynthesisJobResponse(job_id=job.id, status=job.status.upper())


@router.get("/{job_id}/preview", response_model=SynthesisPreviewResponse)
def get_preview(job_id: str, db: Session = Depends(get_db_session)) -> SynthesisPreviewResponse:
    try:
        return SynthesisPreviewResponse(**SynthesisService(db).get_preview(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{job_id}/download-url", response_model=DownloadUrlResponse)
def create_download_url(job_id: str, db: Session = Depends(get_db_session)) -> DownloadUrlResponse:
    try:
        return DownloadUrlResponse(**SynthesisService(db).get_download_url(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

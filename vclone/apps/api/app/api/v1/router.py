from fastapi import APIRouter

from app.api.v1.routes import enrollments, synthesis, voice_profiles


api_router = APIRouter()
api_router.include_router(enrollments.router, prefix="/enrollments", tags=["enrollments"])
api_router.include_router(voice_profiles.router, prefix="/voice-profiles", tags=["voice-profiles"])
api_router.include_router(synthesis.router, prefix="/synthesis", tags=["synthesis"])

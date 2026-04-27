from fastapi import APIRouter

from app.services.engine_registry import EngineRegistry


router = APIRouter()


@router.get("/capabilities")
def get_system_capabilities() -> dict:
    registry = EngineRegistry()
    return {
        "status": "ok",
        "engines": registry.describe(),
        "summary": registry.summary(),
    }
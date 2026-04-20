from app.workers.bootstrap import init_db

init_db()

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Personal voice clone TTS MVP API",
)


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


app.include_router(api_router, prefix=settings.api_v1_prefix)

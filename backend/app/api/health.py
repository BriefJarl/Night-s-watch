from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings


router = APIRouter(
    prefix="/api/v1",
    tags=["System Health"],
)


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
def readiness_check():
    return {
        "status": "ready",
        "demo_mode": settings.DEMO_MODE,
    }
from fastapi import (
    APIRouter,
    Depends,
    Query,
)

from sqlalchemy.orm import Session

from app.core.database import get_db

from app.schemas.dashboard import (
    CameraStatusResponse,
    DashboardSummaryResponse,
    DetectionTypeAnalyticsResponse,
    RecentDetectionResponse,
)

from app.services.dashboard_service import (
    DashboardService,
)


router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"],
)


dashboard_service = DashboardService()


@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
)
def get_dashboard_summary(
    db: Session = Depends(get_db),
):

    return dashboard_service.get_summary(
        db=db,
    )


@router.get(
    "/detections-by-type",
    response_model=DetectionTypeAnalyticsResponse,
)
def get_detections_by_type(
    db: Session = Depends(get_db),
):

    return (
        dashboard_service.get_detections_by_type(
            db=db,
        )
    )


@router.get(
    "/recent-detections",
    response_model=list[RecentDetectionResponse],
)
def get_recent_detections(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
):

    return (
        dashboard_service.get_recent_detections(
            db=db,
            limit=limit,
        )
    )


@router.get(
    "/camera-status",
    response_model=CameraStatusResponse,
)
def get_camera_status(
    db: Session = Depends(get_db),
):

    return dashboard_service.get_camera_status(
        db=db,
    )
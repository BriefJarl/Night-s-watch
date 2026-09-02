from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from sqlalchemy.orm import Session

from app.core.database import get_db

from app.schemas.alert import (
    AlertListResponse,
    AlertResponse,
    AlertResolveResponse,
)

from app.services.alert_service import AlertService


router = APIRouter(
    prefix="/api/v1/alerts",
    tags=["Alerts"],
)


alert_service = AlertService()


@router.get(
    "/",
    response_model=AlertListResponse,
)
def get_alerts(
    skip: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    unresolved_only: bool = Query(
        default=False,
    ),
    db: Session = Depends(get_db),
):

    alerts = alert_service.get_alerts(
        db=db,
        skip=skip,
        limit=limit,
        unresolved_only=unresolved_only,
    )

    return {
        "alerts": alerts,
        "total": len(alerts),
    }


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
)
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
):

    alert = alert_service.get_alert(
        db=db,
        alert_id=alert_id,
    )

    if alert is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    return alert


@router.put(
    "/{alert_id}/resolve",
    response_model=AlertResolveResponse,
)
def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
):

    alert = alert_service.resolve_alert(
        db=db,
        alert_id=alert_id,
    )

    if alert is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    return alert
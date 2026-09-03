from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.detection_event import (
    DetectionEventCreate,
    DetectionEventResponse,
)
from app.services.detection_event_service import (
    DetectionEventService,
)


# NOTE: the video-upload endpoint and the YOLO/OpenCV imports were removed
# from this module for cloud deployment. Importing app.ai.* pulls in torch,
# which does not fit in the 512 MB free-tier memory limit. Live inference
# runs from the local development environment.


router = APIRouter(
    prefix="/api/v1/detection-events",
    tags=["Detection Events"],
)


detection_event_service = DetectionEventService()


@router.post(
    "/",
    response_model=DetectionEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_detection_event(
    detection_event_data: DetectionEventCreate,
    db: Session = Depends(get_db),
):

    try:
        return detection_event_service.create_detection_event(
            db=db,
            detection_event_data=detection_event_data,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.get(
    "/",
    response_model=List[DetectionEventResponse],
)
def get_detection_events(
    skip: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    db: Session = Depends(get_db),
):

    return detection_event_service.get_detection_events(
        db=db,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{event_id}",
    response_model=DetectionEventResponse,
)
def get_detection_event(
    event_id: int,
    db: Session = Depends(get_db),
):

    try:
        event = detection_event_service.get_detection_event(
            db=db,
            event_id=event_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )

    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection event not found",
        )

    return event


@router.delete(
    "/{event_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_detection_event(
    event_id: int,
    db: Session = Depends(get_db),
):

    try:
        deleted = (
            detection_event_service.delete_detection_event(
                db=db,
                event_id=event_id,
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection event not found",
        )

    return None

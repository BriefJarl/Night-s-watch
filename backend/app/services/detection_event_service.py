from sqlalchemy.orm import Session

from app.models.detection_event import DetectionEvent
from app.repositories.detection_event_repository import (
    DetectionEventRepository,
)
from app.schemas.detection_event import DetectionEventCreate


class DetectionEventService:

    def __init__(self) -> None:
        self.repository = DetectionEventRepository()


    def create_detection_event(
        self,
        db: Session,
        detection_event_data: DetectionEventCreate,
    ) -> DetectionEvent:

        object_type = detection_event_data.object_type.strip()

        if not object_type:
            raise ValueError(
                "Object type cannot be empty."
            )

        if detection_event_data.camera_id <= 0:
            raise ValueError(
                "Camera ID must be greater than zero."
            )

        if not 0.0 <= detection_event_data.confidence <= 1.0:
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

        return self.repository.create(
            db=db,
            camera_id=detection_event_data.camera_id,
            object_type=object_type,
            confidence=detection_event_data.confidence,
        )


    def get_detection_event(
        self,
        db: Session,
        event_id: int,
    ) -> DetectionEvent | None:

        if event_id <= 0:
            raise ValueError(
                "Event ID must be greater than zero."
            )

        return self.repository.get_by_id(
            db=db,
            event_id=event_id,
        )


    def get_detection_events(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DetectionEvent]:

        if skip < 0:
            raise ValueError(
                "Skip cannot be negative."
            )

        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero."
            )

        if limit > 500:
            limit = 500

        return self.repository.get_all(
            db=db,
            skip=skip,
            limit=limit,
        )


    def get_camera_detection_events(
        self,
        db: Session,
        camera_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DetectionEvent]:

        if camera_id <= 0:
            raise ValueError(
                "Camera ID must be greater than zero."
            )

        if skip < 0:
            raise ValueError(
                "Skip cannot be negative."
            )

        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero."
            )

        if limit > 500:
            limit = 500

        return self.repository.get_by_camera_id(
            db=db,
            camera_id=camera_id,
            skip=skip,
            limit=limit,
        )


    def delete_detection_event(
        self,
        db: Session,
        event_id: int,
    ) -> bool:

        if event_id <= 0:
            raise ValueError(
                "Event ID must be greater than zero."
            )

        event = self.repository.get_by_id(
            db=db,
            event_id=event_id,
        )

        if event is None:
            return False

        self.repository.delete(
            db=db,
            event=event,
        )

        return True
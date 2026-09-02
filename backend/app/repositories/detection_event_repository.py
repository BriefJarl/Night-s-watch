from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.detection_event import DetectionEvent


class DetectionEventRepository:

    def create(
        self,
        db: Session,
        camera_id: int,
        object_type: str,
        confidence: float,
    ) -> DetectionEvent:

        detection_event = DetectionEvent(
            camera_id=camera_id,
            object_type=object_type,
            confidence=confidence,
        )

        try:
            db.add(detection_event)
            db.commit()
            db.refresh(detection_event)

            return detection_event

        except Exception:
            db.rollback()
            raise


    def get_by_id(
        self,
        db: Session,
        event_id: int,
    ) -> DetectionEvent | None:

        statement = select(DetectionEvent).where(
            DetectionEvent.id == event_id
        )

        result = db.execute(statement)

        return result.scalar_one_or_none()


    def get_all(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DetectionEvent]:

        statement = (
            select(DetectionEvent)
            .order_by(
                DetectionEvent.detected_at.desc()
            )
            .offset(skip)
            .limit(limit)
        )

        result = db.execute(statement)

        return list(result.scalars().all())


    def get_by_camera_id(
        self,
        db: Session,
        camera_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DetectionEvent]:

        statement = (
            select(DetectionEvent)
            .where(
                DetectionEvent.camera_id == camera_id
            )
            .order_by(
                DetectionEvent.detected_at.desc()
            )
            .offset(skip)
            .limit(limit)
        )

        result = db.execute(statement)

        return list(result.scalars().all())


    def delete(
        self,
        db: Session,
        event: DetectionEvent,
    ) -> None:

        try:
            db.delete(event)
            db.commit()

        except Exception:
            db.rollback()
            raise
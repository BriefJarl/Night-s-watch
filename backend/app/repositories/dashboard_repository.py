from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.camera import Camera
from app.models.detection_event import DetectionEvent


class DashboardRepository:

    def get_summary(
        self,
        db: Session,
    ) -> dict:

        total_cameras = db.scalar(
            select(func.count(Camera.id))
        ) or 0

        active_cameras = db.scalar(
            select(func.count(Camera.id)).where(
                Camera.is_active.is_(True)
            )
        ) or 0

        inactive_cameras = total_cameras - active_cameras

        total_detection_events = db.scalar(
            select(func.count(DetectionEvent.id))
        ) or 0

        last_24_hours = (
            datetime.utcnow()
            - timedelta(hours=24)
        )

        detections_last_24_hours = db.scalar(
            select(func.count(DetectionEvent.id)).where(
                DetectionEvent.detected_at >= last_24_hours
            )
        ) or 0

        return {
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "inactive_cameras": inactive_cameras,
            "total_detection_events": total_detection_events,
            "detections_last_24_hours": detections_last_24_hours,
        }


    def get_detections_by_type(
        self,
        db: Session,
    ) -> list[dict]:

        statement = (
            select(
                DetectionEvent.object_type,
                func.count(
                    DetectionEvent.id
                ).label("count"),
                func.avg(
                    DetectionEvent.confidence
                ).label("average_confidence"),
            )
            .group_by(
                DetectionEvent.object_type
            )
            .order_by(
                func.count(
                    DetectionEvent.id
                ).desc()
            )
        )

        results = db.execute(statement).all()

        return [
            {
                "object_type": row.object_type,
                "count": row.count,
                "average_confidence": round(
                    float(row.average_confidence),
                    4,
                ),
            }
            for row in results
        ]


    def get_recent_detections(
        self,
        db: Session,
        limit: int,
    ) -> list[dict]:

        statement = (
            select(
                DetectionEvent.id,
                DetectionEvent.camera_id,
                Camera.name.label("camera_name"),
                DetectionEvent.object_type,
                DetectionEvent.confidence,
                DetectionEvent.detected_at,
            )
            .outerjoin(
                Camera,
                Camera.id == DetectionEvent.camera_id,
            )
            .order_by(
                DetectionEvent.detected_at.desc()
            )
            .limit(limit)
        )

        results = db.execute(statement).all()

        return [
            {
                "id": row.id,
                "camera_id": row.camera_id,
                "camera_name": row.camera_name,
                "object_type": row.object_type,
                "confidence": float(row.confidence),
                "detected_at": row.detected_at,
            }
            for row in results
        ]


    def get_camera_status(
        self,
        db: Session,
    ) -> dict:

        detection_count_subquery = (
            select(
                DetectionEvent.camera_id.label("camera_id"),
                func.count(
                    DetectionEvent.id
                ).label("total_detections"),
            )
            .group_by(
                DetectionEvent.camera_id
            )
            .subquery()
        )

        statement = (
            select(
                Camera.id,
                Camera.name,
                Camera.location,
                Camera.is_active,
                func.coalesce(
                    detection_count_subquery.c.total_detections,
                    0,
                ).label("total_detections"),
            )
            .outerjoin(
                detection_count_subquery,
                Camera.id
                == detection_count_subquery.c.camera_id,
            )
            .order_by(
                Camera.id.asc()
            )
        )

        results = db.execute(statement).all()

        cameras = [
            {
                "id": row.id,
                "name": row.name,
                "location": row.location,
                "is_active": row.is_active,
                "total_detections": int(
                    row.total_detections
                ),
            }
            for row in results
        ]

        total_cameras = len(cameras)

        active_cameras = sum(
            1
            for camera in cameras
            if camera["is_active"]
        )

        return {
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "inactive_cameras": (
                total_cameras
                - active_cameras
            ),
            "cameras": cameras,
        }
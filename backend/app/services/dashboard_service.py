from sqlalchemy.orm import Session

from app.repositories.dashboard_repository import (
    DashboardRepository,
)


class DashboardService:

    def __init__(self):
        self.repository = DashboardRepository()


    def get_summary(
        self,
        db: Session,
    ) -> dict:

        return self.repository.get_summary(
            db=db,
        )


    def get_detections_by_type(
        self,
        db: Session,
    ) -> dict:

        detections = (
            self.repository.get_detections_by_type(
                db=db,
            )
        )

        return {
            "total_detection_types": len(
                detections
            ),
            "detections": detections,
        }


    def get_recent_detections(
        self,
        db: Session,
        limit: int,
    ) -> list[dict]:

        return self.repository.get_recent_detections(
            db=db,
            limit=limit,
        )


    def get_camera_status(
        self,
        db: Session,
    ) -> dict:

        return self.repository.get_camera_status(
            db=db,
        )
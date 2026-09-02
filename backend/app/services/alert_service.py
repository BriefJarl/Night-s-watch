from datetime import datetime

from sqlalchemy.orm import Session

from app.models.alert import Alert
from app.models.detection_event import DetectionEvent


class AlertService:

    def evaluate_risk(
        self,
        object_type: str,
        confidence: float,
    ) -> str:
        """
        Determine alert severity based on detection confidence.

        This MVP uses a transparent and deterministic rule set.
        """

        normalized_object = object_type.lower().strip()

        # High-priority object categories for surveillance MVP
        high_priority_objects = {
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
        }

        if normalized_object in high_priority_objects:
            if confidence >= 0.90:
                return "CRITICAL"

            if confidence >= 0.75:
                return "HIGH"

            if confidence >= 0.60:
                return "MEDIUM"

            return "LOW"

        # Other detected objects
        if confidence >= 0.90:
            return "HIGH"

        if confidence >= 0.75:
            return "MEDIUM"

        return "LOW"

    def create_alert_for_detection(
        self,
        db: Session,
        detection_event: DetectionEvent,
    ) -> Alert:

        alert_level = self.evaluate_risk(
            object_type=detection_event.object_type,
            confidence=detection_event.confidence,
        )

        message = (
            f"{alert_level} alert: "
            f"{detection_event.object_type} detected "
            f"with confidence "
            f"{detection_event.confidence:.2f}"
        )

        alert = Alert(
            detection_event_id=detection_event.id,
            camera_id=detection_event.camera_id,
            alert_level=alert_level,
            message=message,
            is_resolved=False,
        )

        db.add(alert)

        db.commit()

        db.refresh(alert)

        return alert

    def get_alerts(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        unresolved_only: bool = False,
    ) -> list[Alert]:

        query = db.query(Alert)

        if unresolved_only:
            query = query.filter(
                Alert.is_resolved.is_(False)
            )

        return (
            query
            .order_by(Alert.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_alert(
        self,
        db: Session,
        alert_id: int,
    ) -> Alert | None:

        return (
            db.query(Alert)
            .filter(Alert.id == alert_id)
            .first()
        )

    def resolve_alert(
        self,
        db: Session,
        alert_id: int,
    ) -> Alert | None:

        alert = self.get_alert(
            db=db,
            alert_id=alert_id,
        )

        if alert is None:
            return None

        if not alert.is_resolved:

            alert.is_resolved = True

            alert.resolved_at = datetime.utcnow()

            db.commit()

            db.refresh(alert)

        return alert
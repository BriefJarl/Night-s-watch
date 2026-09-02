from app.core.database import SessionLocal
from app.models import Camera
from app.schemas.detection_event import DetectionEventCreate
from app.services.detection_event_service import (
    DetectionEventService,
)


def test_detection_event_service():

    db = SessionLocal()

    service = DetectionEventService()

    try:

        # Check whether at least one camera exists
        camera = db.query(Camera).first()

        # Create test camera if database has no cameras
        if camera is None:

            camera = Camera(
                name="Test Camera",
                location="Test Location",
                stream_url="0",
                is_active=True,
            )

            db.add(camera)

            db.commit()

            db.refresh(camera)

            print(
                "Test camera created successfully!"
            )

        print(
            "Using Camera ID:",
            camera.id,
        )

        detection_event_data = DetectionEventCreate(
            camera_id=camera.id,
            object_type="person",
            confidence=0.95,
        )

        detection_event = (
            service.create_detection_event(
                db=db,
                detection_event_data=detection_event_data,
            )
        )

        print(
            "Detection event created successfully!"
        )

        print(
            "Event ID:",
            detection_event.id,
        )

        fetched_event = (
            service.get_detection_event(
                db=db,
                event_id=detection_event.id,
            )
        )

        if fetched_event is not None:

            print(
                "Detection event fetched successfully!"
            )

            print(
                "Object:",
                fetched_event.object_type,
            )

        events = service.get_detection_events(
            db=db
        )

        print(
            "Total detection events:",
            len(events),
        )

        print(
            "Service tests successful!"
        )

    finally:

        db.close()


if __name__ == "__main__":

    test_detection_event_service()
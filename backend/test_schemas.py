from app.schemas.detection_event import (
    DetectionEventCreate,
    DetectionEventResponse,
)


def test_detection_event_create():
    event = DetectionEventCreate(
        camera_id=1,
        object_type="person",
        confidence=0.95
    )

    print("DetectionEventCreate:")
    print(event.model_dump())


def test_detection_event_response():
    response = DetectionEventResponse(
        id=1,
        camera_id=1,
        object_type="person",
        confidence=0.95,
        detected_at="2026-09-02T10:00:00"
    )

    print("\nDetectionEventResponse:")
    print(response.model_dump())


if __name__ == "__main__":
    test_detection_event_create()
    test_detection_event_response()

    print("\nSchema tests successful!")

from typing import Any

from pydantic import BaseModel, Field


class VideoDetectionEvent(
    BaseModel):

    frame_number: int

    timestamp_seconds: (
        float | None
    ) = None

    detections: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )


class VideoDetectionResponse(
    BaseModel):

    success: bool

    filename: str

    confidence_threshold: float

    total_frames: int

    processed_frames: int

    sampled_frames: int

    fps: float

    duration_seconds: (
        float | None
    ) = None

    total_detections: int

    object_summary: dict[
        str,
        int
    ]

    detection_events: list[
        VideoDetectionEvent
    ]
from datetime import datetime

from pydantic import BaseModel, Field


class DashboardSummaryResponse(BaseModel):
    total_cameras: int
    active_cameras: int
    inactive_cameras: int

    total_detection_events: int
    detections_last_24_hours: int


class DetectionTypeAnalytics(BaseModel):
    object_type: str
    count: int
    average_confidence: float


class DetectionTypeAnalyticsResponse(BaseModel):
    total_detection_types: int
    detections: list[DetectionTypeAnalytics]


class RecentDetectionResponse(BaseModel):
    id: int

    camera_id: int
    camera_name: str | None

    object_type: str
    confidence: float

    detected_at: datetime


class CameraStatusItem(BaseModel):
    id: int

    name: str

    location: str | None

    is_active: bool

    total_detections: int


class CameraStatusResponse(BaseModel):
    total_cameras: int
    active_cameras: int
    inactive_cameras: int

    cameras: list[CameraStatusItem]
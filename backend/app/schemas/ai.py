from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    object_type: str
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )
    bounding_box: BoundingBoxResponse
    detected_at: datetime


class ImageDetectionResponse(BaseModel):
    success: bool
    filename: str
    total_detections: int
    confidence_threshold: float
    processed_at: datetime
    detections: List[DetectionResponse]
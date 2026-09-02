from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DetectionEventBase(BaseModel):
    camera_id: int = Field(
        ...,
        gt=0,
        description="ID of the camera that detected the object"
    )

    object_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of object detected"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence between 0 and 1"
    )


class DetectionEventCreate(DetectionEventBase):
    pass


class DetectionEventResponse(DetectionEventBase):
    id: int
    detected_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
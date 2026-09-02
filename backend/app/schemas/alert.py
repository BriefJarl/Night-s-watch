from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int

    detection_event_id: int

    camera_id: int

    alert_level: str

    message: str

    is_resolved: bool

    created_at: datetime

    resolved_at: datetime | None = None


class AlertResolveResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int

    is_resolved: bool

    resolved_at: datetime | None = None


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]

    total: int
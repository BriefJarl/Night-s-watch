from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class CameraBase(BaseModel):

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    source: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    location: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    is_active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):

    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    source: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    location: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    is_active: Optional[bool] = None


class CameraResponse(BaseModel):

    id: int

    name: str

    source: str = Field(
        validation_alias="stream_url",
        serialization_alias="source",
    )

    location: Optional[str]

    is_active: bool

    created_at: datetime

    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )
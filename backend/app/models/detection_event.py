from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id"),
        nullable=False,
        index=True
    )
    object_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.camera import Camera


class CameraRepository:

    def create(
        self,
        db: Session,
        *,
        name: str,
        source: str,
        location: Optional[str],
        is_active: bool,
    ) -> Camera:

        camera = Camera(
            name=name,
            stream_url=source,
            location=location or "",
            is_active=is_active,
        )

        try:
            db.add(camera)
            db.commit()
            db.refresh(camera)

            return camera

        except Exception:
            db.rollback()
            raise


    def get_by_id(
        self,
        db: Session,
        camera_id: int,
    ) -> Optional[Camera]:

        statement = select(Camera).where(
            Camera.id == camera_id
        )

        result = db.execute(statement)

        return result.scalar_one_or_none()


    def get_all(
        self,
        db: Session,
    ) -> list[Camera]:

        statement = (
            select(Camera)
            .order_by(Camera.id.asc())
        )

        result = db.execute(statement)

        return list(result.scalars().all())


    def update(
        self,
        db: Session,
        *,
        camera: Camera,
        update_data: dict,
    ) -> Camera:

        if "name" in update_data:
            camera.name = update_data["name"]

        if "source" in update_data:
            camera.stream_url = update_data["source"]

        if "location" in update_data:
            camera.location = update_data["location"] or ""

        if "is_active" in update_data:
            camera.is_active = update_data["is_active"]

        try:
            db.commit()
            db.refresh(camera)

            return camera

        except Exception:
            db.rollback()
            raise


    def delete(
        self,
        db: Session,
        *,
        camera: Camera,
    ) -> None:

        try:
            db.delete(camera)
            db.commit()

        except Exception:
            db.rollback()
            raise
from typing import Optional

from sqlalchemy.orm import Session

from app.models.camera import Camera
from app.repositories.camera_repository import CameraRepository
from app.schemas.camera import CameraCreate, CameraUpdate


class CameraService:

    def __init__(self) -> None:
        self.repository = CameraRepository()


    def create_camera(
        self,
        db: Session,
        camera_data: CameraCreate,
    ) -> Camera:

        name = camera_data.name.strip()
        source = camera_data.source.strip()

        location = (
            camera_data.location.strip()
            if camera_data.location
            else None
        )

        if not name:
            raise ValueError(
                "Camera name cannot be empty."
            )

        if not source:
            raise ValueError(
                "Camera source cannot be empty."
            )

        return self.repository.create(
            db=db,
            name=name,
            source=source,
            location=location,
            is_active=camera_data.is_active,
        )


    def get_cameras(
        self,
        db: Session,
    ) -> list[Camera]:

        return self.repository.get_all(
            db=db,
        )


    def get_camera(
        self,
        db: Session,
        camera_id: int,
    ) -> Optional[Camera]:

        if camera_id <= 0:
            raise ValueError(
                "Camera ID must be greater than zero."
            )

        return self.repository.get_by_id(
            db=db,
            camera_id=camera_id,
        )


    def update_camera(
        self,
        db: Session,
        camera_id: int,
        camera_data: CameraUpdate,
    ) -> Optional[Camera]:

        if camera_id <= 0:
            raise ValueError(
                "Camera ID must be greater than zero."
            )

        camera = self.repository.get_by_id(
            db=db,
            camera_id=camera_id,
        )

        if camera is None:
            return None

        update_data = camera_data.model_dump(
            exclude_unset=True,
        )

        if "name" in update_data and update_data["name"] is not None:

            update_data["name"] = update_data["name"].strip()

            if not update_data["name"]:
                raise ValueError(
                    "Camera name cannot be empty."
                )

        if "source" in update_data and update_data["source"] is not None:

            update_data["source"] = update_data["source"].strip()

            if not update_data["source"]:
                raise ValueError(
                    "Camera source cannot be empty."
                )

        if (
            "location" in update_data
            and update_data["location"] is not None
        ):
            update_data["location"] = (
                update_data["location"].strip()
            )

        if not update_data:
            return camera

        return self.repository.update(
            db=db,
            camera=camera,
            update_data=update_data,
        )


    def delete_camera(
        self,
        db: Session,
        camera_id: int,
    ) -> bool:

        if camera_id <= 0:
            raise ValueError(
                "Camera ID must be greater than zero."
            )

        camera = self.repository.get_by_id(
            db=db,
            camera_id=camera_id,
        )

        if camera is None:
            return False

        self.repository.delete(
            db=db,
            camera=camera,
        )

        return True
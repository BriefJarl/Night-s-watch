from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.schemas.camera import (
    CameraCreate,
    CameraResponse,
    CameraUpdate,
)
from app.services.camera_service import CameraService


router = APIRouter(
    prefix="/api/v1/cameras",
    tags=["Cameras"],
)


camera_service = CameraService()


@router.post(
    "/",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_camera(
    camera_data: CameraCreate,
    db: Session = Depends(get_db),
):

    return camera_service.create_camera(
        db=db,
        camera_data=camera_data,
    )


@router.get(
    "/",
    response_model=List[CameraResponse],
)
def get_cameras(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):

    return camera_service.get_cameras(
        db=db,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{camera_id}",
    response_model=CameraResponse,
)
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
):

    camera = camera_service.get_camera(
        db=db,
        camera_id=camera_id,
    )

    if camera is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )

    return camera


@router.put(
    "/{camera_id}",
    response_model=CameraResponse,
)
def update_camera(
    camera_id: int,
    camera_data: CameraUpdate,
    db: Session = Depends(get_db),
):

    camera = camera_service.update_camera(
        db=db,
        camera_id=camera_id,
        camera_data=camera_data,
    )

    if camera is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )

    return camera


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
):

    deleted = camera_service.delete_camera(
        db=db,
        camera_id=camera_id,
    )

    if not deleted:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )

    return None
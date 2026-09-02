from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    status,
)

from sqlalchemy.orm import Session

from app.core.database import get_db

from app.schemas.camera import (
    CameraCreate,
    CameraResponse,
    CameraUpdate,
)

from app.services.camera_service import (
    CameraService,
)


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

    try:
        return camera_service.create_camera(
            db=db,
            camera_data=camera_data,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.get(
    "/",
    response_model=list[CameraResponse],
)
def get_cameras(
    db: Session = Depends(get_db),
):

    return camera_service.get_cameras(
        db=db,
    )


@router.get(
    "/{camera_id}",
    response_model=CameraResponse,
)
def get_camera(
    camera_id: int = Path(
        ...,
        gt=0,
    ),
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


@router.patch(
    "/{camera_id}",
    response_model=CameraResponse,
)
def update_camera(
    camera_id: int = Path(
        ...,
        gt=0,
    ),
    camera_data: CameraUpdate = ...,
    db: Session = Depends(get_db),
):

    try:

        camera = camera_service.update_camera(
            db=db,
            camera_id=camera_id,
            camera_data=camera_data,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
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
    camera_id: int = Path(
        ...,
        gt=0,
    ),
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
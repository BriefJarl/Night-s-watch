from typing import List
from pathlib import Path
from uuid import uuid4
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from app.ai.yolo_detector import (
    YOLODetector
)

from app.ai.video_processor import (
    VideoProcessor
)

from app.schemas.video import (
    VideoDetectionResponse
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.detection_event import (
    DetectionEventCreate,
    DetectionEventResponse,
)
from app.services.detection_event_service import (
    DetectionEventService,
)


router = APIRouter(
    prefix="/api/v1/detection-events",
    tags=["Detection Events"],
)


detection_event_service = DetectionEventService()


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[3]
)


VIDEO_UPLOAD_DIR = (
    BASE_DIR
    / "media"
    / "video_uploads"
)


VIDEO_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MAX_VIDEO_SIZE = (
    100
    * 1024
    * 1024
)


ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


detector = YOLODetector()


video_processor = VideoProcessor(
    detector=detector
)


@router.post(
    "/",
    response_model=DetectionEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_detection_event(
    detection_event_data: DetectionEventCreate,
    db: Session = Depends(get_db),
):

    try:
        return detection_event_service.create_detection_event(
            db=db,
            detection_event_data=detection_event_data,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.get(
    "/",
    response_model=List[DetectionEventResponse],
)
def get_detection_events(
    skip: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    db: Session = Depends(get_db),
):

    return detection_event_service.get_detection_events(
        db=db,
        skip=skip,
        limit=limit,
    )


@router.post(
    "/detect/video",
    response_model=VideoDetectionResponse,
    status_code=status.HTTP_200_OK
)
async def detect_video(

    file: UploadFile = File(...),

    confidence_threshold: float = Query(
        default=0.5,
        ge=0.1,
        le=1.0
    ),

    sample_every_n_frames: int = Query(
        default=15,
        ge=1,
        le=300
    ),

    max_frames: int = Query(
        default=200,
        ge=1,
        le=500
    )
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "Video filename is missing"
            )
        )


    suffix = (
        Path(file.filename)
        .suffix
        .lower()
    )


    if (
        suffix
        not in ALLOWED_VIDEO_EXTENSIONS
    ):

        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported video format. "
                "Allowed formats: "
                "mp4, avi, mov, mkv"
            )
        )


    unique_filename = (
        f"{uuid4().hex}{suffix}"
    )


    video_path = (
        VIDEO_UPLOAD_DIR
        / unique_filename
    )


    total_size = 0

    chunk_size = (
        1024
        * 1024
    )


    try:

        with open(
            video_path,
            "wb"
        ) as buffer:


            while True:

                chunk = (
                    await file.read(
                        chunk_size
                    )
                )


                if not chunk:

                    break


                total_size += (
                    len(chunk)
                )


                if (
                    total_size
                    > MAX_VIDEO_SIZE
                ):

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video file exceeds "
                            "maximum allowed size "
                            "of 100 MB"
                        )
                    )


                buffer.write(
                    chunk
                )


        result = (
            video_processor.process_video(
                video_path=str(
                    video_path
                ),

                confidence_threshold=(
                    confidence_threshold
                ),

                sample_every_n_frames=(
                    sample_every_n_frames
                ),

                max_frames=max_frames
            )
        )


        return {

            "success": True,

            "filename": (
                file.filename
            ),

            "confidence_threshold": (
                confidence_threshold
            ),

            **result
        }


    except HTTPException:

        raise


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Video processing failed"
            )
        ) from error


    finally:

        await file.close()


        if video_path.exists():

            try:

                video_path.unlink()

            except Exception:

                pass


@router.get(
    "/{event_id}",
    response_model=DetectionEventResponse,
)
def get_detection_event(
    event_id: int,
    db: Session = Depends(get_db),
):

    try:
        event = detection_event_service.get_detection_event(
            db=db,
            event_id=event_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )

    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection event not found",
        )

    return event


@router.delete(
    "/{event_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_detection_event(
    event_id: int,
    db: Session = Depends(get_db),
):

    try:
        deleted = (
            detection_event_service.delete_detection_event(
                db=db,
                event_id=event_id,
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection event not found",
        )

    return None

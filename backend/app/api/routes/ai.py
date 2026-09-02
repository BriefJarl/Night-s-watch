from datetime import datetime, timezone

from pathlib import Path

from uuid import uuid4
import os
import shutil

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from PIL import Image, UnidentifiedImageError

from sqlalchemy.orm import Session

from app.ai.yolo_detector import YOLODetector

from app.core.database import get_db

from app.models.detection_event import DetectionEvent

from app.schemas.ai import ImageDetectionResponse

from app.services.alert_service import AlertService

from fastapi import Query

from fastapi.concurrency import run_in_threadpool

from app.ai.video_processor import VideoProcessor

from app.schemas.video import VideoDetectionResponse


router = APIRouter(
    prefix="/api/v1/ai",
    tags=["AI Detection"]
)




# -----------------------------
# Configuration
# -----------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}




# backend/media/uploads
BACKEND_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = (
    BACKEND_DIR
    / "media"
    / "uploads"
)
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIDEO_UPLOAD_DIR = (
    BACKEND_DIR
    / "media"
    / "videos"
)

VIDEO_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "application/octet-stream"
}
ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4"
}




# -----------------------------
# YOLO Singleton
# -----------------------------
detector = YOLODetector()
video_processor = VideoProcessor(
    detector=detector
)
alert_service = AlertService()




@router.post(
    "/detect/image",
    response_model=ImageDetectionResponse,
    status_code=status.HTTP_200_OK
)
async def detect_image(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    db: Session = Depends(get_db),
):
    """
    Upload an image and detect objects using YOLO.
    """
    # -------------------------
    # Validate file metadata
    # -------------------------
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided."
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported file type. "
                "Only JPG, JPEG, PNG and WEBP images are allowed."
            )
        )
    original_filename = Path(
        file.filename
    ).name
    file_extension = Path(
        original_filename
    ).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file extension."
        )




    # -------------------------
    # Read file safely
    # -------------------------
    file_content = await file.read()
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image size exceeds the 10 MB limit."
        )




    # -------------------------
    # Create safe filename
    # -------------------------
    unique_filename = (
        f"{uuid4().hex}"
        f"{file_extension}"
    )
    file_path = (
        UPLOAD_DIR
        / unique_filename
    )




    # -------------------------
    # Save image
    # -------------------------
    try:
        with open(
            file_path,
            "wb"
        ) as buffer:
            buffer.write(
                file_content
            )
        # ---------------------
        # Verify real image
        # ---------------------
        try:
            with Image.open(
                file_path
            ) as image:
                image.verify()
        except (
            UnidentifiedImageError,
            OSError
        ):
            file_path.unlink(
                missing_ok=True
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid image."
            )




        # ---------------------
        # Validate confidence
        # ---------------------
        if not (
            0.0
            <= confidence_threshold
            <= 1.0
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "confidence_threshold "
                    "must be between 0 and 1."
                )
            )




        # ---------------------
        # Run YOLO
        # ---------------------
        detections = detector.detect(
            image_path=str(file_path),
            confidence_threshold=confidence_threshold
        )




        # ---------------------
        # Add timestamp
        # ---------------------
        detection_time = datetime.now(
            timezone.utc
        )
        formatted_detections = []
        for detection in detections:
            object_type = detection["object_type"]
            confidence = detection["confidence"]
            formatted_detections.append(
                {
                    "object_type": object_type,
                    "confidence": confidence,
                    "bounding_box": detection["bounding_box"],
                    "detected_at": detection_time,
                }
            )
            # Save detection event to database.
            # MVP uses camera_id=1 as the default/demo camera.
            detection_event = DetectionEvent(
                camera_id=1,
                object_type=object_type,
                confidence=confidence,
                detected_at=detection_time.replace(tzinfo=None),
            )
            db.add(detection_event)
            db.flush()
            # Automatically evaluate risk and create an alert.
            alert_service.create_alert_for_detection(
                db=db,
                detection_event=detection_event,
            )




        # ---------------------
        # Return response
        # ---------------------
        return {
            "success": True,
            "filename": original_filename,
            "total_detections": len(
                formatted_detections
            ),
            "confidence_threshold": (
                confidence_threshold
            ),
            "processed_at": detection_time,
            "detections": (
                formatted_detections
            )
        }




    finally:
        # ---------------------
        # Close upload
        # ---------------------
        await file.close()


@router.post(
    "/detect/video",
    response_model=VideoDetectionResponse,
    status_code=status.HTTP_200_OK
)
async def detect_video(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0
    ),
    sample_every_n_frames: int = Query(
        default=15,
        ge=1,
        le=300
    ),
    max_frames: int = Query(
        default=300,
        ge=1,
        le=5000
    )
):
    """
    Upload an MP4 video and detect objects using YOLO.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided."
        )

    original_filename = Path(
        file.filename
    ).name

    file_extension = Path(
        original_filename
    ).suffix.lower()

    if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only MP4 videos are allowed."
        )

    if (
        file.content_type
        and file.content_type
        not in ALLOWED_VIDEO_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported video content type."
        )

    unique_filename = (
        f"{uuid4().hex}"
        f"{file_extension}"
    )

    video_path = (
        VIDEO_UPLOAD_DIR
        / unique_filename
    )

    total_size = 0

    try:

        with open(
            video_path,
            "wb"
        ) as buffer:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if total_size > MAX_VIDEO_SIZE:

                    raise HTTPException(
                        status_code=(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        ),
                        detail=(
                            "Video size exceeds "
                            "the 100 MB limit."
                        )
                    )

                buffer.write(
                    chunk
                )

        if total_size == 0:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video is empty."
            )

        try:

            result = await run_in_threadpool(
                video_processor.process_video,
                str(video_path),
                confidence_threshold,
                sample_every_n_frames,
                max_frames
            )

        except ValueError as error:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error)
            )

        return {

            "success": True,

            "filename": original_filename,

            "confidence_threshold": (
                confidence_threshold
            ),

            **result
        }

    finally:

        await file.close()

        video_path.unlink(
            missing_ok=True
        )


@router.post("/detect/video/annotated")
async def detect_annotated_video(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
):
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    filename = file.filename or ""

    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    upload_dir = "media/uploads"
    output_dir = "media/processed"

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_filename = f"{uuid4().hex}{extension}"

    input_path = os.path.join(
        upload_dir,
        input_filename
    )

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        result = video_processor.process_video_annotated(
            input_path=input_path,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
        )

        return {
            "success": True,
            "processed_video_url": (
                f"/media/processed/"
                f"{result['filename']}"
            ),
            "frames_processed": result[
                "frames_processed"
            ],
            "total_detections": result[
                "total_detections"
            ],
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        await file.close()

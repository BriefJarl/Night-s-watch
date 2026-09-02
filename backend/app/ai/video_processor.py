from pathlib import Path

from collections import Counter

import os
import uuid

import cv2

from app.ai.yolo_detector import YOLODetector


class VideoProcessor:

    def __init__(
        self,
        detector: YOLODetector
    ):
        self.detector = detector

    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.5,
        sample_every_n_frames: int = 15,
        max_frames: int = 200
    ):
        path = Path(video_path)
        if not path.exists():
            raise ValueError(
                "Video file does not exist"
            )
        if path.stat().st_size == 0:
            raise ValueError(
                "Video file is empty"
            )

        capture = cv2.VideoCapture(
            str(path)
        )
        if not capture.isOpened():
            raise ValueError(
                "Unable to open video file"
            )

        try:
            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )
            total_frames = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )
            duration_seconds = None
            if fps > 0 and total_frames > 0:
                duration_seconds = round(
                    total_frames / fps,
                    2
                )

            processed_frames = 0
            sampled_frames = 0
            total_detections = 0
            detection_events = []
            object_counter = Counter()
            frame_number = 0

            while processed_frames < max_frames:
                success, frame = (
                    capture.read()
                )

                if not success:
                    break

                processed_frames += 1

                if (
                    frame_number
                    % sample_every_n_frames
                    == 0
                ):
                    sampled_frames += 1

                    detections = (
                        self.detector.detect_frame(
                            frame=frame,
                            confidence_threshold=(
                                confidence_threshold
                            )
                        )
                    )

                    if detections:
                        total_detections += (
                            len(detections)
                        )

                        for detection in detections:
                            object_counter[
                                detection[
                                    "object_type"
                                ]
                            ] += 1

                        timestamp_seconds = None

                        if fps > 0:
                            timestamp_seconds = round(
                                frame_number / fps,
                                2
                            )

                        detection_events.append(
                            {
                                "frame_number": (
                                    frame_number
                                ),
                                "timestamp_seconds": (
                                    timestamp_seconds
                                ),
                                "detections": (
                                    detections
                                )
                            }
                        )

                frame_number += 1

            return {
                "total_frames": (
                    total_frames
                ),
                "processed_frames": (
                    processed_frames
                ),
                "sampled_frames": (
                    sampled_frames
                ),
                "fps": round(
                    fps,
                    2
                ),
                "duration_seconds": (
                    duration_seconds
                ),
                "total_detections": (
                    total_detections
                ),
                "object_summary": dict(
                    object_counter
                ),
                "detection_events": (
                    detection_events
                )
            }

        finally:
            capture.release()

    def process_video_annotated(
        self,
        input_path: str,
        output_dir: str,
        confidence_threshold: float = 0.5,
    ):
        Path(output_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():
            raise ValueError(
                "Unable to open video"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        output_filename = (
            f"processed_{uuid.uuid4().hex}.mp4"
        )
        output_path = os.path.join(
            output_dir,
            output_filename
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height),
        )

        frame_count = 0
        detection_count = 0

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_count += 1

                detections = (
                    self.detector.detect_frame(
                        frame=frame,
                        confidence_threshold=(
                            confidence_threshold
                        )
                    )
                )

                for detection in detections:
                    detection_count += 1

                    object_type = detection[
                        "object_type"
                    ]
                    confidence = detection[
                        "confidence"
                    ]
                    bbox = detection[
                        "bounding_box"
                    ]

                    x1 = int(bbox["x1"])
                    y1 = int(bbox["y1"])
                    x2 = int(bbox["x2"])
                    y2 = int(bbox["y2"])

                    label = (
                        f"{object_type} "
                        f"{confidence:.2f}"
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            max(y1 - 10, 20)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                writer.write(frame)

        finally:
            cap.release()
            writer.release()

        return {
            "output_path": output_path,
            "filename": output_filename,
            "frames_processed": frame_count,
            "total_detections": detection_count,
        }

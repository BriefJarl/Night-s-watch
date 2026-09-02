from datetime import datetime, timezone

from pathlib import Path

from ultralytics import YOLO



class YOLODetector:

    def __init__(self, model_path: str = "yolov8n.pt"):

        self.model = YOLO(model_path)

    def _format_results(self, results):

        detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:

                continue

            for box in boxes:

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                object_type = self.model.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = {

                    "object_type": object_type,

                    "confidence": round(confidence, 4),

                    "bounding_box": {

                        "x1": round(float(x1), 2),

                        "y1": round(float(y1), 2),

                        "x2": round(float(x2), 2),

                        "y2": round(float(y2), 2)

                    },

                    "detected_at": datetime.now(

                        timezone.utc

                    ).isoformat()

                }

                detections.append(detection)

        return detections

    def detect(

        self,

        image_path: str,

        confidence_threshold: float = 0.5

    ):

        path = Path(image_path)

        if not path.exists():

            raise FileNotFoundError(

                f"Image does not exist: {path}"

            )

        if path.stat().st_size == 0:

            raise ValueError(

                f"Image file is empty: {path}"

            )

        results = self.model(

            str(path),

            conf=confidence_threshold,

            verbose=False

        )

        return self._format_results(results)

    def detect_frame(

        self,

        frame,

        confidence_threshold: float = 0.5

    ):

        results = self.model(

            frame,

            conf=confidence_threshold,

            verbose=False

        )

        detections = []

        detected_at = (

            datetime.now(timezone.utc)

            .isoformat()

            .replace("+00:00", "Z")

        )

        for result in results:

            if result.boxes is None:

                continue

            for box in result.boxes:

                class_id = int(

                    box.cls[0].item()

                )

                confidence = round(

                    float(box.conf[0].item()),

                    4

                )

                x1, y1, x2, y2 = (

                    box.xyxy[0]

                    .cpu()

                    .tolist()

                )

                object_type = self.model.names[

                    class_id

                ]

                detections.append(

                    {

                        "object_type": object_type,

                        "confidence": confidence,

                        "bounding_box": {

                            "x1": round(x1, 2),

                            "y1": round(y1, 2),

                            "x2": round(x2, 2),

                            "y2": round(y2, 2)

                        },

                        "detected_at": detected_at

                    }

                )

        return detections

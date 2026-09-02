from pathlib import Path

from app.ai.yolo_detector import YOLODetector


BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "media"
    / "test_images"
    / "test (1).jpg"
)


print(f"Testing image path: {IMAGE_PATH}")

# Check whether file exists
if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

# Check whether file is empty
if IMAGE_PATH.stat().st_size == 0:
    raise ValueError(
        f"Image file is empty: {IMAGE_PATH}"
    )


detector = YOLODetector()


detections = detector.detect(
    image_path=str(IMAGE_PATH),
    confidence_threshold=0.5,
)


print("\nDetections:\n")

for detection in detections:
    print(detection)
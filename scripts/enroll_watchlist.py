import os
import sys
import json
import argparse
from typing import Dict, List, Optional
import cv2
import numpy as np


class WatchlistEnrollment:
    """
    Offline zero-shot face enrollment utility for IBVAP.
    Extracts 128D SFace feature embeddings from suspect mugshots and
    saves them to weights/watchlist_embeddings.json.
    """

    def __init__(
        self,
        detector_path: str = "weights/face_detection_yunet_2023mar.onnx",
        recognizer_path: str = "weights/face_recognition_sface_2021dec.onnx",
        output_json: str = "weights/watchlist_embeddings.json",
    ):
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.output_json = output_json

        if not os.path.exists(detector_path) or not os.path.exists(recognizer_path):
            raise FileNotFoundError(
                f"Face ONNX models not found at {detector_path} or {recognizer_path}."
            )

        self.detector = cv2.FaceDetectorYN.create(
            model=self.detector_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.recognizer_path, config=""
        )

    def extract_embedding_from_image(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Detects the primary face in an image and extracts its 128-dimensional embedding.
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        status, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return None

        # Select the most prominent face (largest bounding box or highest confidence)
        best_face = faces[0]
        if len(faces) > 1:
            # Sort by area (faces[i][2] * faces[i][3])
            best_face = max(faces, key=lambda f: f[2] * f[3])

        aligned_face = self.recognizer.alignCrop(image, best_face)
        feature = self.recognizer.feature(aligned_face)
        return feature.flatten().tolist()

    def enroll_image(self, name: str, image_path: str) -> bool:
        """Enrolls a single suspect image under the given name/ID."""
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return False

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not decode image at {image_path}")
            return False

        embedding = self.extract_embedding_from_image(image)
        if embedding is None:
            print(f"Error: No face detected in {image_path}")
            return False

        watchlist = self.load_watchlist()
        watchlist[name] = {
            "name": name,
            "embedding": embedding,
            "source_image": os.path.basename(image_path),
        }
        self.save_watchlist(watchlist)
        print(f"Successfully enrolled suspect: '{name}' from {image_path}")
        return True

    def enroll_directory(self, directory: str) -> int:
        """
        Scans a directory of suspect photos and enrolls them.
        File name without extension is used as the suspect ID/name.
        """
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return 0

        watchlist = self.load_watchlist()
        enrolled_count = 0

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for filename in sorted(os.listdir(directory)):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in valid_exts:
                continue

            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            embedding = self.extract_embedding_from_image(image)
            if embedding is not None:
                watchlist[name] = {
                    "name": name,
                    "embedding": embedding,
                    "source_image": filename,
                }
                enrolled_count += 1
                print(f"[Watchlist] Enrolled: '{name}'")
            else:
                print(f"[Watchlist] Warning: No face detected in {filename}")

        self.save_watchlist(watchlist)
        print(f"[Watchlist] Total enrolled in {self.output_json}: {len(watchlist)}")
        return enrolled_count

    def load_watchlist(self) -> Dict[str, dict]:
        """Loads existing watchlist embeddings."""
        if os.path.exists(self.output_json):
            try:
                with open(self.output_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read {self.output_json}: {e}. Initializing empty.")
        return {}

    def save_watchlist(self, watchlist: Dict[str, dict]):
        """Saves watchlist embeddings to disk."""
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(watchlist, f, indent=2)


def create_sample_suspect_dataset(output_dir: str = "data/watchlist"):
    """Creates synthetic placeholder suspect images for demonstration if none exist."""
    os.makedirs(output_dir, exist_ok=True)
    sample_files = [f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".png"))]
    if sample_files:
        return

    # Draw a stylized face pattern for demo enrollment
    for suspect_id, name in [("SUSPECT-T101", "Ramesh Kumar"), ("SUSPECT-T102", "Vikram Singh")]:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:] = (230, 230, 230)
        # Face oval
        cv2.ellipse(img, (150, 150), (70, 95), 0, 0, 360, (190, 170, 150), -1)
        # Eyes
        cv2.circle(img, (120, 130), 8, (50, 30, 20), -1)
        cv2.circle(img, (180, 130), 8, (50, 30, 20), -1)
        # Eyebrows
        cv2.line(img, (110, 115), (135, 118), (30, 20, 10), 3)
        cv2.line(img, (165, 118), (190, 115), (30, 20, 10), 3)
        # Nose
        cv2.line(img, (150, 135), (150, 165), (160, 140, 120), 2)
        cv2.line(img, (145, 165), (155, 165), (160, 140, 120), 2)
        # Mouth
        cv2.ellipse(img, (150, 190), (25, 10), 0, 0, 180, (80, 60, 120), 2)
        # Label
        cv2.putText(img, suspect_id, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
        
        file_path = os.path.join(output_dir, f"{suspect_id}_{name.replace(' ', '_')}.jpg")
        cv2.imwrite(file_path, img)
    print(f"Created sample suspect images in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll suspect photos into IBVAP watchlist.")
    parser.add_argument("--dir", default="data/watchlist", help="Directory containing suspect photos.")
    parser.add_argument("--image", default=None, help="Single image path to enroll.")
    parser.add_argument("--name", default=None, help="Suspect name/ID for single image enrollment.")
    parser.add_argument("--create-samples", action="store_true", help="Generate sample demo suspects.")
    args = parser.parse_args()

    if args.create_samples:
        create_sample_suspect_dataset(args.dir)

    enrollment = WatchlistEnrollment()

    if args.image and args.name:
        enrollment.enroll_image(args.name, args.image)
    else:
        # Create sample folder if empty
        if not os.path.exists(args.dir) or not os.listdir(args.dir):
            create_sample_suspect_dataset(args.dir)
        enrollment.enroll_directory(args.dir)

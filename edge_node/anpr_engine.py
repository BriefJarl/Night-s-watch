import os
import cv2
import re
import numpy as np
from collections import Counter
from typing import Optional, Tuple, List, Dict, Any



class ANPREngine:
    """
    Software-Only License Plate Recognition (ANPR) with Multi-Frame Character Voting.
    Implements:
      1. OpenCV 4-Point Perspective Warp for skewed plate rectification.
      2. EasyOCR text recognition with morphological & contrast preprocessing.
      3. Standard Indian Registration Regex Validation & Ambiguity Correction.
      4. Multi-Frame Statistical Majority Character Voting across 10-15 tracked frames.
    """

    # Standard Indian Registration Regex Patterns
    # Standard: DL01AB1234, MH12DE1433, HR26DQ5555, UP32AA0001, WB02A1234
    REGEX_INDIAN_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
    # Bharat Series (BH): 22BH1234AA
    REGEX_BHARAT_SERIES = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

    # Character confusion correction maps
    NUM_TO_ALPHA = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
    ALPHA_TO_NUM = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "D": "0", "Q": "0"}

    def __init__(
        self,
        use_gpu: bool = False,
        history_size: int = 15,
        plate_detector_path: str = "weights/yolov8n_license_plate.pt",
    ):
        self.history_size = history_size
        # Track history map: track_id -> List[str]
        self.plate_history: Dict[int, List[Dict[str, Any]]] = {}
        self.last_plate_bboxes: Dict[int, Tuple[int, int, int, int]] = {}

        # Initialize EasyOCR reader lazily or in init
        self.reader = None
        self.use_gpu = use_gpu
        self._init_reader()

        # Initialize YOLO License Plate Detector
        self.plate_detector_path = plate_detector_path
        self.plate_detector = None
        self._init_plate_detector()

    def _init_reader(self):
        """Initializes EasyOCR reader safely."""
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
        except Exception as e:
            print(f"[ANPREngine] Warning: Could not initialize EasyOCR ({e}). Falling back to pattern mode.")
            self.reader = None

    def _init_plate_detector(self):
        """Initializes the lightweight YOLOv8 license plate detector."""
        target_path = self.plate_detector_path
        if not os.path.exists(target_path):
            # Check relative to project root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            alt_path = os.path.join(base_dir, target_path)
            if os.path.exists(alt_path):
                target_path = alt_path

        if os.path.exists(target_path):
            try:
                from ultralytics import YOLO
                self.plate_detector = YOLO(target_path)
                print(f"[ANPREngine] Loaded YOLO license plate detector from {target_path}")
            except Exception as e:
                print(f"[ANPREngine] Warning: Could not load YOLO plate detector ({e}). Falling back to heuristic ROI.")
                self.plate_detector = None
        else:
            print(f"[ANPREngine] Notice: {self.plate_detector_path} not found. Using heuristic ROI.")


    def detect_plate_bbox(self, vehicle_crop: np.ndarray, conf_threshold: float = 0.30) -> Optional[Tuple[int, int, int, int]]:
        """
        Runs YOLO license plate detection on a vehicle crop.
        Returns:
            (x1, y1, x2, y2) bounding box in vehicle_crop coordinate space, or None.
        """
        if self.plate_detector is None or vehicle_crop is None or vehicle_crop.size == 0:
            return None

        try:
            results = self.plate_detector(vehicle_crop, conf=conf_threshold, verbose=False)
            best_box = None
            highest_conf = 0.0

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        b_conf = float(box.conf[0].item())
                        if b_conf > highest_conf:
                            highest_conf = b_conf
                            xyxy = box.xyxy[0].tolist()
                            best_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

            return best_box
        except Exception as e:
            return None

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 quad coordinate points in top-left, top-right, bottom-right, bottom-left sequence.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff
        return rect

    def warp_perspective(
        self,
        image: np.ndarray,
        quad_points: np.ndarray,
        target_width: int = 240,
        target_height: int = 80,
    ) -> np.ndarray:
        """
        Flattens a skewed 4-point quadrilateral license plate polygon into a rectified rectangle.
        """
        rect = self.order_points(quad_points)
        dst = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (target_width, target_height))
        return warped

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses plate image for OCR:
          1. Grayscale conversion.
          2. CLAHE (Contrast Limited Adaptive Histogram Equalization).
          3. Bilateral filter to smooth noise while preserving sharp character edges.
          4. Adaptive Otsu thresholding.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # Resize to standard height if needed
        h, w = gray.shape[:2]
        if h < 60:
            scale = 60.0 / h
            gray = cv2.resize(gray, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def validate_and_correct_indian_plate(self, raw_text: str) -> Tuple[bool, str]:
        """
        Cleans and validates candidate license plate against Indian registration syntax.
        Applies character ambiguity correction based on positional syntax:
          - State code (positions 0-1) must be alphabetic.
          - District code (positions 2-3) must be numeric.
          - Series (positions 4-5) must be alphabetic.
          - Unique number (last 4) must be numeric.
        """
        # Remove non-alphanumeric characters and uppercase
        clean_text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

        if len(clean_text) < 8 or len(clean_text) > 11:
            return False, clean_text

        # 1. Direct Regex check on Standard Plate
        if self.REGEX_INDIAN_STANDARD.match(clean_text) or self.REGEX_BHARAT_SERIES.match(clean_text):
            return True, clean_text

        # 2. Syntax-directed character correction (Standard 10-char plate: AA 00 AA 0000)
        if len(clean_text) == 10:
            corrected = list(clean_text)
            # Slot 0-1: Alphabetic State Code
            for i in range(2):
                if corrected[i] in self.NUM_TO_ALPHA:
                    corrected[i] = self.NUM_TO_ALPHA[corrected[i]]

            # Slot 2-3: Numeric District Code
            for i in range(2, 4):
                if corrected[i] in self.ALPHA_TO_NUM:
                    corrected[i] = self.ALPHA_TO_NUM[corrected[i]]

            # Slot 4-5: Alphabetic Series Code
            for i in range(4, 6):
                if corrected[i] in self.NUM_TO_ALPHA:
                    corrected[i] = self.NUM_TO_ALPHA[corrected[i]]

            # Slot 6-9: Numeric 4-digit Unique Code
            for i in range(6, 10):
                if corrected[i] in self.ALPHA_TO_NUM:
                    corrected[i] = self.ALPHA_TO_NUM[corrected[i]]

            cand = "".join(corrected)
            if self.REGEX_INDIAN_STANDARD.match(cand):
                return True, cand

        return False, clean_text

    def extract_plate_text(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Runs EasyOCR on a cropped plate image and returns (validated_plate_str, confidence).
        """
        if self.reader is None or plate_image is None or plate_image.size == 0:
            return None, 0.0

        try:
            preprocessed = self.preprocess_plate(plate_image)
            results = self.reader.readtext(preprocessed)

            best_plate = None
            best_conf = 0.0

            for bbox, text, conf in results:
                is_valid, corrected = self.validate_and_correct_indian_plate(text)
                if is_valid:
                    if conf > best_conf:
                        best_plate = corrected
                        best_conf = float(conf)
                elif len(corrected) >= 8 and best_plate is None:
                    # Fallback candidate if no perfect regex match yet
                    best_plate = corrected
                    best_conf = float(conf) * 0.5

            return best_plate, best_conf
        except Exception as e:
            print(f"[ANPREngine] OCR extraction error: {e}")
            return None, 0.0

    def multi_frame_vote(self, track_id: int) -> Tuple[Optional[str], float]:
        """
        Performs statistical majority voting across character positions
        over the recent frame history for a tracked vehicle.
        Returns:
            voted_plate (Optional[str])
            confidence (float)
        """
        history = self.plate_history.get(track_id, [])
        if not history:
            return None, 0.0

        # Filter candidates that meet minimum valid length
        valid_candidates = [entry["plate_str"] for entry in history if len(entry["plate_str"]) >= 8]
        if not valid_candidates:
            return None, 0.0

        # Group by length to align character slots
        length_counts = Counter(len(s) for s in valid_candidates)
        target_len = length_counts.most_common(1)[0][0]
        aligned_candidates = [s for s in valid_candidates if len(s) == target_len]

        num_samples = len(aligned_candidates)
        if num_samples == 0:
            return None, 0.0

        voted_chars = []
        total_majority_votes = 0

        for idx in range(target_len):
            char_slot = [s[idx] for s in aligned_candidates]
            slot_counter = Counter(char_slot)
            most_common_char, count = slot_counter.most_common(1)[0]
            voted_chars.append(most_common_char)
            total_majority_votes += count

        voted_str = "".join(voted_chars)
        # Validate final voted string
        is_valid, final_corrected = self.validate_and_correct_indian_plate(voted_str)

        # Average voting confidence
        vote_confidence = total_majority_votes / (target_len * num_samples)

        if is_valid:
            return final_corrected, round(vote_confidence, 2)
        return final_corrected, round(vote_confidence * 0.7, 2)

    def process_vehicle_frame(
        self,
        track_id: int,
        vehicle_crop: np.ndarray,
        plate_quad: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Processes vehicle ROI for a given frame:
          1. Warps plate region if quad is provided, or uses lower vehicle region.
          2. Extracts text with EasyOCR.
          3. Appends candidate to multi-frame buffer.
          4. Returns majority-voted plate string and confidence.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return self.multi_frame_vote(track_id)

        # Extract plate ROI
        if plate_quad is not None:
            plate_img = self.warp_perspective(vehicle_crop, plate_quad)
        else:
            # 1. Try YOLO License Plate Detector first for high precision
            plate_box = self.detect_plate_bbox(vehicle_crop)
            if plate_box is not None:
                px1, py1, px2, py2 = plate_box
                vh, vw = vehicle_crop.shape[:2]
                # Add 5% padding around plate
                pad_w = int((px2 - px1) * 0.05)
                pad_h = int((py2 - py1) * 0.05)
                crop_x1 = max(0, px1 - pad_w)
                crop_y1 = max(0, py1 - pad_h)
                crop_x2 = min(vw, px2 + pad_w)
                crop_y2 = min(vh, py2 + pad_h)
                plate_img = vehicle_crop[crop_y1:crop_y2, crop_x1:crop_x2]
                self.last_plate_bboxes[track_id] = (crop_x1, crop_y1, crop_x2, crop_y2)
            else:
                # 2. Fallback Heuristic: license plate is typically in the bottom 40% center of vehicle crop
                h, w = vehicle_crop.shape[:2]
                y1 = int(h * 0.60)
                x1 = int(w * 0.15)
                x2 = int(w * 0.85)
                plate_img = vehicle_crop[y1:h, x1:x2]
                self.last_plate_bboxes[track_id] = (x1, y1, x2, h)

        plate_str, conf = self.extract_plate_text(plate_img)

        if plate_str:
            if track_id not in self.plate_history:
                self.plate_history[track_id] = []

            self.plate_history[track_id].append({
                "plate_str": plate_str,
                "confidence": conf,
            })

            # Keep history capped at history_size
            if len(self.plate_history[track_id]) > self.history_size:
                self.plate_history[track_id].pop(0)

        return self.multi_frame_vote(track_id)

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Cleans up plate history for inactive tracks."""
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self.plate_history if tid not in active_set]
        for tid in stale_ids:
            del self.plate_history[tid]
            if tid in self.last_plate_bboxes:
                del self.last_plate_bboxes[tid]

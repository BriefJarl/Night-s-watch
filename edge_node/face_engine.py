import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class FaceEngine:
    """
    Edge-Compatible Zero-Shot Face Recognition & Watchlist Identification Engine.
    Uses OpenCV's native YuNet (FaceDetectorYN) and SFace (FaceRecognizerSF) to
    detect faces, extract 128-dimensional biometric embeddings, and perform
    cosine similarity verification against enrolled suspect profiles.
    """

    # SFace official cosine similarity threshold recommended by OpenCV Zoo is 0.363
    DEFAULT_COSINE_THRESHOLD = 0.38

    def __init__(
        self,
        detector_path: str = "weights/face_detection_yunet_2023mar.onnx",
        recognizer_path: str = "weights/face_recognition_sface_2021dec.onnx",
        watchlist_path: str = "weights/watchlist_embeddings.json",
        similarity_threshold: float = DEFAULT_COSINE_THRESHOLD,
        score_threshold: float = 0.50,
    ):
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.watchlist_path = watchlist_path
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold

        self.detector = None
        self.recognizer = None
        self.watchlist: Dict[str, np.ndarray] = {}
        self.track_face_cache: Dict[int, Dict[str, Any]] = {}

        self._init_models()
        self.reload_watchlist()

    def _resolve_path(self, path: str) -> str:
        if os.path.exists(path):
            return path
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(base_dir, path)
        if os.path.exists(alt_path):
            return alt_path
        return path

    def _init_models(self):
        """Initializes FaceDetectorYN and FaceRecognizerSF safely."""
        det_path = self._resolve_path(self.detector_path)
        rec_path = self._resolve_path(self.recognizer_path)
        self.detector_path = det_path
        self.recognizer_path = rec_path

        if not os.path.exists(det_path) or not os.path.exists(rec_path):
            print(f"[FaceEngine] Warning: Face models not found at {det_path} or {rec_path}. Face recognition disabled.")
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=self.detector_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self.recognizer_path, config=""
            )
            print("[FaceEngine] OpenCV YuNet & SFace models initialized successfully.")
        except Exception as e:
            print(f"[FaceEngine] Initialization error: {e}")
            self.detector = None
            self.recognizer = None

    def reload_watchlist(self) -> int:
        """Loads or reloads suspect embeddings from JSON."""
        self.watchlist.clear()
        watch_path = self._resolve_path(self.watchlist_path)
        self.watchlist_path = watch_path
        if not os.path.exists(watch_path):
            return 0

        try:
            with open(watch_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for name, entry in data.items():
                emb = entry.get("embedding")
                if emb:
                    arr = np.array(emb, dtype=np.float32).reshape(1, -1)
                    self.watchlist[name] = arr

            print(f"[FaceEngine] Loaded {len(self.watchlist)} suspect profile(s) from {self.watchlist_path}.")
            return len(self.watchlist)
        except Exception as e:
            print(f"[FaceEngine] Could not load watchlist: {e}")
            return 0

    def detect_and_match(
        self,
        person_crop: np.ndarray,
        track_id: Optional[int] = None,
    ) -> Tuple[Optional[str], float, Optional[Tuple[int, int, int, int]]]:
        """
        Detects faces in a person crop and matches against the watchlist.
        Returns:
            suspect_name: Optional[str] (Name of matched suspect, or None)
            confidence: float (Cosine similarity score 0.0 - 1.0)
            face_bbox: Optional[Tuple[int, int, int, int]] in person_crop coordinates (x, y, w, h)
        """
        if self.detector is None or self.recognizer is None or person_crop is None:
            return None, 0.0, None

        if person_crop.size == 0 or len(self.watchlist) == 0:
            return None, 0.0, None

        now = time.time()
        if track_id is not None and track_id in self.track_face_cache:
            cache = self.track_face_cache[track_id]
            # Reuse cached result if positive and fresh (< 4.0s)
            if cache.get("suspect_name") and (now - cache.get("timestamp", 0) < 4.0):
                return cache["suspect_name"], cache["confidence"], cache.get("bbox")

        h, w = person_crop.shape[:2]
        if h < 30 or w < 20:
            return None, 0.0, None

        # Determine search regions: for full-body crops (tall), prioritize upper 60%
        search_rois = []
        if h / max(1, w) >= 1.5:
            head_h = max(25, int(h * 0.60))
            search_rois.append((person_crop[0:head_h, 0:w], 0, 0))
        # Full crop fallback
        search_rois.append((person_crop, 0, 0))

        detected_face = None
        face_roi = None
        offset_y = 0

        for roi, ox, oy in search_rois:
            roi_h, roi_w = roi.shape[:2]
            self.detector.setInputSize((roi_w, roi_h))
            try:
                status, faces = self.detector.detect(roi)
                if faces is not None and len(faces) > 0:
                    detected_face = max(faces, key=lambda f: f[2] * f[3])
                    face_roi = roi
                    offset_y = oy
                    break
            except Exception:
                continue

        if detected_face is None or face_roi is None:
            return None, 0.0, None

        fx, fy, fw, fh = map(int, detected_face[:4])

        try:
            aligned_face = self.recognizer.alignCrop(face_roi, detected_face)
            query_feature = self.recognizer.feature(aligned_face)
        except Exception as e:
            return None, 0.0, None

        best_suspect = None
        best_sim = 0.0

        for suspect_name, enrolled_feature in self.watchlist.items():
            try:
                sim = float(self.recognizer.match(query_feature, enrolled_feature, cv2.FaceRecognizerSF_FR_COSINE))
            except Exception:
                norm_q = np.linalg.norm(query_feature)
                norm_e = np.linalg.norm(enrolled_feature)
                if norm_q > 0 and norm_e > 0:
                    sim = float(np.dot(query_feature.flatten(), enrolled_feature.flatten()) / (norm_q * norm_e))
                else:
                    sim = 0.0

            if sim > best_sim:
                best_sim = sim
                best_suspect = suspect_name

        matched_name = None
        matched_score = 0.0
        face_bbox = (fx, fy + offset_y, fw, fh)

        if best_sim >= self.similarity_threshold:
            matched_name = best_suspect
            matched_score = round(best_sim, 2)

        if track_id is not None:
            self.track_face_cache[track_id] = {
                "suspect_name": matched_name,
                "confidence": matched_score,
                "bbox": face_bbox,
                "timestamp": now,
            }

        return matched_name, matched_score, face_bbox

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Cleans up cache for lost tracks."""
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self.track_face_cache if tid not in active_set]
        for tid in stale_ids:
            del self.track_face_cache[tid]

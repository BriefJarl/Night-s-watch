import cv2
import time
import threading
import argparse
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

try:
    import requests as _requests
except ImportError:
    _requests = None

from backend.edge_node.false_alarm_filter import FalseAlarmFilter
from backend.edge_node.rule_engine import RuleEngine
from backend.edge_node.anpr_engine import ANPREngine
from backend.edge_node.edge_queue import EdgeQueue


class VisionEngine:
    """
    Intelligent Border Video Analytics Platform (IBVAP) - Edge Vision Engine.
    Implements:
      - 3-Stage Detection-Gated Pipeline (MOG2 Monitor -> YOLOv8n Gated Verification -> DeepSORT Tracking).
      - Phase 2: False Alarm Suppression Stack, Ground Homography, Behavioral Rules, Tamper Detection.
      - Phase 3: ANPR with Multi-Frame Character Voting, Semantic Compression, SQLite Store-and-Forward.
    """

    def __init__(
        self,
        source: str = "0",
        camera_id: str = "CAM-BOP-01",
        backend_url: str = "http://127.0.0.1:8000",
        headless: bool = False,
        enable_sync: bool = True,
    ):
        self.source = source
        self.camera_id = camera_id
        self.backend_url = backend_url
        self.headless = headless

        # Video stream capture
        try:
            source_int = int(source)
            self.cap = cv2.VideoCapture(source_int)
        except ValueError:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            print(f"[VisionEngine] Warning: Could not open video source '{source}'")
            if not self.headless:
                exit(1)

        # Stage 1: Lightweight Monitor (Background Subtraction)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self.motion_threshold_area = 500

        # Stage 2: Gated Verification (YOLOv8n)
        # Target COCO Classes: person (0), car (2), motorcycle (3), bus (5), truck (7)
        self.target_classes = [0, 2, 3, 5, 7]
        self.model = YOLO("yolov8n.pt")

        # Stage 3: Persistent Tracking (DeepSORT)
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            nms_max_overlap=1.0,
            embedder="mobilenet",
        )

        # Phase 2 Components
        self.false_alarm_filter = FalseAlarmFilter()
        self.rule_engine = RuleEngine()

        # Phase 3 Components
        self.anpr_engine = ANPREngine(use_gpu=False)
        self.edge_queue = EdgeQueue(
            db_path=f"edge_alerts_{self.camera_id}.db",
            backend_url=self.backend_url,
            camera_id=self.camera_id,
        )

        if enable_sync:
            self.edge_queue.start_sync_worker()

        # Operational rate limits (Max 2 FPS as required for Tier 1/2 edge compute)
        self.last_process_time = time.time()
        self.fps_limit = 2.0
        self.process_interval = 1.0 / self.fps_limit

        # Health & Telemetry state
        self.tamper_status = "OK"
        self.is_tampered = False
        self.latest_alerts = []
        self.last_enqueued_timestamps: Dict[int, float] = {}

        # Phase 5: Thread-safe JPEG frame buffer for MJPEG streaming endpoint.
        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()

        # Phase 5: Zone polling — interval (seconds) to fetch user zone from backend.
        self._zone_poll_interval = 30.0
        self._zone_last_polled: float = 0.0

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """
        Phase 5: Returns the most recent processed frame as JPEG bytes.
        Thread-safe — consumed by the FastAPI MJPEG streaming endpoint.
        Returns None if no frame has been processed yet.
        """
        with self._frame_lock:
            return self._latest_jpeg

    def _poll_zone(self) -> None:
        """
        Phase 5: Non-blocking zone poll — fetches the operator-drawn pixel polygon
        from the backend REST API and injects it into the RuleEngine.
        Called at most every _zone_poll_interval seconds from within run().
        """
        if _requests is None:
            return
        try:
            url = f"{self.backend_url}/api/v1/cameras/{self.camera_id}/zones"
            resp = _requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                polygon = data.get("polygon", [])
                zone_label = data.get("zone_label", "Alert zone")
                self.rule_engine.set_user_zone(polygon, zone_label)
        except Exception:
            pass  # Network unavailable — keep existing zone

    def run_background(self) -> threading.Thread:
        """
        Phase 5: Launches run() as a daemon thread so FastAPI's on_startup can
        start engines without blocking the event loop.
        Returns the thread object (already started).
        """
        t = threading.Thread(target=self.run, daemon=True, name=f"VisionEngine-{self.camera_id}")
        t.start()
        return t

    def draw_geofence_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws calibrated ground tripwire and restricted zone overlays projected on the frame.
        """
        h, w = frame.shape[:2]

        # 1. Draw Tripwire Line (A to B) in World Coordinates
        ptA_px = self.false_alarm_filter.world_to_pixel(
            self.false_alarm_filter.tripwire_A[0],
            self.false_alarm_filter.tripwire_A[1],
        )
        ptB_px = self.false_alarm_filter.world_to_pixel(
            self.false_alarm_filter.tripwire_B[0],
            self.false_alarm_filter.tripwire_B[1],
        )

        # Draw Tripwire line
        if 0 <= ptA_px[0] < w * 2 and 0 <= ptA_px[1] < h * 2:
            cv2.line(frame, ptA_px, ptB_px, (0, 0, 255), 2, cv2.LINE_AA)
            mid_w_x = (self.false_alarm_filter.tripwire_A[0] + self.false_alarm_filter.tripwire_B[0]) / 2.0
            mid_w_y = (self.false_alarm_filter.tripwire_A[1] + self.false_alarm_filter.tripwire_B[1]) / 2.0
            mid_px = self.false_alarm_filter.world_to_pixel(mid_w_x, mid_w_y)

            arrow_w_x = mid_w_x + self.false_alarm_filter.inward_normal[0] * 2.0
            arrow_w_y = mid_w_y + self.false_alarm_filter.inward_normal[1] * 2.0
            arrow_px = self.false_alarm_filter.world_to_pixel(arrow_w_x, arrow_w_y)

            cv2.arrowedLine(frame, mid_px, arrow_px, (0, 255, 255), 2, tipLength=0.3)
            cv2.putText(
                frame,
                "VIRTUAL FENCE (INBOUND ->)",
                (ptA_px[0] + 10, ptA_px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )

        # 2. Draw Red Zone Polygon (Zero Line)
        red_pts_px = []
        for pt in self.rule_engine.red_zone:
            px = self.false_alarm_filter.world_to_pixel(pt[0], pt[1])
            red_pts_px.append(px)
        red_pts_arr = np.array([red_pts_px], dtype=np.int32)
        cv2.polylines(frame, red_pts_arr, isClosed=True, color=(0, 0, 200), thickness=1, lineType=cv2.LINE_AA)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        motion_detected: bool,
        active_threat_count: int,
    ) -> np.ndarray:
        """
        Renders the tactical border intelligence HUD on top of the video frame.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 45), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # System Title
        cv2.putText(
            frame,
            f"IBVAP | {self.camera_id}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # Motion Gating Status
        if motion_detected:
            status_text = "STAGE 2: ACTIVE"
            status_color = (0, 220, 255)
        else:
            status_text = "STAGE 1: MONITOR"
            status_color = (180, 180, 180)

        cv2.putText(
            frame,
            status_text,
            (320, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1,
        )

        # Active Threat Indicator
        threat_color = (0, 0, 255) if active_threat_count > 0 else (0, 255, 0)
        cv2.putText(
            frame,
            f"THREATS: {active_threat_count}",
            (520, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            threat_color,
            2,
        )

        # Store-and-Forward Queue & Sync Status
        stats = self.edge_queue.get_queue_stats()
        if stats["is_connected"]:
            sync_text = f"LINK: ONLINE (Synced: {stats['total_synced']})"
            sync_color = (0, 255, 0)
        else:
            sync_text = f"LINK: BUFFERING ({stats['pending_sync']} queued)"
            sync_color = (0, 165, 255)

        cv2.putText(
            frame,
            sync_text,
            (680, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            sync_color,
            1,
        )

        # Camera Health / Tamper Status
        if self.is_tampered:
            cv2.rectangle(frame, (0, 45), (frame.shape[1], 80), (0, 0, 180), -1)
            cv2.putText(
                frame,
                f"!! TAMPER ALERT: {self.tamper_status} !!",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        return frame

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, List[Dict]]:
        """
        Executes full 3-stage pipeline + Phase 2 filtering + Phase 3 ANPR and store-and-forward.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Health Check: Camera Tamper Detection
        self.is_tampered, self.tamper_status = self.false_alarm_filter.check_tampering(gray)

        # Stage 1: Lightweight Motion Monitor (MOG2)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        fg_mask = self.bg_subtractor.apply(gray_blur)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > self.motion_threshold_area for c in contours)

        detections_for_tracker = []

        # Stage 2: Gated Verification (Wake up YOLOv8 only if motion detected)
        if motion_detected:
            results = self.model(frame, classes=self.target_classes, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    bw = x2 - x1
                    bh = y2 - y1
                    detections_for_tracker.append(([x1, y1, bw, bh], conf, class_id))

        # Stage 3: Persistent Tracking (DeepSORT)
        tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)

        active_track_ids = []
        current_alerts = []
        active_threat_count = 0

        # Draw Geofence & Tripwire overlays
        frame = self.draw_geofence_overlays(frame)

        # Phase 2 & 3: Filtering, ANPR, and Alert Enqueueing
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            active_track_ids.append(track_id)

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bottom_center = ((x1 + x2) / 2.0, float(y2))
            class_id = int(track.det_class) if track.det_class is not None else 0
            conf = float(track.get_det_conf()) if track.get_det_conf() is not None else 0.85

            # 1. False Alarm Filter Validation
            val_result = self.false_alarm_filter.validate_track(
                track_id=track_id,
                bbox_bottom_center=bottom_center,
                bbox_width=bbox_w,
                bbox_height=bbox_h,
                object_class=class_id,
                timestamp=timestamp,
            )

            is_valid = val_result["is_valid"]
            x_w, y_w = val_result["world_coords"]
            dist_m = val_result["distance"]

            if is_valid:
                # Vehicle ANPR Extraction
                detected_plate = None
                if class_id in [2, 3, 5, 7]:  # Vehicles
                    crop_y1 = max(0, y1)
                    crop_y2 = min(h, y2)
                    crop_x1 = max(0, x1)
                    crop_x2 = min(w, x2)
                    if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                        veh_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        detected_plate, _ = self.anpr_engine.process_vehicle_frame(track_id, veh_crop)

                # 2. Rule Engine Evaluation
                history = self.false_alarm_filter.track_history.get(track_id, [])
                alert_payload = self.rule_engine.evaluate_track(
                    track_id=track_id,
                    track_history=history,
                    bbox_width=bbox_w,
                    bbox_height=bbox_h,
                    class_id=class_id,
                    confidence=conf,
                    tripwire_event=val_result["tripwire_event"],
                    timestamp=timestamp,
                )

                # Skip if rule engine filtered out the detection (Civilian zone, no rules matched)
                if not alert_payload:
                    box_color = (0, 255, 0)
                    label = f"ID:{track_id} (NOMINAL)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                    cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                    continue

                alert_payload["camera_id"] = self.camera_id
                if detected_plate:
                    alert_payload["license_plate"] = detected_plate

                current_alerts.append(alert_payload)

                priority_level = alert_payload.get("priority_level", "LOW")
                priority_score = alert_payload.get("priority_score", 0.0)
                primary_rule = alert_payload.get("primary_rule", "NOMINAL")
                velocity = alert_payload.get("velocity_mps", 0.0)
                cls_name = alert_payload.get("object_class", "target")

                if alert_payload.get("is_threat", False):
                    active_threat_count += 1

                # 3. Store-and-Forward Enqueueing (Rate limited to 1 alert per 3 sec per track)
                last_enq = self.last_enqueued_timestamps.get(track_id, 0.0)
                if (timestamp - last_enq > 3.0) and (alert_payload.get("is_threat") or priority_score >= 35.0):
                    crop_y1 = max(0, y1)
                    crop_y2 = min(h, y2)
                    crop_x1 = max(0, x1)
                    crop_x2 = min(w, x2)
                    target_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    self.edge_queue.enqueue_alert(
                        alert_payload,
                        image_crop=target_crop,
                        license_plate=detected_plate,
                    )
                    self.last_enqueued_timestamps[track_id] = timestamp

                # Color coding based on Priority Level
                if priority_level == "CRITICAL":
                    box_color = (0, 0, 255)  # Red
                elif priority_level == "HIGH":
                    box_color = (0, 140, 255)  # Amber
                elif priority_level == "MEDIUM":
                    box_color = (0, 230, 255)  # Yellow
                else:
                    box_color = (0, 255, 0)  # Green

                plate_label = f" [{detected_plate}]" if detected_plate else ""
                label = f"ID:{track_id} {cls_name}{plate_label} | {dist_m:.1f}m | {velocity:.1f}m/s | P:{priority_score:.0f}"
            else:
                box_color = (160, 160, 160)
                label = f"ID:{track_id} (FILTERED: {val_result['filter_reason']})"

            # Render Bounding Box & Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                2,
            )

        # Cleanup memory for dead tracks
        self.false_alarm_filter.cleanup_old_tracks(active_track_ids)
        self.rule_engine.cleanup_old_tracks(active_track_ids)
        self.anpr_engine.cleanup_old_tracks(active_track_ids)

        # Render HUD
        frame = self.draw_hud(frame, motion_detected, active_threat_count)

        self.latest_alerts = current_alerts
        return frame, current_alerts

    def run(self):
        """
        Main execution loop for continuous video streaming.
        Phase 5: Loops indefinitely on file-based sources (resets to frame 0 on EOF).
        Also polls the backend for zone updates and writes the processed frame to
        the thread-safe JPEG buffer for the MJPEG streaming endpoint.

        Resolution note: all frames are downscaled to STREAM_W × STREAM_H immediately
        after capture. 4K source → 720p cuts pixel count ~9× and is the single
        biggest CPU saving available without changing the AI pipeline.
        """
        # Target resolution for the processing + streaming pipeline.
        # 1280×720 (720p) is the sweet spot: fluid MJPEG stream, fast YOLO inference.
        # Change to (1920, 1080) on Tier 2/3 hardware if CPU headroom allows.
        STREAM_W, STREAM_H = 1280, 720

        print(f"[VisionEngine] Running IBVAP Engine on source: {self.source} "
              f"(downscaling to {STREAM_W}×{STREAM_H}) ...")

        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 25.0
        frame_delay = 1.0 / source_fps

        try:
            while True:
                # Limit read speed to actual video FPS to prevent buffer bloat
                time.sleep(frame_delay)
                ret, frame = self.cap.read()

                # Phase 5: Loop video files indefinitely — reset on end-of-file.
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print(f"[VisionEngine] Critical: cannot read from source '{self.source}'. Stopping.")
                        break

                # ── Phase 5: Downscale to 720p immediately after capture ──────────
                # Must happen BEFORE MOG2, YOLO, DeepSORT, and JPEG encode so every
                # downstream operation benefits. INTER_LINEAR is the fastest
                # interpolation that avoids blocky artifacts.
                frame = cv2.resize(frame, (STREAM_W, STREAM_H),
                                   interpolation=cv2.INTER_LINEAR)
                # ─────────────────────────────────────────────────────────────────

                current_time = time.time()
                if current_time - self.last_process_time < self.process_interval:
                    # Between AI processing intervals — push raw (resized) frame to
                    # buffer so the MJPEG stream stays fluid between 2-FPS ticks.
                    _, raw_jpeg = cv2.imencode(".jpg", frame,
                                              [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with self._frame_lock:
                        self._latest_jpeg = raw_jpeg.tobytes()
                    continue

                self.last_process_time = current_time

                # Phase 5: Periodic zone poll (non-blocking, every 30 s).
                if current_time - self._zone_last_polled >= self._zone_poll_interval:
                    self._poll_zone()
                    self._zone_last_polled = current_time

                processed_frame, alerts = self.process_frame(frame, current_time)

                # Phase 5: Write AI-annotated frame to JPEG buffer for streaming endpoint.
                _, jpeg_buf = cv2.imencode(".jpg", processed_frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._frame_lock:
                    self._latest_jpeg = jpeg_buf.tobytes()

                if not self.headless:
                    try:
                        cv2.imshow("IBVAP - Intelligent Border Video Analytics", processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error as e:
                        print(f"[VisionEngine] GUI display window error ({e}). Switching to headless mode.")
                        self.headless = True
        finally:
            self.edge_queue.stop_sync_worker()
            self.cap.release()
            if not self.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBVAP Edge Vision Engine")
    parser.add_argument("--source", type=str, default="0", help="RTSP URL, video file path, or webcam index (default: 0)")
    parser.add_argument("--camera-id", type=str, default="CAM-BOP-01", help="Edge Camera Identifier")
    parser.add_argument("--backend-url", type=str, default="http://127.0.0.1:8000", help="Central Command API URL")
    parser.add_argument("--headless", action="store_true", help="Run without cv2 GUI window")
    args = parser.parse_args()

    engine = VisionEngine(
        source=args.source,
        camera_id=args.camera_id,
        backend_url=args.backend_url,
        headless=args.headless,
    )
    engine.run()

import sqlite3
import json
import base64
import time
import threading
import requests
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple


class EdgeQueue:
    """
    Store-and-Forward SQLite Alert Queue & Semantic Compression Worker.
    Provides:
      1. SQLite transactional storage for edge alerts with zero data loss.
      2. Priority-ordered queue retrieval (CRITICAL -> HIGH -> MEDIUM -> LOW).
      3. Base64 JPEG thumbnail compression helper.
      4. Background SyncWorker that polls Central Command /api/v1/health and flushes alerts.
    """

    def __init__(
        self,
        db_path: str = "edge_alerts.db",
        backend_url: str = "http://127.0.0.1:8000",
        camera_id: str = "CAM-BOP-01",
        sync_interval_sec: float = 2.0,
    ):
        self.db_path = db_path
        self.backend_url = backend_url.rstrip("/")
        self.camera_id = camera_id
        self.sync_interval_sec = sync_interval_sec

        self._lock = threading.Lock()
        self.init_db()

        # Sync worker control
        self._stop_event = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None
        self.is_connected = False
        self.total_synced = 0
        self.last_sync_time: Optional[float] = None

    def init_db(self):
        """Initializes the SQLite alerts table with index on priority and sync status."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    object_class TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    world_coords_x REAL NOT NULL,
                    world_coords_y REAL NOT NULL,
                    velocity_mps REAL NOT NULL,
                    heading_deg REAL NOT NULL,
                    zone TEXT NOT NULL,
                    primary_rule TEXT NOT NULL,
                    active_rules_json TEXT NOT NULL,
                    priority_score REAL NOT NULL,
                    priority_level TEXT NOT NULL,
                    license_plate TEXT,
                    thumbnail_b64 TEXT,
                    synced_status INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_sync_priority 
                ON alerts (synced_status, priority_score DESC, created_at ASC)
                """
            )
            conn.commit()
            conn.close()

    @staticmethod
    def compress_image_to_base64(image_crop: np.ndarray, quality: int = 75) -> str:
        """
        Compresses an image crop into a JPEG byte stream and returns a Base64-encoded string.
        Typically reduces image payload from hundreds of KB to 2-8 KB.
        """
        if image_crop is None or image_crop.size == 0:
            return ""

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode(".jpg", image_crop, encode_param)
        if not success:
            return ""

        b64_str = base64.b64encode(encoded_img).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def enqueue_alert(
        self,
        alert_payload: Dict[str, Any],
        image_crop: Optional[np.ndarray] = None,
        license_plate: Optional[str] = None,
    ) -> bool:
        """
        Inserts a new candidate/threat event into the local SQLite store-and-forward queue.
        """
        # Compress thumbnail if image is provided
        thumbnail_b64 = alert_payload.get("thumbnail_b64", "")
        if image_crop is not None and not thumbnail_b64:
            thumbnail_b64 = self.compress_image_to_base64(image_crop)

        alert_id = alert_payload.get("alert_id")
        if not alert_id:
            alert_id = f"ALT-{int(time.time() * 1000)}-{alert_payload.get('track_id', 0)}"

        timestamp = alert_payload.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        camera_id = alert_payload.get("camera_id", self.camera_id)
        track_id = int(alert_payload.get("track_id", 0))
        object_class = alert_payload.get("object_class", "unknown")
        class_id = int(alert_payload.get("class_id", 0))
        confidence = float(alert_payload.get("confidence", 0.85))

        world_coords = alert_payload.get("world_coords", {"x": 0.0, "y": 0.0})
        x_w = float(world_coords.get("x", 0.0))
        y_w = float(world_coords.get("y", 0.0))

        velocity_mps = float(alert_payload.get("velocity_mps", 0.0))
        heading_deg = float(alert_payload.get("heading_deg", 0.0))
        zone = alert_payload.get("zone", "GREEN_ZONE")
        primary_rule = alert_payload.get("primary_rule", "NOMINAL_TRACK")
        active_rules_json = json.dumps(alert_payload.get("active_rules", []))
        priority_score = float(alert_payload.get("priority_score", 0.0))
        priority_level = alert_payload.get("priority_level", "LOW")
        plate = license_plate or alert_payload.get("license_plate")

        created_at = time.time()

        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO alerts (
                        alert_id, timestamp, camera_id, track_id, object_class, class_id,
                        confidence, world_coords_x, world_coords_y, velocity_mps, heading_deg,
                        zone, primary_rule, active_rules_json, priority_score, priority_level,
                        license_plate, thumbnail_b64, synced_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        alert_id,
                        timestamp,
                        camera_id,
                        track_id,
                        object_class,
                        class_id,
                        confidence,
                        x_w,
                        y_w,
                        velocity_mps,
                        heading_deg,
                        zone,
                        primary_rule,
                        active_rules_json,
                        priority_score,
                        priority_level,
                        plate,
                        thumbnail_b64,
                        created_at,
                    ),
                )
                conn.commit()
                conn.close()
            return True
        except Exception as e:
            print(f"[EdgeQueue] Error enqueueing alert {alert_id}: {e}")
            return False

    def get_pending_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves unsynced alerts ordered strictly by Priority Score descending (Critical -> High -> Medium).
        """
        alerts = []
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM alerts 
                WHERE synced_status = 0 
                ORDER BY priority_score DESC, created_at ASC 
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            for row in rows:
                alerts.append({
                    "alert_id": row["alert_id"],
                    "timestamp": row["timestamp"],
                    "camera_id": row["camera_id"],
                    "track_id": row["track_id"],
                    "object_class": row["object_class"],
                    "class_id": row["class_id"],
                    "confidence": row["confidence"],
                    "world_coords": {"x": row["world_coords_x"], "y": row["world_coords_y"]},
                    "velocity_mps": row["velocity_mps"],
                    "heading_deg": row["heading_deg"],
                    "zone": row["zone"],
                    "primary_rule": row["primary_rule"],
                    "active_rules": json.loads(row["active_rules_json"]),
                    "priority_score": row["priority_score"],
                    "priority_level": row["priority_level"],
                    "license_plate": row["license_plate"],
                    "thumbnail_b64": row["thumbnail_b64"],
                    "synced_status": bool(row["synced_status"]),
                })
            conn.close()
        return alerts

    def mark_as_synced(self, alert_ids: List[str]):
        """Marks a batch of alert IDs as synced in the SQLite database."""
        if not alert_ids:
            return
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in alert_ids)
            cursor.execute(
                f"UPDATE alerts SET synced_status = 1 WHERE alert_id IN ({placeholders})",
                alert_ids,
            )
            conn.commit()
            conn.close()

    def get_queue_stats(self) -> Dict[str, Any]:
        """Returns statistics on total alerts, pending sync count, and priority breakdown."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM alerts WHERE synced_status = 0")
            pending_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM alerts WHERE synced_status = 0 AND priority_level = 'CRITICAL'"
            )
            pending_critical = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM alerts WHERE synced_status = 0 AND priority_level = 'HIGH'"
            )
            pending_high = cursor.fetchone()[0]

            conn.close()

        return {
            "total_alerts": total_count,
            "pending_sync": pending_count,
            "pending_critical": pending_critical,
            "pending_high": pending_high,
            "is_connected": self.is_connected,
            "total_synced": self.total_synced,
            "last_sync_time": self.last_sync_time,
        }

    def sync_once(self) -> Tuple[bool, int]:
        """
        Performs a single sync cycle:
          1. Pings backend /api/v1/health.
          2. If HTTP 200, fetches pending alerts in priority order and POSTs to /api/v1/alerts.
          3. On success, marks alerts as synced.
        Returns:
            (is_online, synced_count)
        """
        # 1. Health check ping
        health_url = f"{self.backend_url}/api/v1/health"
        try:
            resp = requests.get(health_url, timeout=2.0)
            if resp.status_code == 200:
                self.is_connected = True
            else:
                self.is_connected = False
                return False, 0
        except Exception:
            self.is_connected = False
            return False, 0

        # 2. Flush pending alerts in priority order
        pending_alerts = self.get_pending_alerts(limit=25)
        if not pending_alerts:
            return True, 0

        ingest_url = f"{self.backend_url}/api/v1/alerts"
        synced_ids = []

        for alert in pending_alerts:
            try:
                post_resp = requests.post(ingest_url, json=alert, timeout=3.0)
                if post_resp.status_code in [200, 201]:
                    synced_ids.append(alert["alert_id"])
                else:
                    # Non-200 response -> stop batch to avoid out-of-order syncing
                    break
            except Exception as e:
                print(f"[EdgeQueue] Sync failed for alert {alert['alert_id']}: {e}")
                self.is_connected = False
                break

        if synced_ids:
            self.mark_as_synced(synced_ids)
            self.total_synced += len(synced_ids)
            self.last_sync_time = time.time()

        return True, len(synced_ids)

    def _sync_worker_loop(self):
        """Continuous background thread loop for store-and-forward synchronization."""
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception as e:
                print(f"[EdgeQueue] SyncWorker loop exception: {e}")

            # Sleep for sync interval or until stopped
            self._stop_event.wait(timeout=self.sync_interval_sec)

    def start_sync_worker(self):
        """Starts the asynchronous background synchronization worker thread."""
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return
        self._stop_event.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_worker_loop, name="EdgeQueueSyncWorker", daemon=True
        )
        self._sync_thread.start()
        print(f"[EdgeQueue] SyncWorker started -> target: {self.backend_url}")

    def stop_sync_worker(self):
        """Stops the background synchronization worker thread."""
        self._stop_event.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=3.0)
            self._sync_thread = None
            print("[EdgeQueue] SyncWorker stopped.")

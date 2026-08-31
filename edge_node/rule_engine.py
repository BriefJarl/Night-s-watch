import cv2
import numpy as np
import math
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any


class RuleEngine:
    """
    Spatiotemporal Rule Engine & Alert Prioritization for Border Video Analytics.
    Implements:
      1. Polygonal Zone Geofencing (Red Zone / Zero Line, Amber Zone / Buffer).
      2. Behavioral Rules:
         - Loitering (dwell time in restricted zone >= threshold).
         - Crawling / Prone Infiltration (aspect ratio < 0.85 + low ground velocity for person).
         - Speeding / Sprinting / Vehicle Breach.
         - Tripwire Inbound / Outbound Breach.
      3. Kinematics Computation (ground velocity m/s, heading angle in degrees).
      4. Actionable Multi-Factor Priority Scoring Function:
         Score = clamp(W_rule * C_zone * C_class * C_time * confidence, 0, 100).
    """

    # Class ID mappings
    CLASS_NAMES = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    # Base rule severity weights
    RULE_WEIGHTS = {
        "TAMPER_OCCLUSION": 95.0,
        "TAMPER_DEFOCUS": 95.0,
        "TRIPWIRE_INBOUND": 90.0,
        "CRAWLING_INTRUSION": 85.0,
        "RESTRICTED_ZONE_INTRUSION_HIGH_ALERT": 95.0,
        "RESTRICTED_ZONE_INTRUSION": 80.0,  # Phase 5: operator-drawn pixel zone
        "LOITERING": 70.0,
        "ZONE_INTRUSION": 65.0,
        "SPEEDING": 60.0,
        "TRIPWIRE_OUTBOUND": 15.0,
        "NOMINAL_TRACK": 10.0,
    }

    # Zone criticality multipliers
    ZONE_MULTIPLIERS = {
        "RED_ZONE": 1.25,
        "AMBER_ZONE": 1.0,
        "GREEN_ZONE": 0.7,
    }

    # Class priority multipliers
    CLASS_MULTIPLIERS = {
        0: 1.10,  # Person
        2: 1.05,  # Car
        3: 1.05,  # Motorcycle
        5: 1.08,  # Bus
        7: 1.08,  # Truck
    }

    def __init__(
        self,
        red_zone_polygon: Optional[List[Tuple[float, float]]] = None,
        amber_zone_polygon: Optional[List[Tuple[float, float]]] = None,
        loiter_time_threshold_sec: float = 10.0,
        person_speed_threshold_mps: float = 4.5,
        vehicle_speed_threshold_mps: float = 12.0,
    ):
        # 1. Zone Polygons in Real-world Ground Coordinates (meters)
        # Red Zone: Zero Line / High Security strip (X: -12 to 12m, Y: 0 to 8m)
        if red_zone_polygon is None:
            self.red_zone = np.array(
                [[-12.0, 0.0], [12.0, 0.0], [12.0, 8.0], [-12.0, 8.0]],
                dtype=np.float32,
            )
        else:
            self.red_zone = np.array(red_zone_polygon, dtype=np.float32)

        # Amber Zone: Buffer Warning zone (X: -15 to 15m, Y: 8 to 16m)
        if amber_zone_polygon is None:
            self.amber_zone = np.array(
                [[-15.0, 8.0], [15.0, 8.0], [15.0, 16.0], [-15.0, 16.0]],
                dtype=np.float32,
            )
        else:
            self.amber_zone = np.array(amber_zone_polygon, dtype=np.float32)

        self.loiter_time_threshold = loiter_time_threshold_sec
        self.person_speed_threshold = person_speed_threshold_mps
        self.vehicle_speed_threshold = vehicle_speed_threshold_mps

        # Phase 5: Operator-drawn pixel-space restricted zone polygon.
        # Set via set_user_zone() after construction; None = no user zone active.
        self.user_zone_polygon: Optional[np.ndarray] = None
        self.user_zone_mode: str = "Alert zone"
        self._user_zone_lock = __import__('threading').Lock()

        # Zone Dwell State Tracking:
        # Map: track_id -> {"zone": str, "entry_time": float, "last_time": float}
        self.track_zone_state: Dict[int, Dict[str, Any]] = {}

    def set_user_zone(self, polygon_points: Optional[List[List[int]]], mode: str = "Alert zone") -> None:
        """
        Phase 5: Sets (or clears) the operator-drawn pixel-space restricted zone and mode.
        Thread-safe — called from VisionEngine's background zone-polling worker.

        Args:
            polygon_points: list of [x, y] integer pixel coordinates,
                            e.g. [[100,200],[300,200],[300,400],[100,400]].
                            Pass None or empty list to disable the user zone.
            mode: Surveillance mode string.
        """
        with self._user_zone_lock:
            self.user_zone_mode = mode
            if polygon_points and len(polygon_points) >= 3:
                self.user_zone_polygon = np.array(polygon_points, dtype=np.float32)
            else:
                self.user_zone_polygon = None

    def is_in_user_zone(self, u: float, v: float) -> bool:
        """
        Phase 5: Tests whether pixel point (u, v) lies inside the operator-drawn zone.
        Uses cv2.pointPolygonTest — returns True if inside or on the boundary.
        Thread-safe read against set_user_zone().
        """
        with self._user_zone_lock:
            if self.user_zone_polygon is None:
                return False
            pt = (float(u), float(v))
            return cv2.pointPolygonTest(self.user_zone_polygon, pt, False) >= 0

    def get_zone_for_point(self, x_w: float, y_w: float) -> str:
        """
        Determines which geofenced zone contains the ground coordinate (X_w, Y_w).
        """
        pt = (float(x_w), float(y_w))
        # cv2.pointPolygonTest returns >= 0 if inside or on contour
        if cv2.pointPolygonTest(self.red_zone, pt, False) >= 0:
            return "RED_ZONE"
        if cv2.pointPolygonTest(self.amber_zone, pt, False) >= 0:
            return "AMBER_ZONE"
        return "GREEN_ZONE"

    def compute_kinematics(
        self,
        track_history: List[Dict[str, Any]],
        current_time: float,
    ) -> Tuple[float, float]:
        """
        Computes smoothed ground velocity (m/s) and heading angle (degrees, 0-360) from recent track history.
        """
        if len(track_history) < 2:
            return 0.0, 0.0

        # Look across up to the last 5 frames for stable velocity calculation
        k = min(len(track_history), 5)
        p_old = track_history[-k]["world_pos"]
        t_old = track_history[-k]["timestamp"]
        p_curr = track_history[-1]["world_pos"]
        t_curr = track_history[-1]["timestamp"]

        dt = t_curr - t_old
        if dt <= 1e-4:
            dt = 0.5  # fallback for 2 FPS assumption

        dx = p_curr[0] - p_old[0]
        dy = p_curr[1] - p_old[1]
        ds = math.sqrt(dx**2 + dy**2)
        velocity_mps = ds / dt

        # Heading angle: 0 deg = +X (East), 90 deg = +Y (North / distant depth)
        heading_rad = math.atan2(dy, dx)
        heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0

        return float(velocity_mps), float(heading_deg)

    @staticmethod
    def get_time_of_day_multiplier(timestamp: Optional[float] = None) -> float:
        """
        Calculates time-of-day threat multiplier:
        Night hours (20:00 to 05:00 local time) have heightened border risk (1.25x).
        Day hours have nominal multiplier (1.0x).
        """
        if timestamp is None:
            dt = datetime.now()
        else:
            dt = datetime.fromtimestamp(timestamp)

        hour = dt.hour
        # Night: 20:00 to 05:00
        if hour >= 20 or hour < 5:
            return 1.25
        return 1.0

    def compute_priority_score(
        self,
        primary_rule: str,
        zone: str,
        class_id: int,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Calculates normalized Priority Score [0.0 - 100.0] and categorizes into Priority Level:
          Priority = clamp(W_rule * C_zone * C_class * C_time * conf, 0, 100)
        """
        w_rule = self.RULE_WEIGHTS.get(primary_rule, 10.0)
        c_zone = self.ZONE_MULTIPLIERS.get(zone, 1.0)
        c_class = self.CLASS_MULTIPLIERS.get(class_id, 1.0)
        c_time = self.get_time_of_day_multiplier(timestamp)
        conf_clamped = max(0.2, min(1.0, confidence))

        raw_score = w_rule * c_zone * c_class * c_time * conf_clamped
        score = max(0.0, min(100.0, raw_score))

        if score >= 75.0:
            level = "CRITICAL"
        elif score >= 50.0:
            level = "HIGH"
        elif score >= 25.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        return round(score, 1), level

    def evaluate_track(
        self,
        track_id: int,
        track_history: List[Dict[str, Any]],
        bbox_width: float,
        bbox_height: float,
        class_id: int,
        confidence: float,
        tripwire_event: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates spatiotemporal behavioral rules over a confirmed track and generates an alert payload.
        Returns:
            alert_dict (Dict[str, Any]): Full alert telemetry with priority scoring.
        """
        if timestamp is None:
            timestamp = time.time()

        if not track_history:
            return {}

        current_entry = track_history[-1]
        x_w, y_w = current_entry["world_pos"]
        u, v = current_entry["bbox_bottom"]

        # 1. Zone Geofencing
        current_zone = self.get_zone_for_point(x_w, y_w)

        # 2. Kinematics (Speed & Heading)
        velocity_mps, heading_deg = self.compute_kinematics(track_history, timestamp)

        # 3. Behavioral Rules Evaluation
        active_rules = []

        # Rule A: Tripwire Inbound Breach
        if tripwire_event == "INBOUND":
            active_rules.append("TRIPWIRE_INBOUND")
        elif tripwire_event == "OUTBOUND":
            active_rules.append("TRIPWIRE_OUTBOUND")

        # Rule B: Loitering Detection
        if track_id not in self.track_zone_state or self.track_zone_state[track_id]["zone"] != current_zone:
            self.track_zone_state[track_id] = {
                "zone": current_zone,
                "entry_time": timestamp,
                "last_time": timestamp,
            }
        else:
            self.track_zone_state[track_id]["last_time"] = timestamp

        aspect_ratio = bbox_height / (bbox_width + 1e-5)

        # Compute dwell time from zone state tracking
        zone_state = self.track_zone_state.get(track_id)
        if zone_state:
            dwell_time = timestamp - zone_state.get("entry_time", timestamp)
        else:
            dwell_time = 0.0

        # -------------------------------------------------------------
        # STRICT SURVEILLANCE MODE RULES (Overrides legacy rules)
        # -------------------------------------------------------------
        mode = getattr(self, "user_zone_mode", "Alert zone")
        if mode == "Civilian zone":
            # Lenient mode: Ignore standard human and vehicle detection — no alert
            if class_id in [0, 2, 3, 5, 7]:
                return None
        elif mode == "No Civilian zone":
            if class_id == 0:
                active_rules.append("RESTRICTED_ZONE_INTRUSION_HIGH_ALERT")
        elif mode == "No vehicle zone":
            if class_id in [2, 3, 5, 7]:
                active_rules.append("RESTRICTED_ZONE_INTRUSION_HIGH_ALERT")
        elif mode == "Emergency/sensitive zone":
            active_rules.append("RESTRICTED_ZONE_INTRUSION_HIGH_ALERT")
        else:
            # Fallback for "Alert zone" or legacy modes
            active_rules.append("RESTRICTED_ZONE_INTRUSION")

        # If the strict surveillance mode did not trigger an alert, ignore the detection.
        if not active_rules:
            return None

        # Determine Primary Rule by highest severity weight
        primary_rule = max(active_rules, key=lambda r: self.RULE_WEIGHTS.get(r, 0.0))

        # 4. Priority Scoring Function
        priority_score, priority_level = self.compute_priority_score(
            primary_rule=primary_rule,
            zone=current_zone,
            class_id=class_id,
            confidence=confidence,
            timestamp=timestamp,
        )

        class_name = self.CLASS_NAMES.get(class_id, f"object_{class_id}")
        iso_timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        alert_payload = {
            "alert_id": f"ALT-{int(timestamp*1000)}-{track_id}",
            "timestamp": iso_timestamp,
            "track_id": track_id,
            "class_id": class_id,
            "object_class": class_name,
            "confidence": round(confidence, 2),
            "world_coords": {"x": round(x_w, 2), "y": round(y_w, 2)},
            "pixel_bottom_center": {"u": int(u), "v": int(v)},
            "aspect_ratio": round(aspect_ratio, 2),
            "velocity_mps": round(velocity_mps, 2),
            "heading_deg": round(heading_deg, 1),
            "zone": current_zone,
            "dwell_time_sec": round(dwell_time, 1),
            "active_rules": active_rules,
            "primary_rule": primary_rule,
            "priority_score": priority_score,
            "priority_level": priority_level,
            "is_threat": (priority_level in ["CRITICAL", "HIGH"]),
            "synced_status": False,
        }

        return alert_payload

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Cleans up zone state for inactive tracks.
        """
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self.track_zone_state if tid not in active_set]
        for tid in stale_ids:
            del self.track_zone_state[tid]

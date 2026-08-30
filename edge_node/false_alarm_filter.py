import cv2
import numpy as np
import math
import time
from typing import Tuple, Optional, Dict, List, Any


class FalseAlarmFilter:
    """
    4-Layer False Alarm Suppression Stack for Edge Video Analytics.
    Implements:
      1. Calibrated Ground Homography mapping (Pixel <-> Real-world Coordinates in meters).
      2. Size-vs-Depth Homography Gate (rejection of forced-perspective bugs/shadows/scale mismatches).
      3. Track Confirmation Gate (Persistence >= 8 frames AND spatial displacement >= 2.0 meters).
      4. Directional Virtual Fence (Vector dot product of trajectory vs tripwire inward normal).
      5. Camera Tamper Detection (Defocus via Laplacian variance and Occlusion via histogram/std dev).
    """

    def __init__(
        self,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        tripwire_a: Tuple[float, float] = (-10.0, 10.0),
        tripwire_b: Tuple[float, float] = (10.0, 10.0),
        inward_normal: Optional[Tuple[float, float]] = None,
        min_persistence_frames: int = 8,
        min_displacement_meters: float = 2.0,
        focal_length_px: float = 1000.0,
    ):
        # 1. Homography Calibration (Standard 1080p ground plane looking forward-down)
        # Default: 4-point ground calibration (image pixels -> ground plane in meters)
        if src_points is None:
            # Calibrated points on standard 1920x1080 frame
            self.src_pts = np.array(
                [
                    [400.0, 1080.0],   # Bottom Left in image
                    [1520.0, 1080.0],  # Bottom Right in image
                    [800.0, 500.0],    # Distant Left in image
                    [1120.0, 500.0],   # Distant Right in image
                ],
                dtype=np.float32,
            )
        else:
            self.src_pts = np.array(src_points, dtype=np.float32)

        if dst_points is None:
            # Real-world metric coordinates: X in [-5, 5] meters, Y (distance) in [2, 15] meters
            self.dst_pts = np.array(
                [
                    [-5.0, 2.0],       # Bottom Left (2m depth)
                    [5.0, 2.0],        # Bottom Right (2m depth)
                    [-5.0, 15.0],      # Distant Left (15m depth)
                    [5.0, 15.0],       # Distant Right (15m depth)
                ],
                dtype=np.float32,
            )
        else:
            self.dst_pts = np.array(dst_points, dtype=np.float32)

        # Compute forward homography (Image -> World) and inverse (World -> Image)
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)
        self.H_inv = np.linalg.inv(self.H) if self.H is not None else None

        self.focal_length_px = focal_length_px

        # 2. Track Persistence & State
        # Map: track_id -> List[dict] where dict contains:
        # {"timestamp": float, "world_pos": (X_w, Y_w), "bbox": [x1, y1, w, h], "class_id": int}
        self.track_history: Dict[int, List[Dict[str, Any]]] = {}
        self.min_persistence_frames = min_persistence_frames
        self.min_displacement_meters = min_displacement_meters

        # 3. Directional Virtual Fence (Tripwire) in World Coordinates
        self.tripwire_A = np.array(tripwire_a, dtype=np.float32)
        self.tripwire_B = np.array(tripwire_b, dtype=np.float32)

        # Compute tripwire direction vector and inward normal (pointing towards domestic zone, Y < 10)
        v_ab = self.tripwire_B - self.tripwire_A
        v_len = np.linalg.norm(v_ab)
        if v_len > 1e-6:
            v_ab_norm = v_ab / v_len
        else:
            v_ab_norm = np.array([1.0, 0.0], dtype=np.float32)

        if inward_normal is not None:
            self.inward_normal = np.array(inward_normal, dtype=np.float32)
            self.inward_normal /= (np.linalg.norm(self.inward_normal) + 1e-6)
        else:
            # Perpendicular vector pointing inwards (towards decreasing Y if tripwire is horizontal at Y=10)
            normal = np.array([-v_ab_norm[1], v_ab_norm[0]], dtype=np.float32)
            # Standardize so it points in the negative Y direction (towards the base/domestic zone)
            if normal[1] > 0:
                normal = -normal
            self.inward_normal = normal

        # State tracking for tripwire crossings per track
        # Map: track_id -> str ("INBOUND" or "OUTBOUND")
        self.crossed_tracks: Dict[int, str] = {}

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Projects 2D image pixel coordinates (u, v) to 2D real-world ground coordinates (X_w, Y_w) in meters.
        """
        if self.H is None:
            return 0.0, 0.0
        pt_img = np.array([[[u, v]]], dtype=np.float32)
        pt_world = cv2.perspectiveTransform(pt_img, self.H)
        x_w, y_w = pt_world[0][0]
        return float(x_w), float(y_w)

    def world_to_pixel(self, x_w: float, y_w: float) -> Tuple[int, int]:
        """
        Projects 2D real-world ground coordinates (X_w, Y_w) in meters to image pixel coordinates (u, v).
        """
        if self.H_inv is None:
            return 0, 0
        pt_world = np.array([[[x_w, y_w]]], dtype=np.float32)
        pt_img = cv2.perspectiveTransform(pt_world, self.H_inv)
        u, v = pt_img[0][0]
        return int(round(u)), int(round(v))

    def estimate_physical_height(self, bbox_height_px: float, distance_m: float) -> float:
        """
        Calculates estimated real-world physical height (in meters) of an object
        given its pixel height and projected distance to camera using the pinhole perspective model:
          H_est = (h_px * D) / f_eff
        """
        if self.focal_length_px <= 0 or distance_m <= 0:
            return 0.0
        h_est = (bbox_height_px * distance_m) / self.focal_length_px
        return float(h_est)

    def check_size_depth_gate(
        self,
        bbox_height_px: float,
        bbox_width_px: float,
        distance_m: float,
        object_class: int,
    ) -> Tuple[bool, float, str]:
        """
        Homography Geometric Gate: Rejects forced-perspective false alarms
        (e.g., bug on lens, distant foliage, or scale-mismatched objects).
        Returns:
            is_valid (bool): True if physical dimensions match expected object class.
            h_est (float): Estimated physical height in meters.
            rejection_reason (str): Reason if rejected, or "OK".
        """
        h_est = self.estimate_physical_height(bbox_height_px, distance_m)

        # 1. Extreme Perspective Outlier Filter
        # Bug crawling on lens: large pixel height at large projected distance -> absurd physical height
        if distance_m > 20.0 and bbox_height_px > 450.0:
            return False, h_est, f"Forced-perspective anomaly (Distance {distance_m:.1f}m, Height {bbox_height_px:.0f}px)"

        # Micro-noise filter at close range
        if distance_m < 3.0 and bbox_height_px < 15.0:
            return False, h_est, f"Micro-noise sub-threshold (Height {bbox_height_px:.0f}px at {distance_m:.1f}m)"

        # 2. Class-Specific Physical Dimension Validation
        # Class 0: Person
        if object_class == 0:
            # Person physical height range: 0.40m (crawling/prone) to 2.60m (tall standing human)
            if h_est < 0.40:
                return False, h_est, f"Physical height underflow for person ({h_est:.2f}m < 0.40m)"
            if h_est > 2.60:
                return False, h_est, f"Physical height overflow for person ({h_est:.2f}m > 2.60m)"

        # Classes 2, 3, 5, 7: Vehicles (Car, Motorcycle, Bus, Truck)
        elif object_class in [2, 3, 5, 7]:
            # Vehicle height range: 0.60m (motorcycle) to 5.20m (large truck/bus)
            if h_est < 0.60:
                return False, h_est, f"Physical height underflow for vehicle ({h_est:.2f}m < 0.60m)"
            if h_est > 5.20:
                return False, h_est, f"Physical height overflow for vehicle ({h_est:.2f}m > 5.20m)"

        return True, h_est, "OK"

    def check_tripwire_crossing(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
    ) -> Optional[str]:
        """
        Determines if a trajectory segment (prev_pos -> curr_pos) crosses the virtual fence tripwire
        and calculates crossing direction using vector dot product with the inward normal.
        Returns:
            "INBOUND"  -> Object crossed into protected area (Threat).
            "OUTBOUND" -> Object crossed out of protected area (Nominal/Patrol).
            None       -> No crossing.
        """
        p1 = np.array(prev_pos, dtype=np.float32)
        p2 = np.array(curr_pos, dtype=np.float32)
        a = self.tripwire_A
        b = self.tripwire_B

        # Check line segment intersection between (p1, p2) and (a, b)
        def ccw(pt_a, pt_b, pt_c):
            return (pt_c[1] - pt_a[1]) * (pt_b[0] - pt_a[0]) > (pt_b[1] - pt_a[1]) * (pt_c[0] - pt_a[0])

        intersect = (ccw(a, p1, p2) != ccw(b, p1, p2)) and (ccw(a, b, p1) != ccw(a, b, p2))

        if not intersect:
            return None

        # Calculate trajectory vector
        v_track = p2 - p1
        v_len = np.linalg.norm(v_track)
        if v_len < 1e-6:
            return None

        v_track_unit = v_track / v_len

        # Directional classification via dot product with inward normal
        dot_product = float(np.dot(v_track_unit, self.inward_normal))

        if dot_product > 0.0:
            return "INBOUND"
        else:
            return "OUTBOUND"

    def validate_track(
        self,
        track_id: int,
        bbox_bottom_center: Tuple[float, float],
        bbox_width: float,
        bbox_height: float,
        object_class: int,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Full 4-layer validation pipeline for a tracked bounding box.
        Returns a rich status dictionary containing:
            is_valid (bool): True if candidate alert passed all gates.
            world_coords (Tuple[float, float]): (X_w, Y_w) in meters.
            distance (float): Ground distance D in meters.
            physical_height (float): Estimated physical height in meters.
            persistence_frames (int): Number of observed consecutive frames.
            displacement_m (float): Total spatial displacement in meters.
            tripwire_event (Optional[str]): "INBOUND", "OUTBOUND", or None.
            filter_reason (str): Reason for filtering if not valid, or "CONFIRMED".
        """
        if timestamp is None:
            timestamp = time.time()

        u, v = bbox_bottom_center
        x_w, y_w = self.pixel_to_world(u, v)
        distance_m = math.sqrt(x_w**2 + y_w**2)

        # 1. Size-vs-Depth Homography Gate
        size_valid, h_est, size_reason = self.check_size_depth_gate(
            bbox_height, bbox_width, distance_m, object_class
        )
        if not size_valid:
            return {
                "is_valid": False,
                "world_coords": (x_w, y_w),
                "distance": distance_m,
                "physical_height": h_est,
                "persistence_frames": 1,
                "displacement_m": 0.0,
                "tripwire_event": None,
                "filter_reason": f"SIZE_DEPTH_REJECT: {size_reason}",
            }

        # 2. Track Persistence & Displacement Gate
        if track_id not in self.track_history:
            self.track_history[track_id] = []

        self.track_history[track_id].append({
            "timestamp": timestamp,
            "world_pos": (x_w, y_w),
            "bbox_bottom": (u, v),
            "bbox_size": (bbox_width, bbox_height),
            "class_id": object_class,
        })

        # Keep history capped to avoid memory growth
        if len(self.track_history[track_id]) > 60:
            self.track_history[track_id].pop(0)

        history = self.track_history[track_id]
        persistence_frames = len(history)

        # Compute net displacement from first detection in history
        init_pos = history[0]["world_pos"]
        displacement_m = math.sqrt((x_w - init_pos[0])**2 + (y_w - init_pos[1])**2)

        # Check Persistence Gate (>= 8 frames)
        if persistence_frames < self.min_persistence_frames:
            return {
                "is_valid": False,
                "world_coords": (x_w, y_w),
                "distance": distance_m,
                "physical_height": h_est,
                "persistence_frames": persistence_frames,
                "displacement_m": displacement_m,
                "tripwire_event": None,
                "filter_reason": f"PERSISTENCE_GATE ({persistence_frames}/{self.min_persistence_frames} frames)",
            }

        # Check Spatial Displacement Gate (>= 2.0 m)
        if displacement_m < self.min_displacement_meters:
            return {
                "is_valid": False,
                "world_coords": (x_w, y_w),
                "distance": distance_m,
                "physical_height": h_est,
                "persistence_frames": persistence_frames,
                "displacement_m": displacement_m,
                "tripwire_event": None,
                "filter_reason": f"DISPLACEMENT_GATE ({displacement_m:.2f}m < {self.min_displacement_meters}m)",
            }

        # 3. Directional Tripwire Crossing Check
        tripwire_event = None
        if len(history) >= 2:
            prev_pos = history[-2]["world_pos"]
            curr_pos = (x_w, y_w)
            crossing = self.check_tripwire_crossing(prev_pos, curr_pos)
            if crossing:
                # Register crossing event
                tripwire_event = crossing
                self.crossed_tracks[track_id] = crossing

        return {
            "is_valid": True,
            "world_coords": (x_w, y_w),
            "distance": distance_m,
            "physical_height": h_est,
            "persistence_frames": persistence_frames,
            "displacement_m": displacement_m,
            "tripwire_event": tripwire_event,
            "filter_reason": "CONFIRMED",
        }

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Removes stored track history for tracks that are no longer active in DeepSORT.
        """
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self.track_history if tid not in active_set]
        for tid in stale_ids:
            del self.track_history[tid]
            if tid in self.crossed_tracks:
                del self.crossed_tracks[tid]

    @staticmethod
    def check_tampering(gray_frame: np.ndarray, blur_threshold: float = 45.0) -> Tuple[bool, str]:
        """
        Camera Tamper Detection Module:
          1. Occlusion / Spray Paint / Solid Cover: Histogram peak collapse (> 0.85) or standard deviation collapse (< 12.0).
          2. Defocus / Severe Blur: Variance of Laplacian < threshold.
        Returns:
            is_tampered (bool)
            reason (str)
        """
        if gray_frame is None or gray_frame.size == 0:
            return True, "EMPTY_FRAME"

        # 1. Occlusion / Spray Paint / Solid Cover Detection (Checked first before blur)
        mean_val, std_val = cv2.meanStdDev(gray_frame)
        std_intensity = float(std_val[0][0])
        if std_intensity < 12.0:
            return True, f"OCCLUSION_UNIFORM (Intensity StdDev: {std_intensity:.1f} < 12.0)"

        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist_sum = float(hist.sum())
        if hist_sum > 0:
            hist_norm = hist / hist_sum
            peak_ratio = float(np.max(hist_norm))
            if peak_ratio > 0.85:
                return True, f"LENS_SPRAY_OCCLUSION (Peak histogram bin: {peak_ratio*100:.1f}% > 85%)"

        # 2. Defocus / Blur Detection (Variance of Laplacian)
        laplacian_var = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
        if laplacian_var < blur_threshold:
            return True, f"DEFOCUS_BLUR (Variance: {laplacian_var:.1f} < {blur_threshold})"

        return False, "OK"

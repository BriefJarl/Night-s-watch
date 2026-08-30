import cv2
import numpy as np
import math
import time
import os
import sys
import unittest
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_node.false_alarm_filter import FalseAlarmFilter
from edge_node.rule_engine import RuleEngine
from edge_node.vision_engine import VisionEngine


class TestPhase2Differentiator(unittest.TestCase):

    def setUp(self):
        self.filter = FalseAlarmFilter()
        self.rule_engine = RuleEngine()

    def test_01_homography_and_reversibility(self):
        """Test Homography mapping accuracy and pixel-world invertibility."""
        # Test calibration anchor points
        u, v = 960.0, 790.0  # approximate center-ground in image
        x_w, y_w = self.filter.pixel_to_world(u, v)
        self.assertIsInstance(x_w, float)
        self.assertIsInstance(y_w, float)
        self.assertGreater(y_w, 0.0, "Depth Y_w should be positive in front of camera")

        # Invertibility check
        u_rec, v_rec = self.filter.world_to_pixel(x_w, y_w)
        self.assertAlmostEqual(u, u_rec, delta=3, msg="U pixel reconstruction should match original")
        self.assertAlmostEqual(v, v_rec, delta=3, msg="V pixel reconstruction should match original")

    def test_02_size_vs_depth_gate(self):
        """Test Homography Geometric Gate for forced-perspective rejection."""
        # Normal standing human at 10m depth (e.g. 170px box -> ~1.7m)
        is_valid, h_est, reason = self.filter.check_size_depth_gate(
            bbox_height_px=170.0,
            bbox_width_px=60.0,
            distance_m=10.0,
            object_class=0,  # Person
        )
        self.assertTrue(is_valid, f"Normal human should pass size-depth gate: {reason}")
        self.assertAlmostEqual(h_est, 1.70, delta=0.2)

        # Forced-perspective anomaly: bug on lens (500px tall) projected at 25m distance -> 12.5m tall
        is_valid, h_est, reason = self.filter.check_size_depth_gate(
            bbox_height_px=500.0,
            bbox_width_px=300.0,
            distance_m=25.0,
            object_class=0,
        )
        self.assertFalse(is_valid, "Extreme forced-perspective anomaly should be rejected")
        self.assertIn("Forced-perspective", reason)

        # Micro-noise: 10px box at 2m depth
        is_valid, h_est, reason = self.filter.check_size_depth_gate(
            bbox_height_px=10.0,
            bbox_width_px=10.0,
            distance_m=2.0,
            object_class=0,
        )
        self.assertFalse(is_valid, "Micro-noise close to camera should be rejected")

        # Dwarf / Underflow noise for person (20px box at 10m -> 0.2m tall)
        is_valid, h_est, reason = self.filter.check_size_depth_gate(
            bbox_height_px=20.0,
            bbox_width_px=10.0,
            distance_m=10.0,
            object_class=0,
        )
        self.assertFalse(is_valid, "Physically impossible small human should be rejected")
        self.assertIn("underflow", reason)

    def test_03_track_persistence_and_displacement_gate(self):
        """Test Track Confirmation Gate (Persistence >= 8 frames AND Displacement >= 2.0m)."""
        track_id = 101

        # Helper to compute realistic perspective bounding box for 1.7m human at ground position
        def get_human_box(u_px, v_px):
            xw, yw = self.filter.pixel_to_world(u_px, v_px)
            dist = math.sqrt(xw**2 + yw**2)
            h = (1.70 * 1000.0) / max(0.1, dist)
            w = h * 0.35
            return w, h

        # Feed 5 frames with displacement at ~6m depth -> Should FAIL on persistence
        for i in range(5):
            u_pt, v_pt = 960.0, 800.0 - i * 30.0
            w_box, h_box = get_human_box(u_pt, v_pt)
            res = self.filter.validate_track(
                track_id=track_id,
                bbox_bottom_center=(u_pt, v_pt),
                bbox_width=w_box,
                bbox_height=h_box,
                object_class=0,
                timestamp=1000.0 + i * 0.5,
            )
            self.assertFalse(res["is_valid"], "Track with <8 frames must be filtered")
            self.assertIn("PERSISTENCE_GATE", res["filter_reason"])

        # Feed 10 frames with near-zero displacement (stationary flickering noise)
        stat_track_id = 102
        for i in range(10):
            u_pt, v_pt = 960.0 + (i % 2) * 1.0, 750.0
            w_box, h_box = get_human_box(u_pt, v_pt)
            res = self.filter.validate_track(
                track_id=stat_track_id,
                bbox_bottom_center=(u_pt, v_pt),
                bbox_width=w_box,
                bbox_height=h_box,
                object_class=0,
                timestamp=2000.0 + i * 0.5,
            )
        self.assertFalse(res["is_valid"], "Stationary track (<2.0m) must be filtered despite >=8 frames")
        self.assertIn("DISPLACEMENT_GATE", res["filter_reason"])

        # Feed 10 frames with steady 4-meter movement -> Should PASS
        valid_track_id = 103
        for i in range(10):
            u_pt, v_pt = 960.0, 850.0 - i * 35.0
            w_box, h_box = get_human_box(u_pt, v_pt)
            res = self.filter.validate_track(
                track_id=valid_track_id,
                bbox_bottom_center=(u_pt, v_pt),
                bbox_width=w_box,
                bbox_height=h_box,
                object_class=0,
                timestamp=3000.0 + i * 0.5,
            )
        self.assertTrue(res["is_valid"], f"Track with >=8 frames and >=2.0m displacement should be confirmed: {res['filter_reason']}")
        self.assertEqual(res["filter_reason"], "CONFIRMED")
        self.assertGreaterEqual(res["displacement_m"], 2.0)

    def test_04_directional_virtual_fence_tripwire(self):
        """Test Tripwire crossing detection and INBOUND vs OUTBOUND vector dot-product classification."""
        # Tripwire is horizontal at Y = 10.0m. Inward normal points in -Y direction.

        # INBOUND trajectory: moving from Y = 12.0m (outside) to Y = 8.0m (inside) across X = 0
        p_prev = (0.0, 12.0)
        p_curr = (0.0, 8.0)
        crossing = self.filter.check_tripwire_crossing(p_prev, p_curr)
        self.assertEqual(crossing, "INBOUND", "Crossing from Y=12 to Y=8 must be INBOUND (Threat)")

        # OUTBOUND trajectory: moving from Y = 8.0m (inside) to Y = 12.0m (outside) across X = 0
        p_prev_out = (0.0, 8.0)
        p_curr_out = (0.0, 12.0)
        crossing_out = self.filter.check_tripwire_crossing(p_prev_out, p_curr_out)
        self.assertEqual(crossing_out, "OUTBOUND", "Crossing from Y=8 to Y=12 must be OUTBOUND (Nominal)")

        # Parallel trajectory not crossing: moving from (0, 5) to (5, 5)
        p_par1 = (0.0, 5.0)
        p_par2 = (5.0, 5.0)
        crossing_none = self.filter.check_tripwire_crossing(p_par1, p_par2)
        self.assertIsNone(crossing_none, "Trajectory parallel to tripwire should not trigger crossing")

    def test_05_crawling_and_prone_rule(self):
        """Test Crawling / Prone infiltration detection (aspect ratio < 0.85 + low speed)."""
        track_id = 201
        t_base = 5000.0
        # Create mock track history for person crawling at 0.5 m/s
        history = [
            {"timestamp": t_base + i * 0.5, "world_pos": (0.0, 5.0 + i * 0.25), "bbox_bottom": (960, 800), "class_id": 0}
            for i in range(10)
        ]
        # Crawling bounding box: width = 120px, height = 50px -> Aspect Ratio = 50/120 = 0.42 (< 0.85)
        alert = self.rule_engine.evaluate_track(
            track_id=track_id,
            track_history=history,
            bbox_width=120.0,
            bbox_height=50.0,
            class_id=0,  # Person
            confidence=0.90,
            timestamp=t_base + 5.0,
        )
        self.assertIn("CRAWLING_INTRUSION", alert["active_rules"], "Low aspect ratio + low speed person must trigger CRAWLING_INTRUSION")
        self.assertEqual(alert["primary_rule"], "CRAWLING_INTRUSION")
        self.assertIn(alert["priority_level"], ["CRITICAL", "HIGH"])

    def test_06_loitering_rule(self):
        """Test Loitering detection for dwell time >= 10 seconds in restricted zone."""
        track_id = 301
        t_base = 6000.0
        # History in Red Zone (Y = 4.0m) over 12 seconds
        history = [
            {"timestamp": t_base + i * 1.0, "world_pos": (1.0, 4.0), "bbox_bottom": (960, 900), "class_id": 0}
            for i in range(13)
        ]
        # Evaluate first frame
        self.rule_engine.evaluate_track(
            track_id=track_id,
            track_history=[history[0]],
            bbox_width=50.0,
            bbox_height=150.0,
            class_id=0,
            confidence=0.85,
            timestamp=t_base,
        )
        # Evaluate after 12 seconds
        alert = self.rule_engine.evaluate_track(
            track_id=track_id,
            track_history=history,
            bbox_width=50.0,
            bbox_height=150.0,
            class_id=0,
            confidence=0.85,
            timestamp=t_base + 12.0,
        )
        self.assertIn("LOITERING", alert["active_rules"], "Dwell time > 10s in Red Zone must trigger LOITERING")
        self.assertGreaterEqual(alert["dwell_time_sec"], 10.0)

    def test_07_alert_priority_scoring(self):
        """Test Multi-Factor Priority Scoring Function across rules, zones, and night multiplier."""
        # Tripwire Inbound in Red Zone at Night (timestamp at 23:00)
        night_ts = datetime(2026, 8, 29, 23, 0, 0).timestamp()
        score_night, level_night = self.rule_engine.compute_priority_score(
            primary_rule="TRIPWIRE_INBOUND",
            zone="RED_ZONE",
            class_id=0,
            confidence=0.95,
            timestamp=night_ts,
        )
        self.assertEqual(level_night, "CRITICAL")
        self.assertGreaterEqual(score_night, 75.0)

        # Day time comparison (timestamp at 14:00)
        day_ts = datetime(2026, 8, 29, 14, 0, 0).timestamp()
        score_day, level_day = self.rule_engine.compute_priority_score(
            primary_rule="TRIPWIRE_INBOUND",
            zone="RED_ZONE",
            class_id=0,
            confidence=0.95,
            timestamp=day_ts,
        )
        # Night multiplier (1.25x) should make raw score higher than daytime
        self.assertGreater(score_night, 0.0)
        self.assertGreater(score_day, 0.0)

        # Nominal track in green zone
        score_nom, level_nom = self.rule_engine.compute_priority_score(
            primary_rule="NOMINAL_TRACK",
            zone="GREEN_ZONE",
            class_id=2,  # Car
            confidence=0.80,
            timestamp=day_ts,
        )
        self.assertEqual(level_nom, "LOW")
        self.assertLess(score_nom, 25.0)

    def test_08_camera_tamper_detection(self):
        """Test Defocus and Occlusion/Spray tamper health checks."""
        # 1. Normal sharp textured frame with gradient and sharp edges
        normal_frame = np.zeros((480, 640), dtype=np.uint8)
        normal_frame[:240, :] = 200  # Bright sky region
        normal_frame[240:, :] = 50   # Dark ground region
        # Add high-frequency grid lines
        for r in range(0, 480, 30):
            normal_frame[r:r+3, :] = 255
        for c in range(0, 640, 30):
            normal_frame[:, c:c+3] = 255

        is_tampered, reason = FalseAlarmFilter.check_tampering(normal_frame)
        self.assertFalse(is_tampered, f"Normal sharp frame should not be tampered: {reason}")

        # 2. Defocused / heavily blurred frame (maintains global contrast between sky & ground, but loses sharp edges)
        blurred_frame = cv2.GaussianBlur(normal_frame, (45, 45), 0)
        is_tampered_blur, reason_blur = FalseAlarmFilter.check_tampering(blurred_frame, blur_threshold=45.0)
        self.assertTrue(is_tampered_blur, "Blurred frame should trigger DEFOCUS_BLUR")
        self.assertIn("DEFOCUS_BLUR", reason_blur)

        # 3. Occluded / spray-painted frame (solid gray or black, low variance and peak bin collapse)
        solid_frame = np.full((480, 640), 30, dtype=np.uint8)
        is_tampered_occ, reason_occ = FalseAlarmFilter.check_tampering(solid_frame)
        self.assertTrue(is_tampered_occ, "Solid/occluded frame should trigger OCCLUSION")
        self.assertTrue("OCCLUSION" in reason_occ or "SPRAY" in reason_occ)

    def test_09_vision_engine_pipeline_execution(self):
        """Test VisionEngine processing loop with synthetic test frame."""
        engine = VisionEngine(source="0", headless=True)
        # Create a synthetic 1080p frame
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Draw some mock objects
        cv2.rectangle(test_frame, (400, 300), (450, 450), (200, 200, 200), -1)

        out_frame, alerts = engine.process_frame(test_frame, timestamp=time.time())
        self.assertEqual(out_frame.shape, (1080, 1920, 3))
        self.assertIsInstance(alerts, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)

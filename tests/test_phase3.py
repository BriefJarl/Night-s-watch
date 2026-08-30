import os
import sys
import time
import json
import numpy as np
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge_node.anpr_engine import ANPREngine
from edge_node.edge_queue import EdgeQueue
from backend.main import app, ALERTS_DB


class TestPhase3TransmissionAndANPR(unittest.TestCase):

    def setUp(self):
        self.anpr = ANPREngine(use_gpu=False)
        self.test_db_path = "test_edge_alerts.db"
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.queue = EdgeQueue(db_path=self.test_db_path)
        self.client = TestClient(app)
        ALERTS_DB.clear()

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_01_anpr_regex_and_syntax_correction(self):
        """Test Indian license plate regex matching and character ambiguity correction."""
        # 1. Valid standard Indian plates
        valid_plates = ["DL01AB1234", "MH12DE1433", "HR26DQ5555", "UP32AA0001"]
        for p in valid_plates:
            is_valid, corrected = self.anpr.validate_and_correct_indian_plate(p)
            self.assertTrue(is_valid, f"Plate {p} should be valid")
            self.assertEqual(corrected, p)

        # 2. Bharat (BH) Series plate
        is_valid_bh, corrected_bh = self.anpr.validate_and_correct_indian_plate("22BH1234AA")
        self.assertTrue(is_valid_bh, "22BH1234AA should be valid Bharat series")

        # 3. Ambiguity correction: Letter 'O' in numeric slot vs '0' in alpha slot
        # Input: "DLO1AB1234" -> Has letter 'O' at slot 2 instead of digit '0'
        is_valid_corr, corrected_str = self.anpr.validate_and_correct_indian_plate("DLO1AB1234")
        self.assertTrue(is_valid_corr, "DLO1AB1234 should be auto-corrected to DL01AB1234")
        self.assertEqual(corrected_str, "DL01AB1234")

        # Input: "0L01AB1234" -> Has digit '0' at slot 0 instead of letter 'D'/'O'
        is_valid_corr2, corrected_str2 = self.anpr.validate_and_correct_indian_plate("0L01AB1234")
        self.assertTrue(is_valid_corr2)
        self.assertEqual(corrected_str2[0], "O")

        # 4. Invalid random text rejection
        is_valid_noise, _ = self.anpr.validate_and_correct_indian_plate("RANDOM_TEXT_123")
        self.assertFalse(is_valid_noise, "Random invalid text must be rejected")

    def test_02_anpr_perspective_warp(self):
        """Test 4-point perspective warp for skewed plate rectification."""
        # Create synthetic test image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        # Define a skewed quad
        quad = np.array([[50, 60], [350, 40], [360, 180], [40, 200]], dtype=np.float32)
        warped = self.anpr.warp_perspective(img, quad, target_width=240, target_height=80)
        self.assertEqual(warped.shape, (80, 240, 3), "Warped plate must match target dimensions")

    def test_03_anpr_multi_frame_voting(self):
        """Test Statistical Character Majority Voting across 10-15 frames."""
        track_id = 99
        # Simulate 10 frames with occasional noise: 8 frames read "DL01AB1234", 2 read "DL01A81234" (8 instead of B)
        noisy_reads = [
            "DL01AB1234", "DL01AB1234", "DL01A81234", "DL01AB1234",
            "DL01AB1234", "DL01AB1234", "DL01A81234", "DL01AB1234",
            "DL01AB1234", "DL01AB1234",
        ]
        for plate_str in noisy_reads:
            if track_id not in self.anpr.plate_history:
                self.anpr.plate_history[track_id] = []
            self.anpr.plate_history[track_id].append({"plate_str": plate_str, "confidence": 0.9})

        voted_plate, conf = self.anpr.multi_frame_vote(track_id)
        self.assertEqual(voted_plate, "DL01AB1234", "Multi-frame voting should resolve majority string 'DL01AB1234'")
        self.assertGreaterEqual(conf, 0.85)

    def test_04_edge_queue_store_and_forward_priority(self):
        """Test SQLite Store-and-Forward queue insertion and strict priority ordering."""
        # Enqueue 3 alerts with different priority scores
        med_alert = {
            "alert_id": "ALT-MED-01",
            "track_id": 1,
            "object_class": "person",
            "class_id": 0,
            "priority_score": 40.0,
            "priority_level": "MEDIUM",
            "primary_rule": "ZONE_INTRUSION",
            "world_coords": {"x": 1.0, "y": 2.0},
        }
        crit_alert = {
            "alert_id": "ALT-CRIT-01",
            "track_id": 2,
            "object_class": "vehicle",
            "class_id": 2,
            "priority_score": 95.0,
            "priority_level": "CRITICAL",
            "primary_rule": "TRIPWIRE_INBOUND",
            "world_coords": {"x": -2.0, "y": 9.0},
        }
        high_alert = {
            "alert_id": "ALT-HIGH-01",
            "track_id": 3,
            "object_class": "person",
            "class_id": 0,
            "priority_score": 75.0,
            "priority_level": "HIGH",
            "primary_rule": "CRAWLING_INTRUSION",
            "world_coords": {"x": 0.0, "y": 5.0},
        }

        # Mock image crop for base64 thumbnail
        crop = np.zeros((100, 100, 3), dtype=np.uint8)

        self.queue.enqueue_alert(med_alert, image_crop=crop)
        self.queue.enqueue_alert(crit_alert, image_crop=crop)
        self.queue.enqueue_alert(high_alert, image_crop=crop)

        # Retrieve pending alerts
        pending = self.queue.get_pending_alerts(limit=10)
        self.assertEqual(len(pending), 3)

        # Verify strict priority ordering: CRITICAL (95) -> HIGH (75) -> MEDIUM (40)
        self.assertEqual(pending[0]["alert_id"], "ALT-CRIT-01")
        self.assertEqual(pending[1]["alert_id"], "ALT-HIGH-01")
        self.assertEqual(pending[2]["alert_id"], "ALT-MED-01")

        # Mark first two as synced
        self.queue.mark_as_synced(["ALT-CRIT-01", "ALT-HIGH-01"])
        remaining = self.queue.get_pending_alerts(limit=10)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["alert_id"], "ALT-MED-01")

    def test_05_backend_fastapi_endpoints(self):
        """Test FastAPI Central Command health, ingestion, filtering, and hard-negative loop."""
        # 1. Health check
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "healthy")

        # 2. Ingest Alert via POST /api/v1/alerts
        alert_payload = {
            "alert_id": "ALT-TEST-99",
            "timestamp": "2026-08-29T21:15:00.000Z",
            "camera_id": "CAM-BOP-02",
            "track_id": 99,
            "object_class": "person",
            "class_id": 0,
            "confidence": 0.92,
            "world_coords": {"x": 2.5, "y": 7.0},
            "velocity_mps": 1.2,
            "heading_deg": 180.0,
            "zone": "RED_ZONE",
            "primary_rule": "TRIPWIRE_INBOUND",
            "active_rules": ["TRIPWIRE_INBOUND", "ZONE_INTRUSION"],
            "priority_score": 92.0,
            "priority_level": "CRITICAL",
            "license_plate": None,
            "thumbnail_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
            "synced_status": True,
        }
        ingest_resp = self.client.post("/api/v1/alerts", json=alert_payload)
        self.assertEqual(ingest_resp.status_code, 201)
        self.assertEqual(ingest_resp.json()["alert_id"], "ALT-TEST-99")

        # 3. Query Alerts via GET /api/v1/alerts
        get_resp = self.client.get("/api/v1/alerts?priority=CRITICAL")
        self.assertEqual(get_resp.status_code, 200)
        data = get_resp.json()
        self.assertEqual(data["total_count"], 1)
        self.assertEqual(data["alerts"][0]["alert_id"], "ALT-TEST-99")

        # 4. Hard-Negative Operator Feedback via POST /api/v1/alerts/{id}/feedback
        feedback_payload = {
            "action": "FALSE_ALARM",
            "operator_id": "SSB-OFFICER-07",
            "notes": "Swaying camouflage net triggered tripwire false alarm.",
        }
        fb_resp = self.client.post("/api/v1/alerts/ALT-TEST-99/feedback", json=feedback_payload)
        self.assertEqual(fb_resp.status_code, 200)
        self.assertEqual(fb_resp.json()["action"], "FALSE_ALARM")

        # Verify feedback persisted
        check_resp = self.client.get("/api/v1/alerts?feedback_status=FALSE_ALARM")
        self.assertEqual(check_resp.status_code, 200)
        self.assertEqual(check_resp.json()["total_count"], 1)

        # 5. Stats endpoint
        stats_resp = self.client.get("/api/v1/stats")
        self.assertEqual(stats_resp.status_code, 200)
        stats_data = stats_resp.json()
        self.assertEqual(stats_data["total_alerts"], 1)
        self.assertEqual(stats_data["review_breakdown"]["FALSE_ALARM"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

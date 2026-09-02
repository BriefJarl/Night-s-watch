import os
import sys
import json
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from edge_node.anpr_engine import ANPREngine
from edge_node.face_engine import FaceEngine
from edge_node.rule_engine import RuleEngine
from edge_node.edge_queue import EdgeQueue
from backend.main import AlertPayload, AlertResponse
from backend.genai_copilot import translate_alert_to_text


class TestFaceAndPlateUpgrade(unittest.TestCase):

    def test_01_yolo_plate_detector_init_and_box(self):
        """Verify that ANPREngine loads YOLO plate detector and handles crops safely."""
        anpr = ANPREngine(use_gpu=False)
        self.assertIsNotNone(anpr.plate_detector, "YOLO plate detector should be loaded")

        # Test on dummy vehicle crop
        dummy_veh = np.zeros((200, 300, 3), dtype=np.uint8)
        box = anpr.detect_plate_bbox(dummy_veh)
        # On blank image it should return None safely
        self.assertIsNone(box)

        # Test process_vehicle_frame does not crash
        plate, conf = anpr.process_vehicle_frame(101, dummy_veh)
        self.assertIsInstance(conf, float)

    def test_02_face_engine_matching(self):
        """Verify FaceEngine loads enrolled suspects and correctly matches an enrolled image."""
        face_engine = FaceEngine()
        self.assertIsNotNone(face_engine.detector)
        self.assertIsNotNone(face_engine.recognizer)
        self.assertGreater(len(face_engine.watchlist), 0, "Watchlist should contain enrolled suspects")

        import cv2
        sample_path = "data/watchlist/SUSPECT-T101_Ramesh_Kumar.jpg"
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
            matched, score, bbox = face_engine.detect_and_match(img)
            self.assertEqual(matched, "SUSPECT-T101_Ramesh_Kumar")
            self.assertGreaterEqual(score, 0.40)
            self.assertIsNotNone(bbox)

    def test_03_rule_engine_watchlist_suspect_breach(self):
        """Verify that RuleEngine triggers WATCHLIST_SUSPECT_BREACH and elevates to CRITICAL."""
        rule_engine = RuleEngine()
        track_history = [
            {"world_pos": (2.0, 5.0), "bbox_bottom": (100, 200), "timestamp": 1000.0},
            {"world_pos": (2.1, 5.2), "bbox_bottom": (102, 201), "timestamp": 1001.0},
        ]

        # Normal person in Civilian zone is suppressed (None)
        rule_engine.user_zone_mode = "Civilian zone"
        normal_alert = rule_engine.evaluate_track(
            track_id=1,
            track_history=track_history,
            bbox_width=50,
            bbox_height=120,
            class_id=0,
            confidence=0.85,
        )
        self.assertIsNone(normal_alert, "Normal civilian person should be suppressed in Civilian zone")

        # But a watchlisted suspect in Civilian zone triggers immediate CRITICAL alert
        suspect_alert = rule_engine.evaluate_track(
            track_id=2,
            track_history=track_history,
            bbox_width=50,
            bbox_height=120,
            class_id=0,
            confidence=0.85,
            suspect_id="SUSPECT-T101_Ramesh_Kumar",
            face_confidence=0.92,
        )
        self.assertIsNotNone(suspect_alert)
        self.assertEqual(suspect_alert["primary_rule"], "WATCHLIST_SUSPECT_BREACH")
        self.assertEqual(suspect_alert["priority_level"], "CRITICAL")
        self.assertEqual(suspect_alert["suspect_id"], "SUSPECT-T101_Ramesh_Kumar")
        self.assertEqual(suspect_alert["face_confidence"], 0.92)
        self.assertTrue(suspect_alert["is_threat"])

    def test_04_edge_queue_suspect_persistence(self):
        """Verify SQLite EdgeQueue correctly serializes and retrieves suspect details."""
        db_path = "test_suspect_edge_queue.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        queue = EdgeQueue(db_path=db_path)
        payload = {
            "alert_id": "ALT-TEST-SUSPECT-01",
            "timestamp": "2026-09-02T12:00:00.000000Z",
            "camera_id": "CAM-BOP-01",
            "track_id": 99,
            "object_class": "person",
            "class_id": 0,
            "confidence": 0.95,
            "world_coords": {"x": 3.5, "y": 6.2},
            "velocity_mps": 1.2,
            "heading_deg": 90.0,
            "zone": "RED_ZONE",
            "primary_rule": "WATCHLIST_SUSPECT_BREACH",
            "active_rules": ["WATCHLIST_SUSPECT_BREACH"],
            "priority_score": 98.0,
            "priority_level": "CRITICAL",
            "license_plate": None,
            "suspect_id": "SUSPECT-T101_Ramesh_Kumar",
            "face_confidence": 0.88,
        }

        enqueued = queue.enqueue_alert(payload)
        self.assertTrue(enqueued)

        pending = queue.get_pending_alerts(limit=10)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["suspect_id"], "SUSPECT-T101_Ramesh_Kumar")
        self.assertEqual(pending[0]["face_confidence"], 0.88)
        self.assertEqual(pending[0]["priority_level"], "CRITICAL")

        if os.path.exists(db_path):
            os.remove(db_path)

    def test_05_backend_payload_and_genai_rag_description(self):
        """Verify Pydantic schemas and semantic text generation for RAG."""
        alert_dict = {
            "alert_id": "ALT-TEST-RAG-01",
            "timestamp": "2026-09-02T12:00:00.000000Z",
            "camera_id": "CAM-BOP-01",
            "track_id": 12,
            "object_class": "person",
            "class_id": 0,
            "confidence": 0.91,
            "world_coords": {"x": 4.1, "y": 8.2},
            "velocity_mps": 1.5,
            "heading_deg": 180.0,
            "zone": "RED_ZONE",
            "primary_rule": "WATCHLIST_SUSPECT_BREACH",
            "active_rules": ["WATCHLIST_SUSPECT_BREACH"],
            "priority_score": 98.0,
            "priority_level": "CRITICAL",
            "license_plate": None,
            "suspect_id": "SUSPECT-T101_Ramesh_Kumar",
            "face_confidence": 0.95,
            "thumbnail_b64": None,
            "is_threat": True,
        }

        payload = AlertPayload(**alert_dict)
        self.assertEqual(payload.suspect_id, "SUSPECT-T101_Ramesh_Kumar")

        response = AlertResponse(**alert_dict)
        self.assertEqual(response.suspect_id, "SUSPECT-T101_Ramesh_Kumar")

        rag_text = translate_alert_to_text(alert_dict)
        self.assertIn("CRITICAL BIOMETRIC HIT", rag_text)
        self.assertIn("SUSPECT-T101_Ramesh_Kumar", rag_text)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""
SignSpeak AI - Unit & Integration Tests for Stream Recognition Adapter & WebSocket
==================================================================================
Tests cover:
  1. Stream adapter lifecycle (process_frame, reset, close, diagnostics).
  2. WebSocket connection and initial handshake (init_ack).
  3. WebSocket ping/pong and control messages (clear / reset / status).
  4. WebSocket frame processing (JSON base64 and binary JPEG).
  5. WebSocket error handling for malformed payloads and invalid images.
  6. Calibration and thresholds under web stream parameters.
"""

import base64
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app, get_recognition_engine
from src.recognition.stream_adapter import LiveWebcamRecognitionSession, analyze_frame_posture


class MockLM:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.9):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


class TestStreamRecognition(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.engine = get_recognition_engine()

    def _create_synthetic_frame(self, width: int = 320, height: int = 240) -> np.ndarray:
        """Generates a dummy 3-channel frame for testing without physical webcam."""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _frame_to_base64_jpeg(self, frame: np.ndarray) -> str:
        """Encodes a BGR numpy array to base64 JPEG data URL."""
        success, buffer = cv2.imencode(".jpg", frame)
        self.assertTrue(success, "Failed to encode frame to JPEG")
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _frame_to_binary_jpeg(self, frame: np.ndarray) -> bytes:
        """Encodes a BGR numpy array to raw JPEG bytes."""
        success, buffer = cv2.imencode(".jpg", frame)
        self.assertTrue(success, "Failed to encode frame to JPEG")
        return buffer.tobytes()

    # ── 1. Stream Adapter Direct Tests ────────────────────────────────────────

    def test_adapter_process_frame_and_reset(self):
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            frame = self._create_synthetic_frame()
            result = session.process_frame(frame, timestamp=100.0)

            self.assertEqual(result["type"], "frame_result")
            self.assertEqual(result["frame_count"], 1)
            self.assertEqual(result["state"], "IDLE")
            self.assertFalse(result["hands_detected"])
            self.assertIn("landmarks", result)
            self.assertIn("diagnostics", result)
            self.assertEqual(result["diagnostics"]["frame_count"], 1)
            self.assertIn("current_glosses", result)
            self.assertEqual(result["current_glosses"], [])

            session.reset()
            self.assertEqual(session.seg_state, "IDLE")
            self.assertEqual(session.gloss_buffer.current_sequence, [])
        finally:
            session.close()

    def test_adapter_invalid_frame_raises(self):
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            with self.assertRaises(ValueError):
                session.process_frame(None)
            with self.assertRaises(ValueError):
                session.process_frame(np.array([]))
        finally:
            session.close()

    # ── 2. WebSocket Connection & Controls ────────────────────────────────────

    def test_websocket_connect_receives_init_ack(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            init_ack = ws.receive_json()
            self.assertEqual(init_ack.get("type"), "init_ack")
            self.assertEqual(init_ack.get("status"), "ready")
            self.assertTrue(init_ack.get("gallery_loaded"))
            self.assertEqual(init_ack.get("enrolled_classes"), 25)

    def test_websocket_ping_pong(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            self.assertEqual(resp, {"type": "pong"})

    def test_websocket_status_query(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_json({"type": "status"})
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "status_response")
            self.assertTrue(resp.get("gallery_loaded"))
            self.assertEqual(resp.get("enrolled_classes"), 25)

    def test_websocket_clear_command(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_json({"type": "clear"})
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "reset_ack")
            self.assertEqual(resp.get("current_glosses"), [])

    def test_websocket_invalid_json(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_text("not-a-json-object")
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "error")
            self.assertIn("Invalid JSON", resp.get("message", ""))

    def test_websocket_missing_frame_data(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_json({"type": "frame"})
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "error")
            self.assertIn("Missing frame data", resp.get("message", ""))

    def test_websocket_invalid_base64_data(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_json({"type": "frame", "data": "not-valid-base64-image-bytes!!!"})
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "error")

    # ── 3. WebSocket Frame Processing ─────────────────────────────────────────

    def test_websocket_json_frame_processing(self):
        frame = self._create_synthetic_frame()
        b64_data = self._frame_to_base64_jpeg(frame)

        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_json({
                "type": "frame",
                "data": b64_data
            })
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "frame_result")
            self.assertIn("state", resp)
            self.assertIn("hands_detected", resp)
            self.assertIn("landmarks", resp)
            self.assertIn("diagnostics", resp)
            self.assertIn("current_glosses", resp)

    def test_websocket_binary_frame_processing(self):
        frame = self._create_synthetic_frame()
        binary_data = self._frame_to_binary_jpeg(frame)

        with self.client.websocket_connect("/ws/recognize") as ws:
            _ = ws.receive_json()  # init_ack
            ws.send_bytes(binary_data)
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "frame_result")
            self.assertIn("state", resp)
            self.assertIn("landmarks", resp)

    # ── 4. Multi-Condition Gesture Boundaries & Sentence Pipeline Tests ───────

    def test_analyze_frame_posture_motion_and_rest_zone(self):
        """Verifies wrist motion displacement and neutral rest zone detection."""
        # 1. Wrists in active signing zone (chest level: shoulder=0.35, wrist=0.42)
        pose_active = type("PoseMock", (), {
            "landmark": [MockLM(x=0.5, y=0.5, z=0.0) for _ in range(33)]
        })()
        pose_active.landmark[11].y = 0.35  # left shoulder
        pose_active.landmark[12].y = 0.35  # right shoulder
        pose_active.landmark[15].y = 0.42  # left wrist
        pose_active.landmark[16].y = 0.42  # right wrist

        lw, rw, motion, in_rest = analyze_frame_posture(
            pose_landmarks=pose_active,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
            prev_left_wrist=(0.5, 0.42),
            prev_right_wrist=(0.5, 0.42),
        )
        self.assertFalse(in_rest, "Wrists in front of chest must not be classified as rest zone")
        self.assertAlmostEqual(motion, 0.0, places=4)

        # 2. Moving wrists (active displacement)
        _, _, motion_moving, _ = analyze_frame_posture(
            pose_landmarks=pose_active,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
            prev_left_wrist=(0.4, 0.42),  # 0.10 horizontal displacement
            prev_right_wrist=(0.5, 0.42),
        )
        self.assertGreater(motion_moving, 0.08, "Moving wrist must produce high motion energy")

        # 3. Wrists in neutral rest zone (hands lowered below chest: y = 0.85)
        pose_resting = type("PoseMock", (), {
            "landmark": [MockLM(x=0.5, y=0.5, z=0.0) for _ in range(33)]
        })()
        pose_resting.landmark[11].y = 0.35
        pose_resting.landmark[12].y = 0.35
        pose_resting.landmark[15].y = 0.85
        pose_resting.landmark[16].y = 0.85

        _, _, _, in_rest_down = analyze_frame_posture(
            pose_landmarks=pose_resting,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )
        self.assertTrue(in_rest_down, "Wrists lowered below chest must be detected as neutral rest zone")

    def test_three_consecutive_signs_accumulate_into_sentence(self):
        """
        Consecutive signs (I_Me -> Want_Desire_Wish_For -> Water) performed in sequence
        must each be recognized separately, stored in order, and translated as a complete sentence.
        """
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            min_gesture_len=4,
            max_gesture_len=20,
            cooldown_threshold=1,
            stillness_threshold=0.04,
            pause_threshold=2,
        )
        try:
            # Helper to simulate a gesture stroke with active motion, followed by a pause hold
            def simulate_sign(label: str, base_time: float):
                # Mock engine matching for this gesture
                session.engine.recognize_features = lambda **kwargs: {
                    "candidates": [{"label": label, "similarity": 0.92, "distance": 1.0}]
                }

                # Onset frames (hands present, moving in signing zone)
                for i in range(2):
                    t = base_time + i * 0.08
                    mock_res = type("HolisticMock", (), {
                        "pose_landmarks": type("Pose", (), {
                            "landmark": [MockLM(x=0.5, y=0.4, z=0.0) for _ in range(33)]
                        })(),
                        "face_landmarks": None,
                        "left_hand_landmarks": None,
                        "right_hand_landmarks": type("Hand", (), {
                            "landmark": [MockLM(x=0.4 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                        })(),
                    })()
                    session._holistic.process = lambda rgb: mock_res
                    frame = self._create_synthetic_frame()
                    session.process_frame(frame, timestamp=t)

                self.assertEqual(session.seg_state, "ACTIVE")

                # Active signing frames (2 moving frames)
                for i in range(2, 4):
                    t = base_time + i * 0.08
                    mock_res = type("HolisticMock", (), {
                        "pose_landmarks": type("Pose", (), {
                            "landmark": [MockLM(x=0.5, y=0.4, z=0.0) for _ in range(33)]
                        })(),
                        "face_landmarks": None,
                        "left_hand_landmarks": None,
                        "right_hand_landmarks": type("Hand", (), {
                            "landmark": [MockLM(x=0.5 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                        })(),
                    })()
                    session._holistic.process = lambda rgb: mock_res
                    session.process_frame(frame, timestamp=t)

                # Pause / stillness frames (hands hold stroke still at apex: dx < 0.04)
                for i in range(4, 7):
                    t = base_time + i * 0.08
                    mock_res = type("HolisticMock", (), {
                        "pose_landmarks": type("Pose", (), {
                            "landmark": [MockLM(x=0.5, y=0.4, z=0.0) for _ in range(33)]
                        })(),
                        "face_landmarks": None,
                        "left_hand_landmarks": None,
                        "right_hand_landmarks": type("Hand", (), {
                            "landmark": [MockLM(x=0.70, y=0.4, z=0.0) for _ in range(21)]
                        })(),
                    })()
                    session._holistic.process = lambda rgb: mock_res
                    session.process_frame(frame, timestamp=t)

                # Cooldown drain frame (evaluates DTW)
                t_drain = base_time + 8 * 0.08
                mock_idle = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None,
                    "left_hand_landmarks": None, "right_hand_landmarks": None
                })()
                session._holistic.process = lambda rgb: mock_idle
                res = session.process_frame(frame, timestamp=t_drain)
                return res

            # Gesture 1: "I_Me"
            simulate_sign("I_Me", base_time=1.0)
            self.assertEqual(session.displayed_sequence, ["I_Me"])
            self.assertEqual(session.seg_state, "IDLE")

            # Gesture 2: "Want_Desire_Wish_For"
            simulate_sign("Want_Desire_Wish_For", base_time=3.0)
            self.assertEqual(session.displayed_sequence, ["I_Me", "Want_Desire_Wish_For"])

            # Gesture 3: "Water"
            simulate_sign("Water", base_time=5.0)
            self.assertEqual(session.displayed_sequence, ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(session.displayed_translation, "I want water.")

            # Idle finalization after pause (at t = 7.5 >= 5.64 + 1.5s timeout)
            frame = self._create_synthetic_frame()
            mock_rest = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None,
                "left_hand_landmarks": None, "right_hand_landmarks": None
            })()
            session._holistic.process = lambda rgb: mock_rest
            res_idle = session.process_frame(frame, timestamp=7.5)

            self.assertTrue(session.is_finalized)
            self.assertEqual(res_idle["current_glosses"], ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(res_idle["current_translation"], "I want water.")
            self.assertEqual(res_idle["finalized_sequence"], ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(res_idle["finalized_translation"], "I want water.")

            # Subsequent idle frame preserves sequence and does NOT re-finalize
            res_idle_2 = session.process_frame(frame, timestamp=8.5)
            self.assertTrue(session.is_finalized)
            self.assertEqual(res_idle_2["current_glosses"], ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(res_idle_2["finalized_translation"], "I want water.")

        finally:
            session.close()

    def test_consecutive_identical_signs_both_recorded(self):
        """Verifies that two consecutive distinct gestures of the same sign are both recorded."""
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            min_gesture_len=4,
            max_gesture_len=20,
            cooldown_threshold=1,
            stillness_threshold=0.04,
            pause_threshold=2,
        )
        try:
            session.engine.recognize_features = lambda **kwargs: {
                "candidates": [{"label": "One", "similarity": 0.90, "distance": 1.0}]
            }

            def run_gesture(base_time):
                # Onset (2 frames)
                for i in range(2):
                    t = base_time + i * 0.08
                    mock_res = type("HolisticMock", (), {
                        "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                        "right_hand_landmarks": type("Hand", (), {
                            "landmark": [MockLM(x=0.4 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                        })(),
                    })()
                    session._holistic.process = lambda rgb: mock_res
                    session.process_frame(self._create_synthetic_frame(), timestamp=t)

                # Active motion (2 frames)
                for i in range(2, 4):
                    t = base_time + i * 0.08
                    mock_res = type("HolisticMock", (), {
                        "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                        "right_hand_landmarks": type("Hand", (), {
                            "landmark": [MockLM(x=0.5 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                        })(),
                    })()
                    session._holistic.process = lambda rgb: mock_res
                    session.process_frame(self._create_synthetic_frame(), timestamp=t)

                # Stillness pause hold (3 frames: dx = 0 < 0.04)
                for i in range(4, 7):
                    t = base_time + i * 0.08
                    mock_res = type("HolisticMock", (), {
                        "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                        "right_hand_landmarks": type("Hand", (), {
                            "landmark": [MockLM(x=0.70, y=0.4, z=0.0) for _ in range(21)]
                        })(),
                    })()
                    session._holistic.process = lambda rgb: mock_res
                    session.process_frame(self._create_synthetic_frame(), timestamp=t)

                # Cooldown drain frame (DTW matching)
                mock_idle = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None,
                    "left_hand_landmarks": None, "right_hand_landmarks": None
                })()
                session._holistic.process = lambda rgb: mock_idle
                session.process_frame(self._create_synthetic_frame(), timestamp=base_time + 8 * 0.08)

            # Gesture 1: "One"
            run_gesture(base_time=1.0)
            self.assertEqual(session.displayed_sequence, ["One"])

            # Gesture 2: "One" (repetition after boundary)
            run_gesture(base_time=3.0)
            self.assertEqual(session.displayed_sequence, ["One", "One"])

        finally:
            session.close()

    def test_new_sentence_starts_cleanly_after_finalization(self):
        """When a new gesture completes after finalization, a clean new sentence begins."""
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            min_gesture_len=4,
            max_gesture_len=20,
            cooldown_threshold=1,
            stillness_threshold=0.04,
            pause_threshold=2,
        )
        try:
            # Seed finalized sentence
            session.displayed_sequence = ["I_Me", "Water"]
            session.displayed_translation = "I want water."
            session.is_finalized = True

            # Start and complete new sign "Food"
            session.engine.recognize_features = lambda **kwargs: {
                "candidates": [{"label": "Food", "similarity": 0.95, "distance": 1.0}]
            }

            # Onset (2 frames)
            for i in range(2):
                t = 10.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.4 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                # During onset, previous sequence remains visible
                session.process_frame(self._create_synthetic_frame(), timestamp=t)
                self.assertEqual(session.displayed_sequence, ["I_Me", "Water"])

            # Active motion (2 frames)
            for i in range(2, 4):
                t = 10.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.5 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=t)

            # Stillness pause hold (3 frames: dx = 0 < 0.04)
            for i in range(4, 7):
                t = 10.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.70, y=0.4, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=t)

            # Drain to trigger DTW evaluation
            mock_idle = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None,
                "left_hand_landmarks": None, "right_hand_landmarks": None
            })()
            session._holistic.process = lambda rgb: mock_idle
            res = session.process_frame(self._create_synthetic_frame(), timestamp=10.0 + 8 * 0.08)

            # Fresh sentence started
            self.assertFalse(session.is_finalized)
            self.assertEqual(session.displayed_sequence, ["Food"])
            self.assertEqual(res["current_glosses"], ["Food"])

        finally:
            session.close()

    def test_clear_resets_sequence_and_translation(self):
        """Reset clears sequence and translation while keeping session alive in IDLE."""
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            session.displayed_sequence = ["Help", "Food"]
            session.displayed_translation = "Please help with food."
            session.is_finalized = True
            session.seg_state = "ACTIVE"

            session.reset()
            self.assertEqual(session.displayed_sequence, [])
            self.assertIsNone(session.displayed_translation)
            self.assertFalse(session.is_finalized)
            self.assertEqual(session.seg_state, "IDLE")
            self.assertEqual(session.gloss_buffer.current_sequence, [])
        finally:
            session.close()

    def test_valid_gesture_below_chest_not_classified_as_rest(self):
        """
        Signs performed below chest level (e.g. wrist_y > 0.80 or shoulder_y + 0.32)
        must NOT be classified as resting when there is active hand/wrist movement.
        """
        pose_low = type("PoseMock", (), {
            "landmark": [MockLM(x=0.5, y=0.5, z=0.0) for _ in range(33)]
        })()
        pose_low.landmark[11].y = 0.35  # left shoulder
        pose_low.landmark[12].y = 0.35  # right shoulder
        pose_low.landmark[15].y = 0.85  # left wrist below chest
        pose_low.landmark[16].y = 0.85  # right wrist below chest

        # Wrists moving horizontally: left wrist moved from 0.40 to 0.50 (dx = 0.10)
        posture = analyze_frame_posture(
            pose_landmarks=pose_low,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
            prev_left_wrist=(0.40, 0.85),
            prev_right_wrist=(0.50, 0.85),
            stillness_threshold=0.04,
        )

        self.assertTrue(posture.in_low_zone, "Wrists at y=0.85 must be flagged as in low zone")
        self.assertGreater(posture.motion_energy, 0.04, "Moving wrist must produce active motion")
        self.assertFalse(posture.is_resting, "Active moving hands in low zone must NOT be marked resting")

        # Verify 4-tuple unpack backward compatibility: (lw, rw, motion, is_resting)
        lw, rw, motion, in_rest = posture
        self.assertFalse(in_rest, "4th tuple element must be False for moving hands")

    def test_moving_hand_in_lower_frame_continues_gesture(self):
        """
        A valid gesture performed below chest level must start and run its full duration
        without being prematurely cut off by resting zone detection.
        """
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            min_gesture_len=4,
            max_gesture_len=20,
            cooldown_threshold=1,
            stillness_threshold=0.04,
            pause_threshold=3,
            rest_threshold=3,
        )
        try:
            # Onset: 2 frames of active motion below chest (y = 0.85)
            for i in range(2):
                t = 1.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": type("Pose", (), {
                        "landmark": [MockLM(x=0.5, y=0.4, z=0.0) for _ in range(33)]
                    })(),
                    "face_landmarks": None,
                    "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.4 + i * 0.05, y=0.85, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=t)

            self.assertEqual(session.seg_state, "ACTIVE", "Active motion in low zone must trigger onset")

            # Continue moving for 8 more frames (total 10 frames > min_len=4 and > rest_threshold=3)
            for i in range(2, 10):
                t = 1.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": type("Pose", (), {
                        "landmark": [MockLM(x=0.5, y=0.4, z=0.0) for _ in range(33)]
                    })(),
                    "face_landmarks": None,
                    "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.4 + i * 0.05, y=0.85, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                res = session.process_frame(self._create_synthetic_frame(), timestamp=t)
                self.assertEqual(session.seg_state, "ACTIVE", f"Frame {i}: gesture must not be terminated while moving")
                self.assertFalse(res["in_rest_zone"], "Moving hand in low frame must not be reported as in rest zone")

        finally:
            session.close()

    def test_stationary_hand_in_lower_frame_triggers_idle_finalization(self):
        """
        When hands rest motionless in the lower frame after a gesture,
        the idle timer must finalize the sentence after timeout even though hands are visible.
        """
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            idle_timeout_seconds=1.0,
            onset_threshold=2,
            min_gesture_len=4,
            max_gesture_len=20,
            cooldown_threshold=1,
            stillness_threshold=0.04,
            pause_threshold=2,
        )
        try:
            # Seed an active sequence in session
            session.gloss_buffer.add_gloss("Water", confidence=0.90, timestamp=1.0)
            session.displayed_sequence = ["Water"]
            session.displayed_translation = "Water."
            session.seg_state = "IDLE"

            # Signer rests hands in the lower frame (y = 0.85, stationary dx = 0)
            # Send frames for 1.2 seconds (> idle_timeout_seconds = 1.0)
            for i in range(15):
                t = 1.0 + i * 0.08  # up to t = 2.12
                mock_rest = type("HolisticMock", (), {
                    "pose_landmarks": type("Pose", (), {
                        "landmark": [MockLM(x=0.5, y=0.85, z=0.0) for _ in range(33)]
                    })(),
                    "face_landmarks": None,
                    "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.50, y=0.85, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_rest
                res = session.process_frame(self._create_synthetic_frame(), timestamp=t)

            self.assertTrue(session.is_finalized, "Sentence must be finalized even with hands resting in low frame")
            self.assertEqual(res["finalized_sequence"], ["Water"])
            self.assertEqual(res["finalized_translation"], "Water.")
            self.assertTrue(res["in_rest_zone"], "Stationary low hands must be reported in rest zone")

        finally:
            session.close()

    def test_brief_pause_during_gesture_does_not_terminate_prematurely(self):
        """
        A 1-frame micro-pause within an active gesture must NOT cause premature termination
        thanks to temporal smoothing and hysteresis.
        """
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            min_gesture_len=4,
            max_gesture_len=20,
            cooldown_threshold=1,
            stillness_threshold=0.04,
            pause_threshold=3,
        )
        try:
            # Onset (2 frames moving)
            for i in range(2):
                t = 1.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.4 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=t)

            self.assertEqual(session.seg_state, "ACTIVE")

            # Active frame 1 (i=2, moving)
            mock_mov = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                "right_hand_landmarks": type("Hand", (), {
                    "landmark": [MockLM(x=0.55, y=0.4, z=0.0) for _ in range(21)]
                })(),
            })()
            session._holistic.process = lambda rgb: mock_mov
            session.process_frame(self._create_synthetic_frame(), timestamp=1.16)

            # Active frame 2 (i=3, moving)
            mock_mov2 = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                "right_hand_landmarks": type("Hand", (), {
                    "landmark": [MockLM(x=0.62, y=0.4, z=0.0) for _ in range(21)]
                })(),
            })()
            session._holistic.process = lambda rgb: mock_mov2
            session.process_frame(self._create_synthetic_frame(), timestamp=1.24)

            # Micro-pause: 1 frame still (i=4, x=0.62, dx=0.0)
            session.process_frame(self._create_synthetic_frame(), timestamp=1.32)
            self.assertEqual(session.seg_state, "ACTIVE", "1-frame micro-pause must not terminate gesture")

            # Active motion resumes immediately (i=5, x=0.70, dx=0.08)
            mock_mov3 = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                "right_hand_landmarks": type("Hand", (), {
                    "landmark": [MockLM(x=0.70, y=0.4, z=0.0) for _ in range(21)]
                })(),
            })()
            session._holistic.process = lambda rgb: mock_mov3
            session.process_frame(self._create_synthetic_frame(), timestamp=1.40)
            self.assertEqual(session.seg_state, "ACTIVE", "Gesture must remain ACTIVE when motion resumes")
            self.assertEqual(session.pause_counter, 0, "Pause counter must be reset by hysteresis")

        finally:
            session.close()

    def test_max_gesture_len_acts_as_safety_cutoff(self):
        """
        Continuous motion beyond max_gesture_len triggers safety termination.
        """
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            min_gesture_len=4,
            max_gesture_len=8,  # small max cap for test
            cooldown_threshold=1,
            stillness_threshold=0.04,
            pause_threshold=5,
        )
        try:
            # Onset (2 frames)
            for i in range(2):
                t = 1.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.1 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=t)

            self.assertEqual(session.seg_state, "ACTIVE")

            # Keep moving continuously for exactly 8 frames (reaching max_gesture_len=8)
            for i in range(2, 10):
                t = 1.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.1 + i * 0.05, y=0.4, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=t)

            # Reached max_gesture_len -> transitioned to COOLDOWN
            self.assertEqual(session.seg_state, "COOLDOWN", "Reaching max_gesture_len must trigger cooldown")

            # Next frame drains cooldown and returns to IDLE
            session.process_frame(self._create_synthetic_frame(), timestamp=2.0)
            self.assertEqual(session.seg_state, "IDLE")

        finally:
            session.close()

    def test_posture_stationary_hand_in_low_zone_is_resting(self):
        """Stationary hands in low zone (dx=0.0 < 0.02) must evaluate to is_resting=True."""
        pose_low = type("PoseMock", (), {
            "landmark": [MockLM(x=0.5, y=0.5, z=0.0) for _ in range(33)]
        })()
        pose_low.landmark[11].y = 0.35  # left shoulder
        pose_low.landmark[12].y = 0.35  # right shoulder
        pose_low.landmark[15].y = 0.85  # left wrist below chest
        pose_low.landmark[16].y = 0.85  # right wrist below chest

        posture = analyze_frame_posture(
            pose_landmarks=pose_low,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
            prev_left_wrist=(0.50, 0.85),
            prev_right_wrist=(0.50, 0.85),
            stillness_threshold=0.02,
        )
        self.assertTrue(posture.in_low_zone)
        self.assertAlmostEqual(posture.motion_energy, 0.0)
        self.assertTrue(posture.is_resting)

    def test_moving_hand_triggers_onset(self):
        """Moving hand below chest level triggers transition from IDLE to ACTIVE."""
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            stillness_threshold=0.02,
        )
        try:
            self.assertEqual(session.seg_state, "IDLE")
            for i in range(2):
                t = 1.0 + i * 0.08
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": type("Pose", (), {
                        "landmark": [MockLM(x=0.5, y=0.4, z=0.0) for _ in range(33)]
                    })(),
                    "face_landmarks": None,
                    "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.4 + i * 0.05, y=0.85, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=t)
            self.assertEqual(session.seg_state, "ACTIVE")
        finally:
            session.close()

    def test_pause_during_gesture_triggers_stroke_completion_and_dtw(self):
        """Active motion followed by pause triggers stroke completion, COOLDOWN, and DTW."""
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=2,
            min_gesture_len=4,
            cooldown_threshold=1,
            stillness_threshold=0.02,
            pause_threshold=2,
        )
        try:
            session.engine.recognize_features = lambda **kwargs: {
                "candidates": [{"label": "Water", "similarity": 0.91, "distance": 1.0}]
            }
            # Onset (2 frames moving)
            for i in range(2):
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.4 + i * 0.05, y=0.5, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=1.0 + i * 0.08)
            self.assertEqual(session.seg_state, "ACTIVE")

            # Active motion (2 frames)
            for i in range(2, 4):
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.5 + i * 0.05, y=0.5, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=1.0 + i * 0.08)

            # Pause / Stillness hold (frames held still at x=0.70)
            for i in range(4, 7):
                mock_res = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.70, y=0.5, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_res
                session.process_frame(self._create_synthetic_frame(), timestamp=1.0 + i * 0.08)

            self.assertEqual(session.seg_state, "COOLDOWN")

            # Cooldown drain frame triggers DTW evaluation
            mock_idle = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None,
                "left_hand_landmarks": None, "right_hand_landmarks": None
            })()
            session._holistic.process = lambda rgb: mock_idle
            res = session.process_frame(self._create_synthetic_frame(), timestamp=1.6)

            self.assertEqual(session.seg_state, "IDLE")
            self.assertTrue(session.reached_matcher)
            self.assertIsNone(session.last_rejection_reason)
            self.assertEqual(session.displayed_sequence, ["Water"])
        finally:
            session.close()

    def test_hand_source_tracking(self):
        """Verifies correct tracking of hand detection source."""
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            # 1. No hands or pose -> "none"
            session._holistic.process = lambda rgb: None
            res1 = session.process_frame(self._create_synthetic_frame(), timestamp=1.0)
            self.assertEqual(res1["hand_source"], "none")
            self.assertFalse(res1["hands_detected"])

            # 2. Hand landmarks present -> "hand_landmarks"
            mock_hand = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                "right_hand_landmarks": type("Hand", (), {
                    "landmark": [MockLM(x=0.5, y=0.5, z=0.0) for _ in range(21)]
                })(),
            })()
            session._holistic.process = lambda rgb: mock_hand
            res2 = session.process_frame(self._create_synthetic_frame(), timestamp=1.1)
            self.assertEqual(res2["hand_source"], "hand_landmarks")
            self.assertTrue(res2["hands_detected"])

            # 3. Pose wrists only, moving -> "pose_wrists_only"
            pose_moving = type("PoseMock", (), {
                "landmark": [MockLM(x=0.5, y=0.5, z=0.0) for _ in range(33)]
            })()
            pose_moving.landmark[15].x = 0.3  # left wrist moved
            pose_moving.landmark[15].y = 0.6
            pose_moving.landmark[15].visibility = 0.9
            pose_moving.landmark[16].visibility = 0.9
            mock_pose = type("HolisticMock", (), {
                "pose_landmarks": pose_moving,
                "face_landmarks": None,
                "left_hand_landmarks": None,
                "right_hand_landmarks": None,
            })()
            session._holistic.process = lambda rgb: mock_pose
            res3 = session.process_frame(self._create_synthetic_frame(), timestamp=1.2)
            self.assertEqual(res3["hand_source"], "pose_wrists_only")
        finally:
            session.close()

    def test_rejection_reason_short_duration(self):
        """A gesture ending with fewer than min_gesture_len frames is rejected as short_duration."""
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            onset_threshold=1,
            min_gesture_len=6,
            cooldown_threshold=1,
        )
        try:
            # Onset (1 frame)
            mock_hand = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                "right_hand_landmarks": type("Hand", (), {
                    "landmark": [MockLM(x=0.5, y=0.5, z=0.0) for _ in range(21)]
                })(),
            })()
            session._holistic.process = lambda rgb: mock_hand
            session.process_frame(self._create_synthetic_frame(), timestamp=1.0)
            self.assertEqual(session.seg_state, "ACTIVE")

            # Cut off immediately: hands dropped (3 frames of offset)
            mock_idle = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None,
                "left_hand_landmarks": None, "right_hand_landmarks": None
            })()
            session._holistic.process = lambda rgb: mock_idle
            session.process_frame(self._create_synthetic_frame(), timestamp=1.08)
            session.process_frame(self._create_synthetic_frame(), timestamp=1.16)
            session.process_frame(self._create_synthetic_frame(), timestamp=1.24)

            # Cooldown drain
            res = session.process_frame(self._create_synthetic_frame(), timestamp=1.32)
            self.assertFalse(session.reached_matcher)
            self.assertEqual(session.last_rejection_reason, "short_duration")
            self.assertEqual(res["rejection_reason"], "short_duration")
        finally:
            session.close()

    def test_rejection_reason_low_confidence(self):
        """When DTW similarity is below min_confidence, rejection_reason is low_confidence."""
        session = LiveWebcamRecognitionSession(
            engine=self.engine,
            min_confidence=0.75,
            onset_threshold=2,
            min_gesture_len=4,
            cooldown_threshold=1,
            stillness_threshold=0.02,
            pause_threshold=2,
        )
        try:
            # Mock DTW candidate with similarity 0.45 (< 0.75)
            session.engine.recognize_features = lambda **kwargs: {
                "candidates": [{"label": "Help", "similarity": 0.45, "distance": 2.5}]
            }

            # Onset (2 frames)
            for i in range(2):
                mock_hand = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.4 + i * 0.05, y=0.5, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_hand
                session.process_frame(self._create_synthetic_frame(), timestamp=1.0 + i * 0.08)

            # Active (2 frames)
            for i in range(2, 4):
                mock_hand = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.5 + i * 0.05, y=0.5, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_hand
                session.process_frame(self._create_synthetic_frame(), timestamp=1.0 + i * 0.08)

            # Pause / Stillness hold (frames held still at x=0.70)
            for i in range(4, 7):
                mock_hand = type("HolisticMock", (), {
                    "pose_landmarks": None, "face_landmarks": None, "left_hand_landmarks": None,
                    "right_hand_landmarks": type("Hand", (), {
                        "landmark": [MockLM(x=0.70, y=0.5, z=0.0) for _ in range(21)]
                    })(),
                })()
                session._holistic.process = lambda rgb: mock_hand
                session.process_frame(self._create_synthetic_frame(), timestamp=1.0 + i * 0.08)

            # Cooldown drain
            mock_idle = type("HolisticMock", (), {
                "pose_landmarks": None, "face_landmarks": None,
                "left_hand_landmarks": None, "right_hand_landmarks": None
            })()
            session._holistic.process = lambda rgb: mock_idle
            res = session.process_frame(self._create_synthetic_frame(), timestamp=1.6)

            self.assertTrue(session.reached_matcher)
            self.assertEqual(session.last_rejection_reason, "low_confidence")
            self.assertEqual(res["rejection_reason"], "low_confidence")
            self.assertEqual(session.displayed_sequence, [])
        finally:
            session.close()

    def test_pause_and_resume_session(self):
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            self.assertFalse(session.is_paused)
            # Pause session
            res_pause = session.set_paused(True)
            self.assertTrue(res_pause)
            self.assertTrue(session.is_paused)

            frame = self._create_synthetic_frame()
            result = session.process_frame(frame, timestamp=10.0)
            self.assertEqual(result["state"], "PAUSED")
            self.assertTrue(result["is_paused"])
            self.assertTrue(result["diagnostics"]["is_paused"])

            # Resume session
            res_resume = session.set_paused(False)
            self.assertFalse(res_resume)
            self.assertFalse(session.is_paused)

            result2 = session.process_frame(frame, timestamp=10.1)
            self.assertEqual(result2["state"], "IDLE")
            self.assertFalse(result2["is_paused"])
        finally:
            session.close()

    def test_pop_last_gloss_undo(self):
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            # Empty pop
            res_empty = session.pop_last_gloss()
            self.assertIsNone(res_empty["removed_gloss"])
            self.assertEqual(res_empty["current_glosses"], [])

            # Add glosses to session buffer
            session.gloss_buffer.add_gloss("I_Me", confidence=0.9, timestamp=1.0)
            session.gloss_buffer.add_gloss("Want_Desire_Wish_For", confidence=0.85, timestamp=2.0)
            session.gloss_buffer.add_gloss("Water", confidence=0.92, timestamp=3.0)
            session.displayed_sequence = list(session.gloss_buffer.current_sequence)
            session.displayed_translation = session.translator.translate(session.displayed_sequence)

            # Pop "Water"
            res_pop = session.pop_last_gloss()
            self.assertEqual(res_pop["removed_gloss"], "Water")
            self.assertEqual(res_pop["current_glosses"], ["I_Me", "Want_Desire_Wish_For"])
            self.assertEqual(res_pop["current_translation"], "I want.")

            # Pop "Want_Desire_Wish_For"
            res_pop2 = session.pop_last_gloss()
            self.assertEqual(res_pop2["removed_gloss"], "Want_Desire_Wish_For")
            self.assertEqual(res_pop2["current_glosses"], ["I_Me"])
            self.assertEqual(res_pop2["current_translation"], "Me.")

            # Pop "I_Me"
            res_pop3 = session.pop_last_gloss()
            self.assertEqual(res_pop3["removed_gloss"], "I_Me")
            self.assertEqual(res_pop3["current_glosses"], [])
            self.assertIsNone(res_pop3["current_translation"])
        finally:
            session.close()

    def test_websocket_pause_resume_undo_messages(self):
        with self.client.websocket_connect("/ws/recognize") as ws:
            init_ack = ws.receive_json()
            self.assertEqual(init_ack.get("type"), "init_ack")

            # Test pause
            ws.send_json({"type": "pause"})
            pause_ack = ws.receive_json()
            self.assertEqual(pause_ack.get("type"), "pause_ack")
            self.assertTrue(pause_ack.get("is_paused"))
            self.assertEqual(pause_ack.get("state"), "PAUSED")

            # Test resume
            ws.send_json({"type": "resume"})
            resume_ack = ws.receive_json()
            self.assertEqual(resume_ack.get("type"), "resume_ack")
            self.assertFalse(resume_ack.get("is_paused"))
            self.assertEqual(resume_ack.get("state"), "IDLE")

            # Test undo on empty session
            ws.send_json({"type": "undo"})
            undo_ack = ws.receive_json()
            self.assertEqual(undo_ack.get("type"), "undo_ack")
            self.assertIsNone(undo_ack.get("removed_gloss"))
            self.assertEqual(undo_ack.get("current_glosses"), [])


    def test_sentence_persistence_across_pause_resume(self):
        """Verifies completed sentences persist across spacebar pause/resume and are only cleared by reset."""
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            # Finalize a sentence
            session.gloss_buffer.add_gloss("I_Me", confidence=0.95, timestamp=1.0)
            session.gloss_buffer.add_gloss("Want_Desire_Wish_For", confidence=0.90, timestamp=2.0)
            session.gloss_buffer.add_gloss("Water", confidence=0.92, timestamp=3.0)
            session.completed_sequence = ["I_Me", "Want_Desire_Wish_For", "Water"]
            session.completed_translation = "I want water."
            session.displayed_sequence = list(session.completed_sequence)
            session.displayed_translation = session.completed_translation
            session.is_finalized = True

            # 1. Pause recognition (Spacebar pause)
            session.set_paused(True)
            self.assertTrue(session.is_paused)
            self.assertEqual(session.completed_sequence, ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(session.completed_translation, "I want water.")

            # Frame during pause should return completed sentence
            frame = self._create_synthetic_frame()
            res_paused = session.process_frame(frame, timestamp=4.0)
            self.assertEqual(res_paused["state"], "PAUSED")
            self.assertEqual(res_paused["completed_sequence"], ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(res_paused["completed_translation"], "I want water.")

            # 2. Resume recognition (Spacebar resume)
            session.set_paused(False)
            self.assertFalse(session.is_paused)
            self.assertEqual(session.completed_sequence, ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(session.completed_translation, "I want water.")

            # 3. In-progress active sequence for next sentence
            session.in_progress_sequence = ["Food"]
            session.in_progress_translation = "Food."
            res_active = session.process_frame(frame, timestamp=5.0)
            self.assertEqual(res_active["completed_sequence"], ["I_Me", "Want_Desire_Wish_For", "Water"])
            self.assertEqual(res_active["in_progress_sequence"], ["Food"])

            # 4. Explicit reset (Clear Sentence button)
            session.reset()
            self.assertEqual(session.completed_sequence, [])
            self.assertIsNone(session.completed_translation)
            self.assertEqual(session.in_progress_sequence, [])
            self.assertIsNone(session.in_progress_translation)
        finally:
            session.close()

    def test_continuous_signing_chaining_without_rests(self):
        """Verifies immediate chaining from recognized sign into next active sign when hands remain moving."""
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            # Set up session in ACTIVE state with gesture frames accumulated
            session.seg_state = "ACTIVE"
            session.gesture_feats = [np.zeros(210, dtype=np.float32) for _ in range(16)]
            session.gesture_pres = [0.9 for _ in range(16)]
            session.in_progress_sequence = ["I_Me"]

            # Hands remain actively in signing zone with high energy
            session.smoothed_motion = 0.05
            session.rest_counter = 0

            # Mark gesture boundary for continuous chaining
            boundary_result = session.mark_gesture_boundary()
            self.assertEqual(boundary_result["state"], "ACTIVE")
            self.assertEqual(len(session.gesture_feats), 0)
            self.assertEqual(session.seg_state, "ACTIVE")
            self.assertFalse(session._safety_cutoff_triggered)
        finally:
            session.close()

    def test_undo_backspace_preserves_completed_sentence(self):
        """Verifies backspace undoes from editable active sequence first without wiping completed sentence."""
        session = LiveWebcamRecognitionSession(engine=self.engine)
        try:
            session.completed_sequence = ["I_Me", "Want_Desire_Wish_For", "Water"]
            session.completed_translation = "I want water."
            session.in_progress_sequence = ["Food", "Eat"]
            session.in_progress_translation = "Eat food."

            # Pop 1: removes 'Eat' from in-progress sequence
            pop1 = session.pop_last_gloss()
            self.assertEqual(pop1["removed_gloss"], "Eat")
            self.assertEqual(pop1["in_progress_sequence"], ["Food"])
            self.assertEqual(session.completed_sequence, ["I_Me", "Want_Desire_Wish_For", "Water"])

            # Pop 2: removes 'Food' from in-progress sequence
            pop2 = session.pop_last_gloss()
            self.assertEqual(pop2["removed_gloss"], "Food")
            self.assertEqual(pop2["in_progress_sequence"], [])
            self.assertEqual(session.completed_sequence, ["I_Me", "Want_Desire_Wish_For", "Water"])

            # Pop 3: now undoes from completed sentence
            pop3 = session.pop_last_gloss()
            self.assertEqual(pop3["removed_gloss"], "Water")
            self.assertEqual(pop3["current_glosses"], ["I_Me", "Want_Desire_Wish_For"])
            self.assertEqual(session.completed_sequence, ["I_Me", "Want_Desire_Wish_For"])
        finally:
            session.close()

    def test_websocket_clear_only_resets_completed_sentence(self):
        """Verifies websocket 'clear' explicitly resets completed sequence, while pause/resume does not."""
        with self.client.websocket_connect("/ws/recognize") as ws:
            init_ack = ws.receive_json()
            self.assertEqual(init_ack.get("type"), "init_ack")

            # Pause and resume
            ws.send_json({"type": "pause"})
            p_ack = ws.receive_json()
            self.assertTrue(p_ack.get("is_paused"))

            ws.send_json({"type": "resume"})
            r_ack = ws.receive_json()
            self.assertFalse(r_ack.get("is_paused"))

            # Clear
            ws.send_json({"type": "clear"})
            c_ack = ws.receive_json()
            self.assertEqual(c_ack.get("type"), "reset_ack")
            self.assertEqual(c_ack.get("completed_sequence"), [])
            self.assertEqual(c_ack.get("in_progress_sequence"), [])


if __name__ == "__main__":
    unittest.main()



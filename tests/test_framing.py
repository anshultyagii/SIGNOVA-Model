# -*- coding: utf-8 -*-
"""
SignSpeak AI - Unit Tests for Framing Validation (Phase 2)
=========================================================
Tests pose and wrist positioning rules, consecutive stability,
and warning detection during sign recording.
"""

import unittest
from types import SimpleNamespace
import numpy as np

from src.recording.framing import (
    evaluate_pose_framing,
    FramingStatus,
    FramingStateMachine,
    draw_framing_hud
)


def _make_dummy_landmark(x: float, y: float, z: float = 0.0, visibility: float = 0.9):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _make_dummy_pose(
    ls_x: float = 0.4, ls_y: float = 0.4,
    rs_x: float = 0.6, rs_y: float = 0.4,
    lw_x: float = 0.4, lw_y: float = 0.7,
    rw_x: float = 0.6, rw_y: float = 0.7,
    wrist_vis: float = 0.9
):
    """Constructs a minimal dummy MediaPipe pose landmark structure with 17 keypoints."""
    landmarks = [_make_dummy_landmark(0.5, 0.2)] * 17
    # Left shoulder (11), Right shoulder (12)
    landmarks[11] = _make_dummy_landmark(ls_x, ls_y, visibility=0.9)
    landmarks[12] = _make_dummy_landmark(rs_x, rs_y, visibility=0.9)
    # Left wrist (15), Right wrist (16)
    landmarks[15] = _make_dummy_landmark(lw_x, lw_y, visibility=wrist_vis)
    landmarks[16] = _make_dummy_landmark(rw_x, rw_y, visibility=wrist_vis)
    return SimpleNamespace(landmark=landmarks)


class TestFramingValidation(unittest.TestCase):
    """Tests evaluate_pose_framing with various body postures."""

    def test_no_pose_detected(self):
        eval_res = evaluate_pose_framing(None)
        self.assertEqual(eval_res.status, FramingStatus.NO_POSE)
        self.assertFalse(eval_res.is_acceptable)
        self.assertIn("Position yourself", eval_res.user_message)

    def test_wrists_below_frame_detected(self):
        """Simulates the recorded videos where wrists were at y = 1.25 (below bottom edge)."""
        pose = _make_dummy_pose(lw_y=1.25, rw_y=1.35, wrist_vis=0.1)
        eval_res = evaluate_pose_framing(pose, has_left_hand=False, has_right_hand=False)
        self.assertEqual(eval_res.status, FramingStatus.HANDS_NOT_VISIBLE)
        self.assertFalse(eval_res.is_acceptable)
        self.assertIn("Hands not visible", eval_res.user_message)

    def test_wrists_in_frame_acceptable(self):
        """Wrists positioned comfortably in the signing area (y = 0.65)."""
        pose = _make_dummy_pose(lw_y=0.65, rw_y=0.65, wrist_vis=0.95)
        eval_res = evaluate_pose_framing(pose)
        self.assertEqual(eval_res.status, FramingStatus.READY_TO_RECORD)
        self.assertTrue(eval_res.is_acceptable)
        self.assertTrue(eval_res.left_wrist_in_frame)
        self.assertTrue(eval_res.right_wrist_in_frame)

    def test_one_handed_wrist_in_frame(self):
        """One hand raised in frame (right wrist at 0.5), other hand at rest below frame."""
        pose = _make_dummy_pose(lw_y=1.1, rw_y=0.5, wrist_vis=0.9)
        eval_res = evaluate_pose_framing(pose)
        self.assertEqual(eval_res.status, FramingStatus.READY_TO_RECORD)
        self.assertTrue(eval_res.is_acceptable)
        self.assertTrue(eval_res.right_wrist_in_frame)
        self.assertFalse(eval_res.left_wrist_in_frame)

    def test_hand_detection_overrides_boundary_wrists(self):
        """When MediaPipe hand detector actively sees a hand, readiness is confirmed."""
        pose = _make_dummy_pose(lw_y=0.96, rw_y=0.97, wrist_vis=0.4)
        eval_res = evaluate_pose_framing(pose, has_right_hand=True)
        self.assertTrue(eval_res.is_acceptable)
        self.assertTrue(eval_res.hands_detected)

    def test_shoulders_out_of_frame(self):
        """Camera positioned too high so shoulders are cropped at the bottom."""
        pose = _make_dummy_pose(ls_y=0.95, rs_y=0.95)
        eval_res = evaluate_pose_framing(pose)
        self.assertEqual(eval_res.status, FramingStatus.SHOULDERS_OUT_OF_FRAME)
        self.assertFalse(eval_res.is_acceptable)


class TestFramingStateMachine(unittest.TestCase):
    """Tests stability counters and recording warning triggers."""

    def setUp(self):
        self.sm = FramingStateMachine(required_stable_frames=5, max_warning_frames_during_rec=3)

    def test_consecutive_frames_required_before_ready(self):
        pose = _make_dummy_pose(lw_y=0.6, rw_y=0.6)
        acceptable = evaluate_pose_framing(pose)

        # First 4 frames: stabilizing
        for i in range(4):
            status, msg = self.sm.update(acceptable)
            self.assertEqual(status, FramingStatus.STABILIZING)
            self.assertIn("Hold position", msg)

        # 5th frame: ready
        status, msg = self.sm.update(acceptable)
        self.assertEqual(status, FramingStatus.READY_TO_RECORD)
        self.assertIn("Ready to record", msg)

    def test_instability_resets_counter(self):
        pose_good = _make_dummy_pose(lw_y=0.6, rw_y=0.6)
        pose_bad = _make_dummy_pose(lw_y=1.2, rw_y=1.2, wrist_vis=0.1)

        good_eval = evaluate_pose_framing(pose_good)
        bad_eval = evaluate_pose_framing(pose_bad)

        self.sm.update(good_eval)
        self.sm.update(good_eval)
        self.assertEqual(self.sm.consecutive_stable_frames, 2)

        # Bad frame resets counter
        self.sm.update(bad_eval)
        self.assertEqual(self.sm.consecutive_stable_frames, 0)

    def test_recording_framing_violation_tracked(self):
        self.sm.start_recording()
        pose_good = _make_dummy_pose(lw_y=0.6, rw_y=0.6)
        pose_bad = _make_dummy_pose(lw_y=1.2, rw_y=1.2, wrist_vis=0.1)

        good_eval = evaluate_pose_framing(pose_good)
        bad_eval = evaluate_pose_framing(pose_bad)

        # 2 good frames
        self.sm.update(good_eval)
        self.sm.update(good_eval)

        # 3 bad frames (reaches max_warning_frames_during_rec=3)
        self.sm.update(bad_eval)
        self.sm.update(bad_eval)
        status, msg = self.sm.update(bad_eval)

        self.assertEqual(status, FramingStatus.RECORDING_WARNING)
        self.assertIn("WARNING", msg)

        summary = self.sm.stop_recording()
        self.assertEqual(summary["total_frames"], 5)
        self.assertEqual(summary["warning_frames"], 3)
        self.assertFalse(summary["is_valid"])


class TestHudRendering(unittest.TestCase):
    """Tests overlay drawing on test canvas."""

    def test_draw_hud_runs_cleanly(self):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = draw_framing_hud(
            frame=canvas,
            status=FramingStatus.READY_TO_RECORD,
            message="Ready to record",
            target_word="Drink",
            countdown_secs=2
        )
        self.assertEqual(rendered.shape, (720, 1280, 3))


class TestRecordingUtils(unittest.TestCase):
    """Tests non-destructive indexing and camera initialization handling."""

    def test_get_next_sample_index_empty_folder(self):
        import tempfile
        from pathlib import Path
        from collect_signs import get_next_sample_index

        with tempfile.TemporaryDirectory() as tmpdir:
            idx = get_next_sample_index(Path(tmpdir))
            self.assertEqual(idx, 1)

    def test_get_next_sample_index_existing_samples(self):
        import tempfile
        from pathlib import Path
        from collect_signs import get_next_sample_index

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "sample_001.mp4").touch()
            (p / "sample_002.mp4").touch()
            (p / "sample_010.mp4").touch()
            idx = get_next_sample_index(p)
            self.assertEqual(idx, 11)

    def test_open_webcam_invalid_index(self):
        from collect_signs import open_webcam
        cap, success, err = open_webcam(-999)
        self.assertFalse(success)
        self.assertIsNone(cap)

    def test_main_graceful_exit_on_frame_read_fail(self):
        """Ensures collect_signs.main exits gracefully without traceback when camera feed read fails."""
        import io
        from unittest.mock import patch, MagicMock
        import collect_signs

        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_holistic_instance = MagicMock()
        mock_holistic_class = MagicMock()
        mock_holistic_class.return_value.__enter__.return_value = mock_holistic_instance

        with patch("collect_signs.open_webcam", return_value=(mock_cap, True, "")), \
             patch("mediapipe.solutions.holistic.Holistic", mock_holistic_class), \
             patch("sys.stdout", new_callable=io.StringIO):
            collect_signs.main([])
            mock_cap.release.assert_called()

    def test_main_graceful_exit_on_interrupt(self):
        """Ensures collect_signs.main exits gracefully without traceback on KeyboardInterrupt."""
        import io
        from unittest.mock import patch, MagicMock
        import collect_signs

        mock_cap = MagicMock()
        mock_holistic_instance = MagicMock()
        mock_holistic_class = MagicMock()
        mock_holistic_class.return_value.__enter__.return_value = mock_holistic_instance

        with patch("collect_signs.open_webcam", return_value=(mock_cap, True, "")), \
             patch("mediapipe.solutions.holistic.Holistic", mock_holistic_class), \
             patch("collect_signs.run_interactive_recorder", side_effect=KeyboardInterrupt), \
             patch("sys.stdout", new_callable=io.StringIO):
            collect_signs.main([])
            mock_cap.release.assert_called()

    def test_run_interactive_recorder_smoke_test(self):
        """Ensures run_interactive_recorder runs and exits cleanly in smoke-test mode."""
        from unittest.mock import patch, MagicMock
        import collect_signs
        import numpy as np

        mock_cap = MagicMock()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)

        mock_holistic = MagicMock()
        mock_holistic.process.return_value = MagicMock(
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        mock_draw = MagicMock()

        with patch("cv2.imshow"), patch("cv2.namedWindow"), patch("cv2.destroyWindow"):
            res = collect_signs.run_interactive_recorder(
                cap=mock_cap,
                holistic=mock_holistic,
                mp_draw=mock_draw,
                smoke_test=True
            )
            self.assertTrue(res)

    def test_resolve_initial_sign_index(self):
        """Ensures sign name or number argument resolves correctly to vocabulary index."""
        from collect_signs import resolve_initial_sign_index, WORDS
        self.assertEqual(resolve_initial_sign_index(None), 0)
        self.assertEqual(resolve_initial_sign_index("1"), 0)
        self.assertEqual(resolve_initial_sign_index("11"), 10)
        self.assertEqual(resolve_initial_sign_index("Drink"), WORDS.index("Drink"))
        self.assertEqual(resolve_initial_sign_index("water"), WORDS.index("Water"))
        self.assertEqual(resolve_initial_sign_index("invalid_sign_xyz"), 0)


if __name__ == "__main__":
    unittest.main()

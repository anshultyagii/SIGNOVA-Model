"""
SignSpeak AI - Unit Tests for Recognition Package
Tests feature extraction, normalization, missing hand handling,
DTW distance calculation, and prediction output structure.
"""

import unittest
import numpy as np

from src.recognition.features import (
    TOTAL_RAW_DIM,
    parse_raw_landmarks_frame,
    normalize_frame,
    normalize_sequence,
    frame_distance,
    NormalizedFrame,
    NormalizedSequence
)
from src.recognition.matcher import (
    DTWSequenceMatcher,
    PredictionResult,
    _pack_sequence,
    _resample_matrix,
    _RH_FLAG_IDX,
    _LH_FLAG_IDX,
    MAX_DTW_LEN
)


class TestFeatureProcessing(unittest.TestCase):
    """Tests feature parsing, dimensions, and numerical validity."""

    def setUp(self):
        # Create a valid synthetic raw frame
        self.raw_frame = np.ones(TOTAL_RAW_DIM, dtype=np.float32) * 0.5
        # Set distinct shoulder positions: left shoulder (index 11), right shoulder (index 12)
        # Pose starts at index 0. Index 11 is [33:36], index 12 is [36:39]
        self.raw_frame[33:36] = np.array([0.4, 0.5, 0.0], dtype=np.float32)
        self.raw_frame[36:39] = np.array([0.6, 0.5, 0.0], dtype=np.float32)

        self.presence = np.array([True, True, True, True], dtype=bool)

    def test_valid_parsing(self):
        p_raw, p_pres, lh_raw, lh_pres, rh_raw, rh_pres = parse_raw_landmarks_frame(
            self.raw_frame, self.presence
        )
        self.assertEqual(p_raw.shape, (33, 3))
        self.assertEqual(lh_raw.shape, (21, 3))
        self.assertEqual(rh_raw.shape, (21, 3))
        self.assertTrue(p_pres)
        self.assertTrue(lh_pres)
        self.assertTrue(rh_pres)

    def test_invalid_dimension_raises(self):
        invalid_row = np.ones(100, dtype=np.float32)
        with self.assertRaises(ValueError):
            parse_raw_landmarks_frame(invalid_row, self.presence)

    def test_non_finite_values_raise(self):
        row_with_nan = self.raw_frame.copy()
        row_with_nan[10] = np.nan
        with self.assertRaises(ValueError):
            parse_raw_landmarks_frame(row_with_nan, self.presence)

        row_with_inf = self.raw_frame.copy()
        row_with_inf[25] = np.inf
        with self.assertRaises(ValueError):
            parse_raw_landmarks_frame(row_with_inf, self.presence)

    def test_empty_sequence_raises(self):
        empty_feats = np.empty((0, TOTAL_RAW_DIM), dtype=np.float32)
        empty_pres = np.empty((0, 4), dtype=bool)
        with self.assertRaises(ValueError):
            normalize_sequence(empty_feats, empty_pres)

    def test_missing_hand_handling(self):
        """Verifies that missing hands are explicitly handled and not treated as (0, 0, 0)."""
        # Frame A: Right hand present
        f_a = normalize_frame(
            pose_raw=np.zeros((33, 3)),
            pose_present=True,
            lh_raw=np.zeros((21, 3)),
            lh_present=False,
            rh_raw=np.ones((21, 3)) * 0.5,
            rh_present=True
        )

        # Frame B: Right hand absent
        f_b = normalize_frame(
            pose_raw=np.zeros((33, 3)),
            pose_present=True,
            lh_raw=np.zeros((21, 3)),
            lh_present=False,
            rh_raw=np.zeros((21, 3)),
            rh_present=False
        )

        # Frame C: Right hand also absent (identical rest state to B)
        f_c = normalize_frame(
            pose_raw=np.zeros((33, 3)),
            pose_present=True,
            lh_raw=np.zeros((21, 3)),
            lh_present=False,
            rh_raw=np.zeros((21, 3)),
            rh_present=False
        )

        # 1. Present vs Absent should incur penalty
        dist_ab = frame_distance(f_a, f_b, missing_hand_penalty=2.0)
        self.assertGreaterEqual(dist_ab, 2.0)

        # 2. Absent vs Absent should have zero hand penalty
        dist_bc = frame_distance(f_b, f_c, missing_hand_penalty=2.0)
        self.assertEqual(dist_bc, 0.0)

    def test_translation_invariance(self):
        """Verifies that shifting signer position globally does not alter wrist-normalized hand shape."""
        raw_f1 = self.raw_frame.copy()
        raw_f2 = self.raw_frame.copy()

        # Shift raw_f2 by +0.3 in X across all coordinates (different position in camera)
        raw_f2 += 0.3

        p1, pp1, lh1, lp1, rh1, rp1 = parse_raw_landmarks_frame(raw_f1, self.presence)
        p2, pp2, lh2, lp2, rh2, rp2 = parse_raw_landmarks_frame(raw_f2, self.presence)

        norm_f1 = normalize_frame(p1, pp1, lh1, lp1, rh1, rp1)
        norm_f2 = normalize_frame(p2, pp2, lh2, lp2, rh2, rp2)

        # Hand local shape should be identical
        np.testing.assert_allclose(norm_f1.rh_shape, norm_f2.rh_shape, atol=1e-5)
        # Upper body pose relative to shoulders should be identical
        np.testing.assert_allclose(norm_f1.pose, norm_f2.pose, atol=1e-5)


class TestDTWMatcher(unittest.TestCase):
    """Tests DTW distance computation and prediction output."""

    def _create_synthetic_sequence(
        self,
        length: int,
        label: str,
        offset: float = 0.0,
        rh_present_pattern: list = None
    ) -> NormalizedSequence:
        frames = []
        for i in range(length):
            t = (i + offset) / max(length, 1)
            # Create a synthetic pose
            pose = np.zeros((7, 3), dtype=np.float32) + t
            is_rh = rh_present_pattern[i] if rh_present_pattern is not None else True
            # Synthetic right hand shape
            rh_shape = np.ones((21, 3), dtype=np.float32) * t if is_rh else None
            rh_pos = np.array([t, 0.5, 0.0], dtype=np.float32) if is_rh else None
            frames.append(
                NormalizedFrame(
                    pose=pose,
                    lh_shape=None,
                    lh_pos=None,
                    lh_present=False,
                    rh_shape=rh_shape,
                    rh_pos=rh_pos,
                    rh_present=is_rh
                )
            )
        return NormalizedSequence(
            frames=frames,
            label=label,
            video_name=f"{label}.mp4",
            active_window=(0, length - 1)
        )

    def test_identity_distance_is_zero(self):
        seq = self._create_synthetic_sequence(20, "SignA")
        matcher = DTWSequenceMatcher()
        dist = matcher.compute_dtw_distance(seq, seq)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_symmetry(self):
        seq1 = self._create_synthetic_sequence(15, "SignA", offset=0.1)
        seq2 = self._create_synthetic_sequence(25, "SignB", offset=0.3)
        matcher = DTWSequenceMatcher()

        dist12 = matcher.compute_dtw_distance(seq1, seq2)
        dist21 = matcher.compute_dtw_distance(seq2, seq1)
        self.assertAlmostEqual(dist12, dist21, places=5)

    def test_different_sequence_lengths(self):
        """DTW should smoothly handle sequences of varying lengths (e.g. 10 vs 35)."""
        seq_short = self._create_synthetic_sequence(10, "FastSign")
        seq_long = self._create_synthetic_sequence(35, "SlowSign")
        matcher = DTWSequenceMatcher()

        dist = matcher.compute_dtw_distance(seq_short, seq_long)
        self.assertGreater(dist, 0.0)
        self.assertTrue(np.isfinite(dist))

    def test_prediction_output_structure(self):
        matcher = DTWSequenceMatcher()
        matcher.add_template("WordA", self._create_synthetic_sequence(20, "WordA", offset=0.0))
        matcher.add_template("WordB", self._create_synthetic_sequence(20, "WordB", offset=0.5))
        matcher.add_template("WordC", self._create_synthetic_sequence(20, "WordC", offset=1.0))

        # Query identical to WordA
        query = self._create_synthetic_sequence(20, "Query", offset=0.0)
        results = matcher.predict(query, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].label, "WordA")
        self.assertAlmostEqual(results[0].distance, 0.0, places=5)
        self.assertAlmostEqual(results[0].similarity, 1.0, places=5)

        # Check ranking ordering
        self.assertGreaterEqual(results[0].similarity, results[1].similarity)
        self.assertGreaterEqual(results[1].similarity, results[2].similarity)

        # Check serialization
        d = results[0].to_dict()
        self.assertIn("rank", d)
        self.assertIn("label", d)
        self.assertIn("similarity", d)
        self.assertIn("distance", d)

    def test_resampling_long_sequence_identity_is_zero(self):
        """A sequence longer than MAX_DTW_LEN (60) with hand transitions compared with itself must have exact distance 0.0."""
        # 80 frames: 10 frames no hand, 50 frames hand present, 20 frames no hand
        pattern = [False] * 10 + [True] * 50 + [False] * 20
        seq_long = self._create_synthetic_sequence(80, "LongSign", rh_present_pattern=pattern)
        matcher = DTWSequenceMatcher()
        dist = matcher.compute_dtw_distance(seq_long, seq_long)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_resampled_presence_flags_remain_binary(self):
        """Resampling sequences longer than MAX_DTW_LEN must keep hand flags strictly binary (0.0 or 1.0)."""
        pattern = [False] * 15 + [True] * 45 + [False] * 15
        seq_long = self._create_synthetic_sequence(75, "LongSign", rh_present_pattern=pattern)
        packed = _pack_sequence(seq_long)
        resampled = _resample_matrix(packed, MAX_DTW_LEN)

        unique_rh = set(np.unique(resampled[:, _RH_FLAG_IDX]))
        unique_lh = set(np.unique(resampled[:, _LH_FLAG_IDX]))
        self.assertTrue(unique_rh.issubset({0.0, 1.0}))
        self.assertTrue(unique_lh.issubset({0.0, 1.0}))

    def test_genuine_missing_hand_mismatch_penalized(self):
        """Genuine hand presence mismatches must still incur the intended missing hand penalty."""
        seq_with_hand = self._create_synthetic_sequence(20, "WithHand", rh_present_pattern=[True] * 20)
        seq_no_hand = self._create_synthetic_sequence(20, "NoHand", rh_present_pattern=[False] * 20)
        matcher = DTWSequenceMatcher()
        dist = matcher.compute_dtw_distance(seq_with_hand, seq_no_hand)
        # Missing hand penalty is 2.0 per frame on right hand
        self.assertGreaterEqual(dist, 1.0)


if __name__ == "__main__":
    unittest.main()

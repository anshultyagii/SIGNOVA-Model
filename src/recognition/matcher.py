"""
SignSpeak AI - Dynamic Time Warping (DTW) Sequence Matcher Module
Compares query gesture sequences against enrolled reference templates.
Produces top-k candidate classifications with similarity scores.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .features import (
    NormalizedSequence,
    normalize_sequence,
    frame_distance,
    UPPER_BODY_POSE_INDICES,
    HAND_COUNT
)

# --- Vectorized sequence representation ---
# Instead of calling frame_distance() in a Python loop (slow), we pack each
# NormalizedSequence into a single float32 matrix (T, D) and run DTW using
# NumPy batch distance calculations.

# Maximum number of frames to run DTW on per sequence.
# If the trimmed sequence is longer, it is linearly resampled to this cap.
# Using natural sequence lengths (not forcing both to the same length) allows
# DTW to exploit genuine temporal-length differences between signs (e.g. Drink
# is naturally ~27 frames while Water is ~42 frames after active-window trim).
MAX_DTW_LEN = 60   # cap — keeps cost O(60*60)=3600 per template pair

# Keep DTW_RESAMPLE_LEN as an alias so any external imports don't break.
DTW_RESAMPLE_LEN = MAX_DTW_LEN


def _pack_sequence(seq: NormalizedSequence) -> np.ndarray:
    """
    Converts a NormalizedSequence into a (T, D) float32 numpy array.
    Layout per row:
      - pose   : (7*3 = 21) upper-body keypoints, wrist-normalized
      - rh_shape: (21*3 = 63) right-hand shape  (zeros if absent)
      - rh_pos  : (3)         right-wrist body-relative position
      - rh_pres : (1)         binary presence flag
      - lh_shape: (21*3 = 63) left-hand shape   (zeros if absent)
      - lh_pos  : (3)         left-wrist body-relative position
      - lh_pres : (1)         binary presence flag
    Total D = 21 + 63 + 3 + 1 + 63 + 3 + 1 = 155
    """
    rows = []
    for f in seq.frames:
        pose_flat = f.pose.flatten()                              # (21,)
        rh_shape = f.rh_shape.flatten() if f.rh_present else np.zeros(63, np.float32)  # (63,)
        rh_pos   = f.rh_pos if f.rh_present else np.zeros(3, np.float32)               # (3,)
        rh_flag  = np.array([1.0 if f.rh_present else 0.0], np.float32)                # (1,)
        lh_shape = f.lh_shape.flatten() if f.lh_present else np.zeros(63, np.float32)  # (63,)
        lh_pos   = f.lh_pos if f.lh_present else np.zeros(3, np.float32)               # (3,)
        lh_flag  = np.array([1.0 if f.lh_present else 0.0], np.float32)                # (1,)
        rows.append(np.concatenate([pose_flat, rh_shape, rh_pos, rh_flag,
                                     lh_shape, lh_pos,   lh_flag]))
    return np.stack(rows).astype(np.float32)                      # (T, 155)


def _resample_matrix(mat: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resamples a (T, D) matrix to (target_len, D)."""
    T = len(mat)
    if T == target_len:
        return mat
    indices = np.linspace(0, T - 1, target_len)
    lo = np.floor(indices).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    alpha = (indices - lo)[:, None]   # (target_len, 1)
    res = (1.0 - alpha) * mat[lo] + alpha * mat[hi]
    # Hand presence flags must remain strictly binary (0.0 or 1.0)
    res[:, [_RH_FLAG_IDX, _LH_FLAG_IDX]] = np.where(
        res[:, [_RH_FLAG_IDX, _LH_FLAG_IDX]] >= 0.5, 1.0, 0.0
    )
    return res


# Weights matching frame_distance() defaults, applied to packed vector slices
_W_POSE   = 0.25
_W_SHAPE  = 0.50
_W_POS    = 0.25
_MISSING_PENALTY = 2.0

# Slice positions in the 155-D packed row
_POSE_SLICE      = slice(0, 21)
_RH_SHAPE_SLICE  = slice(21, 84)
_RH_POS_SLICE    = slice(84, 87)
_RH_FLAG_IDX     = 87
_LH_SHAPE_SLICE  = slice(88, 151)
_LH_POS_SLICE    = slice(151, 154)
_LH_FLAG_IDX     = 154


def _batch_frame_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes a (len_A, len_B) distance matrix between all pairs of frames
    using vectorized NumPy, matching frame_distance() logic.
    A: (Ta, 155), B: (Tb, 155)
    Returns: (Ta, Tb) float32 distance matrix
    """
    Ta, Tb = len(A), len(B)

    # Pose distance (L2 of 21 values, normalised by sqrt(7))
    d_pose = np.linalg.norm(
        A[:, None, _POSE_SLICE] - B[None, :, _POSE_SLICE], axis=2
    ) / np.sqrt(7)   # (Ta, Tb)

    # Right-hand distances (using boolean logic so identical frames cannot trigger penalty)
    rh_a = A[:, _RH_FLAG_IDX] > 0.5            # (Ta,) bool
    rh_b = B[:, _RH_FLAG_IDX] > 0.5            # (Tb,) bool
    both_rh = rh_a[:, None] & rh_b[None, :]     # (Ta, Tb) bool
    only_one_rh = rh_a[:, None] ^ rh_b[None, :] # (Ta, Tb) bool (XOR)

    rh_shape_dist = np.linalg.norm(
        A[:, None, _RH_SHAPE_SLICE] - B[None, :, _RH_SHAPE_SLICE], axis=2
    ) / np.sqrt(HAND_COUNT)  # (Ta, Tb)
    rh_pos_dist = np.linalg.norm(
        A[:, None, _RH_POS_SLICE] - B[None, :, _RH_POS_SLICE], axis=2
    )
    d_rh = np.where(
        both_rh,
        _W_SHAPE * rh_shape_dist + _W_POS * rh_pos_dist,
        np.where(only_one_rh, _MISSING_PENALTY, 0.0)
    )

    # Left-hand distances (using boolean logic so identical frames cannot trigger penalty)
    lh_a = A[:, _LH_FLAG_IDX] > 0.5            # (Ta,) bool
    lh_b = B[:, _LH_FLAG_IDX] > 0.5            # (Tb,) bool
    both_lh = lh_a[:, None] & lh_b[None, :]     # (Ta, Tb) bool
    only_one_lh = lh_a[:, None] ^ lh_b[None, :] # (Ta, Tb) bool (XOR)

    lh_shape_dist = np.linalg.norm(
        A[:, None, _LH_SHAPE_SLICE] - B[None, :, _LH_SHAPE_SLICE], axis=2
    ) / np.sqrt(HAND_COUNT)
    lh_pos_dist = np.linalg.norm(
        A[:, None, _LH_POS_SLICE] - B[None, :, _LH_POS_SLICE], axis=2
    )
    d_lh = np.where(
        both_lh,
        _W_SHAPE * lh_shape_dist + _W_POS * lh_pos_dist,
        np.where(only_one_lh, _MISSING_PENALTY, 0.0)
    )

    return (_W_POSE * d_pose + d_rh + d_lh).astype(np.float32)


@dataclass
class PredictionResult:
    """Represents a classification candidate with similarity score and distance."""
    rank: int
    label: str
    similarity: float
    distance: float

    def to_dict(self):
        return {
            "rank": self.rank,
            "label": self.label,
            "similarity": round(self.similarity, 4),
            "distance": round(self.distance, 4)
        }


class DTWSequenceMatcher:
    """
    Exemplar-based recognizer using Dynamic Time Warping distance.
    """

    def __init__(
        self,
        templates: Optional[Dict[str, NormalizedSequence]] = None,
        w_pose: float = 0.25,
        w_shape: float = 0.50,
        w_pos: float = 0.25,
        missing_hand_penalty: float = 2.0
    ):
        self.templates: Dict[str, NormalizedSequence] = templates or {}
        self.w_pose = w_pose
        self.w_shape = w_shape
        self.w_pos = w_pos
        self.missing_hand_penalty = missing_hand_penalty

    def add_template(self, label: str, sequence: NormalizedSequence):
        """Adds a reference template to the gallery."""
        self.templates[label] = sequence

    def load_gallery_from_npz_dir(
        self,
        npz_dir: Path,
        trim_to_active: bool = True
    ) -> int:
        """
        Loads and pre-normalizes all .npz files in a directory into the template gallery.
        Pre-packs and pre-resamples all template matrices for maximum inference throughput.
        """
        npz_files = sorted(Path(npz_dir).glob("*_landmarks.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz landmark files found in {npz_dir}")

        self.templates.clear()
        for f in npz_files:
            with np.load(f) as data:
                features = data["features"]
                presence = data["presence"]
                label = str(data["label"]) if "label" in data else f.stem.replace("_landmarks", "")
                video_name = str(data["video_name"]) if "video_name" in data else f.stem + ".mp4"

                norm_seq = normalize_sequence(
                    features=features,
                    presence=presence,
                    label=label,
                    video_name=video_name,
                    trim_to_active=trim_to_active
                )
                # Pre-pack and pre-resample to numpy matrix so live queries avoid repeated work
                packed = _pack_sequence(norm_seq)
                norm_seq._packed = packed
                norm_seq._resampled = _resample_matrix(packed, min(len(packed), MAX_DTW_LEN))
                self.templates[label] = norm_seq

        return len(self.templates)

    @staticmethod
    def _dtw_distance_matrices(A: np.ndarray, B: np.ndarray) -> float:
        """Fast DTW calculation on pre-resampled (Ta, D) and (Tb, D) float32 matrices."""
        Ta, Tb = len(A), len(B)
        cost_matrix = _batch_frame_distances(A, B)

        dtw = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float32)
        dtw[0, 0] = 0.0
        for i in range(1, Ta + 1):
            prev = dtw[i - 1, :]
            cur = dtw[i, :]
            c = cost_matrix[i - 1, :]
            for j in range(1, Tb + 1):
                cur[j] = c[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])

        return float(dtw[Ta, Tb]) / (Ta + Tb)

    def compute_dtw_distance(
        self,
        seq_a: NormalizedSequence,
        seq_b: NormalizedSequence
    ) -> float:
        """
        Computes normalized DTW distance using vectorized NumPy frame distances.
        Supports pre-packed/resampled sequence objects transparently.
        """
        if seq_a.length == 0 or seq_b.length == 0:
            raise ValueError("Cannot compute DTW distance on zero-length sequences.")

        A = getattr(seq_a, "_resampled", None)
        if A is None:
            mat_a = getattr(seq_a, "_packed", None)
            if mat_a is None:
                mat_a = _pack_sequence(seq_a)
            A = _resample_matrix(mat_a, min(len(mat_a), MAX_DTW_LEN))

        B = getattr(seq_b, "_resampled", None)
        if B is None:
            mat_b = getattr(seq_b, "_packed", None)
            if mat_b is None:
                mat_b = _pack_sequence(seq_b)
            B = _resample_matrix(mat_b, min(len(mat_b), MAX_DTW_LEN))

        return self._dtw_distance_matrices(A, B)

    def predict(
        self,
        query: NormalizedSequence,
        top_k: int = 3
    ) -> List[PredictionResult]:
        """
        Scores query sequence against all enrolled gallery templates.
        Pre-packs and pre-resamples the query ONCE across the entire gallery.
        Returns top-k candidate labels sorted by highest similarity.
        """
        if not self.templates:
            raise RuntimeError("Template gallery is empty. Call load_gallery_from_npz_dir first.")
        if query.length == 0:
            raise ValueError("Query sequence has zero frames.")

        # Prepare query matrix ONCE before iterating over gallery
        A = getattr(query, "_resampled", None)
        if A is None:
            mat_q = getattr(query, "_packed", None)
            if mat_q is None:
                mat_q = _pack_sequence(query)
            A = _resample_matrix(mat_q, min(len(mat_q), MAX_DTW_LEN))

        scores = []
        for label, template_seq in self.templates.items():
            B = getattr(template_seq, "_resampled", None)
            if B is None:
                mat_b = getattr(template_seq, "_packed", None)
                if mat_b is None:
                    mat_b = _pack_sequence(template_seq)
                B = _resample_matrix(mat_b, min(len(mat_b), MAX_DTW_LEN))

            dist = self._dtw_distance_matrices(A, B)
            # Similarity mapped monotonically to (0, 1]
            similarity = 1.0 / (1.0 + dist)
            scores.append((label, similarity, dist))

        # Sort by similarity descending (distance ascending)
        scores.sort(key=lambda x: x[1], reverse=True)

        k = min(top_k, len(scores))
        results = [
            PredictionResult(
                rank=i + 1,
                label=scores[i][0],
                similarity=scores[i][1],
                distance=scores[i][2]
            )
            for i in range(k)
        ]
        return results

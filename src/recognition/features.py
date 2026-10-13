"""
SignSpeak AI - Feature Extraction and Landmark Normalization Module
Handles landmark parsing, upper-body/wrist normalization, and explicit missing-hand logic.
Reduces dependence on dense 1,404 face-mesh coordinates while preserving key anchors.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

# Landmark layout constants
POSE_COUNT = 33
FACE_COUNT = 468
HAND_COUNT = 21

POSE_DIM = POSE_COUNT * 3      # 99
FACE_DIM = FACE_COUNT * 3      # 1404
HAND_DIM = HAND_COUNT * 3      # 63
TOTAL_RAW_DIM = POSE_DIM + FACE_DIM + HAND_DIM + HAND_DIM  # 1629

# Selected upper-body pose indices for robust sign posture
# 0: Nose, 11: Left Shoulder, 12: Right Shoulder,
# 13: Left Elbow, 14: Right Elbow, 15: Left Wrist, 16: Right Wrist
UPPER_BODY_POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]


@dataclass
class NormalizedFrame:
    """Represents a single video frame with body-anchored and wrist-normalized features."""
    pose: np.ndarray             # (7, 3) Upper body keypoints normalized to shoulder center and width
    lh_shape: Optional[np.ndarray]  # (21, 3) Left hand shape normalized to wrist and palm scale, or None
    lh_pos: Optional[np.ndarray]    # (3,) Left wrist position relative to shoulder center, or None
    lh_present: bool
    rh_shape: Optional[np.ndarray]  # (21, 3) Right hand shape normalized to wrist and palm scale, or None
    rh_pos: Optional[np.ndarray]    # (3,) Right wrist position relative to shoulder center, or None
    rh_present: bool


@dataclass
class NormalizedSequence:
    """A full gesture sequence composed of normalized frames."""
    frames: List[NormalizedFrame]
    label: str
    video_name: str
    active_window: Tuple[int, int]  # (start_frame, end_frame)

    @property
    def length(self) -> int:
        return len(self.frames)


def parse_raw_landmarks_frame(features_row: np.ndarray, presence_row: np.ndarray):
    """
    Splits a raw 1629-element row and presence flags into distinct anatomical components.
    """
    if len(features_row) != TOTAL_RAW_DIM:
        raise ValueError(
            f"Expected raw feature dimension {TOTAL_RAW_DIM}, got {len(features_row)}"
        )
    if not np.isfinite(features_row).all():
        raise ValueError("Raw feature row contains non-finite values (NaN or Inf).")

    # Slice raw landmarks
    pose_raw = features_row[:POSE_DIM].reshape(POSE_COUNT, 3)
    # face_raw = features_row[POSE_DIM:POSE_DIM + FACE_DIM].reshape(FACE_COUNT, 3)  # Omitted to reduce dimensionality
    lh_raw = features_row[POSE_DIM + FACE_DIM:POSE_DIM + FACE_DIM + HAND_DIM].reshape(HAND_COUNT, 3)
    rh_raw = features_row[POSE_DIM + FACE_DIM + HAND_DIM:].reshape(HAND_COUNT, 3)

    # Presence flags
    pose_present = bool(presence_row[0]) if len(presence_row) > 0 else True
    lh_flag = bool(presence_row[2]) if len(presence_row) > 2 else False
    rh_flag = bool(presence_row[3]) if len(presence_row) > 3 else False

    # Check non-zero coordinates to prevent fabricated zeros from being treated as detections
    lh_present = lh_flag and bool((np.abs(lh_raw).sum() > 1e-4))
    rh_present = rh_flag and bool((np.abs(rh_raw).sum() > 1e-4))

    return pose_raw, pose_present, lh_raw, lh_present, rh_raw, rh_present


def normalize_frame(
    pose_raw: np.ndarray,
    pose_present: bool,
    lh_raw: np.ndarray,
    lh_present: bool,
    rh_raw: np.ndarray,
    rh_present: bool
) -> NormalizedFrame:
    """
    Normalizes upper-body pose and hands using torso and wrist anchors.
    """
    # 1. Torso anchor: Left Shoulder (index 11), Right Shoulder (index 12)
    l_shoulder = pose_raw[11]
    r_shoulder = pose_raw[12]
    shoulder_center = (l_shoulder + r_shoulder) / 2.0
    shoulder_width = float(np.linalg.norm(l_shoulder - r_shoulder))
    if shoulder_width < 1e-4:
        shoulder_width = 1.0  # Fallback scale if shoulders degenerate

    # Upper body pose (7 keypoints x 3)
    upper_body = pose_raw[UPPER_BODY_POSE_INDICES]
    norm_pose = (upper_body - shoulder_center) / shoulder_width

    # 2. Left hand normalization
    if lh_present:
        lh_wrist = lh_raw[0]
        lh_middle_mcp = lh_raw[9]
        lh_scale = float(np.linalg.norm(lh_middle_mcp - lh_wrist))
        if lh_scale < 1e-4:
            lh_scale = 1.0

        # Local hand shape (invariant to body position)
        lh_shape = (lh_raw - lh_wrist) / lh_scale
        # Global hand position relative to body
        lh_pos = (lh_wrist - shoulder_center) / shoulder_width
    else:
        lh_shape = None
        lh_pos = None

    # 3. Right hand normalization
    if rh_present:
        rh_wrist = rh_raw[0]
        rh_middle_mcp = rh_raw[9]
        rh_scale = float(np.linalg.norm(rh_middle_mcp - rh_wrist))
        if rh_scale < 1e-4:
            rh_scale = 1.0

        # Local hand shape (invariant to body position)
        rh_shape = (rh_raw - rh_wrist) / rh_scale
        # Global hand position relative to body
        rh_pos = (rh_wrist - shoulder_center) / shoulder_width
    else:
        rh_shape = None
        rh_pos = None

    return NormalizedFrame(
        pose=norm_pose.astype(np.float32),
        lh_shape=lh_shape.astype(np.float32) if lh_shape is not None else None,
        lh_pos=lh_pos.astype(np.float32) if lh_pos is not None else None,
        lh_present=lh_present,
        rh_shape=rh_shape.astype(np.float32) if rh_shape is not None else None,
        rh_pos=rh_pos.astype(np.float32) if rh_pos is not None else None,
        rh_present=rh_present
    )


def normalize_sequence(
    features: np.ndarray,
    presence: np.ndarray,
    label: str = "",
    video_name: str = "",
    trim_to_active: bool = False
) -> NormalizedSequence:
    """
    Converts raw sequence features and presence flags into a NormalizedSequence.
    """
    if len(features) == 0:
        raise ValueError("Cannot normalize an empty sequence.")
    if features.ndim != 2 or features.shape[1] != TOTAL_RAW_DIM:
        raise ValueError(
            f"Expected feature shape (T, {TOTAL_RAW_DIM}), got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Features array contains non-finite values (NaN/Inf).")

    total_frames = len(features)
    frames: List[NormalizedFrame] = []
    active_indices = []

    for idx in range(total_frames):
        p_raw, p_pres, lh_raw, lh_pres, rh_raw, rh_pres = parse_raw_landmarks_frame(
            features[idx], presence[idx]
        )
        norm_f = normalize_frame(p_raw, p_pres, lh_raw, lh_pres, rh_raw, rh_pres)
        frames.append(norm_f)

        if norm_f.lh_present or norm_f.rh_present:
            active_indices.append(idx)

    if active_indices:
        act_start = active_indices[0]
        act_end = active_indices[-1]
    else:
        act_start = 0
        act_end = total_frames - 1

    if trim_to_active and active_indices:
        # Buffer of 2 frames on either side
        start_idx = max(0, act_start - 2)
        end_idx = min(total_frames, act_end + 3)
        trimmed_frames = frames[start_idx:end_idx]
    else:
        trimmed_frames = frames

    return NormalizedSequence(
        frames=trimmed_frames,
        label=label,
        video_name=video_name,
        active_window=(act_start, act_end)
    )


def frame_distance(
    f_a: NormalizedFrame,
    f_b: NormalizedFrame,
    w_pose: float = 0.25,
    w_shape: float = 0.50,
    w_pos: float = 0.25,
    missing_hand_penalty: float = 2.0
) -> float:
    """
    Computes distance between two normalized frames.
    Explicitly handles missing hands:
    - If both present: compares shape and body-relative position.
    - If both absent: 0 penalty (both in rest position).
    - If one present, one absent: applies missing_hand_penalty.
    """
    # 1. Pose distance (normalized Frobenius norm)
    d_pose = float(np.linalg.norm(f_a.pose - f_b.pose)) / float(np.sqrt(len(UPPER_BODY_POSE_INDICES)))

    # 2. Right hand distance
    if f_a.rh_present and f_b.rh_present:
        d_rh_shape = float(np.linalg.norm(f_a.rh_shape - f_b.rh_shape)) / float(np.sqrt(HAND_COUNT))
        d_rh_pos = float(np.linalg.norm(f_a.rh_pos - f_b.rh_pos))
        d_rh = w_shape * d_rh_shape + w_pos * d_rh_pos
    elif not f_a.rh_present and not f_b.rh_present:
        d_rh = 0.0
    else:
        d_rh = missing_hand_penalty

    # 3. Left hand distance
    if f_a.lh_present and f_b.lh_present:
        d_lh_shape = float(np.linalg.norm(f_a.lh_shape - f_b.lh_shape)) / float(np.sqrt(HAND_COUNT))
        d_lh_pos = float(np.linalg.norm(f_a.lh_pos - f_b.lh_pos))
        d_lh = w_shape * d_lh_shape + w_pos * d_lh_pos
    elif not f_a.lh_present and not f_b.lh_present:
        d_lh = 0.0
    else:
        d_lh = missing_hand_penalty

    total_dist = w_pose * d_pose + d_rh + d_lh
    return total_dist

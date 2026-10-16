# -*- coding: utf-8 -*-
"""
SignSpeak AI - Real-Time Framing & Wrist Validation Module (Phase 2)
===================================================================
Provides modular, testable logic for:
  1. Validating upper-body pose and wrist positioning within camera boundaries.
  2. Confirming hand visibility to prevent recordings with hands outside field of view.
  3. Managing the consecutive stability state machine before recording is permitted.
  4. Detecting and logging framing violations during recording.
  5. Rendering HUD guide overlays, status indicators, and bounding boxes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np


class FramingStatus(str, Enum):
    NO_POSE = "NO_POSE"
    SHOULDERS_OUT_OF_FRAME = "SHOULDERS_OUT_OF_FRAME"
    TOO_CLOSE_OR_FAR = "TOO_CLOSE_OR_FAR"
    HANDS_NOT_VISIBLE = "HANDS_NOT_VISIBLE"
    STABILIZING = "STABILIZING"
    READY_TO_RECORD = "READY_TO_RECORD"
    RECORDING = "RECORDING"
    RECORDING_WARNING = "RECORDING_WARNING"


@dataclass
class FramingEvaluation:
    status: FramingStatus
    is_acceptable: bool
    user_message: str
    left_wrist_in_frame: bool
    right_wrist_in_frame: bool
    hands_detected: bool
    shoulder_width_ratio: float
    details: Dict[str, Any]


def evaluate_pose_framing(
    pose_landmarks: Optional[Any],
    has_left_hand: bool = False,
    has_right_hand: bool = False,
    min_wrist_visibility: float = 0.5,
    max_wrist_y: float = 0.95,
    min_shoulder_width_ratio: float = 0.12,
    max_shoulder_width_ratio: float = 0.85
) -> FramingEvaluation:
    """
    Evaluates whether signer pose and wrists are within acceptable framing boundaries.

    Coordinates from MediaPipe: x in [0.0, 1.0], y in [0.0, 1.0].
    y > 1.0 means the point is physically below the bottom edge of the frame.
    """
    if pose_landmarks is None or not hasattr(pose_landmarks, "landmark"):
        return FramingEvaluation(
            status=FramingStatus.NO_POSE,
            is_acceptable=False,
            user_message="Position yourself in frame",
            left_wrist_in_frame=False,
            right_wrist_in_frame=False,
            hands_detected=False,
            shoulder_width_ratio=0.0,
            details={"reason": "No pose landmarks detected"}
        )

    lms = pose_landmarks.landmark
    if len(lms) < 17:
        return FramingEvaluation(
            status=FramingStatus.NO_POSE,
            is_acceptable=False,
            user_message="Position yourself in frame",
            left_wrist_in_frame=False,
            right_wrist_in_frame=False,
            hands_detected=False,
            shoulder_width_ratio=0.0,
            details={"reason": "Incomplete pose landmark list"}
        )

    # 1. Torso validation: Left Shoulder (11), Right Shoulder (12)
    ls = lms[11]
    rs = lms[12]
    ls_vis = getattr(ls, "visibility", 1.0)
    rs_vis = getattr(rs, "visibility", 1.0)

    # Check shoulders in frame
    if ls_vis < 0.4 or rs_vis < 0.4 or ls.y > 0.90 or rs.y > 0.90 or ls.y < 0.05 or rs.y < 0.05:
        return FramingEvaluation(
            status=FramingStatus.SHOULDERS_OUT_OF_FRAME,
            is_acceptable=False,
            user_message="Position shoulders and torso in frame",
            left_wrist_in_frame=False,
            right_wrist_in_frame=False,
            hands_detected=has_left_hand or has_right_hand,
            shoulder_width_ratio=0.0,
            details={"reason": "Shoulders out of frame or low visibility", "ls_y": ls.y, "rs_y": rs.y}
        )

    # Shoulder width ratio (distance in x normalized)
    sh_dist = np.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)
    if sh_dist < min_shoulder_width_ratio:
        return FramingEvaluation(
            status=FramingStatus.TOO_CLOSE_OR_FAR,
            is_acceptable=False,
            user_message="Too far from camera - step closer",
            left_wrist_in_frame=False,
            right_wrist_in_frame=False,
            hands_detected=has_left_hand or has_right_hand,
            shoulder_width_ratio=float(sh_dist),
            details={"reason": "Signer too far", "shoulder_width": float(sh_dist)}
        )
    if sh_dist > max_shoulder_width_ratio:
        return FramingEvaluation(
            status=FramingStatus.TOO_CLOSE_OR_FAR,
            is_acceptable=False,
            user_message="Too close to camera - step back",
            left_wrist_in_frame=False,
            right_wrist_in_frame=False,
            hands_detected=has_left_hand or has_right_hand,
            shoulder_width_ratio=float(sh_dist),
            details={"reason": "Signer too close", "shoulder_width": float(sh_dist)}
        )

    # 2. Wrists validation: Left Wrist (15), Right Wrist (16)
    lw = lms[15]
    rw = lms[16]
    lw_vis = getattr(lw, "visibility", 1.0)
    rw_vis = getattr(rw, "visibility", 1.0)

    # A wrist is in frame if x in [0.0, 1.0], y in [0.0, max_wrist_y], and visibility >= min_visibility
    lw_ok = (0.0 <= lw.x <= 1.0) and (0.0 <= lw.y <= max_wrist_y) and (lw_vis >= min_wrist_visibility)
    rw_ok = (0.0 <= rw.x <= 1.0) and (0.0 <= rw.y <= max_wrist_y) and (rw_vis >= min_wrist_visibility)

    # Hand landmark detections provide strong positive confirmation
    hands_confirmed = has_left_hand or has_right_hand

    # At least one wrist must be clearly in the frame (for single-handed signs),
    # or hand landmarks must be actively detected.
    wrists_acceptable = lw_ok or rw_ok or hands_confirmed

    if not wrists_acceptable:
        return FramingEvaluation(
            status=FramingStatus.HANDS_NOT_VISIBLE,
            is_acceptable=False,
            user_message="Hands not visible - raise hands or tilt camera down",
            left_wrist_in_frame=lw_ok,
            right_wrist_in_frame=rw_ok,
            hands_detected=hands_confirmed,
            shoulder_width_ratio=float(sh_dist),
            details={"reason": "Wrists below frame", "rw_y": rw.y, "lw_y": lw.y, "rw_vis": rw_vis, "lw_vis": lw_vis}
        )

    return FramingEvaluation(
        status=FramingStatus.READY_TO_RECORD,
        is_acceptable=True,
        user_message="Ready to record",
        left_wrist_in_frame=lw_ok,
        right_wrist_in_frame=rw_ok,
        hands_detected=hands_confirmed,
        shoulder_width_ratio=float(sh_dist),
        details={"rw_y": rw.y, "lw_y": lw.y, "rw_vis": rw_vis, "lw_vis": lw_vis}
    )


class FramingStateMachine:
    """
    Manages consecutive stability before allowing recording to start,
    and detects framing drops during recording.
    """

    def __init__(self, required_stable_frames: int = 15, max_warning_frames_during_rec: int = 8):
        self.required_stable_frames = required_stable_frames
        self.max_warning_frames_during_rec = max_warning_frames_during_rec

        self.consecutive_stable_frames = 0
        self.is_recording = False
        self.recording_frames_total = 0
        self.recording_warning_frames = 0
        self.had_framing_violation = False

    def reset_stability(self):
        self.consecutive_stable_frames = 0

    def start_recording(self):
        self.is_recording = True
        self.recording_frames_total = 0
        self.recording_warning_frames = 0
        self.had_framing_violation = False

    def stop_recording(self) -> Dict[str, Any]:
        result = {
            "total_frames": self.recording_frames_total,
            "warning_frames": self.recording_warning_frames,
            "is_valid": not self.had_framing_violation,
            "warning_ratio": (self.recording_warning_frames / max(1, self.recording_frames_total))
        }
        self.is_recording = False
        self.consecutive_stable_frames = 0
        return result

    def update(self, evaluation: FramingEvaluation) -> Tuple[FramingStatus, str]:
        """
        Updates the state machine with a new frame's evaluation.
        Returns (current_status, display_message).
        """
        if self.is_recording:
            self.recording_frames_total += 1
            if not evaluation.is_acceptable:
                self.recording_warning_frames += 1
                if self.recording_warning_frames >= self.max_warning_frames_during_rec:
                    self.had_framing_violation = True
                return FramingStatus.RECORDING_WARNING, "WARNING: Hands out of frame!"
            return FramingStatus.RECORDING, "RECORDING"

        # Not recording: manage readiness
        if evaluation.is_acceptable:
            self.consecutive_stable_frames += 1
            if self.consecutive_stable_frames >= self.required_stable_frames:
                return FramingStatus.READY_TO_RECORD, "Ready to record - Press SPACE to start"
            remaining = self.required_stable_frames - self.consecutive_stable_frames
            return FramingStatus.STABILIZING, f"Hold position... ({remaining})"
        else:
            self.consecutive_stable_frames = 0
            return evaluation.status, evaluation.user_message


# ─────────────────────────── HUD Rendering Helper ───────────────────────────

def draw_framing_hud(
    frame: np.ndarray,
    status: FramingStatus,
    message: str,
    target_word: str = "",
    countdown_secs: Optional[int] = None,
    recording_secs_left: Optional[int] = None,
    show_signing_box: bool = True
) -> np.ndarray:
    """
    Renders on-screen guidance, signing boundary box, and status headers
    on the mirrored display frame.
    """
    h, w = frame.shape[:2]

    # Color scheme (BGR)
    COLOR_RED = (40, 40, 230)
    COLOR_GREEN = (40, 220, 80)
    COLOR_YELLOW = (40, 215, 235)
    COLOR_ORANGE = (30, 140, 240)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (180, 180, 180)

    # 1. Signing Guide Box (recommended active signing volume)
    if show_signing_box:
        box_x1 = int(w * 0.15)
        box_x2 = int(w * 0.85)
        box_y1 = int(h * 0.15)
        box_y2 = int(h * 0.90)

        box_color = COLOR_GREEN if status == FramingStatus.READY_TO_RECORD else (
            COLOR_RED if status == FramingStatus.RECORDING_WARNING else COLOR_YELLOW
        )
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "SIGNING REGION (Keep hands inside)", (box_x1 + 8, box_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # 2. Top Header Bar (dark translucent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Target Word Label
    if target_word:
        cv2.putText(frame, f"Sign: {target_word.upper()}", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Status Pill on Top-Right
    if status == FramingStatus.READY_TO_RECORD:
        pill_color = COLOR_GREEN
        status_text = "READY"
    elif status == FramingStatus.RECORDING:
        pill_color = COLOR_RED
        status_text = f"REC {recording_secs_left}s" if recording_secs_left is not None else "REC"
    elif status == FramingStatus.RECORDING_WARNING:
        pill_color = COLOR_RED
        status_text = "REC (HANDS OUT)"
    elif status == FramingStatus.STABILIZING:
        pill_color = COLOR_YELLOW
        status_text = "STABILIZING"
    else:
        pill_color = COLOR_ORANGE
        status_text = "CHECK POSITION"

    tw = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
    px = w - tw - 36
    cv2.rectangle(frame, (px - 10, 10), (w - 14, 44), pill_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (px, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # 3. Bottom Status / Message Bar
    cv2.rectangle(overlay, (0, h - 46), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    msg_color = COLOR_RED if "WARNING" in message else (
        COLOR_GREEN if "Ready" in message else COLOR_WHITE
    )
    cv2.putText(frame, message, (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, msg_color, 2, cv2.LINE_AA)

    # 4. Countdown Overlay (Centered)
    if countdown_secs is not None and countdown_secs > 0:
        cd_text = str(countdown_secs)
        font_scale = 3.5
        thick = 5
        cd_w, cd_h = cv2.getTextSize(cd_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)[0]
        cx = (w - cd_w) // 2
        cy = (h + cd_h) // 2
        cv2.putText(frame, cd_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
        cv2.putText(frame, cd_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_GREEN, thick, cv2.LINE_AA)

    return frame

# -*- coding: utf-8 -*-
"""
SignSpeak AI - Live Stream Recognition Adapter
==============================================
Encapsulates real-time video frame processing, MediaPipe landmark extraction,
gesture boundary segmentation, DTW exemplar matching, gloss sequence buffering,
and sentence translation into a reusable session adapter for WebSocket and live streaming.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional
import cv2
import mediapipe as mp
import numpy as np

from src.recognition.pipeline import ExemplarRecognitionEngine
from src.recognition.gloss_buffer import GlossSequenceBuffer, canonicalize_gloss
from src.translation.model_translator import ConfigurableISLTranslator


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Constants (calibrated for real-time web streams at ~12-15 FPS)
# ─────────────────────────────────────────────────────────────────────────────
_STATE_IDLE = "IDLE"
_STATE_ACTIVE = "ACTIVE"
_STATE_COOLDOWN = "COOLDOWN"

# Web-stream tuned thresholds (~12-15 FPS):
DEFAULT_ONSET_THRESHOLD = 2    # ~0.15s of active hands to confirm sign start
DEFAULT_OFFSET_THRESHOLD = 3   # ~0.25s without hands to confirm sign end
DEFAULT_MIN_GESTURE_LEN = 4    # discard gestures shorter than ~0.3s
DEFAULT_MAX_GESTURE_LEN = 45   # hard cap (~3.0 sec) safety limit to prevent runaway gestures
DEFAULT_COOLDOWN_THRESHOLD = 1 # ~0.08s drain before DTW evaluation
DEFAULT_STILLNESS_THRESHOLD = 0.02 # motion energy threshold for stillness
DEFAULT_PAUSE_THRESHOLD = 3    # consecutive still frames to detect stroke completion
DEFAULT_REST_THRESHOLD = 4     # consecutive resting frames (low zone + stillness)


def extract_landmarks_array(landmark_list, count: int) -> np.ndarray:
    """Extract flattened coordinates or zeros — identical to extract_dataset.py."""
    if landmark_list is None:
        return np.zeros(count * 3, dtype=np.float32)
    points = np.array(
        [[getattr(p, "x", 0.0), getattr(p, "y", 0.0), getattr(p, "z", 0.0)] for p in landmark_list.landmark],
        dtype=np.float32
    )
    if len(points) != count:
        return np.zeros(count * 3, dtype=np.float32)
    return points.flatten()


def format_landmarks_for_ui(landmark_list) -> Optional[List[Dict[str, float]]]:
    """Converts a MediaPipe landmark list to lightweight UI coordinate dictionaries."""
    if landmark_list is None:
        return None
    return [{"x": float(p.x), "y": float(p.y)} for p in landmark_list.landmark]


def extract_hand_motion_points(hand_landmarks, pose_landmarks, is_left: bool) -> List[tuple]:
    """
    Extracts key articulated landmarks for multi-point motion tracking.
    When hand landmarks are available, extracts wrist (0) and fingertips:
    thumb (4), index (8), middle (12), ring (16), pinky (20).
    Falls back to pose wrist (15/16) and index (19/20) if hand landmarks are unavailable.
    """
    pts = []
    if hand_landmarks is not None and getattr(hand_landmarks, "landmark", None):
        lm = hand_landmarks.landmark
        key_indices = [0, 4, 8, 12, 16, 20]
        for idx in key_indices:
            if idx < len(lm):
                pts.append((float(lm[idx].x), float(lm[idx].y)))
        if pts:
            return pts

    # Fallback to pose wrist and index
    if pose_landmarks is not None and getattr(pose_landmarks, "landmark", None):
        p = pose_landmarks.landmark
        w_idx = 15 if is_left else 16
        i_idx = 19 if is_left else 20
        if len(p) > max(w_idx, i_idx):
            w = p[w_idx]
            if getattr(w, "visibility", 1.0) > 0.3:
                pts.append((float(w.x), float(w.y)))
            idx_lm = p[i_idx]
            if getattr(idx_lm, "visibility", 1.0) > 0.3:
                pts.append((float(idx_lm.x), float(idx_lm.y)))
    return pts


class PostureResult(tuple):
    """
    4-element tuple for backward compatibility with unpack patterns:
        curr_lw, curr_rw, motion_energy, is_resting = posture
    Also exposes direct properties:
        posture.curr_lw, posture.curr_rw, posture.motion_energy,
        posture.is_resting, posture.in_low_zone, posture.curr_l_pts, posture.curr_r_pts
    """
    def __new__(
        cls,
        curr_lw: Optional[tuple],
        curr_rw: Optional[tuple],
        motion_energy: float,
        is_resting: bool,
        in_low_zone: bool = False,
        curr_l_pts: Optional[List[tuple]] = None,
        curr_r_pts: Optional[List[tuple]] = None,
    ):
        return super().__new__(cls, (curr_lw, curr_rw, motion_energy, is_resting))

    def __init__(
        self,
        curr_lw: Optional[tuple],
        curr_rw: Optional[tuple],
        motion_energy: float,
        is_resting: bool,
        in_low_zone: bool = False,
        curr_l_pts: Optional[List[tuple]] = None,
        curr_r_pts: Optional[List[tuple]] = None,
    ):
        self.curr_lw = curr_lw
        self.curr_rw = curr_rw
        self.motion_energy = motion_energy
        self.is_resting = is_resting
        self.in_low_zone = in_low_zone
        self.curr_l_pts = curr_l_pts or []
        self.curr_r_pts = curr_r_pts or []


def analyze_frame_posture(
    pose_landmarks,
    left_hand_landmarks,
    right_hand_landmarks,
    prev_left_wrist: Optional[Any] = None,
    prev_right_wrist: Optional[Any] = None,
    stillness_threshold: float = DEFAULT_STILLNESS_THRESHOLD,
    prev_left_pts: Optional[List[tuple]] = None,
    prev_right_pts: Optional[List[tuple]] = None,
) -> PostureResult:
    """
    Extracts wrist and hand positions, computes frame-to-frame motion energy across
    articulated hand landmarks and wrists, and evaluates resting posture.

    CRITICAL RULE:
    Low wrist position alone (wrist_y > shoulder_y + 0.32 or wrist_y > 0.80) does NOT
    mean the signer is resting. Legitimate ISL signs performed below chest level remain
    active as long as there is movement. Rest is detected ONLY when hands are in the
    lower frame AND exhibit sustained stillness (motion_energy < stillness_threshold).

    Returns:
        PostureResult: (curr_lw, curr_rw, motion_energy, is_resting)
    """
    lw = None
    rw = None
    shoulder_y = None

    if pose_landmarks is not None and getattr(pose_landmarks, "landmark", None):
        p = pose_landmarks.landmark
        if len(p) > 16:
            ls = p[11]
            rs = p[12]
            ls_vis = getattr(ls, "visibility", 1.0)
            rs_vis = getattr(rs, "visibility", 1.0)
            if ls_vis > 0.3 and rs_vis > 0.3:
                shoulder_y = min(ls.y, rs.y)
            elif ls_vis > 0.3:
                shoulder_y = ls.y
            elif rs_vis > 0.3:
                shoulder_y = rs.y

            lw_p = p[15]
            rw_p = p[16]
            if getattr(lw_p, "visibility", 1.0) > 0.3:
                lw = (float(lw_p.x), float(lw_p.y))
            if getattr(rw_p, "visibility", 1.0) > 0.3:
                rw = (float(rw_p.x), float(rw_p.y))

    if left_hand_landmarks is not None and getattr(left_hand_landmarks, "landmark", None):
        lm = left_hand_landmarks.landmark
        if len(lm) > 0:
            lw = (float(lm[0].x), float(lm[0].y))

    if right_hand_landmarks is not None and getattr(right_hand_landmarks, "landmark", None):
        lm = right_hand_landmarks.landmark
        if len(lm) > 0:
            rw = (float(lm[0].x), float(lm[0].y))

    # Extract multi-point articulation keypoints
    curr_l_pts = extract_hand_motion_points(left_hand_landmarks, pose_landmarks, is_left=True)
    curr_r_pts = extract_hand_motion_points(right_hand_landmarks, pose_landmarks, is_left=False)

    # Frame-to-frame displacement calculation
    displacements = []

    # 1. Direct wrist displacement
    if lw is not None and prev_left_wrist is not None and isinstance(prev_left_wrist, (tuple, list)) and len(prev_left_wrist) >= 2:
        if isinstance(prev_left_wrist[0], (int, float)):
            displacements.append(float(np.hypot(lw[0] - prev_left_wrist[0], lw[1] - prev_left_wrist[1])))
    if rw is not None and prev_right_wrist is not None and isinstance(prev_right_wrist, (tuple, list)) and len(prev_right_wrist) >= 2:
        if isinstance(prev_right_wrist[0], (int, float)):
            displacements.append(float(np.hypot(rw[0] - prev_right_wrist[0], rw[1] - prev_right_wrist[1])))

    # 2. Multi-point finger articulation displacement (pairs up to min available keypoints)
    if curr_l_pts and prev_left_pts:
        for c_pt, p_pt in zip(curr_l_pts, prev_left_pts):
            displacements.append(float(np.hypot(c_pt[0] - p_pt[0], c_pt[1] - p_pt[1])))
    if curr_r_pts and prev_right_pts:
        for c_pt, p_pt in zip(curr_r_pts, prev_right_pts):
            displacements.append(float(np.hypot(c_pt[0] - p_pt[0], c_pt[1] - p_pt[1])))

    if displacements:
        motion_energy = float(np.max(displacements))
    else:
        motion_energy = 0.0

    # Low zone check: hands lowered below chest level (shoulder_y + 0.32) or absolute y > 0.80
    ref_y = (shoulder_y + 0.32) if shoulder_y is not None else 0.80
    present_wrists = []
    if lw is not None:
        present_wrists.append(lw[1] > ref_y or lw[1] > 0.80)
    if rw is not None:
        present_wrists.append(rw[1] > ref_y or rw[1] > 0.80)

    in_low_zone = bool(present_wrists and all(present_wrists))

    # RESTING DEFINITION:
    # A hand in the low zone is resting ONLY IF motion is below the stillness threshold.
    # If hands are moving in the lower frame, is_resting is False (actively signing).
    is_resting = in_low_zone and (motion_energy < stillness_threshold)

    return PostureResult(
        curr_lw=lw,
        curr_rw=rw,
        motion_energy=motion_energy,
        is_resting=is_resting,
        in_low_zone=in_low_zone,
        curr_l_pts=curr_l_pts,
        curr_r_pts=curr_r_pts,
    )


class LiveWebcamRecognitionSession:
    """
    Stateful session processing a stream of video frames from a single webcam connection.
    Extracts landmarks, tracks gesture segmentation, triggers DTW exemplar matching,
    accumulates gloss sequences, and emits finalized translations.
    """

    def __init__(
        self,
        engine: ExemplarRecognitionEngine,
        translator: Optional[Any] = None,
        idle_timeout_seconds: float = 1.5,
        min_confidence: float = 0.50,
        onset_threshold: int = DEFAULT_ONSET_THRESHOLD,
        offset_threshold: int = DEFAULT_OFFSET_THRESHOLD,
        min_gesture_len: int = DEFAULT_MIN_GESTURE_LEN,
        max_gesture_len: int = DEFAULT_MAX_GESTURE_LEN,
        cooldown_threshold: int = DEFAULT_COOLDOWN_THRESHOLD,
        stillness_threshold: float = DEFAULT_STILLNESS_THRESHOLD,
        pause_threshold: int = DEFAULT_PAUSE_THRESHOLD,
        rest_threshold: int = DEFAULT_REST_THRESHOLD,
    ):
        self.engine = engine
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_gesture_len = min_gesture_len
        self.max_gesture_len = max_gesture_len
        self.cooldown_threshold = cooldown_threshold
        self.stillness_threshold = stillness_threshold
        self.pause_threshold = pause_threshold
        self.rest_threshold = rest_threshold

        self.gloss_buffer = GlossSequenceBuffer(
            idle_timeout_seconds=idle_timeout_seconds,
            min_confidence=min_confidence
        )
        self.translator = translator if translator is not None else ConfigurableISLTranslator()

        # Segmentation state machine
        self.seg_state = _STATE_IDLE
        self.onset_counter = 0
        self.offset_counter = 0
        self.pause_counter = 0
        self.rest_counter = 0
        self.cooldown_counter = 0
        self.gesture_feats: List[np.ndarray] = []
        self.gesture_pres: List[np.ndarray] = []

        # Motion & posture tracking
        self._prev_lw: Optional[tuple] = None
        self._prev_rw: Optional[tuple] = None
        self._prev_l_pts: Optional[List[tuple]] = None
        self._prev_r_pts: Optional[List[tuple]] = None
        self.smoothed_motion: float = 0.0

        # Persistent Sequence & Translation state
        self.completed_sequence: List[str] = []
        self.completed_translation: Optional[str] = None
        self.in_progress_sequence: List[str] = []
        self.in_progress_translation: Optional[str] = None
        self.displayed_sequence: List[str] = []
        self.displayed_translation: Optional[str] = None
        self.is_finalized: bool = False

        # Continuous signing kinematic history (rolling motion window)
        self.motion_history: List[float] = []

        # Session metrics
        self.frame_count = 0

        # Latest predictions and outputs
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_finalized: Optional[Dict[str, Any]] = None

        # Diagnostics & Rejection tracking
        self.reached_matcher: bool = False
        self.last_rejection_reason: Optional[str] = None
        self.hand_source: str = "none"

        # Recognition control (Spacebar pause / resume)
        self.is_paused: bool = False

        # Live in-flight candidate tracking
        self.active_candidate: Optional[Dict[str, Any]] = None

        # Robustness & Diagnostic metrics
        self.missed_hands_count: int = 0
        self.rejected_segments_count: int = 0
        self.last_latency_ms: float = 0.0
        self._safety_cutoff_triggered: bool = False

        # MediaPipe Holistic instance for this session
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._is_closed = False

    def set_paused(self, paused: bool) -> bool:
        """
        Pauses or resumes gesture recognition.
        When paused: camera frames still stream landmarks for UI rendering,
        but sign accumulation and sequence finalization are suspended.
        When resumed: keeps existing recognized sentences intact without clobbering text.
        """
        self.is_paused = bool(paused)
        # Reset current partial gesture accumulation
        self.seg_state = _STATE_IDLE
        self.onset_counter = 0
        self.offset_counter = 0
        self.pause_counter = 0
        self.rest_counter = 0
        self.cooldown_counter = 0
        self.gesture_feats.clear()
        self.gesture_pres.clear()
        self.active_candidate = None

        # PRESERVE all recognized sentences! Do NOT clear sequences on pause/resume.
        if not self.is_paused:
            self.gloss_buffer.record_activity(time.time())
        return self.is_paused

    def mark_gesture_boundary(self) -> Dict[str, Any]:
        """
        Signals an explicit gesture boundary, preparing for the next chained sign
        without dropping to idle or resting.
        """
        self.gloss_buffer.mark_gesture_boundary()
        self.gesture_feats.clear()
        self.gesture_pres.clear()
        self.seg_state = _STATE_ACTIVE
        self.active_candidate = None
        self._safety_cutoff_triggered = False
        return {"state": self.seg_state}

    def pop_last_gloss(self) -> Dict[str, Any]:
        """
        Removes the last accepted gloss from the current sequence buffer and
        updates the displayed sentence. Safely does nothing if the buffer is empty.
        """
        removed = self.gloss_buffer.remove_last_gloss()
        removed_label = removed.gloss if removed else None

        if not removed_label and self.in_progress_sequence:
            removed_label = self.in_progress_sequence.pop()

        if removed_label:
            if removed is not None:
                self.in_progress_sequence = list(self.gloss_buffer.current_sequence)
            self.displayed_sequence = list(self.in_progress_sequence)
            if self.displayed_sequence:
                self.displayed_translation = self.translator.translate(self.displayed_sequence)
                self.in_progress_translation = self.displayed_translation
            else:
                self.displayed_translation = None
                self.in_progress_translation = None
        elif self.displayed_sequence or self.completed_sequence:
            # Sequence was finalized or in displayed_sequence: pop from displayed/completed sequence
            seq_to_pop = self.displayed_sequence if self.displayed_sequence else self.completed_sequence
            removed_label = seq_to_pop.pop()
            if self.completed_sequence:
                self.completed_sequence = list(seq_to_pop)
            self.displayed_sequence = list(seq_to_pop)
            if self.displayed_sequence:
                self.displayed_translation = self.translator.translate(self.displayed_sequence)
                self.completed_translation = self.displayed_translation
                if self.last_finalized:
                    self.last_finalized["sequence"] = list(self.displayed_sequence)
                    self.last_finalized["translation"] = self.displayed_translation
            else:
                self.displayed_translation = None
                self.completed_translation = None
                self.last_finalized = None
                self.is_finalized = False

        return {
            "removed_gloss": removed_label,
            "current_glosses": list(self.displayed_sequence),
            "current_translation": self.displayed_translation,
            "is_finalized": self.is_finalized,
            "completed_sequence": list(self.completed_sequence),
            "completed_translation": self.completed_translation,
            "in_progress_sequence": list(self.in_progress_sequence),
            "in_progress_translation": self.in_progress_translation,
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a single BGR camera frame.

        Args:
            frame_bgr: BGR frame from webcam.
            timestamp: Injected timestamp or time.time().

        Returns:
            Dict containing state, detected landmarks, active sequence,
            and any new prediction or finalized translation.
        """
        if self._is_closed:
            raise RuntimeError("Recognition session is closed.")
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Invalid or empty image frame.")

        self.frame_count += 1
        t = timestamp if timestamp is not None else time.time()
        t_start = time.perf_counter()

        # Downscale if wider than 960px to preserve performance
        h, w = frame_bgr.shape[:2]
        if w > 960:
            proc_w = 960
            proc_h = int(h * 960 / w)
            proc_frame = cv2.resize(frame_bgr, (proc_w, proc_h))
        else:
            proc_frame = frame_bgr

        # Landmark extraction on UNFLIPPED frame (geometry matching templates)
        try:
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            results = self._holistic.process(rgb)
        except Exception as exc:
            print(f"[LiveRecognitionSession] MediaPipe error: {exc}")
            results = None

        if results is None:
            has_pose = False
            has_face = False
            has_lh = False
            has_rh = False
            hands_detected = False
            hand_source = "none"
            row = np.zeros(1629, dtype=np.float32)
            pres_row = np.zeros(4, dtype=bool)
            ui_landmarks = {
                "left_hand": None,
                "right_hand": None,
                "pose": None,
                "has_pose": False,
                "has_face": False,
            }
        else:
            has_pose = results.pose_landmarks is not None
            has_face = results.face_landmarks is not None
            has_lh = results.left_hand_landmarks is not None
            has_rh = results.right_hand_landmarks is not None
            pres_row = np.array([has_pose, has_face, has_lh, has_rh], dtype=bool)

            row = np.concatenate([
                extract_landmarks_array(results.pose_landmarks, 33),
                extract_landmarks_array(results.face_landmarks, 468),
                extract_landmarks_array(results.left_hand_landmarks, 21),
                extract_landmarks_array(results.right_hand_landmarks, 21),
            ], dtype=np.float32)

            ui_landmarks = {
                "left_hand": format_landmarks_for_ui(results.left_hand_landmarks),
                "right_hand": format_landmarks_for_ui(results.right_hand_landmarks),
                "pose": format_landmarks_for_ui(results.pose_landmarks),
                "has_pose": has_pose,
                "has_face": has_face,
            }

        # Analyze motion energy and rest zone posture
        posture = analyze_frame_posture(
            results.pose_landmarks if results else None,
            results.left_hand_landmarks if results else None,
            results.right_hand_landmarks if results else None,
            prev_left_wrist=self._prev_lw,
            prev_right_wrist=self._prev_rw,
            stillness_threshold=self.stillness_threshold,
            prev_left_pts=self._prev_l_pts,
            prev_right_pts=self._prev_r_pts,
        )
        curr_lw, curr_rw, raw_motion, in_rest_zone = posture
        self._prev_lw = curr_lw
        self._prev_rw = curr_rw
        self._prev_l_pts = posture.curr_l_pts
        self._prev_r_pts = posture.curr_r_pts

        # Apply exponential temporal smoothing to motion energy
        self.smoothed_motion = 0.70 * raw_motion + 0.30 * self.smoothed_motion
        motion_energy = self.smoothed_motion

        # Track motion velocity history for continuous signing inflection detection
        self.motion_history.append(motion_energy)
        if len(self.motion_history) > 8:
            self.motion_history.pop(0)

        # Determine hand detection status and source:
        # Priority 1: Hand landmarks directly detected by MediaPipe
        if has_lh or has_rh:
            hands_detected = True
            hand_source = "hand_landmarks"
        elif has_pose and results and results.pose_landmarks:
            # Priority 2: Fallback to pose wrists if visible
            pose_wrists_visible = False
            plm = getattr(results.pose_landmarks, "landmark", [])
            if len(plm) > 16:
                lw_vis = getattr(plm[15], "visibility", 0.0)
                rw_vis = getattr(plm[16], "visibility", 0.0)
                if lw_vis > 0.5 or rw_vis > 0.5:
                    pose_wrists_visible = True

            if pose_wrists_visible:
                hand_source = "pose_wrists_only"
                # Moving wrists indicate active signing even if MediaPipe hand detector misses finger crops
                hands_detected = bool(raw_motion >= self.stillness_threshold or motion_energy >= self.stillness_threshold)
            else:
                hand_source = "none"
                hands_detected = False
        else:
            hand_source = "none"
            hands_detected = False

        self.hand_source = hand_source

        # If recognition is paused, skip segmentation, matching, and buffering while streaming landmarks
        if self.is_paused:
            self.last_latency_ms = (time.perf_counter() - t_start) * 1000
            diagnostics = {
                "frame_count": self.frame_count,
                "is_paused": True,
                "has_pose": has_pose,
                "has_face": has_face,
                "has_lh": has_lh,
                "has_rh": has_rh,
                "hand_source": self.hand_source,
                "reached_matcher": False,
                "rejection_reason": None,
                "in_rest_zone": posture.is_resting,
                "in_low_zone": posture.in_low_zone,
                "motion_energy": round(motion_energy, 4),
                "latency_ms": round(self.last_latency_ms, 2),
                "missed_hands_count": self.missed_hands_count,
                "rejected_segments_count": self.rejected_segments_count,
                "active_candidate": None,
                "gesture_frames": 0,
            }
            return {
                "type": "frame_result",
                "frame_count": self.frame_count,
                "state": "PAUSED",
                "is_paused": True,
                "hands_detected": hands_detected,
                "hand_source": self.hand_source,
                "reached_matcher": False,
                "rejection_reason": None,
                "in_rest_zone": posture.is_resting,
                "in_low_zone": posture.in_low_zone,
                "motion_energy": round(motion_energy, 4),
                "gesture_frames": 0,
                "landmarks": ui_landmarks,
                "diagnostics": diagnostics,
                "new_prediction": None,
                "active_candidate": None,
                "latest_prediction": self.last_prediction,
                "current_glosses": self.displayed_sequence,
                "current_translation": self.displayed_translation,
                "is_finalized": self.is_finalized,
                "completed_sequence": list(self.completed_sequence),
                "completed_translation": self.completed_translation,
                "in_progress_sequence": list(self.in_progress_sequence),
                "in_progress_translation": self.in_progress_translation,
                "finalized_sequence": self.last_finalized["sequence"] if self.last_finalized else (list(self.completed_sequence) if self.completed_sequence else None),
                "finalized_translation": self.last_finalized["translation"] if self.last_finalized else self.completed_translation,
            }

        new_prediction = None
        emitted_seq = None
        finalized_translation = None

        # ── Multi-Condition Gesture Segmentation State Machine ────────────────
        if self.seg_state == _STATE_IDLE:
            # Gesture onset: hands detected and not resting (actively gesturing or in upper zone)
            # Low hands that are actively moving can begin a valid gesture.
            if hands_detected and not posture.is_resting:
                self.onset_counter += 1
                if self.onset_counter >= self.onset_threshold:
                    self.seg_state = _STATE_ACTIVE
                    self.onset_counter = 0
                    self.offset_counter = 0
                    self.pause_counter = 0
                    self.rest_counter = 0
                    self.gesture_feats = []
                    self.gesture_pres = []
                    self.gloss_buffer.mark_gesture_boundary()
            else:
                self.onset_counter = max(0, self.onset_counter - 1)

        elif self.seg_state == _STATE_ACTIVE:
            self.gesture_feats.append(row)
            self.gesture_pres.append(pres_row)
            n_frames = len(self.gesture_feats)

            trigger_end = False

            # Condition 1: Hands completely dropped / absent
            if not hands_detected:
                self.missed_hands_count += 1
                self.offset_counter += 1
                if self.offset_counter >= self.offset_threshold:
                    trigger_end = True
            else:
                self.offset_counter = 0

            # Live candidate tracking during active gesture (throttled every 5 frames)
            if n_frames >= 5 and n_frames % 5 == 0 and self.engine.gallery_loaded:
                try:
                    curr_feats = np.stack(self.gesture_feats)
                    curr_pres = np.stack(self.gesture_pres)
                    quick_rec = self.engine.recognize_features(curr_feats, curr_pres, top_k=1)
                    if quick_rec.get("candidates"):
                        top = quick_rec["candidates"][0]
                        self.active_candidate = {
                            "label": top["label"],
                            "similarity": float(top["similarity"]),
                            "distance": float(top["distance"]),
                            "gesture_frames": n_frames,
                        }
                except Exception:
                    pass

            # Condition 2: Sustained resting posture (hands lowered AND still)
            # IMPORTANT: Low hand position alone does NOT terminate a gesture; signs performed below
            # chest level remain active as long as there is movement.
            if posture.is_resting and n_frames >= self.min_gesture_len:
                self.rest_counter += 1
                if self.rest_counter >= self.rest_threshold:
                    trigger_end = True
            else:
                self.rest_counter = 0

            # Condition 3: Motion stillness dip (apex hold or stroke completion)
            # Uses hysteresis: active motion drains pause_counter smoothly so micro-pauses
            # do not prematurely terminate gestures.
            if n_frames >= self.min_gesture_len:
                if motion_energy < self.stillness_threshold:
                    self.pause_counter += 1
                    if self.pause_counter >= self.pause_threshold:
                        trigger_end = True
                else:
                    self.pause_counter = max(0, self.pause_counter - 2)
            else:
                self.pause_counter = 0

            # Condition 4: Kinematic motion trough / velocity inflection (rapid continuous signing)
            # Triggers boundary when candidate confidence is high (>= 0.70) and motion inflects at trough
            if not trigger_end and n_frames >= (self.min_gesture_len + 3) and len(self.motion_history) >= 4:
                if self.active_candidate and self.active_candidate.get("similarity", 0.0) >= 0.70:
                    m_prev2 = self.motion_history[-3]
                    m_prev1 = self.motion_history[-2]
                    m_curr = self.motion_history[-1]
                    if (m_prev1 <= m_prev2 and m_curr > m_prev1 and m_prev1 < 0.035):
                        trigger_end = True

            # Condition 5: Max gesture duration cap (safety limit)
            if n_frames >= self.max_gesture_len:
                trigger_end = True
                self._safety_cutoff_triggered = True

            if trigger_end:
                self.seg_state = _STATE_COOLDOWN
                self.cooldown_counter = 0
                self.offset_counter = 0
                self.pause_counter = 0
                self.rest_counter = 0

        elif self.seg_state == _STATE_COOLDOWN:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_threshold:
                self.cooldown_counter = 0
                self.offset_counter = 0
                self.onset_counter = 0
                self.pause_counter = 0
                self.rest_counter = 0

                n_frames = len(self.gesture_feats)
                added = False
                if n_frames < self.min_gesture_len:
                    self.reached_matcher = False
                    self.last_rejection_reason = "short_duration"
                    self.rejected_segments_count += 1
                    self.active_candidate = None
                else:
                    self.reached_matcher = True
                    feats_arr = np.stack(self.gesture_feats)
                    pres_arr = np.stack(self.gesture_pres)

                    try:
                        rec_res = self.engine.recognize_features(
                            features=feats_arr,
                            presence=pres_arr,
                            query_name="live_stream",
                            top_k=3
                        )
                        candidates = rec_res.get("candidates", [])
                        if not candidates:
                            self.last_rejection_reason = "no_candidates"
                            self.rejected_segments_count += 1
                            self.active_candidate = None
                        else:
                            top_c = candidates[0]
                            label = top_c["label"]
                            conf = float(top_c["similarity"])
                            dist = float(top_c["distance"])

                            if conf < self.gloss_buffer.min_confidence:
                                self.last_rejection_reason = "low_confidence"
                                self.rejected_segments_count += 1
                                self.active_candidate = None
                            else:
                                # If previous sentence was finalized, archive it in completed_sequence and start new active sequence
                                if self.is_finalized:
                                    self.completed_sequence = list(self.displayed_sequence)
                                    self.completed_translation = self.displayed_translation
                                    self.gloss_buffer.reset()
                                    self.in_progress_sequence = []
                                    self.in_progress_translation = None
                                    self.displayed_sequence = []
                                    self.displayed_translation = None
                                    self.is_finalized = False
                                    self.last_finalized = None

                                added = self.gloss_buffer.add_gloss(label, confidence=conf, timestamp=t)
                                if added:
                                    self.last_rejection_reason = None
                                    self.in_progress_sequence = list(self.gloss_buffer.current_sequence)
                                    self.displayed_sequence = list(self.in_progress_sequence)
                                    self.displayed_translation = self.translator.translate(self.displayed_sequence)
                                    self.in_progress_translation = self.displayed_translation
                                else:
                                    self.last_rejection_reason = "duplicate_gloss"
                                    self.rejected_segments_count += 1

                                self.active_candidate = None

                                new_prediction = {
                                    "label": label,
                                    "confidence": conf,
                                    "distance": dist,
                                    "added_to_buffer": added,
                                    "candidates": [
                                        {"label": c["label"], "similarity": c["similarity"]}
                                        for c in candidates[:3]
                                    ]
                                }
                                self.last_prediction = new_prediction
                    except Exception as exc:
                        print(f"[LiveRecognitionSession] Matching error: {exc}")
                        self.last_rejection_reason = "matcher_error"
                        self.rejected_segments_count += 1
                        self.active_candidate = None

                # Continuous signing chaining:
                # If a valid gesture was recognized and added, and hands are STILL actively present
                # and moving in the signing zone (not dropped/resting), and safety cutoff was NOT triggered,
                # seamlessly transition into ACTIVE for the next sign immediately with zero dropped frames.
                if added and hands_detected and not posture.is_resting and not self._safety_cutoff_triggered:
                    self.seg_state = _STATE_ACTIVE
                    self.gloss_buffer.mark_gesture_boundary()
                    self.gesture_feats = [row]
                    self.gesture_pres = [pres_row]
                else:
                    self.seg_state = _STATE_IDLE
                    self.gesture_feats = []
                    self.gesture_pres = []

                self._safety_cutoff_triggered = False

        # Record activity: active while executing a gesture, or while hands are actively moving
        is_actively_signing = (self.seg_state in (_STATE_ACTIVE, _STATE_COOLDOWN)) or (
            hands_detected and not posture.is_resting
        )
        if is_actively_signing:
            self.gloss_buffer.record_activity(t)

        # Idle sequence finalization (when in IDLE and timeout elapsed)
        # Finalizes even if hands remain in the lower frame, provided they are resting/inactive!
        if self.seg_state == _STATE_IDLE and not self.is_finalized:
            emitted_seq = self.gloss_buffer.check_idle_timeout(t)
            if emitted_seq:
                finalized_translation = self.translator.translate(emitted_seq)
                self.is_finalized = True
                self.completed_sequence = list(emitted_seq)
                self.completed_translation = finalized_translation
                self.displayed_sequence = list(emitted_seq)
                self.displayed_translation = finalized_translation
                self.in_progress_sequence = []
                self.in_progress_translation = None
                self.last_finalized = {
                    "sequence": list(emitted_seq),
                    "translation": finalized_translation
                }

        self.last_latency_ms = (time.perf_counter() - t_start) * 1000

        diagnostics = {
            "frame_count": self.frame_count,
            "is_paused": self.is_paused,
            "has_pose": has_pose,
            "has_face": has_face,
            "has_lh": has_lh,
            "has_rh": has_rh,
            "hand_source": self.hand_source,
            "reached_matcher": self.reached_matcher,
            "rejection_reason": self.last_rejection_reason,
            "in_rest_zone": posture.is_resting,
            "in_low_zone": posture.in_low_zone,
            "motion_energy": round(motion_energy, 4),
            "onset_counter": self.onset_counter,
            "offset_counter": self.offset_counter,
            "pause_counter": self.pause_counter,
            "rest_counter": self.rest_counter,
            "gesture_frames": len(self.gesture_feats),
            "latency_ms": round(self.last_latency_ms, 2),
            "missed_hands_count": self.missed_hands_count,
            "rejected_segments_count": self.rejected_segments_count,
            "active_candidate": self.active_candidate,
        }

        return {
            "type": "frame_result",
            "frame_count": self.frame_count,
            "state": self.seg_state,
            "is_paused": self.is_paused,
            "hands_detected": hands_detected,
            "hand_source": self.hand_source,
            "reached_matcher": self.reached_matcher,
            "rejection_reason": self.last_rejection_reason,
            "in_rest_zone": posture.is_resting,
            "in_low_zone": posture.in_low_zone,
            "motion_energy": round(motion_energy, 4),
            "gesture_frames": len(self.gesture_feats),
            "landmarks": ui_landmarks,
            "diagnostics": diagnostics,
            "new_prediction": new_prediction,
            "active_candidate": self.active_candidate,
            "latest_prediction": self.last_prediction,
            "current_glosses": self.displayed_sequence,
            "current_translation": self.displayed_translation,
            "is_finalized": self.is_finalized,
            "completed_sequence": list(self.completed_sequence),
            "completed_translation": self.completed_translation,
            "in_progress_sequence": list(self.in_progress_sequence),
            "in_progress_translation": self.in_progress_translation,
            "finalized_sequence": self.last_finalized["sequence"] if self.last_finalized else (list(self.completed_sequence) if self.completed_sequence else None),
            "finalized_translation": self.last_finalized["translation"] if self.last_finalized else self.completed_translation,
        }

    def reset(self):
        """Resets active sequence buffer, state machine, and accumulated frames."""
        self.gloss_buffer.clear_history()
        self.seg_state = _STATE_IDLE
        self.onset_counter = 0
        self.offset_counter = 0
        self.pause_counter = 0
        self.rest_counter = 0
        self.cooldown_counter = 0
        self.gesture_feats.clear()
        self.gesture_pres.clear()
        self.last_prediction = None
        self.last_finalized = None
        self.reached_matcher = False
        self.last_rejection_reason = None
        self.hand_source = "none"
        self.in_progress_sequence.clear()
        self.in_progress_translation = None
        self.completed_sequence.clear()
        self.completed_translation = None
        self.displayed_sequence.clear()
        self.displayed_translation = None
        self.is_finalized = False
        self._prev_lw = None
        self._prev_rw = None
        self._prev_l_pts = None
        self._prev_r_pts = None
        self.smoothed_motion = 0.0
        self.motion_history.clear()
        self.active_candidate = None
        self.missed_hands_count = 0
        self.rejected_segments_count = 0
        self.last_latency_ms = 0.0
        self._safety_cutoff_triggered = False

    def close(self):
        """Closes MediaPipe Holistic model instance and releases resources."""
        if not self._is_closed:
            self._holistic.close()
            self._is_closed = True


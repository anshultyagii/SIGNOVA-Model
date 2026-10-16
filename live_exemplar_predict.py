"""
SignSpeak AI - Live Webcam 25-Class Exemplar Recognition Prototype
Captures live webcam feed, tracks pose and hand landmarks with MediaPipe Holistic,
segments the active gesture window (onset to offset), and runs Dynamic Time Warping
(DTW) recognition against the 25 enrolled dictionary templates.

Key fixes applied in this version:
  1. FLIP BUG FIXED: Display frame is mirrored for UX, but MediaPipe always
     processes the ORIGINAL (unflipped) camera frame — matching extract_dataset.py.
  2. GESTURE SEGMENTATION: Only frames from hand-raise to hand-lower are recognised.
     Rest frames no longer contaminate the DTW query sequence.
  3. SINGLE PREDICTION: One dominant sign result with confidence is shown.
  4. FAST DTW: Vectorised NumPy DTW in matcher.py reduces inference to ~100ms.
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import threading
import time
from typing import List, Optional
import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.pipeline import ExemplarRecognitionEngine
from src.recognition.gloss_buffer import GlossSequenceBuffer
from src.translation import ISLToEnglishTranslator

DEFAULT_GALLERY_DIR = PROJECT_ROOT / "dataset" / "processed_corrected"

# ─────────────────────────── Gesture Segmentation ───────────────────────────
# We define three states for the signing buffer state-machine:
#   IDLE     - hands not detected; wait for sign onset
#   ACTIVE   - hands detected; accumulate gesture frames
#   COOLDOWN - hands just disappeared; wait N frames before recognising
#              (prevents recognising a partial sign if hands briefly drop)

_STATE_IDLE     = "IDLE"
_STATE_ACTIVE   = "ACTIVE"
_STATE_COOLDOWN = "COOLDOWN"

ONSET_THRESHOLD  = 8    # frames with hands needed to confirm sign start
OFFSET_THRESHOLD = 18   # frames without hands needed to confirm sign end → trigger
MAX_GESTURE_LEN  = 120  # hard cap (~4 sec at 30 fps)
MIN_GESTURE_LEN  = 10   # discard twitches shorter than this

# ─────────────────────────── Landmark Extraction ────────────────────────────

def get_landmarks(landmark_list, count: int) -> np.ndarray:
    """Extract flattened coordinates or zeros — identical to extract_dataset.py."""
    if landmark_list is None:
        return np.zeros(count * 3, dtype=np.float32)
    points = np.array(
        [[p.x, p.y, p.z] for p in landmark_list.landmark],
        dtype=np.float32
    )
    if len(points) != count:
        return np.zeros(count * 3, dtype=np.float32)
    return points.flatten()

# ─────────────────────────── Async Predictor ────────────────────────────────

class AsyncPredictor:
    """
    Runs DTW recognition in a background thread so the main capture loop
    stays at full camera FPS. Because DTW is now ~100ms we rarely need to
    queue more than one job at a time.
    """

    def __init__(self, engine: ExemplarRecognitionEngine):
        self.engine = engine
        self.lock = threading.Lock()
        self.is_busy = False
        self.last_label = ""
        self.last_confidence = 0.0
        self.last_update_time = 0.0

    def submit(self, features: np.ndarray, presence: np.ndarray):
        """Submit a gesture segment for recognition. Dropped if a job is running."""
        with self.lock:
            if self.is_busy:
                return False
            self.is_busy = True

        def worker():
            try:
                res = self.engine.recognize_features(
                    features=features,
                    presence=presence,
                    query_name="webcam_live",
                    top_k=3          # always get top-3 for console diagnostics
                )
                candidates = res.get("candidates", [])

                # ── Diagnostic: print top-3 to console ──────────────────────
                # The UI only shows candidates[0]; this output helps diagnose
                # confusion without changing the displayed prediction.
                if candidates:
                    print(f"[Recognition] Top-3 results  (query {res.get('query_frames','?')}f):")
                    for c in candidates:
                        bar = '#' * int(c['similarity'] * 20)
                        print(f"   #{c['rank']}  {c['label']:<28}  "
                              f"sim={c['similarity']:.3f}  dist={c['distance']:.4f}  [{bar:<20}]")
                else:
                    print("[Recognition] No candidates returned.")

                with self.lock:
                    if candidates:
                        self.last_label      = candidates[0]["label"]
                        self.last_confidence = candidates[0]["similarity"]
                    else:
                        self.last_label      = "-"
                        self.last_confidence = 0.0
                    self.last_update_time = time.time()
            except Exception as exc:
                print(f"[AsyncPredictor] Recognition error: {exc}")
            finally:
                with self.lock:
                    self.is_busy = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def get_result(self):
        with self.lock:
            return self.last_label, self.last_confidence, self.is_busy, self.last_update_time

    def clear(self):
        with self.lock:
            self.last_label      = ""
            self.last_confidence = 0.0
            self.last_update_time = 0.0

# ─────────────────────────── HUD Rendering ──────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    state: str,
    label: str,
    confidence: float,
    is_busy: bool,
    gesture_frames: int,
    fps: float,
    gloss_sequence: Optional[List[str]] = None,
    last_translation: str = ""
) -> np.ndarray:
    """Draws a clean single-result HUD on the display frame."""
    h, w = frame.shape[:2]

    panel_w = min(400, int(w * 0.40))
    panel_h = 280
    panel_x = w - panel_w - 16
    panel_y = 16

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (70, 70, 88), 1)

    # Title
    cv2.putText(frame, "SignSpeak AI",
                (panel_x + 12, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 255), 2)
    cv2.putText(frame, "ISL 25-Class Prototype",
                (panel_x + 12, panel_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 175), 1)

    # Separator
    cv2.line(frame, (panel_x + 12, panel_y + 50),
             (panel_x + panel_w - 12, panel_y + 50), (55, 55, 68), 1)

    # State badge
    if state == _STATE_IDLE:
        badge_text  = "○  Waiting for sign..."
        badge_color = (100, 180, 255)
    elif state == _STATE_ACTIVE:
        badge_text  = f"●  Signing  [{gesture_frames}f]"
        badge_color = (0, 255, 130)
    elif state == _STATE_COOLDOWN:
        badge_text  = "◌  Finishing sign..."
        badge_color = (0, 230, 255)
    else:
        badge_text  = "..."
        badge_color = (180, 180, 180)

    cv2.putText(frame, badge_text,
                (panel_x + 12, panel_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, badge_color, 1)

    # Separator
    cv2.line(frame, (panel_x + 12, panel_y + 82),
             (panel_x + panel_w - 12, panel_y + 82), (55, 55, 68), 1)

    # Main prediction area
    if is_busy:
        cv2.putText(frame, "Matching...",
                    (panel_x + 12, panel_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2)
    elif label:
        # Large label
        font_scale = 0.9 if len(label) <= 14 else 0.65
        cv2.putText(frame, label,
                    (panel_x + 12, panel_y + 118),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 195), 2)
        # Confidence bar
        bar_x  = panel_x + 12
        bar_y  = panel_y + 128
        bar_mw = panel_w - 24
        bar_fw = max(4, int(bar_mw * confidence))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_mw, bar_y + 8),
                      (40, 40, 52), -1)
        # colour: green >80%, amber >55%, red below
        if confidence >= 0.80:
            bar_col = (0, 215, 90)
        elif confidence >= 0.55:
            bar_col = (0, 190, 255)
        else:
            bar_col = (80, 80, 200)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fw, bar_y + 8),
                      bar_col, -1)
        cv2.putText(frame, f"{confidence*100:.0f}% similarity",
                    (bar_x, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 190), 1)
    else:
        cv2.putText(frame, "—  No result yet",
                    (panel_x + 12, panel_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (110, 110, 125), 1)
        cv2.putText(frame, "Perform a clear isolated sign,",
                    (panel_x + 12, panel_y + 138),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (110, 110, 125), 1)
        cv2.putText(frame, "then lower your hands.",
                    (panel_x + 12, panel_y + 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (110, 110, 125), 1)

    # Gloss Sequence accumulator display
    cv2.line(frame, (panel_x + 12, panel_y + 170),
             (panel_x + panel_w - 12, panel_y + 170), (55, 55, 68), 1)
    if gloss_sequence:
        seq_str = " -> ".join(gloss_sequence)
        if len(seq_str) > 28:
            seq_str = "..." + seq_str[-25:]
        cv2.putText(frame, f"Seq: {seq_str}", (panel_x + 12, panel_y + 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Seq: (idle / waiting)", (panel_x + 12, panel_y + 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 135), 1, cv2.LINE_AA)

    # English Translation display
    if last_translation:
        trans_str = last_translation
        if len(trans_str) > 28:
            trans_str = trans_str[:25] + "..."
        cv2.putText(frame, f"Eng: {trans_str}", (panel_x + 12, panel_y + 212),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Eng: (waiting for pause)", (panel_x + 12, panel_y + 212),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 135), 1, cv2.LINE_AA)

    # Footer: FPS + controls
    cv2.line(frame, (panel_x + 12, panel_y + panel_h - 28),
             (panel_x + panel_w - 12, panel_y + panel_h - 28), (55, 55, 68), 1)
    cv2.putText(frame, f"{fps:.0f} FPS  |  [C] Clear  [SPC] Pause  [Q] Quit",
                (panel_x + 12, panel_y + panel_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 130, 145), 1)

    return frame


# ─────────────────────────── Main Loop ──────────────────────────────────────

def run_live_webcam(
    camera_index: int = 0,
    gallery_dir: Path = DEFAULT_GALLERY_DIR,
    smoke_test: bool = False
):
    print("=" * 72)
    print("SignSpeak AI — Live Webcam 25-Class Exemplar Recognition")
    print("=" * 72)
    print(f"Gallery Dir  : {gallery_dir}")
    print(f"Camera Index : {camera_index}")
    print("=" * 72)

    if not gallery_dir.exists():
        print(f"ERROR: Gallery directory not found: {gallery_dir}")
        sys.exit(1)

    # ── 1. Load gallery ──────────────────────────────────────────────────────
    print("\n[1/3] Loading 25-Class Reference Gallery...")
    engine = ExemplarRecognitionEngine(gallery_dir=gallery_dir, trim_to_active=True)
    labels = engine.get_gallery_labels()
    print(f"      {len(labels)} classes enrolled successfully.")
    print("      Classes: " + ", ".join(labels[:8]) + f"... (+{len(labels)-8} more)")

    # ── 2. Open webcam ───────────────────────────────────────────────────────
    print(f"\n[2/3] Connecting to webcam (index {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("\n" + "!" * 72)
        print(f"ERROR: Could not open camera at index {camera_index}.")
        print("Check that no other program is using the webcam (Zoom, Teams, etc.).")
        print("!" * 72)
        sys.exit(1)

    # Match dataset extraction: do NOT force a fixed resolution here.
    # extract_dataset.py resizes frames >960px wide but uses whatever the camera
    # delivers natively.  We apply the same 960px cap below.
    cam_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps <= 0 or cam_fps > 120:
        cam_fps = 25.0
    print(f"      Camera: {cam_w}x{cam_h} @ ~{cam_fps:.1f} FPS")

    # ── 3. Setup ─────────────────────────────────────────────────────────────
    predictor = AsyncPredictor(engine)
    mp_holistic = mp.solutions.holistic
    mp_draw     = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles

    window_title = "SignSpeak AI — Live Exemplar Recognition"
    if not smoke_test:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1150, 700)

    print("\n[3/3] Live recognition started.")
    print("      Perform ONE sign clearly, then lower your hands to trigger matching.")
    print("      Press 'q' to quit, 'c' to clear, SPACE to pause.\n")

    # State machine variables
    seg_state       = _STATE_IDLE     # current segmentation state
    onset_counter   = 0               # frames with hands seen in IDLE phase
    offset_counter  = 0               # frames without hands seen in ACTIVE/COOLDOWN
    gesture_feats   = []              # accumulated frames during ACTIVE segment
    gesture_pres    = []

    # Display/timing
    is_paused       = False
    frame_counter   = 0
    fps_estimate    = cam_fps
    # Remember latest results from the holistic call across pause states
    results         = None
    hands_detected  = False

    # Variables used only inside the with-block but needed for smoke test
    last_label      = ""
    last_conf       = 0.0
    is_busy         = False
    gloss_buffer    = GlossSequenceBuffer(idle_timeout_seconds=1.5, min_confidence=0.50)
    translator      = ISLToEnglishTranslator()
    last_translation = ""
    last_consumed_update_time = 0.0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            t0 = time.perf_counter()

            ret, raw_frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            # ── CRITICAL FIX: landmark extraction on original frame ───────────
            # extract_dataset.py never flips frames.  We must NOT flip before
            # MediaPipe so that left/right hand slots match the gallery templates.

            # Processing frame = raw (optionally shrunk to ≤960 wide)
            orig_h, orig_w = raw_frame.shape[:2]
            if orig_w > 960:
                proc_w = 960
                proc_h = int(orig_h * 960 / orig_w)
                proc_frame = cv2.resize(raw_frame, (proc_w, proc_h))
            else:
                proc_frame = raw_frame

            # draw_frame starts as proc_frame (same geometry MediaPipe processed);
            # we will flip it AFTER drawing landmarks so the overlay is exact.
            draw_frame = proc_frame.copy()

            if not is_paused:
                rgb     = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Extract 1629-D feature row — identical to extract_dataset.py
                row = np.concatenate([
                    get_landmarks(results.pose_landmarks,      33),
                    get_landmarks(results.face_landmarks,     468),
                    get_landmarks(results.left_hand_landmarks,  21),
                    get_landmarks(results.right_hand_landmarks, 21),
                ], dtype=np.float32)

                has_pose = results.pose_landmarks         is not None
                has_face = results.face_landmarks         is not None
                has_lh   = results.left_hand_landmarks    is not None
                has_rh   = results.right_hand_landmarks   is not None
                hands_detected = has_lh or has_rh

                pres_row = np.array([has_pose, has_face, has_lh, has_rh], dtype=bool)

                # ── Gesture segmentation state machine ────────────────────────
                if seg_state == _STATE_IDLE:
                    if hands_detected:
                        onset_counter += 1
                        if onset_counter >= ONSET_THRESHOLD:
                            # Confirmed gesture start
                            seg_state     = _STATE_ACTIVE
                            onset_counter = 0
                            gesture_feats = []
                            gesture_pres  = []
                            gloss_buffer.mark_gesture_boundary()
                            # Seed gesture buffer with the onset frames already seen
                            # (they have been discarded; we simply start fresh here)
                    else:
                        onset_counter = max(0, onset_counter - 1)

                elif seg_state == _STATE_ACTIVE:
                    gesture_feats.append(row)
                    gesture_pres.append(pres_row)

                    if not hands_detected:
                        offset_counter += 1
                        if offset_counter >= OFFSET_THRESHOLD:
                            seg_state      = _STATE_COOLDOWN
                            offset_counter = 0
                    else:
                        offset_counter = 0

                    # Hard cap: force recognition if sign is very long
                    if len(gesture_feats) >= MAX_GESTURE_LEN:
                        seg_state      = _STATE_COOLDOWN
                        offset_counter = OFFSET_THRESHOLD  # skip cooldown wait

                elif seg_state == _STATE_COOLDOWN:
                    # Wait for the cooldown period, then fire recognition
                    offset_counter += 1
                    if offset_counter >= 4:  # brief drain before recognising
                        seg_state      = _STATE_IDLE
                        offset_counter = 0
                        onset_counter  = 0

                        n_frames = len(gesture_feats)
                        if n_frames >= MIN_GESTURE_LEN:
                            feats_arr = np.stack(gesture_feats)
                            pres_arr  = np.stack(gesture_pres)
                            predictor.submit(feats_arr, pres_arr)
                            print(f"[Segmentation] Gesture submitted: {n_frames} frames.")
                        else:
                            print(f"[Segmentation] Discarded short segment: {n_frames} frames.")

                        gesture_feats = []
                        gesture_pres  = []

            # ── Draw landmarks on UNFLIPPED draw_frame, then flip for display ──
            # MediaPipe landmark coords are normalized [0,1] relative to proc_frame
            # (the unflipped frame).  mp_draw.draw_landmarks multiplies lm.x * W
            # and lm.y * H to get pixel positions.  Drawing must therefore happen
            # on a frame with the SAME geometry as proc_frame — otherwise the
            # x-coordinates are reflected and every landmark appears on the wrong
            # side of the hand.
            #
            # Fix: draw on draw_frame (a copy of proc_frame), then mirror the
            # whole annotated frame once → display_frame.  This way:
            #   • landmarks sit exactly where MediaPipe found them, and
            #   • the final view is still mirrored (natural webcam UX).

            if results is not None:
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        draw_frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )
                if results.left_hand_landmarks:
                    mp_draw.draw_landmarks(
                        draw_frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2)
                    )
                if results.right_hand_landmarks:
                    mp_draw.draw_landmarks(
                        draw_frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(60, 130, 255), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(40, 100, 230), thickness=2)
                    )

            # Flip the fully-annotated frame once → mirrored display (natural UX)
            display_frame = cv2.flip(draw_frame, 1)

            # ── Retrieve prediction and feed GlossSequenceBuffer ──────────────
            last_label, last_conf, is_busy, last_update_time = predictor.get_result()

            if last_update_time > last_consumed_update_time:
                if last_label and last_label != "-":
                    added = gloss_buffer.add_gloss(last_label, confidence=last_conf, timestamp=last_update_time)
                    if added:
                        print(f"[Gloss Buffer] Added '{last_label}' (conf={last_conf:.2f}) -> Active: {' -> '.join(gloss_buffer.current_sequence)}")
                last_consumed_update_time = last_update_time

            if hands_detected:
                gloss_buffer.record_activity()

            if seg_state == _STATE_IDLE and not is_paused and not is_busy:
                emitted_seq = gloss_buffer.check_idle_timeout()
                if emitted_seq:
                    seq_str = " -> ".join(emitted_seq)
                    last_translation = translator.translate(emitted_seq)
                    print(f"\n[Finalized Sequence]")
                    print(f"Gloss sequence: {seq_str}")
                    print(f"English output: {last_translation}\n")

            gesture_frame_count = len(gesture_feats)
            annotated = draw_hud(
                frame            = display_frame,
                state            = seg_state if not is_paused else "PAUSED",
                label            = last_label,
                confidence       = last_conf,
                is_busy          = is_busy,
                gesture_frames   = gesture_frame_count,
                fps              = fps_estimate,
                gloss_sequence   = gloss_buffer.current_sequence,
                last_translation = last_translation
            )

            # Show PAUSED overlay
            if is_paused:
                cv2.putText(annotated, "PAUSED",
                            (30, annotated.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3)

            # Smoke test exit
            if smoke_test:
                frame_counter += 1
                if frame_counter >= 15:
                    print(f"Smoke test: {frame_counter} frames processed. OK.")
                    break
                continue

            cv2.imshow(window_title, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                print("Session ended by user.")
                break
            elif key == ord('c'):
                gesture_feats    = []
                gesture_pres     = []
                seg_state        = _STATE_IDLE
                onset_counter    = 0
                offset_counter   = 0
                predictor.clear()
                gloss_buffer.reset()
                last_translation = ""
                print("[User] Buffer cleared.")
            elif key == ord(' '):
                is_paused = not is_paused
                if not is_paused:
                    gloss_buffer.record_activity()
                print(f"[User] {'PAUSED' if is_paused else 'RESUMED'}.")

            frame_counter += 1
            dt = time.perf_counter() - t0
            if dt > 0:
                fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / dt)

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Session complete.")


# ─────────────────────────── Entry Point ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SignSpeak AI — Live Webcam 25-Class Exemplar Recognition"
    )
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--gallery-dir",  type=str,
                        default=str(DEFAULT_GALLERY_DIR),
                        help="Directory containing reference .npz templates")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Non-interactive 15-frame test (CI/sanity)")
    args = parser.parse_args()

    run_live_webcam(
        camera_index = args.camera_index,
        gallery_dir  = Path(args.gallery_dir),
        smoke_test   = args.smoke_test
    )


if __name__ == "__main__":
    main()

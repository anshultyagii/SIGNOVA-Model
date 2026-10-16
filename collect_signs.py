# -*- coding: utf-8 -*-
"""
SignSpeak AI - Interactive Sign Language Data Collection with Real-Time Framing Validation
==========================================================================================
Records high-quality sign samples with real-time pose and wrist tracking to prevent
recordings with hands out of frame.

Features:
  - Live webcam preview with MediaPipe Holistic pose skeleton and hand landmarks.
  - Active signing region guide box.
  - On-screen framing status: "Position yourself in frame", "Hands not visible",
    "Hold position...", and "Ready to record".
  - Interactive on-screen sign selector:
      * 'n' / '+' / Right-Arrow : Next sign in vocabulary
      * 'p' / '-' / Left-Arrow  : Previous sign in vocabulary
      * Digits (0-9) + ENTER    : Jump directly to sign index (1-25)
  - Consecutive stable framing required before recording can be initiated (SPACE).
  - 3-second live on-screen countdown over camera feed.
  - Real-time framing violation detection during recording.
  - Post-recording quality verification (re-record option if hands dropped below frame).
  - Safe exit ('q' / ESC) and clean resource release.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recording.framing import (
    evaluate_pose_framing,
    FramingStateMachine,
    FramingStatus,
    draw_framing_hud
)

WORDS = [
    "One", "Two", "Three", "Four", "Five", "Six", "Seven",
    "Eight", "Nine", "Ten", "Drink", "Eat", "Food", "Give",
    "Go", "Help", "I_Me", "Milk", "My",
    "Want_Desire_Wish_For", "Water", "We", "What", "Where", "You"
]

BASE_DIR = PROJECT_ROOT / "dataset" / "my_recordings"
BASE_DIR.mkdir(parents=True, exist_ok=True)


def open_webcam(camera_index: int = 0, width: int = 1280, height: int = 720) -> Tuple[Optional[cv2.VideoCapture], bool, str]:
    """Safely initializes webcam with requested resolution."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None, False, (
            f"ERROR: Could not open webcam at index {camera_index}. "
            "Please check that your camera is connected and not in use by another program."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Webcam] Opened camera index {camera_index} ({actual_w}x{actual_h})")
    return cap, True, ""


def get_next_sample_index(folder: Path) -> int:
    """Finds the next unused sample index to prevent overwriting existing recordings."""
    existing_samples = list(folder.glob("sample_*.mp4"))
    max_idx = 0
    for s in existing_samples:
        parts = s.stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            idx = int(parts[-1])
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def resolve_initial_sign_index(sign_arg: Optional[str]) -> int:
    """Resolves optional sign name or number argument into a valid 0-based WORDS index."""
    if not sign_arg:
        return 0
    sign_str = str(sign_arg).strip()
    if sign_str.isdigit():
        val = int(sign_str)
        if 1 <= val <= len(WORDS):
            return val - 1
    sign_lower = sign_str.lower()
    for idx, w in enumerate(WORDS):
        if w.lower() == sign_lower:
            return idx
    print(f"[Recorder] Warning: '{sign_arg}' not found in vocabulary. Defaulting to '{WORDS[0]}'.")
    return 0


def run_interactive_recorder(
    cap: cv2.VideoCapture,
    holistic: Any,
    mp_draw: Any,
    initial_word: Optional[str] = None,
    single_session: bool = False,
    smoke_test: bool = False
) -> bool:
    """
    Unified OpenCV interactive event loop for webcam preview, framing validation,
    sign selection, 3s countdown, recording, review, and auto-cleanup.
    """
    sign_idx = resolve_initial_sign_index(initial_word)
    jump_buffer = ""
    last_saved_msg = ""
    last_saved_time = 0.0

    state_machine = FramingStateMachine(
        required_stable_frames=15,  # ~0.5s of stable framing before enabling recording
        max_warning_frames_during_rec=6
    )

    window_name = f"SignSpeakAI Recorder — {WORDS[sign_idx]}" if single_session else "SignSpeakAI Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Modes: PREVIEW -> COUNTDOWN -> RECORDING -> REVIEW
    current_mode = "PREVIEW"
    countdown_start_time = 0.0
    recording_start_time = 0.0
    video_writer = None
    output_path = None
    RECORDING_DURATION = 3.0  # seconds
    frame_counter = 0

    mp_holistic = mp.solutions.holistic
    mp_styles = mp.solutions.drawing_styles

    # Cache sample counts
    last_cached_sign_idx = -1
    sample_count = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            print("[Recorder] Camera feed ended or frame read failed.")
            break

        orig_h, orig_w = raw_frame.shape[:2]
        word = WORDS[sign_idx]

        if sign_idx != last_cached_sign_idx:
            folder = BASE_DIR / word
            folder.mkdir(parents=True, exist_ok=True)
            sample_count = len(list(folder.glob("sample_*.mp4")))
            last_cached_sign_idx = sign_idx
            if single_session:
                cv2.setWindowTitle(window_name, f"SignSpeakAI Recorder — {word}")

        # 1. MediaPipe Holistic processing on raw (unflipped) frame
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        has_lh = results.left_hand_landmarks is not None
        has_rh = results.right_hand_landmarks is not None

        # 2. Evaluate framing using pose and wrists
        eval_res = evaluate_pose_framing(
            pose_landmarks=results.pose_landmarks,
            has_left_hand=has_lh,
            has_right_hand=has_rh
        )

        status, status_message = state_machine.update(eval_res)

        # 3. Draw landmarks on draw_frame before flipping
        draw_frame = raw_frame.copy()
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

        # 4. Flip to mirrored display for natural UX
        display_frame = cv2.flip(draw_frame, 1)

        # 5. Handle Modes
        countdown_secs = None
        recording_secs_left = None

        if current_mode == "PREVIEW":
            if jump_buffer:
                status_message = f"Jump to sign #{jump_buffer}_  [ENTER to jump, ESC to cancel]"
            elif last_saved_msg and (time.time() - last_saved_time < 3.5):
                status_message = last_saved_msg
            elif status == FramingStatus.READY_TO_RECORD:
                status_message = "Ready! Press SPACE to start countdown"

        elif current_mode == "COUNTDOWN":
            elapsed = time.time() - countdown_start_time
            countdown_secs = max(1, 3 - int(elapsed))
            status_message = f"Starting in {countdown_secs}s... Prepare to sign '{word}'"

            if elapsed >= 3.0:
                folder = BASE_DIR / word
                folder.mkdir(parents=True, exist_ok=True)
                next_sample_idx = get_next_sample_index(folder)
                output_path = folder / f"sample_{next_sample_idx:03d}.mp4"

                video_writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25.0,
                    (orig_w, orig_h)
                )

                if not video_writer.isOpened():
                    print(f"ERROR: Could not initialize video writer for {output_path}")
                    break

                current_mode = "RECORDING"
                recording_start_time = time.time()
                state_machine.start_recording()
                print(f"[Recorder] Recording started -> {output_path.name}")

        elif current_mode == "RECORDING":
            rec_elapsed = time.time() - recording_start_time
            recording_secs_left = max(0, int(np.ceil(RECORDING_DURATION - rec_elapsed)))
            status = FramingStatus.RECORDING if eval_res.is_acceptable else FramingStatus.RECORDING_WARNING

            # Write clean camera frame without HUD burn-in
            if video_writer is not None:
                video_writer.write(raw_frame)

            if rec_elapsed >= RECORDING_DURATION:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None

                rec_summary = state_machine.stop_recording()
                print(f"[Recorder] Recorded {rec_summary['total_frames']} frames.")

                if not rec_summary["is_valid"]:
                    print(f"[Recorder] WARNING: Hands were out of frame for {rec_summary['warning_frames']} frames!")
                    current_mode = "REVIEW"
                else:
                    print(f"[Recorder] Successfully saved: {output_path.name}")
                    sample_count += 1
                    last_saved_msg = f"Saved {output_path.name}! SPACE: record again | [N] next sign"
                    last_saved_time = time.time()
                    if single_session:
                        cv2.destroyWindow(window_name)
                        return True
                    current_mode = "PREVIEW"
                    state_machine.reset_stability()

        # 6. Render HUD
        target_display = f"[{sign_idx + 1:02d}/{len(WORDS)}] {word} ({sample_count} recs)"
        hud_frame = draw_framing_hud(
            frame=display_frame,
            status=status,
            message=status_message,
            target_word=target_display,
            countdown_secs=countdown_secs,
            recording_secs_left=recording_secs_left,
            show_signing_box=True
        )

        # On-screen navigation hint
        if current_mode == "PREVIEW":
            cv2.putText(
                hud_frame,
                "Controls: [N] Next Sign  |  [P] Prev Sign  |  [0-9+Enter] Jump  |  [SPACE] Record  |  [Q] Exit",
                (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

        if current_mode == "REVIEW":
            h, w = hud_frame.shape[:2]
            cv2.rectangle(hud_frame, (w // 2 - 320, h // 2 - 60), (w // 2 + 320, h // 2 + 60), (30, 30, 30), -1)
            cv2.rectangle(hud_frame, (w // 2 - 320, h // 2 - 60), (w // 2 + 320, h // 2 + 60), (40, 40, 230), 2)
            cv2.putText(hud_frame, "WARNING: Hands were out of frame!", (w // 2 - 280, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 240), 2, cv2.LINE_AA)
            cv2.putText(hud_frame, "Press 'r' to re-record, or 'k' / SPACE to keep sample", (w // 2 - 280, h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Smoke test exit (headless/CI test)
        if smoke_test:
            frame_counter += 1
            if frame_counter >= 5:
                print(f"[Recorder] Smoke test: {frame_counter} frames processed successfully.")
                cv2.destroyWindow(window_name)
                return True
            continue

        cv2.imshow(window_name, hud_frame)

        # Window closed by clicking 'X'
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[Recorder] Preview window closed.")
                break
        except Exception:
            pass

        key = cv2.waitKey(1) & 0xFF

        # Keyboard Controls
        if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
            if current_mode == "PREVIEW" and jump_buffer:
                jump_buffer = ""  # cancel jump buffer first
            else:
                if video_writer is not None:
                    video_writer.release()
                    if output_path and output_path.exists():
                        output_path.unlink()
                print("[Recorder] Cancelled / Exited by user.")
                cv2.destroyWindow(window_name)
                return False

        if current_mode == "PREVIEW":
            if key == 32:  # SPACE
                if status == FramingStatus.READY_TO_RECORD:
                    current_mode = "COUNTDOWN"
                    countdown_start_time = time.time()
                    jump_buffer = ""
                else:
                    print(f"[Recorder] Cannot start yet: {status_message}")

            elif key in (ord("n"), ord("N"), ord("+"), ord("=")):
                if not single_session:
                    sign_idx = (sign_idx + 1) % len(WORDS)
                    state_machine.reset_stability()
                    last_saved_msg = ""
                    jump_buffer = ""
                    print(f"[Recorder] Switched to sign: {WORDS[sign_idx]}")

            elif key in (ord("p"), ord("P"), ord("-"), ord("_")):
                if not single_session:
                    sign_idx = (sign_idx - 1) % len(WORDS)
                    state_machine.reset_stability()
                    last_saved_msg = ""
                    jump_buffer = ""
                    print(f"[Recorder] Switched to sign: {WORDS[sign_idx]}")

            elif ord("0") <= key <= ord("9"):
                if not single_session and len(jump_buffer) < 2:
                    jump_buffer += chr(key)

            elif key in (13, 10):  # ENTER
                if jump_buffer and not single_session:
                    target_num = int(jump_buffer)
                    if 1 <= target_num <= len(WORDS):
                        sign_idx = target_num - 1
                        state_machine.reset_stability()
                        last_saved_msg = ""
                        print(f"[Recorder] Jumped to sign #{target_num}: {WORDS[sign_idx]}")
                    else:
                        print(f"[Recorder] Invalid sign #{target_num}. Range is 1-{len(WORDS)}.")
                    jump_buffer = ""

            elif key == 8:  # Backspace
                jump_buffer = jump_buffer[:-1]

        elif current_mode == "REVIEW":
            if key in (ord("r"), ord("R")):  # Re-record
                if output_path and output_path.exists():
                    output_path.unlink()
                print("[Recorder] Discarded flawed sample. Re-starting preview...")
                current_mode = "PREVIEW"
                state_machine.reset_stability()
                last_saved_msg = "Discarded flawed sample. Ready to re-record."
                last_saved_time = time.time()
            elif key in (ord("k"), ord("K"), 32):  # Keep sample ('k' or SPACE)
                print(f"[Recorder] Kept sample with warning: {output_path.name}")
                sample_count += 1
                if single_session:
                    cv2.destroyWindow(window_name)
                    return True
                current_mode = "PREVIEW"
                state_machine.reset_stability()
                last_saved_msg = f"Kept {output_path.name}! SPACE: record again | [N] next sign"
                last_saved_time = time.time()

    cv2.destroyWindow(window_name)
    return False


def record_sign_session(cap: cv2.VideoCapture, word: str, holistic: Any, mp_draw: Any) -> bool:
    """
    Runs a single recording session for the specified word.
    Used by collect_batch.py for batch data collection.
    """
    return run_interactive_recorder(
        cap=cap,
        holistic=holistic,
        mp_draw=mp_draw,
        initial_word=word,
        single_session=True
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SignSpeak AI — Interactive Sign Language Recorder with Real-Time Framing"
    )
    parser.add_argument(
        "sign",
        nargs="?",
        default=None,
        help="Optional initial sign to record by name (e.g. Drink) or index (1-25)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Non-interactive test (processes 5 frames and exits)"
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("SignSpeak AI — Interactive Sign Recorder with Real-Time Framing")
    print("=" * 70)

    cap, success, err = open_webcam(args.camera_index, 1280, 720)
    if not success:
        print(f"\n{err}\n")
        sys.exit(1)

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    print("\n[Preview] Opening live camera preview window...")
    print("  • Look at the preview window and position yourself inside the SIGNING REGION.")
    print("  • On-screen Controls:")
    print("      [N] or [+]       : Next sign in vocabulary")
    print("      [P] or [-]       : Previous sign in vocabulary")
    print("      [0-9] + [ENTER]  : Jump directly to sign index (1-25)")
    print("      [SPACE]          : Start 3-second countdown & record")
    print("      [Q] or [ESC]     : Exit recorder\n")

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            run_interactive_recorder(
                cap=cap,
                holistic=holistic,
                mp_draw=mp_draw,
                initial_word=args.sign,
                single_session=False,
                smoke_test=args.smoke_test
            )
    except KeyboardInterrupt:
        print("\n[Recorder] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[Recorder] Camera released. Goodbye.")


if __name__ == "__main__":
    main()

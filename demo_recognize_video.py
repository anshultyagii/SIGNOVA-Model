"""
SignSpeak AI - Visual Video Recognition Demonstration
Plays a sign-language video, visualizes MediaPipe pose and hand landmarks,
and displays live Top-3 recognition predictions from the 25-class exemplar engine.
"""

import argparse
from pathlib import Path
import sys
import time
import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.pipeline import ExemplarRecognitionEngine

DEFAULT_GALLERY_DIR = PROJECT_ROOT / "dataset" / "processed_corrected"


def get_landmarks(landmark_list, count: int) -> np.ndarray:
    """Extract flattened coordinates or zeros matching extract_dataset.py."""
    if landmark_list is None:
        return np.zeros(count * 3, dtype=np.float32)
    points = np.array(
        [[p.x, p.y, p.z] for p in landmark_list.landmark],
        dtype=np.float32
    )
    if len(points) != count:
        return np.zeros(count * 3, dtype=np.float32)
    return points.flatten()


def draw_hud(
    frame: np.ndarray,
    video_name: str,
    frame_idx: int,
    total_frames: int,
    hands_detected: bool,
    candidates: list,
    is_paused: bool = False,
    is_final: bool = False
) -> np.ndarray:
    """Draws a clean, semi-transparent Heads-Up Display (HUD) overlay on the frame."""
    h, w = frame.shape[:2]
    
    # Overlay panel parameters
    panel_w = min(420, int(w * 0.45))
    panel_h = min(280, int(h * 0.70))
    panel_x = w - panel_w - 20
    panel_y = 20

    # Draw semi-transparent background card
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (25, 25, 30),
        -1
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (80, 80, 95),
        2
    )

    # 1. Header & Source File
    cv2.putText(
        frame,
        "SignSpeak AI - 25-Class Matcher",
        (panel_x + 15, panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2
    )
    cv2.putText(
        frame,
        f"File: {video_name[:24]}",
        (panel_x + 15, panel_y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )

    # Frame Counter & State
    state_str = "FINISHED" if is_final else ("PAUSED" if is_paused else "PLAYING")
    cv2.putText(
        frame,
        f"Frame: {frame_idx + 1}/{total_frames} [{state_str}]",
        (panel_x + 15, panel_y + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1
    )

    # 2. Hand Tracking Status Badge
    badge_y = panel_y + 88
    if hands_detected:
        badge_text = "● HANDS DETECTED (ACTIVE)"
        badge_color = (0, 255, 120)  # Green
    else:
        badge_text = "○ NO HANDS IN FRAME (REST)"
        badge_color = (0, 165, 255)  # Orange

    cv2.putText(
        frame,
        badge_text,
        (panel_x + 15, badge_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        badge_color,
        1
    )

    cv2.line(
        frame,
        (panel_x + 15, panel_y + 98),
        (panel_x + panel_w - 15, panel_y + 98),
        (60, 60, 75),
        1
    )

    # 3. Top-3 Prediction Results
    cv2.putText(
        frame,
        "TOP CANDIDATES (Similarity Metric):",
        (panel_x + 15, panel_y + 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 210),
        1
    )

    bar_start_y = panel_y + 140
    if candidates:
        for idx, cand in enumerate(candidates[:3]):
            rank = cand["rank"]
            label = cand["label"]
            sim = cand["similarity"]
            pct = int(sim * 100)

            # Colors for ranks: 1st=Cyan/Green, 2nd=Yellow, 3rd=Light Gray
            if rank == 1:
                text_color = (0, 255, 200)
                bar_color = (0, 220, 180)
            elif rank == 2:
                text_color = (0, 220, 255)
                bar_color = (0, 180, 220)
            else:
                text_color = (200, 200, 200)
                bar_color = (140, 140, 150)

            row_y = bar_start_y + idx * 36
            # Label line
            cv2.putText(
                frame,
                f"#{rank} {label}",
                (panel_x + 15, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                text_color,
                1
            )
            cv2.putText(
                frame,
                f"{pct}% ({sim:.3f})",
                (panel_x + panel_w - 110, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1
            )

            # Similarity Progress Bar
            bar_x = panel_x + 15
            bar_y = row_y + 6
            bar_max_w = panel_w - 30
            bar_fill_w = max(4, int(bar_max_w * sim))
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_max_w, bar_y + 6),
                (45, 45, 55),
                -1
            )
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_fill_w, bar_y + 6),
                bar_color,
                -1
            )
    else:
        cv2.putText(
            frame,
            "Buffering gesture sequence...",
            (panel_x + 15, bar_start_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (140, 140, 140),
            1
        )

    # 4. Footer Disclaimer & Controls Hint
    footer_y = panel_y + panel_h - 12
    cv2.putText(
        frame,
        "[SPACE] Pause | [R] Replay | [Q] Quit",
        (panel_x + 15, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (130, 130, 140),
        1
    )

    return frame


def run_video_demo(video_path: Path, gallery_dir: Path, no_gui: bool = False):
    """Executes the visual demonstration pipeline on a query video."""
    print("=" * 75)
    print("SignSpeak AI — Visual Recognition Demo")
    print("=" * 75)
    print(f"Target Video : {video_path}")
    print(f"Gallery Dir  : {gallery_dir}")
    print(f"Display Mode : {'Headless / Console' if no_gui else 'OpenCV GUI Window'}")
    print("=" * 75)

    # 1. Validation
    if not video_path.exists():
        print(f"ERROR: Video file does not exist: {video_path}")
        sys.exit(1)

    if not gallery_dir.exists():
        print(f"ERROR: Gallery directory does not exist: {gallery_dir}")
        sys.exit(1)

    # 2. Load Recognition Gallery
    print("\n[1/3] Loading 25-Class Exemplar Recognition Engine...")
    engine = ExemplarRecognitionEngine(gallery_dir=gallery_dir, trim_to_active=True)
    labels = engine.get_gallery_labels()
    print(f"      Enrolled templates: {len(labels)} classes loaded successfully.")

    # 3. Open Video Stream
    print(f"\n[2/3] Opening video stream: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_delay = max(1, int(1000 / fps))

    print(f"      Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"      Total Frames: {total_frames} (~{total_frames / fps:.2f} sec at {fps:.1f} FPS)")

    # 4. MediaPipe Holistic Setup (Identical to extract_dataset.py)
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    features_buffer = []
    presence_buffer = []
    candidates = []
    last_predict_frame = 0

    if not no_gui:
        window_title = "SignSpeak AI — Exemplar Recognition Demo"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1100, 680)

    print("\n[3/3] Processing video frames...")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame_idx = 0
        is_paused = False

        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Standardize frame width (same as extract_dataset.py)
                orig_h, orig_w = frame.shape[:2]
                if orig_w > 960:
                    proc_frame = cv2.resize(frame, (960, int(orig_h * 960 / orig_w)))
                else:
                    proc_frame = frame

                rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Extract raw 1629 features exactly matching dataset schema
                row = np.concatenate([
                    get_landmarks(results.pose_landmarks, 33),
                    get_landmarks(results.face_landmarks, 468),
                    get_landmarks(results.left_hand_landmarks, 21),
                    get_landmarks(results.right_hand_landmarks, 21)
                ])

                has_pose = results.pose_landmarks is not None
                has_lh = results.left_hand_landmarks is not None
                has_rh = results.right_hand_landmarks is not None
                has_either_hand = has_lh or has_rh

                presence_row = [
                    has_pose,
                    results.face_landmarks is not None,
                    has_lh,
                    has_rh
                ]

                features_buffer.append(row)
                presence_buffer.append(presence_row)

                # Draw skeleton landmarks on display frame
                display_frame = proc_frame.copy()
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )
                if results.left_hand_landmarks:
                    mp_draw.draw_landmarks(
                        display_frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2)
                    )
                if results.right_hand_landmarks:
                    mp_draw.draw_landmarks(
                        display_frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 160, 255), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 120, 240), thickness=2)
                    )

                # Periodic sliding window update (every 12 frames once active sequence has >= 18 frames)
                if len(features_buffer) >= 18 and (frame_idx - last_predict_frame >= 12):
                    try:
                        feats_np = np.stack(features_buffer)
                        pres_np = np.asarray(presence_buffer, dtype=bool)
                        res = engine.recognize_features(feats_np, pres_np, query_name=video_path.stem, top_k=3)
                        candidates = res["candidates"]
                        last_predict_frame = frame_idx
                    except Exception as e:
                        print(f"Warning during intermediate prediction: {e}")

                # Draw HUD
                annotated = draw_hud(
                    display_frame,
                    video_name=video_path.name,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    hands_detected=has_either_hand,
                    candidates=candidates,
                    is_paused=is_paused,
                    is_final=False
                )

                if not no_gui:
                    cv2.imshow(window_title, annotated)
                    key = cv2.waitKey(frame_delay) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("\nPlayback stopped by user.")
                        break
                    elif key == ord(' '):  # SPACE
                        is_paused = not is_paused
                    elif key == ord('r'):  # 'r' to restart
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        features_buffer.clear()
                        presence_buffer.clear()
                        candidates.clear()
                        frame_idx = 0
                        continue

                frame_idx += 1
            else:
                # Paused state loop
                if not no_gui:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord(' '):
                        is_paused = not is_paused

        cap.release()

    # 5. Final Full-Sequence Recognition
    print("\n" + "=" * 75)
    print("FINAL RECOGNITION RESULTS (Full Video Sequence)")
    print("=" * 75)

    if not features_buffer:
        print("ERROR: No frames were successfully processed from the video.")
        sys.exit(1)

    final_feats = np.stack(features_buffer)
    final_pres = np.asarray(presence_buffer, dtype=bool)

    # Check hand presence in sequence
    rh_frames = int(final_pres[:, 3].sum())
    lh_frames = int(final_pres[:, 2].sum())
    print(f"Processed Frames     : {len(final_feats)}")
    print(f"Hand Detection Stats : Right Hand: {rh_frames} frames | Left Hand: {lh_frames} frames")

    if rh_frames == 0 and lh_frames == 0:
        print("WARNING: Zero hand landmarks detected in this video!")
        print("         The signer's hands may be outside the camera frame.")

    final_result = engine.recognize_features(final_feats, final_pres, query_name=video_path.stem, top_k=3)
    final_candidates = final_result["candidates"]

    print("\nTop-3 Matches (Ranked by DTW Normalized Similarity):")
    print("-" * 75)
    for c in final_candidates:
        bar = "=" * int(c["similarity"] * 20)
        print(f"  Rank {c['rank']}: {c['label']:<20} | Similarity: {c['similarity']:.4f} ({c['similarity']*100:.1f}%) | Dist: {c['distance']:.4f} | [{bar:<20}]")
    print("-" * 75)
    print("  * Note: 'Similarity' is 1 / (1 + DTW_dist). It indicates trajectory similarity,")
    print("    NOT a calibrated probability.")

    # In GUI mode, hold final frame with final prediction until user presses 'q'
    if not no_gui and 'display_frame' in locals():
        final_display = draw_hud(
            display_frame,
            video_name=video_path.name,
            frame_idx=len(features_buffer) - 1,
            total_frames=total_frames,
            hands_detected=(rh_frames > 0 or lh_frames > 0),
            candidates=final_candidates,
            is_paused=False,
            is_final=True
        )
        cv2.imshow(window_title, final_display)
        print("\nDisplaying final prediction on screen. Press 'q' or close the window to exit.")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 27 or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('r'):
                cv2.destroyAllWindows()
                return run_video_demo(video_path, gallery_dir, no_gui=False)

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="SignSpeak AI — Visual Video Recognition Demo"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "raw" / "Drink.mp4"),
        help="Path to the video file to recognize (default: dataset/raw/Drink.mp4)"
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default=str(DEFAULT_GALLERY_DIR),
        help="Path to directory containing reference .npz templates"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless/console mode without opening an OpenCV GUI window"
    )

    args = parser.parse_args()
    video_path = Path(args.video)
    gallery_dir = Path(args.gallery_dir)

    run_video_demo(video_path=video_path, gallery_dir=gallery_dir, no_gui=args.no_gui)


if __name__ == "__main__":
    main()

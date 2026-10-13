"""
SignSpeak AI - Hand Detection Diagnostic Tool
Inspects recordings, tests MediaPipe Holistic and Hands under various configurations,
and saves annotated preview frames to debug hand detection issues.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import sys

# Default videos to inspect
TARGET_VIDEOS = [
    Path(r"F:\SignSpeakAI\dataset\my_recordings\Drink\sample_001.mp4"),
    Path(r"F:\SignSpeakAI\dataset\my_recordings\Eat\sample_001.mp4"),
    Path(r"F:\SignSpeakAI\dataset\my_recordings\Water\sample_001.mp4"),
    Path(r"F:\SignSpeakAI\dataset\my_recordings\Help\sample_001.mp4"),
]

OUTPUT_DIR = Path(r"F:\SignSpeakAI\debug_output\hand_previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def inspect_video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "fourcc": fourcc_str,
        "duration": total_frames / fps if fps > 0 else 0,
    }


def analyze_video(video_path: Path, save_previews: bool = True):
    print(f"\n{'='*70}")
    print(f"ANALYZING: {video_path.parent.name} / {video_path.name}")
    print(f"Path: {video_path}")
    print(f"{'='*70}")

    meta = inspect_video_metadata(video_path)
    if not meta:
        print(f"ERROR: Could not open {video_path}")
        return

    print(
        f"Dimensions: {meta['width']}x{meta['height']} | "
        f"FPS: {meta['fps']:.1f} | "
        f"Frames: {meta['total_frames']} | "
        f"Duration: {meta['duration']:.2f}s | "
        f"Codec: {meta['fourcc']}"
    )

    video_preview_dir = OUTPUT_DIR / f"{video_path.parent.name}_{video_path.stem}"
    if save_previews:
        video_preview_dir.mkdir(parents=True, exist_ok=True)

    # We will test:
    # 1. Fresh Holistic (static_image_mode=False, conf=0.5)
    # 2. Fresh Holistic (static_image_mode=False, conf=0.3)
    # 3. Fresh Holistic (static_image_mode=True, conf=0.5)
    # 4. Dedicated mp.solutions.hands.Hands (static_image_mode=False, conf=0.3)
    # 5. Dedicated mp.solutions.hands.Hands (static_image_mode=True, conf=0.3)

    # Collect frames first to ensure fair, identical comparison
    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
    cap.release()

    frame_count = len(raw_frames)
    if frame_count == 0:
        print("ERROR: 0 frames read from video!")
        return

    # Run Test 1: Standard Holistic (Fresh instance, static=False, conf=0.5)
    h_fresh_pose = 0
    h_fresh_face = 0
    h_fresh_left = 0
    h_fresh_right = 0
    h_fresh_either = 0
    h_fresh_results = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as model:
        for f in raw_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = model.process(rgb)
            h_fresh_results.append(res)
            has_pose = res.pose_landmarks is not None
            has_face = res.face_landmarks is not None
            has_left = res.left_hand_landmarks is not None
            has_right = res.right_hand_landmarks is not None
            if has_pose:
                h_fresh_pose += 1
            if has_face:
                h_fresh_face += 1
            if has_left:
                h_fresh_left += 1
            if has_right:
                h_fresh_right += 1
            if has_left or has_right:
                h_fresh_either += 1

    # Run Test 2: Holistic with lower confidence (conf=0.3)
    h_low_either = 0
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as model:
        for f in raw_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = model.process(rgb)
            if res.left_hand_landmarks or res.right_hand_landmarks:
                h_low_either += 1

    # Run Test 3: Holistic static_image_mode=True (each frame evaluated independently)
    h_static_either = 0
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as model:
        for f in raw_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = model.process(rgb)
            if res.left_hand_landmarks or res.right_hand_landmarks:
                h_static_either += 1

    # Run Test 4: Dedicated MediaPipe Hands (static=False, max_num_hands=2, conf=0.3)
    hands_detected_frames = 0
    hands_results = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands_model:
        for f in raw_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = hands_model.process(rgb)
            hands_results.append(res)
            if res.multi_hand_landmarks:
                hands_detected_frames += 1

    # Run Test 5: Dedicated MediaPipe Hands (static=True, max_num_hands=2, conf=0.3)
    hands_static_detected = 0
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3
    ) as hands_model:
        for f in raw_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = hands_model.process(rgb)
            if res.multi_hand_landmarks:
                hands_static_detected += 1

    # Check wrist visibility from pose landmarks
    wrist_visible_count = 0
    for res in h_fresh_results:
        if res.pose_landmarks:
            # MediaPipe Pose: Landmark 15 is LEFT_WRIST, 16 is RIGHT_WRIST
            l_wrist = res.pose_landmarks.landmark[15]
            r_wrist = res.pose_landmarks.landmark[16]
            if (l_wrist.visibility > 0.5) or (r_wrist.visibility > 0.5):
                wrist_visible_count += 1

    print("\n--- DETECTION SUMMARY ---")
    print(f"Total Frames Analyzed: {frame_count}")
    print(f"Holistic Pose detected:            {h_fresh_pose:3d} / {frame_count} ({h_fresh_pose/frame_count*100:5.1f}%)")
    print(f"Holistic Face detected:            {h_fresh_face:3d} / {frame_count} ({h_fresh_face/frame_count*100:5.1f}%)")
    print(f"Holistic Wrists visible (in pose): {wrist_visible_count:3d} / {frame_count} ({wrist_visible_count/frame_count*100:5.1f}%)")
    print(f"Holistic Left hand detected:       {h_fresh_left:3d} / {frame_count} ({h_fresh_left/frame_count*100:5.1f}%)")
    print(f"Holistic Right hand detected:      {h_fresh_right:3d} / {frame_count} ({h_fresh_right/frame_count*100:5.1f}%)")
    print(f"Holistic Either hand (conf=0.5):   {h_fresh_either:3d} / {frame_count} ({h_fresh_either/frame_count*100:5.1f}%)")
    print(f"Holistic Either hand (conf=0.3):   {h_low_either:3d} / {frame_count} ({h_low_either/frame_count*100:5.1f}%)")
    print(f"Holistic Either hand (static=True):{h_static_either:3d} / {frame_count} ({h_static_either/frame_count*100:5.1f}%)")
    print(f"Dedicated mp.Hands (video mode):   {hands_detected_frames:3d} / {frame_count} ({hands_detected_frames/frame_count*100:5.1f}%)")
    print(f"Dedicated mp.Hands (static mode):  {hands_static_detected:3d} / {frame_count} ({hands_static_detected/frame_count*100:5.1f}%)")

    # SAVE PREVIEWS
    if save_previews and frame_count > 0:
        # Choose representative frame indices:
        # e.g., 5 evenly spaced frames, plus up to 3 frames where hands were detected (if any),
        # plus up to 3 frames where hands were NOT detected.
        chosen_indices = set()
        # Evenly spaced
        for step in [0.1, 0.25, 0.5, 0.75, 0.9]:
            chosen_indices.add(int(step * (frame_count - 1)))

        # Frames with hands detected by Holistic
        hands_detected_idx = [
            i for i, r in enumerate(h_fresh_results)
            if r.left_hand_landmarks or r.right_hand_landmarks
        ]
        for idx in hands_detected_idx[:3]:
            chosen_indices.add(idx)

        # Frames with hands detected by mp.Hands but NOT by Holistic
        hands_only_idx = [
            i for i, (hr, hands_r) in enumerate(zip(h_fresh_results, hands_results))
            if hands_r.multi_hand_landmarks and not (hr.left_hand_landmarks or hr.right_hand_landmarks)
        ]
        for idx in hands_only_idx[:3]:
            chosen_indices.add(idx)

        # Frames with NO hands detected
        no_hands_idx = [
            i for i, r in enumerate(h_fresh_results)
            if not r.left_hand_landmarks and not r.right_hand_landmarks
        ]
        for idx in no_hands_idx[:3]:
            chosen_indices.add(idx)

        sorted_indices = sorted(list(chosen_indices))
        print(f"\nSaving {len(sorted_indices)} preview frames to: {video_preview_dir}")

        for idx in sorted_indices:
            frame = raw_frames[idx].copy()
            res_holistic = h_fresh_results[idx]
            res_hands = hands_results[idx]

            # Draw Holistic pose
            if res_holistic.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res_holistic.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

            # Draw Holistic hands in Green (left) and Magenta (right)
            if res_holistic.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res_holistic.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
                )
            if res_holistic.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res_holistic.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(200, 0, 200), thickness=2)
                )

            # Draw dedicated mp.Hands in Bright Red/Orange circles if detected
            if res_hands.multi_hand_landmarks:
                for hand_lms in res_hands.multi_hand_landmarks:
                    # Draw a distinct circle around wrist / center
                    wrist_pt = hand_lms.landmark[0]
                    cx, cy = int(wrist_pt.x * frame.shape[1]), int(wrist_pt.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 12, (0, 165, 255), 2)
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 140, 255), thickness=1, circle_radius=2),
                        connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 100, 255), thickness=1)
                    )

            # Status Banner at bottom or top
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            l_str = "YES" if res_holistic.left_hand_landmarks else "NO"
            r_str = "YES" if res_holistic.right_hand_landmarks else "NO"
            p_str = "YES" if res_holistic.pose_landmarks else "NO"
            hands_cnt = len(res_hands.multi_hand_landmarks) if res_hands.multi_hand_landmarks else 0

            line1 = f"Frame {idx+1}/{frame_count} | Holistic Pose: {p_str} | Left Hand: {l_str} | Right Hand: {r_str}"
            line2 = f"Dedicated mp.Hands detected: {hands_cnt} hand(s)"

            cv2.putText(frame, line1, (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(
                frame, line2, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 200, 255) if hands_cnt > 0 else (150, 150, 150), 2
            )

            out_file = video_preview_dir / f"frame_{idx:03d}.jpg"
            cv2.imwrite(str(out_file), frame)

        print(f"Preview frames saved successfully to: {video_preview_dir}")


def main():
    print("=" * 70)
    print("SignSpeak AI - Hand Detection Diagnostic Tool")
    print("=" * 70)

    # Allow custom video path or list passed as CLI arguments
    videos_to_test = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.is_file():
                videos_to_test.append(p)
            elif p.is_dir():
                videos_to_test.extend(sorted(p.glob("*.mp4")))
    else:
        videos_to_test = TARGET_VIDEOS

    print(f"Videos queued for inspection: {len(videos_to_test)}")
    for v in videos_to_test:
        print(f" - {v}")

    for v in videos_to_test:
        if not v.exists():
            print(f"\nWARNING: File does not exist: {v}")
            continue
        analyze_video(v, save_previews=True)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC RUN COMPLETED!")
    print(f"Preview images saved under: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()


from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

RAW_DIR = Path(r"F:\SignSpeakAI\dataset\raw")
OUT_DIR = Path(r"F:\SignSpeakAI\dataset\processed_corrected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic


def get_landmarks(landmark_list, count):
    if landmark_list is None:
        return np.zeros(count * 3, dtype=np.float32)

    points = np.array(
        [[p.x, p.y, p.z] for p in landmark_list.landmark],
        dtype=np.float32
    )

    if len(points) != count:
        return np.zeros(count * 3, dtype=np.float32)

    return points.flatten()


def extract_video(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR opening: {video_path.name}")
        return False

    features = []
    presence = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        height, width = frame.shape[:2]

        if width > 960:
            frame = cv2.resize(
                frame, (960, int(height * 960 / width))
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        row = np.concatenate([
            get_landmarks(results.pose_landmarks, 33),
            get_landmarks(results.face_landmarks, 468),
            get_landmarks(results.left_hand_landmarks, 21),
            get_landmarks(results.right_hand_landmarks, 21)
        ])

        features.append(row)

        presence.append([
            results.pose_landmarks is not None,
            results.face_landmarks is not None,
            results.left_hand_landmarks is not None,
            results.right_hand_landmarks is not None
        ])

    cap.release()

    if not features:
        print(f"ERROR: No frames in {video_path.name}")
        return False

    output_path = OUT_DIR / f"{video_path.stem}_landmarks.npz"

    np.savez_compressed(
        output_path,
        features=np.stack(features),
        presence=np.asarray(presence, dtype=bool),
        label=np.array(video_path.stem),
        video_name=np.array(video_path.name)
    )

    print(
        f"OK: {video_path.name} | "
        f"{len(features)} frames | "
        f"right hand: {np.asarray(presence)[:, 3].sum()} frames | "
        f"left hand: {np.asarray(presence)[:, 2].sum()} frames"
    )

    return True


videos = sorted(RAW_DIR.glob("*.mp4"))

if not videos:
    raise FileNotFoundError(f"No MP4 files found in {RAW_DIR}")

success_count = 0

for video_path in videos:
    output_path = OUT_DIR / f"{video_path.stem}_landmarks.npz"

    if output_path.exists():
        print(f"SKIP (already extracted): {video_path.name}")
        success_count += 1
        continue

    # Fresh MediaPipe Holistic instance per video to prevent tracking leakage
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        if extract_video(video_path, holistic):
            success_count += 1

print(f"\nFinished: {success_count}/{len(videos)} videos processed or already extracted.")
print(f"Output folder: {OUT_DIR}")

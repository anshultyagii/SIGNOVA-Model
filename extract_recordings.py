
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

BASE = Path(r"F:\SignSpeakAI\dataset\my_recordings")
OUT = Path(r"F:\SignSpeakAI\dataset\recorded_processed")
OUT.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=False
) as holistic:

    videos = sorted(BASE.rglob("*.mp4"))
    print("Videos found:", len(videos))

    for video in videos:
        output = OUT / f"{video.parent.name}_{video.stem}_landmarks.npz"

        if output.exists():
            continue

        cap = cv2.VideoCapture(str(video))
        features, presence = [], []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            def coords(landmarks, count):
                if landmarks:
                    return np.array(
                        [[p.x, p.y, p.z] for p in landmarks.landmark],
                        dtype=np.float32
                    ).flatten()
                return np.zeros(count * 3, dtype=np.float32)

            pose = coords(result.pose_landmarks, 33)
            face = coords(result.face_landmarks, 468)
            left = coords(result.left_hand_landmarks, 21)
            right = coords(result.right_hand_landmarks, 21)

            features.append(np.concatenate([pose, face, left, right]))
            presence.append([
                result.pose_landmarks is not None,
                result.face_landmarks is not None,
                result.left_hand_landmarks is not None,
                result.right_hand_landmarks is not None
            ])

        cap.release()

        if features:
            np.savez_compressed(
                output,
                features=np.array(features, dtype=np.float32),
                presence=np.array(presence, dtype=bool),
                label=video.parent.name,
                video_name=video.name
            )
            print("OK:", video.parent.name, "|", len(features), "frames")
        else:
            print("WARNING: No frames read:", video)

print("Landmark extraction complete.")

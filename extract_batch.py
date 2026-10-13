
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

BASE = Path(r"F:\SignSpeakAI\dataset\my_recordings")
OUT = Path(r"F:\SignSpeakAI\dataset\recorded_processed")
OUT.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic
WORDS = {"Drink", "Eat", "Help", "Water", "I_Me"}

videos = sorted(
    video for video in BASE.rglob("*.mp4")
    if video.parent.name in WORDS
)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=False
) as model:

    for video in videos:
        output = OUT / f"{video.parent.name}_{video.stem}_landmarks.npz"

        if output.exists():
            continue

        cap = cv2.VideoCapture(str(video))
        features, presence = [], []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            def get_coords(obj, count):
                if obj:
                    return np.array(
                        [[p.x, p.y, p.z] for p in obj.landmark],
                        dtype=np.float32
                    ).flatten()
                return np.zeros(count * 3, dtype=np.float32)

            features.append(np.concatenate([
                get_coords(result.pose_landmarks, 33),
                get_coords(result.face_landmarks, 468),
                get_coords(result.left_hand_landmarks, 21),
                get_coords(result.right_hand_landmarks, 21)
            ]))

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
                features=np.asarray(features, dtype=np.float32),
                presence=np.asarray(presence, dtype=bool),
                label=video.parent.name,
                video_name=video.name
            )
            print("OK:", video.parent.name, video.name)

print("Done!")

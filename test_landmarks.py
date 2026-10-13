
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

video_path = Path(r"F:\SignSpeakAI\dataset\raw\1_One.mp4")
output_dir = Path(r"F:\SignSpeakAI\dataset\processed")
output_dir.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic

def get_landmarks(landmark_list, count):
    """Return x, y, z coordinates or zeros if not detected."""
    if landmark_list is None:
        return np.zeros(count * 3, dtype=np.float32)

    points = np.array(
        [[p.x, p.y, p.z] for p in landmark_list.landmark],
        dtype=np.float32
    )

    if len(points) != count:
        return np.zeros(count * 3, dtype=np.float32)

    return points.flatten()


cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

features = []
presence = []

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Reduce processing cost while preserving the full frame.
        height, width = frame.shape[:2]
        if width > 960:
            frame = cv2.resize(
                frame,
                (960, int(height * 960 / width))
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        pose = get_landmarks(
            results.pose_landmarks, 33
        )
        face = get_landmarks(
            results.face_landmarks, 468
        )
        left_hand = get_landmarks(
            results.left_hand_landmarks, 21
        )
        right_hand = get_landmarks(
            results.right_hand_landmarks, 21
        )

        # 99 pose + 1404 face + 63 left hand + 63 right hand
        row = np.concatenate([
            pose, face, left_hand, right_hand
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
    raise RuntimeError("No frames were extracted from the video.")

features = np.stack(features)
presence = np.asarray(presence, dtype=bool)

output_path = output_dir / "1_One_landmarks.npz"

np.savez_compressed(
    output_path,
    features=features,
    presence=presence,
    label=np.array("ONE"),
    video_name=np.array(video_path.name)
)

print("\nLandmark extraction complete!")
print("Video:", video_path.name)
print("Frames processed:", len(features))
print("Features per frame:", features.shape[1])
print("Pose detected:", presence[:, 0].sum(), "/", len(presence))
print("Face detected:", presence[:, 1].sum(), "/", len(presence))
print("Left hand detected:", presence[:, 2].sum(), "/", len(presence))
print("Right hand detected:", presence[:, 3].sum(), "/", len(presence))
print("Saved to:", output_path)

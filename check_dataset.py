
import numpy as np
from pathlib import Path

folder = Path(r"F:\SignSpeakAI\dataset\processed")
files = sorted(folder.glob("*_landmarks.npz"))

print("Total processed videos:", len(files))
print()

for file in files:
    with np.load(file) as data:
        features = data["features"]
        presence = data["presence"]
        right_hand_frames = int(presence[:, 3].sum())

        print(
            f"{file.name}: "
            f"features={features.shape}, "
            f"right_hand={right_hand_frames}/{len(presence)} frames"
        )

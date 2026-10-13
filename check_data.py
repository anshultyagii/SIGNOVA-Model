
from pathlib import Path
import numpy as np

folder = Path(r"F:\SignSpeakAI\dataset\recorded_processed")
words = ["Drink", "Eat", "Help", "Water", "I_Me"]

for word in words:
    files = sorted(folder.glob(f"{word}_sample_*_landmarks.npz"))
    print(f"\n{word}: {len(files)} recordings")

    for file in files[:2]:
        data = np.load(file)
        x = data["features"]

        # Pose = first 99 values; hands = last 126 values
        pose = x[:, :99]
        hands = x[:, -126:]

        print(
            file.name,
            "frames =", len(x),
            "pose variation =", round(float(np.std(pose)), 5),
            "hand variation =", round(float(np.std(hands)), 5),
            "hand nonzero fraction =", round(float(np.mean(hands != 0)), 3)
        )

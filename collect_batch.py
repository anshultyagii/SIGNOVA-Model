
import cv2
import time
from pathlib import Path

WORDS = ["Drink", "Eat", "Help", "Water", "I_Me"]

BASE = Path("dataset/my_recordings")
TARGET = 10
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not available")

try:
    for word in WORDS:
        folder = BASE / word
        folder.mkdir(parents=True, exist_ok=True)

        existing = len(list(folder.glob("*.mp4")))
        for sample in range(existing + 1, TARGET + 1):
            input(
                f"\n{word} | recording {sample}/{TARGET}. "
                "Press Enter when ready (q to quit): "
            )
            # The previous input's return value isn't needed.
            print("Starting in 3 seconds...")
            time.sleep(3)

            ret, frame = cap.read()
            if not ret:
                print("Webcam frame unavailable; skipping.")
                continue

            h, w = frame.shape[:2]
            path = folder / f"sample_{sample:03d}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
            )

            start = time.time()
            print(f"Sign NOW: {word}")
            while time.time() - start < 3:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, f"{word} | RECORDING",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.imshow("SignSpeakAI", frame)
                writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            writer.release()
            cv2.destroyAllWindows()
            print("Saved:", path)

finally:
    cap.release()
    cv2.destroyAllWindows()


import cv2
import time
from pathlib import Path

WORDS = [
    "One", "Two", "Three", "Four", "Five", "Six", "Seven",
    "Eight", "Nine", "Ten", "Drink", "Eat", "Food", "Give",
    "Go", "Help", "I_Me", "Milk", "My",
    "Want_Desire_Wish_For", "Water", "We", "What", "Where", "You"
]

BASE = Path("dataset/my_recordings")
BASE.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nSignSpeakAI Data Collection")
print("Select a word number, or enter q to quit.")

try:
    while True:
        for i, word in enumerate(WORDS, 1):
            print(f"{i}. {word}")

        choice = input("\nWord number: ").strip()

        if choice.lower() == "q":
            break

        if not choice.isdigit() or not 1 <= int(choice) <= len(WORDS):
            print("Invalid choice.")
            continue

        word = WORDS[int(choice) - 1]
        folder = BASE / word
        folder.mkdir(parents=True, exist_ok=True)

        existing = list(folder.glob("*.mp4"))
        output = folder / f"sample_{len(existing) + 1:03d}.mp4"

        print(f"\nGet ready to sign: {word}")
        print("Starting in 3 seconds...")
        time.sleep(3)

        ret, frame = cap.read()
        if not ret:
            print("Could not read webcam.")
            continue

        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (width, height)
        )

        if not writer.isOpened():
            print("Could not create video file.")
            continue

        start = time.time()
        print("RECORDING! Perform the sign naturally.")

        while time.time() - start < 3:
            ret, frame = cap.read()
            if not ret:
                break

            remaining = max(0, 3 - int(time.time() - start))
            cv2.putText(
                frame, f"SIGN: {word} | {remaining}s",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

            cv2.imshow("SignSpeakAI Recorder", frame)
            writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.release()
        cv2.destroyAllWindows()
        print(f"Saved: {output}")

finally:
    cap.release()
    cv2.destroyAllWindows()


import cv2
import mediapipe as mp
from pathlib import Path

BASE = Path(r"F:\SignSpeakAI\dataset\my_recordings")
WORDS = ["Drink", "Eat", "Help", "Water", "I_Me"]

mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as model:

    for word in WORDS:
        videos = sorted((BASE / word).glob("*.mp4"))
        detected = 0
        total = 0

        for video in videos:
            cap = cv2.VideoCapture(str(video))
            video_detected = 0
            frames = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                if (result.left_hand_landmarks or
                        result.right_hand_landmarks):
                    video_detected += 1

                frames += 1

            cap.release()
            total += frames

            if video_detected > 0:
                detected += 1

            print(
                f"{word}/{video.name}: "
                f"{video_detected}/{frames} frames with hands"
            )

        print(f"\n{word}: hands detected in {detected}/{len(videos)} videos\n")

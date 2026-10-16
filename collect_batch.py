# -*- coding: utf-8 -*-
"""
SignSpeak AI - Batch Sign Language Data Collection with Framing Validation (Phase 2)
===================================================================================
Iterates through target words and records multiple samples per class using real-time
framing validation to guarantee hands and wrists remain within camera boundaries.
"""

from pathlib import Path
import sys
import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collect_signs import record_sign_session, open_webcam, BASE_DIR

BATCH_WORDS = ["Drink", "Eat", "Help", "Water", "I_Me"]
TARGET_SAMPLES_PER_WORD = 10


def main():
    print("=" * 70)
    print("SignSpeak AI — Batch Data Collection with Framing Validation")
    print("=" * 70)

    cap, success, err = open_webcam(0, 1280, 720)
    if not success:
        print(f"\n{err}\n")
        sys.exit(1)

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            for word in BATCH_WORDS:
                folder = BASE_DIR / word
                folder.mkdir(parents=True, exist_ok=True)
                existing = len(list(folder.glob("sample_*.mp4")))

                print(f"\nTarget: '{word}' — Currently {existing}/{TARGET_SAMPLES_PER_WORD} samples recorded.")
                if existing >= TARGET_SAMPLES_PER_WORD:
                    print(f"  Already reached {TARGET_SAMPLES_PER_WORD} samples for '{word}', skipping.")
                    continue

                for sample_num in range(existing + 1, TARGET_SAMPLES_PER_WORD + 1):
                    print(f"\n--> {word} [Sample {sample_num}/{TARGET_SAMPLES_PER_WORD}]")
                    input("Press ENTER to open camera preview (or 'q' to abort batch): ")

                    recorded = record_sign_session(cap, word, holistic, mp_draw)
                    if not recorded:
                        print("\nBatch recording stopped by user.")
                        return

        print("\n[Batch Collector] All target samples completed successfully!")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Batch Collector] Camera released.")


if __name__ == "__main__":
    main()

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset_raw")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

classes = [
    "10_Ten", "1_One", "2_Two", "3_Three", "4_Four", "5_Five",
    "6_Six", "7_Seven", "8_Eight", "9_Nine", "Drink", "Eat",
    "Food", "Give", "Go", "Help", "I_Me", "Milk", "My",
    "Want_Desire_Wish_For", "Water", "We", "What", "Where", "You"
]

print(f"{'Class':<22} | {'Hands':<5} | {'Extended Fingers [T,I,M,R,P]':<30} | {'Y pos (0=top, 1=bot)':<20}")
print("-" * 85)

for c in classes:
    vname = f"{c}.mp4"
    vpath = os.path.join(RAW_DIR, vname)
    if not os.path.exists(vpath):
        continue
    cap = cv2.VideoCapture(vpath)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, tot // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    h, w, _ = frame.shape
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)
    if res.hand_landmarks:
        num_h = len(res.hand_landmarks)
        hand = res.hand_landmarks[0]
        # Check fingers: Thumb (4 vs 2), Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
        ext = [
            hand[4].x < hand[3].x if hand[0].x > hand[9].x else hand[4].x > hand[3].x, # Thumb
            hand[8].y < hand[6].y,   # Index
            hand[12].y < hand[10].y, # Middle
            hand[16].y < hand[14].y, # Ring
            hand[20].y < hand[18].y  # Pinky
        ]
        ext_str = "".join(["1" if e else "0" for e in ext])
        wrist_y = hand[0].y
        print(f"{c:<22} | {num_h:<5} | {ext_str:<30} | {wrist_y:.3f}")
    else:
        print(f"{c:<22} | 0     | None                           | N/A")

detector.close()

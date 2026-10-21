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

# Test 5 distinct sign videos
test_videos = ["1_One.mp4", "2_Two.mp4", "3_Three.mp4", "4_Four.mp4", "5_Five.mp4", "Eat.mp4", "Drink.mp4"]

for tv in test_videos:
    vpath = os.path.join(RAW_DIR, tv)
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
        
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)
    if res.hand_landmarks:
        hand = res.hand_landmarks[0]
        # Check finger tip vs pip heights
        # Landmarks: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # PIPs: 6 (index pip), 10 (middle pip), 14 (ring pip), 18 (pinky pip)
        extended = []
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            extended.append(hand[tip].y < hand[pip].y)
        print(f"{tv}: Fingers extended [index, middle, ring, pinky] = {extended}")
    else:
        print(f"{tv}: No hand detected")

detector.close()

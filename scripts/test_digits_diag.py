import os
import glob
import sys
import cv2
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

sys.path.append('.')
from scripts.realtime_isl_translator import (
    CLASS_NAMES, MODEL_TASK_PATH, MODEL_WEIGHTS_PATH,
    SwinMishAttentionNet, extract_advanced_gesture_features,
    analyze_hand_geometry, get_symbolic_prior
)

base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.25,
    min_hand_presence_confidence=0.25,
    min_tracking_confidence=0.25
)
detector = vision.HandLandmarker.create_from_options(options)

model = SwinMishAttentionNet(in_features=160, num_classes=25)
ckpt = torch.load(MODEL_WEIGHTS_PATH, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def evaluate_all_numbers():
    print("================ FULL ACCURACY BENCHMARK ON ALL NUMBERS 1-10 ================")
    for num in ['1_One', '2_Two', '3_Three', '4_Four', '5_Five', '6_Six', '7_Seven', '8_Eight', '9_Nine', '10_Ten']:
        files = sorted(glob.glob(f'dataset/images/*/*{num}*'))
        correct = 0
        total = 0
        confused = {}
        for f in files:
            img = cv2.imread(f)
            h, w, _ = img.shape
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp_img)
            if not res.hand_landmarks:
                continue
            total += 1
            feats = extract_advanced_gesture_features(res.hand_landmarks, w, h)
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor(feats, dtype=torch.float32).unsqueeze(0)), dim=1)[0].numpy()
            prior = get_symbolic_prior(res.hand_landmarks, w, h)
            fused = probs * prior
            best = CLASS_NAMES[int(np.argmax(fused))]
            if best == num:
                correct += 1
            else:
                confused[best] = confused.get(best, 0) + 1
        acc = (correct / max(1, total)) * 100
        conf_str = ", ".join([f"{k}:{v}" for k, v in sorted(confused.items(), key=lambda x: -x[1])[:3]])
        print(f"  {num:<10}: {correct:2d}/{total:2d} ({acc:5.1f}%) | Top misclassifications: [{conf_str}]")

evaluate_all_numbers()

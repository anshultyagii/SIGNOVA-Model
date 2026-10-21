import cv2
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

samples_dir = os.path.join(PROJECT_ROOT, "samples")
sample_files = ["1_One_mid.jpg", "Eat_mid.jpg", "Eat1.mp4.jpg", "Eat2.mp4.jpg", "Drink_mid.jpg", "Help_mid.jpg", "Want_Desire_Wish_For_mid.jpg"]

for sf in sample_files:
    img_path = os.path.join(samples_dir, sf)
    if not os.path.exists(img_path):
        continue
    
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # MediaPipe requires RGB and mediapipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    annotated = img.copy()
    if detection_result.hand_landmarks:
        print(f"{sf}: Detected {len(detection_result.hand_landmarks)} hand(s)")
        all_x, all_y = [], []
        for hand in detection_result.hand_landmarks:
            for lm in hand:
                all_x.append(lm.x * w)
                all_y.append(lm.y * h)
                
        x_min = max(0, int(min(all_x)))
        x_max = min(w, int(max(all_x)))
        y_min = max(0, int(min(all_y)))
        y_max = min(h, int(max(all_y)))
        
        # Add 15% margin
        bw_raw = x_max - x_min
        bh_raw = y_max - y_min
        pad_x = int(0.20 * bw_raw)
        pad_y = int(0.20 * bh_raw)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        out_path = os.path.join(samples_dir, f"hand_bbox_{sf}")
        cv2.imwrite(out_path, annotated)
        print(f"  Saved hand_bbox_{sf}: [{x_min}, {y_min}, {x_max}, {y_max}]")
    else:
        print(f"{sf}: No hands detected")

detector.close()
print("Hand landmark test completed.")

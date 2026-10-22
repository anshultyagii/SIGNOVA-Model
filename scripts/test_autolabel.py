import cv2
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def test_hand_detection():
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    samples_dir = os.path.join(PROJECT_ROOT, "samples")
    img_files = [f for f in os.listdir(samples_dir) if f.endswith(('.jpg', '.png')) and not f.startswith("bbox_")]
    
    print(f"Testing on {len(img_files)} sample images...")
    
    for fname in img_files:
        fpath = os.path.join(samples_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        annotated = img.copy()
        
        if results.multi_hand_landmarks:
            print(f"  {fname}: Detected {len(results.multi_hand_landmarks)} hand(s)")
            all_x = []
            all_y = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_x.append(lm.x * w)
                    all_y.append(lm.y * h)
            
            x_min = max(0, int(min(all_x)))
            x_max = min(w, int(max(all_x)))
            y_min = max(0, int(min(all_y)))
            y_max = min(h, int(max(all_y)))
            
            # Add padding/margin around the gesture (e.g. 15%)
            pad_w = int(0.15 * (x_max - x_min))
            pad_h = int(0.15 * (y_max - y_min))
            x_min = max(0, x_min - pad_w)
            x_max = min(w, x_max + pad_w)
            y_min = max(0, y_min - pad_h)
            y_max = min(h, y_max + pad_h)
            
            # YOLO format normalized
            cx = ((x_min + x_max) / 2.0) / w
            cy = ((y_min + y_max) / 2.0) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(annotated, f"BBox: {cx:.2f},{cy:.2f},{bw:.2f},{bh:.2f}", 
                        (x_min, max(25, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out_name = os.path.join(samples_dir, f"bbox_{fname}")
            cv2.imwrite(out_name, annotated)
            print(f"    Saved bbox visualization: bbox_{fname} -> YOLO: {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
        else:
            print(f"  {fname}: No hands detected with static mode.")
            
    hands.close()

if __name__ == "__main__":
    test_hand_detection()

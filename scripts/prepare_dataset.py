import os
import re
import cv2
import yaml
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset_raw")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "hand_landmarker.task")

# 25 classes in alphabetical order
CLASS_NAMES = [
    "10_Ten",
    "1_One",
    "2_Two",
    "3_Three",
    "4_Four",
    "5_Five",
    "6_Six",
    "7_Seven",
    "8_Eight",
    "9_Nine",
    "Drink",
    "Eat",
    "Food",
    "Give",
    "Go",
    "Help",
    "I_Me",
    "Milk",
    "My",
    "Want_Desire_Wish_For",
    "Water",
    "We",
    "What",
    "Where",
    "You"
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def parse_filename(fname):
    name_no_ext = os.path.splitext(fname)[0]
    # Check if ends with '1' or '2'
    if name_no_ext.endswith('2') and not name_no_ext == '2_Two':
        base = name_no_ext[:-1]
        split = 'val'
        take = 2
    elif name_no_ext.endswith('1') and not name_no_ext == '1_One':
        base = name_no_ext[:-1]
        split = 'train'
        take = 1
    else:
        base = name_no_ext
        split = 'train'
        take = 0
    return base, split, take

def create_dirs():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

def mask_reference_video_overlays(frame):
    # For ISLRTC reference video (Take 0):
    # Top-left has English text (e.g. "eat", "1 one")
    # Bottom-right has ISLRTC logo
    h, w, _ = frame.shape
    masked = frame.copy()
    
    # Mask top-left text box (approx x: 0..0.35*w, y: 0..0.45*h)
    # Fill with the background grey color (sampled from near corner)
    bg_color = [128, 128, 128]
    if h > 100 and w > 100:
        sample_bg = frame[20:60, int(w*0.5):int(w*0.6)]
        if sample_bg.size > 0:
            bg_color = np.median(sample_bg, axis=(0, 1)).astype(int).tolist()
            
    cv2.rectangle(masked, (0, 0), (int(w * 0.35), int(h * 0.45)), bg_color, -1)
    # Mask bottom-right logo
    cv2.rectangle(masked, (int(w * 0.80), int(h * 0.70)), (w, h), bg_color, -1)
    return masked

def process_dataset(target_frames_per_video=25):
    create_dirs()
    
    # Initialize HandLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Total raw videos: {len(files)}")
    
    stats = {'train': 0, 'val': 0, 'dropped_no_hands': 0}
    
    for fname in tqdm(sorted(files), desc="Processing Videos"):
        base_class, split, take = parse_filename(fname)
        if base_class not in CLASS_TO_ID:
            print(f"Warning: Unknown class for {fname} (detected base: {base_class})")
            continue
        
        class_id = CLASS_TO_ID[base_class]
        vpath = os.path.join(RAW_DIR, fname)
        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
            
        # We want frames primarily from the middle 70% where gesture takes place
        start_frame = int(total_frames * 0.15)
        end_frame = int(total_frames * 0.85)
        if end_frame <= start_frame:
            start_frame = 0
            end_frame = total_frames
            
        step = max(1, (end_frame - start_frame) // target_frames_per_video)
        frame_indices = list(range(start_frame, end_frame, step))[:target_frames_per_video]
        
        extracted_for_video = 0
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            orig_h, orig_w, _ = frame.shape
            
            # Mask overlays if it's the reference video (take == 0)
            if take == 0:
                frame_clean = mask_reference_video_overlays(frame)
            else:
                frame_clean = frame
                
            # Detect hands using MediaPipe
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp_img)
            
            if not res.hand_landmarks:
                stats['dropped_no_hands'] += 1
                continue
                
            # Compute bounding box
            all_x, all_y = [], []
            for hand in res.hand_landmarks:
                for lm in hand:
                    all_x.append(lm.x * orig_w)
                    all_y.append(lm.y * orig_h)
                    
            x_min = max(0, int(min(all_x)))
            x_max = min(orig_w, int(max(all_x)))
            y_min = max(0, int(min(all_y)))
            y_max = min(orig_h, int(max(all_y)))
            
            # Add 20% margin to encompass full hand gesture
            bw_raw = x_max - x_min
            bh_raw = y_max - y_min
            pad_x = int(0.20 * bw_raw)
            pad_y = int(0.20 * bh_raw)
            
            x_min = max(0, x_min - pad_x)
            x_max = min(orig_w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(orig_h, y_max + pad_y)
            
            # Resize image to 640x640 (standard YOLO size)
            target_size = 640
            scale_x = target_size / orig_w
            scale_y = target_size / orig_h
            resized_frame = cv2.resize(frame_clean, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            # Rescale normalized YOLO coordinates
            norm_cx = ((x_min + x_max) / 2.0) / orig_w
            norm_cy = ((y_min + y_max) / 2.0) / orig_h
            norm_w = (x_max - x_min) / orig_w
            norm_h = (y_max - y_min) / orig_h
            
            # Clamp to [0, 1]
            norm_cx = min(max(norm_cx, 0.0), 1.0)
            norm_cy = min(max(norm_cy, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.01), 1.0)
            norm_h = min(max(norm_h, 0.01), 1.0)
            
            # Save image and label
            img_id = f"{base_class}_take{take}_f{f_idx:04d}"
            img_out = os.path.join(DATASET_DIR, 'images', split, f"{img_id}.jpg")
            lbl_out = os.path.join(DATASET_DIR, 'labels', split, f"{img_id}.txt")
            
            cv2.imwrite(img_out, resized_frame)
            with open(lbl_out, 'w') as lf:
                lf.write(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
            stats[split] += 1
            extracted_for_video += 1
            
        cap.release()
        
    detector.close()
    
    print("\nDataset generation completed!")
    print(f"  Train frames: {stats['train']}")
    print(f"  Val frames:   {stats['val']}")
    print(f"  Dropped (idle/no hand): {stats['dropped_no_hands']}")
    
    # Generate data.yaml for YOLO
    yaml_data = {
        'path': DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {idx: name for idx, name in enumerate(CLASS_NAMES)}
    }
    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as yf:
        yaml.dump(yaml_data, yf, default_flow_style=False)
    print(f"Saved {os.path.join(DATASET_DIR, 'data.yaml')}")

if __name__ == '__main__':
    process_dataset()

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset_raw")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "hand_landmarker.task")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

CLASS_NAMES = [
    "10_Ten", "1_One", "2_Two", "3_Three", "4_Four",
    "5_Five", "6_Six", "7_Seven", "8_Eight", "9_Nine",
    "Drink", "Eat", "Food", "Give", "Go",
    "Help", "I_Me", "Milk", "My", "Want_Desire_Wish_For",
    "Water", "We", "What", "Where", "You"
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def build_dataset(stride=3, val_split=0.20):
    print("=" * 70)
    print("  YOLOv10-ST Research Paper Dataset Builder (25 ISL Signs)")
    print("=" * 70)
    print(f"Source Raw Videos: {RAW_DIR}")
    print(f"Target YOLO Dataset: {OUT_DIR}")

    train_img_dir = os.path.join(OUT_DIR, "images", "train")
    val_img_dir = os.path.join(OUT_DIR, "images", "val")
    train_lbl_dir = os.path.join(OUT_DIR, "labels", "train")
    val_lbl_dir = os.path.join(OUT_DIR, "labels", "val")

    for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(p, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25
    )
    detector = vision.HandLandmarker.create_from_options(options)

    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found {len(files)} video takes across 25 signs.\n")

    total_extracted = 0
    total_train = 0
    total_val = 0

    for fname in tqdm(sorted(files), desc="Processing Videos"):
        name_no_ext = os.path.splitext(fname)[0]
        # Resolve base class name
        if name_no_ext.endswith('2') and not name_no_ext == '2_Two':
            base = name_no_ext[:-1]
            is_val_take = True  # Hold out take2 for validation
        elif name_no_ext.endswith('1') and not name_no_ext == '1_One':
            base = name_no_ext[:-1]
            is_val_take = False
        else:
            base = name_no_ext
            is_val_take = False

        if base not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[base]

        cap = cv2.VideoCapture(os.path.join(RAW_DIR, fname))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        start_f = int(total_frames * 0.10)
        end_f = int(total_frames * 0.90)

        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or f_idx > end_f:
                break

            if f_idx >= start_f and (f_idx - start_f) % stride == 0:
                h, w, _ = frame.shape
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = detector.detect(mp_img)

                if res.hand_landmarks:
                    # Enclose all detected hands in the gesture bounding box
                    all_xs = []
                    all_ys = []
                    for hand in res.hand_landmarks:
                        for lm in hand:
                            all_xs.append(lm.x)
                            all_ys.append(lm.y)

                    min_x, max_x = max(0.0, min(all_xs)), min(1.0, max(all_xs))
                    min_y, max_y = max(0.0, min(all_ys)), min(1.0, max(all_ys))

                    # 15% margin for gesture context
                    bw = max_x - min_x
                    bh = max_y - min_y
                    pad_x = bw * 0.15
                    pad_y = bh * 0.15

                    box_min_x = max(0.0, min_x - pad_x)
                    box_max_x = min(1.0, max_x + pad_x)
                    box_min_y = max(0.0, min_y - pad_y)
                    box_max_y = min(1.0, max_y + pad_y)

                    cx = (box_min_x + box_max_x) / 2.0
                    cy = (box_min_y + box_max_y) / 2.0
                    box_w = box_max_x - box_min_x
                    box_h = box_max_y - box_min_y

                    # Resize image to standard 640x640
                    img_640 = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

                    # Determine split: take2 is validation, take0 & take1 are train
                    # If random fallback: 80/20
                    is_val = is_val_take or (np.random.rand() < val_split)

                    target_img_dir = val_img_dir if is_val else train_img_dir
                    target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                    sample_id = f"{base}_{fname[:fname.rfind('.')]}_f{f_idx:04d}"
                    out_img_path = os.path.join(target_img_dir, f"{sample_id}.jpg")
                    out_lbl_path = os.path.join(target_lbl_dir, f"{sample_id}.txt")

                    cv2.imwrite(out_img_path, img_640, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(out_lbl_path, "w") as lf:
                        lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}\n")

                    total_extracted += 1
                    if is_val:
                        total_val += 1
                    else:
                        total_train += 1

            f_idx += 1

        cap.release()

    detector.close()

    # Generate data.yaml matching the 25 classes
    data_yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for i, cname in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {cname}\n")

    print(f"\n[+] Dataset Generation Complete!")
    print(f"    Total Images: {total_extracted}")
    print(f"    Train: {total_train} images")
    print(f"    Val:   {total_val} images")
    print(f"    YAML:  {data_yaml_path}\n")

if __name__ == "__main__":
    build_dataset(stride=2, val_split=0.20)

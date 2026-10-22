import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import re
import cv2
from collections import defaultdict

DATASET_RAW = os.path.join(PROJECT_ROOT, "dataset_raw")

def group_videos():
    files = [f for f in os.listdir(DATASET_RAW) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Total video files found: {len(files)}")
    
    # Class grouping regex or mapping
    # Note files like:
    # 1_One.mp4, 1_One1.mp4, 1_One2.mp4
    # Drink.mp4, Drink1.mp4, Drink2.mp4
    # Want_Desire_Wish_For.mp4, Want_Desire_Wish_For1.mp4, Want_Desire_Wish_For2.mp4
    
    classes = defaultdict(list)
    for f in sorted(files):
        name_no_ext = os.path.splitext(f)[0]
        # Match trailing 1 or 2 if preceded by word, but preserve numbers like 10_Ten
        # Notice:
        # 1_One.mp4, 1_One1.mp4, 1_One2.mp4 -> base class is "1_One"
        # 10_Ten.mp4, 10_Ten1.mp4, 10_Ten2.mp4 -> base class is "10_Ten"
        # Drink.mp4, Drink1.mp4, Drink2.mp4 -> base class is "Drink"
        m = re.match(r"^(.*?)(1|2)?$", name_no_ext)
        if m:
            base_class = m.group(1)
            # Edge case check: if base_class ends with number that is part of name, check if file with exact base_class exists
            if not os.path.exists(os.path.join(DATASET_RAW, f"{base_class}.mp4")):
                # Check without stripping digit
                base_class = name_no_ext
        else:
            base_class = name_no_ext
        classes[base_class].append(f)
        
    print(f"\nUnique classes detected: {len(classes)}")
    for cls_name, vlist in sorted(classes.items()):
        print(f"  [{len(vlist)} takes] {cls_name}: {vlist}")
        
    # Check video specs for the first few
    sample_files = files[:3]
    print("\nSample Video Properties:")
    for sf in sample_files:
        cap = cv2.VideoCapture(os.path.join(DATASET_RAW, sf))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        print(f"  {sf}: {w}x{h} @ {fps:.1f} fps | {frame_count} frames (~{duration:.2f}s)")

if __name__ == "__main__":
    group_videos()

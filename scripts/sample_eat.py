import cv2
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

takes = ["Eat.mp4", "Eat1.mp4", "Eat2.mp4"]
for t in takes:
    cap = cv2.VideoCapture(os.path.join(os.path.join(PROJECT_ROOT, "dataset_raw"), t))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(os.path.join(PROJECT_ROOT, "samples"), f"{t}.jpg"), frame)
    cap.release()
print("Saved takes for Eat")

import cv2
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.makedirs(os.path.join(PROJECT_ROOT, "samples"), exist_ok=True)
sample_videos = ["1_One.mp4", "Eat.mp4", "Drink.mp4", "Help.mp4", "Want_Desire_Wish_For.mp4"]

for vname in sample_videos:
    vpath = os.path.join(os.path.join(PROJECT_ROOT, "dataset_raw"), vname)
    if not os.path.exists(vpath):
        continue
    cap = cv2.VideoCapture(vpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # grab middle frame (where the sign is in full action)
    mid = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    if ret:
        out_path = os.path.join(os.path.join(PROJECT_ROOT, "samples"), f"{os.path.splitext(vname)[0]}_mid.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved mid-frame for {vname}: {frame.shape}")
    cap.release()
print("Done sampling preview frames.")

import cv2
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
samples_dir = os.path.join(PROJECT_ROOT, "samples")
sample_files = ["1_One_mid.jpg", "Eat_mid.jpg", "Eat1.mp4.jpg", "Eat2.mp4.jpg", "Drink_mid.jpg"]

for sf in sample_files:
    img_path = os.path.join(samples_dir, sf)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    results = model(img, verbose=False)
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0].cpu().numpy() # [17, 3] (x, y, conf)
            # 9: left wrist, 10: right wrist
            # 7: left elbow, 8: right elbow
            lw = kpts[9]
            rw = kpts[10]
            le = kpts[7]
            re = kpts[8]
            
            annotated = img.copy()
            print(f"{sf}:")
            print(f"  Left wrist conf: {lw[2]:.2f}, Right wrist conf: {rw[2]:.2f}")
            
            # Find active hands
            active_points = []
            if lw[2] > 0.4:
                # extrapolate hand from elbow -> wrist
                dx = lw[0] - le[0]
                dy = lw[1] - le[1]
                hand_center_x = lw[0] + 0.35 * dx
                hand_center_y = lw[1] + 0.35 * dy
                active_points.append((hand_center_x, hand_center_y))
            if rw[2] > 0.4:
                dx = rw[0] - re[0]
                dy = rw[1] - re[1]
                hand_center_x = rw[0] + 0.35 * dx
                hand_center_y = rw[1] + 0.35 * dy
                active_points.append((hand_center_x, hand_center_y))
                
            if active_points:
                xs = [p[0] for p in active_points]
                ys = [p[1] for p in active_points]
                
                # Determine gesture bounding box
                # Ensure box has minimum size (e.g. at least 15% of image width/height)
                min_box_w = 0.20 * w
                min_box_h = 0.25 * h
                
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                
                span_x = max(xs) - min(xs) + 0.15 * w
                span_y = max(ys) - min(ys) + 0.15 * h
                
                bw = max(min_box_w, span_x)
                bh = max(min_box_h, span_y)
                
                x1 = int(max(0, cx - bw / 2))
                y1 = int(max(0, cy - bh / 2))
                x2 = int(min(w, cx + bw / 2))
                y2 = int(min(h, cy + bh / 2))
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.circle(annotated, (int(cx), int(cy)), 6, (255, 0, 0), -1)
                
                out_path = os.path.join(samples_dir, f"pose_bbox_{sf}")
                cv2.imwrite(out_path, annotated)
                print(f"  -> Saved pose_bbox_{sf}: ({x1},{y1}) to ({x2},{y2})")

print("Pose testing finished.")

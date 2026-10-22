import os
import sys
import time
import json
import torch
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.yolov10_st import build_yolov10_st

def evaluate_yolov10_st():
    data_yaml = os.path.join(BASE_DIR, "dataset", "data.yaml")
    weights_path = os.path.join(BASE_DIR, "weights", "best_yolov10_st.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(BASE_DIR, "runs", "yolov10_st_training", "weights", "best.pt")
        if not os.path.exists(weights_path):
            print(f"Error: Trained weights not found at {weights_path}. Train the model first.")
            return

    print("=" * 75)
    print("      YOLOv10-ST RESEARCH PAPER EVALUATION & BENCHMARKING (NATURE 2025)   ")
    print("=" * 75)
    print(f"Data:    {data_yaml}")
    print(f"Weights: {weights_path}")
    print("-" * 75)

    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=640, split='val', plots=True)

    prec = metrics.box.mp * 100.0
    rec = metrics.box.mr * 100.0
    map50 = metrics.box.map50 * 100.0
    map50_95 = metrics.box.map * 100.0
    f1 = (2 * prec * rec) / max(1e-6, (prec + rec))

    # Benchmark real-time latency and FPS on batch=1
    dummy = torch.randn(1, 3, 640, 640)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dummy = dummy.to(device)

    # Warmup
    for _ in range(30):
        _ = model.predict(dummy, verbose=False)

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_lat_ms = float(np.mean(times) * 1000.0)
    fps = 1000.0 / avg_lat_ms

    print("\n" + "=" * 75)
    print("                       EVALUATION METRICS SUMMARY                        ")
    print("=" * 75)
    print(f"Precision (P):       {prec:.2f}% (Eq. 5)")
    print(f"Recall (R):          {rec:.2f}% (Eq. 6)")
    print(f"F1-Score:            {f1:.2f}% (Eq. 7)")
    print(f"mAP@50:              {map50:.2f}% (Table 4)")
    print(f"mAP@50-95:           {map50_95:.2f}%")
    print(f"Inference Latency:   {avg_lat_ms:.2f} ms")
    print(f"Throughput (FPS):    {fps:.1f} FPS")
    print("=" * 75)

if __name__ == "__main__":
    evaluate_yolov10_st()

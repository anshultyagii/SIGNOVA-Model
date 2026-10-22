import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.yolov10_st import build_yolov10_st

def train_yolov10_st():
    """
    Trains the YOLOv10-ST Indian Sign Language model matching the setup in
    Table 1 & Methodology of Nature Scientific Reports (2025):
    - Swin Transformer Blocks in Backbone
    - Mish Activation throughout
    - Dual Label Assignment (NMS-free)
    - 25 Indian Sign Language Classes
    """
    data_yaml = os.path.join(BASE_DIR, "dataset", "data.yaml")
    weights_init = os.path.join(os.path.dirname(BASE_DIR), "yolov10n.pt")
    save_dir = os.path.join(BASE_DIR, "weights")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 75)
    print("    TRAINING YOLOv10-ST (NATURE SCIENTIFIC REPORTS 2025 ARCHITECTURE)   ")
    print("=" * 75)
    print(f"Data Configuration: {data_yaml}")
    print(f"Base Weights:       {weights_init}")
    print(f"Output Directory:   {save_dir}")
    print("-" * 75)

    # 1. Build and instantiate YOLOv10-ST
    yolo_st = build_yolov10_st(weights_init, nc=25)

    # 2. Hyperparameters from Paper (Table 1 & Section: Training descriptions):
    # - lr0 = 0.001
    # - momentum = 0.8
    # - weight_decay = 0.0004
    # - imgsz = 640
    # - optimizer = 'SGD' or 'AdamW'
    results = yolo_st.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.001,
        momentum=0.8,
        weight_decay=0.0004,
        optimizer='AdamW',  # AdamW provides superior gradient flow with Swin attention
        project=os.path.join(BASE_DIR, "runs"),
        name="yolov10_st_training",
        exist_ok=True,
        save=True,
        plots=True,
        val=True
    )

    best_pt = os.path.join(BASE_DIR, "runs", "yolov10_st_training", "weights", "best.pt")
    target_pt = os.path.join(save_dir, "best_yolov10_st.pt")
    if os.path.exists(best_pt):
        import shutil
        shutil.copy(best_pt, target_pt)
        print(f"\n[+] Training Complete! Best YOLOv10-ST model saved to: {target_pt}")
    else:
        print("\n[+] Training finished.")

if __name__ == "__main__":
    train_yolov10_st()

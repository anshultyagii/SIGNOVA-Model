import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from ultralytics import YOLO
from models.build_yolov10_st import build_yolov10_st, replace_activations_with_mish, SwinEnhancedC2f

DATA_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")
PROJECT_DIR = os.path.join(PROJECT_ROOT, "runs")
EXPERIMENT_NAME = "yolov10_st_isl"

def train():
    # Set PyTorch CPU multi-threading for AMD Ryzen 5 (6 cores / 12 threads)
    torch.set_num_threads(8)
    print(f"Configured PyTorch CPU threads: {torch.get_num_threads()}")
    
    print("=" * 60)
    print("Initializing YOLOv10-ST (Swin Transformer + YOLOv10 + Mish)")
    print("Indian Sign Language Recognition System (25 Classes)")
    print("=" * 60)
    
    model = YOLO('yolov10n.pt')
    
    # 1. Replace SiLU activations with Mish (as in paper Section 'Backbone' & 'Methodology')
    mish_count = replace_activations_with_mish(model.model)
    print(f"Replaced {mish_count} activations with Mish activation.")
    
    # 2. Inject Swin Transformer Blocks into Backbone Layer 2 and Layer 4
    c2 = model.model.model[2].cv1.conv.in_channels
    c4 = model.model.model[4].cv1.conv.in_channels
    
    m2 = SwinEnhancedC2f(model.model.model[2], c2, num_heads=4, window_size=8)
    m2.f = getattr(model.model.model[2], 'f', -1)
    m2.i = getattr(model.model.model[2], 'i', 2)
    m2.type = 'SwinEnhancedC2f'
    model.model.model[2] = m2
    
    m4 = SwinEnhancedC2f(model.model.model[4], c4, num_heads=4, window_size=8)
    m4.f = getattr(model.model.model[4], 'f', -1)
    m4.i = getattr(model.model.model[4], 'i', 4)
    m4.type = 'SwinEnhancedC2f'
    model.model.model[4] = m4
    print("Successfully integrated Swin Transformer blocks into YOLOv10 backbone.")
    
    print("\nTraining configuration:")
    print("  Optimizer: SGD (momentum=0.8, weight_decay=0.0004)")
    print("  Learning Rate: 0.001")
    print("  Resolution: 480x480 (CPU optimized)")
    print("  Batch size: 16")
    print("  Cache: RAM enabled")
    print("=" * 60)
    
    # Train model
    results = model.train(
        data=DATA_YAML,
        epochs=10,
        batch=16,
        imgsz=480,
        cache=True,
        optimizer='SGD',
        lr0=0.001,
        momentum=0.8,
        weight_decay=0.0004,
        device='cpu',
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        workers=0,
        verbose=True,
        plots=True
    )
    
    print("\nTraining completed successfully!")
    print(f"Trained model saved in: {os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'best.pt')}")
    return results

if __name__ == "__main__":
    train()

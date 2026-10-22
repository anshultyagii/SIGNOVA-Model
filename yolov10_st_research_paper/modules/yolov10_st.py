import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import torch
import torch.nn as nn
from ultralytics import YOLO
from c2f_swin import SwinEnhancedC2f

def replace_activations_with_mish(module):
    """
    Recursively replaces all standard activations (SiLU, ReLU, LeakyReLU)
    with Mish activation as described in the YOLOv10-ST paper.
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, (nn.SiLU, nn.ReLU, nn.LeakyReLU)):
            setattr(module, name, nn.Mish())
            count += 1
        else:
            count += replace_activations_with_mish(child)
    return count

def build_yolov10_st(weights_path='yolov10n.pt', nc=25):
    """
    Constructs the exact YOLOv10-ST architecture described in Nature Scientific Reports (2025):
    - Swin Transformer Blocks (W-MSA + SW-MSA) integrated into Backbone Layer 2 and Layer 4
    - Mish activations replacing standard activation functions
    - Configured for 25 Indian Sign Language classes
    """
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), weights_path)

    print(f"[YOLOv10-ST] Loading base YOLOv10 weights from: {weights_path}")
    yolo = YOLO(weights_path)
    core_model = yolo.model

    # 1. Replace activations with Mish
    mish_count = replace_activations_with_mish(core_model)
    print(f"[YOLOv10-ST] Successfully replaced {mish_count} activations with Mish activation.")

    # 2. Inspect channels at Backbone stages 2 and 4
    c2_channels = core_model.model[2].cv1.conv.in_channels
    c4_channels = core_model.model[4].cv1.conv.in_channels

    # 3. Inject Swin Transformer into Backbone Layer 2 (160x160) and Layer 4 (80x80)
    mod2 = SwinEnhancedC2f(core_model.model[2], c2_channels, num_heads=4, window_size=8)
    mod2.f = getattr(core_model.model[2], 'f', -1)
    mod2.i = getattr(core_model.model[2], 'i', 2)
    mod2.type = 'SwinEnhancedC2f'
    core_model.model[2] = mod2

    mod4 = SwinEnhancedC2f(core_model.model[4], c4_channels, num_heads=4, window_size=8)
    mod4.f = getattr(core_model.model[4], 'f', -1)
    mod4.i = getattr(core_model.model[4], 'i', 4)
    mod4.type = 'SwinEnhancedC2f'
    core_model.model[4] = mod4

    print("[YOLOv10-ST] Injected Swin Transformer (W-MSA + SW-MSA) blocks into Backbone Layer 2 and 4.")

    # 4. Update number of classes to 25
    yolo.overrides['nc'] = nc
    return yolo

if __name__ == "__main__":
    model = build_yolov10_st('yolov10n.pt', nc=25)
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model.model(dummy)
    print("YOLOv10-ST forward pass verified on 640x640 input!")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from ultralytics import YOLO
from models.yolov10_st_modules import SwinTransformerBlock

class SwinEnhancedC2f(nn.Module):
    """
    Synergistic integration of Swin Transformer with C2f block in YOLOv10 backbone.
    Applies W-MSA and SW-MSA with Mish activation, fusing local convolutional
    and global transformer attention features.
    """
    def __init__(self, original_c2f, channels, num_heads=4, window_size=8):
        super().__init__()
        self.original_c2f = original_c2f
        self.swin = SwinTransformerBlock(channels, num_heads=num_heads, window_size=window_size)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish()
        )

    def forward(self, x):
        c2f_out = self.original_c2f(x)
        swin_out = self.swin(x)
        cat = torch.cat([c2f_out, swin_out], dim=1)
        return self.fusion(cat)

def replace_activations_with_mish(module):
    """Recursively replaces SiLU / ReLU / LeakyReLU activations with Mish."""
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
    Builds YOLOv10-ST architecture:
    1. Loads YOLOv10 base architecture.
    2. Replaces activations with Mish.
    3. Integrates Swin Transformer blocks at Backbone stages.
    """
    print(f"Loading base YOLOv10 weights from {weights_path}...")
    yolo = YOLO(weights_path)
    core_model = yolo.model
    
    # Replace activations with Mish
    mish_count = replace_activations_with_mish(core_model)
    print(f"Replaced {mish_count} activations with Mish activation.")
    
    # Inspect Backbone layers
    # Layer 2: C2f (160x160, channels = core_model.model[2].cv2.conv.out_channels)
    # Layer 4: C2f (80x80, channels = core_model.model[4].cv2.conv.out_channels)
    c2_channels = core_model.model[2].cv1.conv.in_channels
    c4_channels = core_model.model[4].cv1.conv.in_channels
        
    print(f"Backbone Layer 2 channels: {c2_channels}")
    print(f"Backbone Layer 4 channels: {c4_channels}")
    
    # Inject Swin Transformer into Layer 2 and Layer 4
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
    print("Injected Swin Transformer blocks into Backbone Layer 2 and Layer 4 with proper graph indexing.")
    
    return yolo

if __name__ == "__main__":
    yolo_st = build_yolov10_st('yolov10n.pt', nc=25)
    
    # Test forward pass with dummy image [1, 3, 640, 640]
    dummy = torch.randn(1, 3, 640, 640)
    print("Running forward pass through YOLOv10-ST...")
    with torch.no_grad():
        out = yolo_st.model(dummy)
    print("Forward pass successful!")
    if isinstance(out, (list, tuple)):
        print(f"Output type: tuple/list of {len(out)} elements")
        for i, o in enumerate(out):
            if isinstance(o, torch.Tensor):
                print(f"  Element {i}: {o.shape}")
            elif isinstance(o, (list, tuple)):
                print(f"  Element {i}: list of {len(o)} items")
    else:
        print("Output shape:", out.shape)

import os
import sys
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import torch
import torch.nn as nn
from swin_transformer import SwinTransformerBlock

class SwinEnhancedC2f(nn.Module):
    """
    Synergistic integration of Swin Transformer with C2f module in YOLOv10 Backbone.
    F_fused = Concat(Z_ST, C2f(F)) -> Conv1x1 with Mish activation.
    Matches Page 6 and Figure 2 of the YOLOv10-ST research paper.
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

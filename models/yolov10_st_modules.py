import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """Window based multi-head self-attention (W-MSA / SW-MSA) module."""
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Coordinate map for relative position indices
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with Mish activation as specified in the YOLOv10-ST paper.
    Contains 2 consecutive sub-blocks:
    Sub-block 1: W-MSA + MLP (Mish)
    Sub-block 2: SW-MSA (Shifted Window) + MLP (Mish)
    """
    def __init__(self, dim, num_heads=4, window_size=8, shift_size=4, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Norm layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        # Self-attention modules
        self.attn_w = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.attn_sw = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        # MLPs with Mish activation
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.Mish(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.0)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.Mish(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        # x is [B, C, H, W]
        B, C, H, W = x.shape
        # Permute to [B, H, W, C]
        x_in = x.permute(0, 2, 3, 1).contiguous()

        # Pad if H or W not divisible by window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_in = F.pad(x_in, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x_in.shape

        # ----- Sub-block 1: W-MSA -----
        res1 = x_in
        x_norm1 = self.norm1(x_in)
        windows1 = window_partition(x_norm1, self.window_size)
        attn_windows1 = self.attn_w(windows1)
        x_w = window_reverse(attn_windows1, self.window_size, pad_H, pad_W)
        x_in = res1 + x_w
        x_in = x_in + self.mlp1(self.norm2(x_in))

        # ----- Sub-block 2: SW-MSA (Shifted Window) -----
        res2 = x_in
        x_norm2 = self.norm3(x_in)
        # Shift
        shifted_x = torch.roll(x_norm2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows2 = window_partition(shifted_x, self.window_size)
        attn_windows2 = self.attn_sw(windows2)
        shifted_x = window_reverse(attn_windows2, self.window_size, pad_H, pad_W)
        # Reverse shift
        x_sw = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x_in = res2 + x_sw
        x_in = x_in + self.mlp2(self.norm4(x_in))

        # Crop if padded
        if pad_r > 0 or pad_b > 0:
            x_in = x_in[:, :H, :W, :].contiguous()

        # Permute back to [B, C, H, W]
        return x_in.permute(0, 3, 1, 2).contiguous()

class YOLOv10ST_BackboneFusion(nn.Module):
    """
    Swin Transformer Block combined with C2f module and residual fusion
    F_fused = Concat(Z_ST, C2f(F)) -> Conv1x1(2*C, C) with Mish activation
    """
    def __init__(self, c_in, c_out, num_heads=4, window_size=8):
        super().__init__()
        self.st_block = SwinTransformerBlock(c_in, num_heads=num_heads, window_size=window_size)
        # 1x1 conv to project concatenated channels back to c_out with Mish
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(c_in * 2, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.Mish()
        )

    def forward(self, x, c2f_out):
        st_out = self.st_block(x)
        cat = torch.cat([st_out, c2f_out], dim=1)
        return self.fuse_conv(cat)

if __name__ == "__main__":
    # Test on synthetic tensor of size [2, 128, 40, 40]
    dummy_input = torch.randn(2, 128, 40, 40)
    st = SwinTransformerBlock(128, num_heads=4, window_size=8)
    out = st(dummy_input)
    print("SwinTransformerBlock test successful! Output shape:", out.shape)
    
    fusion = YOLOv10ST_BackboneFusion(128, 128)
    fused_out = fusion(dummy_input, dummy_input)
    print("BackboneFusion test successful! Output shape:", fused_out.shape)

"""
DBF-UNet Building Blocks (Low-Dose Ablation)
Contains: RCAB, CBAM, Downsample, Upsample
(Removed: MS-ASPP, Fourier Convolution, Fusion Gate)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CBAM: Convolutional Block Attention Module
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        w = torch.sigmoid(self.conv(cat))
        return x * w

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ============================================================
# RCAB: Residual Channel Attention Block
# ============================================================
class RCAB(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attn = CBAM(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.attn(out)
        out = self.drop(out)
        return out + res

class RCABGroup(nn.Module):
    def __init__(self, channels, num_blocks=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            RCAB(channels, dropout) for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)

# ============================================================
# Down / Up sampling with conv
# ============================================================
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.op(x)
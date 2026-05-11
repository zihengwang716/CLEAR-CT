"""
DBF-UNet Building Blocks (Sparse-View Ablation)
Contains: BasicResGroup, MS-ASPP, Fourier Convolution, Fusion Gate
(Removed: RCAB, CBAM)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Basic Residual Block (Replaces RCAB for Sparse-View Ablation)
# ============================================================
class BasicResBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.drop(out)
        return out + res

class BasicResGroup(nn.Module):
    """Multiple BasicResBlocks in sequence."""
    def __init__(self, channels, num_blocks=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            BasicResBlock(channels, dropout) for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)

# ============================================================
# MS-ASPP: Multi-Scale Atrous Spatial Pyramid Pooling
# ============================================================
class MSASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(2, 4, 8, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        ))
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            ))
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        total = out_channels * (len(dilations) + 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(total, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        g = self.global_branch(x)
        g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        feats.append(g)
        out = torch.cat(feats, dim=1)
        return self.fuse(out)

# ============================================================
# Fourier Convolution Block (Frequency Branch Core)
# ============================================================
class FourierBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.freq_conv_real = nn.Conv2d(channels, channels, 1, bias=False)
        self.freq_conv_imag = nn.Conv2d(channels, channels, 1, bias=False)
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        orig_dtype = x.dtype
        x_fp32 = x.float()

        x_fft = torch.fft.rfft2(x_fp32, norm='ortho') 
        real = x_fft.real
        imag = x_fft.imag

        with torch.cuda.amp.autocast(enabled=False):
            new_real = self.freq_conv_real(real) - self.freq_conv_imag(imag)
            new_imag = self.freq_conv_real(imag) + self.freq_conv_imag(real)
            x_fft_new = torch.complex(new_real, new_imag)
            x_spatial = torch.fft.irfft2(x_fft_new, s=x_fp32.shape[-2:], norm='ortho')

        x_spatial = x_spatial.to(orig_dtype)
        out = self.spatial_conv(x_spatial)
        out = self.norm(out)
        out = self.act(out)
        return out + res

class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_blocks=3, out_scales=(1, 2, 4, 8, 16)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            FourierBlock(base_channels) for _ in range(num_blocks)
        ])
        self.out_scales = out_scales
        self.base_channels = base_channels

    def forward(self, x):
        feat = self.stem(x)
        for blk in self.blocks:
            feat = blk(feat)
        multi_scale = {}
        for s in self.out_scales:
            if s == 1:
                multi_scale[s] = feat
            else:
                multi_scale[s] = F.avg_pool2d(feat, kernel_size=s, stride=s)
        return multi_scale

# ============================================================
# Fusion Gate
# ============================================================
class FusionGate(nn.Module):
    def __init__(self, spatial_ch, freq_ch):
        super().__init__()
        self.freq_proj = nn.Conv2d(freq_ch, spatial_ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(spatial_ch * 2, spatial_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(spatial_ch, spatial_ch, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(spatial_ch * 2, spatial_ch, 1)

    def forward(self, x_spatial, x_freq):
        if x_freq.shape[-2:] != x_spatial.shape[-2:]:
            x_freq = F.interpolate(x_freq, size=x_spatial.shape[-2:],
                                   mode='bilinear', align_corners=False)
        x_freq = self.freq_proj(x_freq)
        cat = torch.cat([x_spatial, x_freq], dim=1)
        g = self.gate(cat)
        fused = torch.cat([x_spatial, x_freq * g], dim=1)
        return self.fuse(fused)

# ============================================================
# Down / Up sampling
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
"""
DBF-UNet Building Blocks
Contains: RCAB, CBAM, MS-ASPP, Fourier Convolution, Fusion Gate
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
    """
    Residual block with CBAM attention.
    Input [C, H, W] -> Output [C, H, W]
    """
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
    """Multiple RCABs in sequence."""
    def __init__(self, channels, num_blocks=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            RCAB(channels, dropout) for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


# ============================================================
# MS-ASPP: Multi-Scale Atrous Spatial Pyramid Pooling
# ============================================================
class MSASPP(nn.Module):
    """
    Multi-scale ASPP for the bottleneck.
    Captures global context for streak artifacts and cupping.
    """
    def __init__(self, in_channels, out_channels, dilations=(2, 4, 8, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        ))
        # dilated branches
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            ))
        # global pooling branch
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
    """
    Fourier convolution: FFT -> complex 1x1 conv -> iFFT -> spatial conv.
    Natural global receptive field, great for sparse-view streaks.
    """
    def __init__(self, channels):
        super().__init__()
        # Complex weight is represented as 2 real weights (real + imag)
        # Use 1x1 conv in frequency domain (i.e., pointwise complex multiplication learned)
        self.freq_conv_real = nn.Conv2d(channels, channels, 1, bias=False)
        self.freq_conv_imag = nn.Conv2d(channels, channels, 1, bias=False)
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        # ============================================================
        # CRITICAL: Force FFT computation in FP32 to avoid NaN under AMP.
        # ComplexHalf (complex32) support in PyTorch is experimental and
        # produces NaN with many operators including Conv2d.
        # ============================================================
        orig_dtype = x.dtype
        x_fp32 = x.float()

        # FFT in FP32
        x_fft = torch.fft.rfft2(x_fp32, norm='ortho')  # complex64
        real = x_fft.real
        imag = x_fft.imag

        # Apply conv separately to real and imag parts (learnable complex 1x1)
        # Conv weights stay in FP32 since input is FP32 here
        with torch.cuda.amp.autocast(enabled=False):
            new_real = self.freq_conv_real(real) - self.freq_conv_imag(imag)
            new_imag = self.freq_conv_real(imag) + self.freq_conv_imag(real)
            x_fft_new = torch.complex(new_real, new_imag)
            # iFFT in FP32
            x_spatial = torch.fft.irfft2(x_fft_new, s=x_fp32.shape[-2:], norm='ortho')

        # Cast back to the original dtype (may be FP16 under AMP)
        x_spatial = x_spatial.to(orig_dtype)

        # Post spatial conv (normal AMP behavior)
        out = self.spatial_conv(x_spatial)
        out = self.norm(out)
        out = self.act(out)
        return out + res


class FrequencyBranch(nn.Module):
    """
    Frequency branch that produces multi-scale features
    to be fused with the spatial decoder.
    """
    def __init__(self, in_channels=1, base_channels=32, num_blocks=3, out_scales=(1, 2, 4, 8, 16)):
        """
        out_scales: downsample factors at which to output features
                    (matching the spatial encoder levels).
        """
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
        # Project to different channel counts for each scale
        # (will be set externally to match spatial branch channels)
        self.base_channels = base_channels

    def forward(self, x):
        """
        Returns a dict of features at different scales:
        {scale: feature_map}
        """
        feat = self.stem(x)
        for blk in self.blocks:
            feat = blk(feat)
        # Produce multi-scale features by pooling
        multi_scale = {}
        for s in self.out_scales:
            if s == 1:
                multi_scale[s] = feat
            else:
                multi_scale[s] = F.avg_pool2d(feat, kernel_size=s, stride=s)
        return multi_scale


# ============================================================
# Fusion Gate: fuse spatial + frequency features at each scale
# ============================================================
class FusionGate(nn.Module):
    """
    Gated fusion of spatial and frequency features.
    """
    def __init__(self, spatial_ch, freq_ch):
        super().__init__()
        # Project frequency features to match spatial channels
        self.freq_proj = nn.Conv2d(freq_ch, spatial_ch, 1, bias=False)
        # Gate network
        self.gate = nn.Sequential(
            nn.Conv2d(spatial_ch * 2, spatial_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(spatial_ch, spatial_ch, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(spatial_ch * 2, spatial_ch, 1)

    def forward(self, x_spatial, x_freq):
        # Align spatial size
        if x_freq.shape[-2:] != x_spatial.shape[-2:]:
            x_freq = F.interpolate(x_freq, size=x_spatial.shape[-2:],
                                   mode='bilinear', align_corners=False)
        x_freq = self.freq_proj(x_freq)
        cat = torch.cat([x_spatial, x_freq], dim=1)
        g = self.gate(cat)
        # Gated blend: g controls how much frequency info to inject
        fused = torch.cat([x_spatial, x_freq * g], dim=1)
        return self.fuse(fused)


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
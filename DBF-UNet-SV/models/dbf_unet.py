"""
Frequency UNet (Sparse-View Ablation Only)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    BasicResGroup, MSASPP, FrequencyBranch, FusionGate,
    Downsample, Upsample,
)

class FreqUNet_SparseView(nn.Module):
    """
    消融实验：仅保留频率分支和全局 MS-ASPP 模块。
    局部注意力机制 (RCAB) 被替换为基础的残差卷积 (BasicResGroup)。
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        num_levels=5,
        freq_channels=32,
        freq_blocks=3,
        dropout=0.0,
    ):
        super().__init__()
        self.num_levels = num_levels

        channels = []
        for i in range(num_levels):
            c = base_channels * (2 ** i)
            c = min(c, base_channels * 8) 
            channels.append(c)
        self.channels = channels

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.GELU(),
        )

        # ---- Spatial Encoder (Replaced RCAB with BasicResGroup) ----
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(num_levels):
            self.enc_blocks.append(BasicResGroup(channels[i], num_blocks=2, dropout=dropout))
            if i < num_levels - 1:
                self.downs.append(Downsample(channels[i], channels[i + 1]))

        # ---- Bottleneck: MS-ASPP (Kept for Sparse-view) ----
        self.bottleneck = nn.Sequential(
            MSASPP(channels[-1], channels[-1]),
            BasicResGroup(channels[-1], num_blocks=2, dropout=dropout),
        )

        # ---- Frequency Branch (Kept for Sparse-view) ----
        scales = [2 ** i for i in range(num_levels)]
        self.freq_branch = FrequencyBranch(
            in_channels=in_channels,
            base_channels=freq_channels,
            num_blocks=freq_blocks,
            out_scales=tuple(scales),
        )
        self.fusion_gates = nn.ModuleList([
            FusionGate(spatial_ch=channels[i], freq_ch=freq_channels)
            for i in range(num_levels)
        ])

        # ---- Decoder (Replaced RCAB with BasicResGroup) ----
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(num_levels - 1)):
            self.ups.append(Upsample(channels[i + 1], channels[i]))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(channels[i] * 2, channels[i], 3, padding=1),
                nn.GroupNorm(8, channels[i]),
                nn.GELU(),
                BasicResGroup(channels[i], num_blocks=2, dropout=dropout),
            ))

        # ---- Output Head ----
        self.head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0] // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels[0] // 2, out_channels, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head[-1].weight)
        if self.head[-1].bias is not None:
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        # Always run frequency branch for this ablation
        freq_feats = self.freq_branch(x) 

        feat = self.stem(x)
        skips = []
        
        # Encoder
        for i in range(self.num_levels):
            feat = self.enc_blocks[i](feat)
            # Fuse frequency features
            scale = 2 ** i
            freq_f = freq_feats[scale]
            feat = self.fusion_gates[i](feat, freq_f)
            
            if i < self.num_levels - 1:
                skips.append(feat)
                feat = self.downs[i](feat)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):
            feat = up(feat)
            skip = skips[-(i + 1)]
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            feat = torch.cat([feat, skip], dim=1)
            feat = dec(feat)

        # Output
        delta = self.head(feat)
        return x + delta


if __name__ == "__main__":
    model = FreqUNet_SparseView(base_channels=64, num_levels=5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f} M")
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Output: {y.shape}")
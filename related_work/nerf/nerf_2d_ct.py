#!/usr/bin/env python3
"""
2D NeRF/INR for fan-beam CT reconstruction (per-slice).

Pipeline per slice:
  1. Load sinogram + flat + dark, do Beer-Lambert preprocessing
  2. Apply degradation (sparse view, limited angle, etc.)
  3. Train 2D MLP (with positional encoding) to fit sinogram via fan-beam ray integration
  4. Sample MLP on rec_size×rec_size grid → reconstruction
  5. Save .npy + evaluate against GT + visualize

Usage:
  python nerf_2d_ct.py \\
      --data_dir /ibex/.../2DeteCT_slicesAll \\
      --gt_dir /ibex/.../2DeteCT_slices_RecSeg_All \\
      --slices 1 100 \\
      --mode 3p \\
      --ang_subsamp 6 \\
      --n_iter 2000

Geometry follows reconstruct.py / ASTRA fanflat:
  - source at (-SOD sin θ, SOD cos θ)
  - detector pixel d at (d cos θ + (SDD-SOD) sin θ, d sin θ - (SDD-SOD) cos θ)
  - All in scaled coords (rec_size 1024 = full FOV)
"""

import argparse
import glob
import json
import os
import sys
import time
import uuid
import warnings
from datetime import datetime

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")   # 必须在 pyplot 导入前
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
#  Constants (match reconstruct.py / ASTRA)
# ============================================================
TOTAL_PROJECTIONS = 3601
DET_PIX = 0.0748
SOD = 431.019989
SDD = 529.000488
CORR = np.array([1.00, 0.0])
N_DET_BINNED = 956


def mode_dir_name(mode):
    return "mode3_Poisson_noisy" if mode == "3p" else f"mode{mode}"


# ============================================================
#  Data Loading (copy of reconstruct.py's load_and_preprocess)
# ============================================================
def load_and_preprocess(current_path, slice_idx, mode="1"):
    """Load raw data → 2x binning → dark/flat normalization → Beer-Lambert.

    Returns:
        data: (3600, 956) array, line integral p = -log((sino-dark)/(flat-dark))
    """
    if mode == "3p":
        dark = imageio.imread(os.path.join(current_path, 'dark.tif')).astype('float32')
        flat1 = imageio.imread(os.path.join(current_path, 'flat1_scaled.tif')).astype('float32')
        flat2 = imageio.imread(os.path.join(current_path, 'flat2_scaled.tif')).astype('float32')
        sinogram = imageio.imread(os.path.join(current_path, 'sinogram_with_poisson.tif')).astype('float32')
    else:
        dark = imageio.imread(os.path.join(current_path, 'dark.tif')).astype('float32')
        flat1 = imageio.imread(os.path.join(current_path, 'flat1.tif')).astype('float32')
        flat2 = imageio.imread(os.path.join(current_path, 'flat2.tif')).astype('float32')
        sinogram = imageio.imread(os.path.join(current_path, 'sinogram.tif')).astype('float32')

    flat = np.mean(np.array([flat1, flat2]), axis=0)

    # 2x pixel binning
    sino_binned = sinogram[:, 0::2] + sinogram[:, 1::2]
    dark_binned = dark[0, 0::2] + dark[0, 1::2]
    flat_binned = flat[0, 0::2] + flat[0, 1::2]

    # Dark sub + flat norm
    data = (sino_binned - dark_binned) / (flat_binned - dark_binned)
    data = data[:-1, :]

    # Detector shift correction (in linear domain, before -log)
    if slice_idx <= 2830 or (5521 <= slice_idx <= 5870):
        det_shift = CORR[0] * DET_PIX
    else:
        det_shift = CORR[1] * DET_PIX
    det_grid = np.arange(0, N_DET_BINNED) * DET_PIX
    data = interp1d(det_grid, data, kind='linear',
                    fill_value='extrapolate')(det_grid + det_shift)

    # Beer-Lambert
    data = -np.log(np.clip(data, 1e-6, None))
    return data  # (3600, 956)


def apply_degradation(data, all_angles, args):
    """Apply sparse view, limited angle, Poisson noise on sinogram."""
    angles = all_angles.copy()
    tag_parts = []

    # Limited angle
    if args.max_ang < 360.0:
        max_rad = np.deg2rad(args.max_ang)
        mask = angles <= max_rad
        angles = angles[mask]
        data = data[mask, :]
        tag_parts.append(f"limang{int(args.max_ang)}")

    # Sparse view
    if args.ang_subsamp > 1:
        angles = angles[::args.ang_subsamp]
        data = data[::args.ang_subsamp, :]
        tag_parts.append(f"sparse{args.ang_subsamp}x")

    tag = "_".join(tag_parts) if tag_parts else "full"
    return data, angles, tag


# ============================================================
#  Geometry (fan-beam ray construction)
# ============================================================
def compute_scale():
    """Match reconstruct.py's setup_geometry scale factor."""
    det_pix_sz = 2 * DET_PIX  # after binning
    return 1.0 / ((det_pix_sz * N_DET_BINNED * SOD / SDD) / 1024)


def get_sources_and_detectors(angles, n_det=N_DET_BINNED):
    """Compute source and detector pixel positions (in scaled coords).

    Convention VERIFIED against ASTRA's fanflat default (via geom_2vec):
      angle=0 → source at (0, -SOD), detector center at (0, +(SDD-SOD)),
      detector pixel +d direction along +X axis at angle 0.
      Rotate CCW by angle (math convention).

    Verification (at angle 0):
      ASTRA: source = (0, -3787.6),  detector_d=0 = (-627.7, +861.0)  ✓
      Mine:  source = (0, -3787.6),  detector_d=0 = (-627.7, +861.0)  ✓

    Returns:
        sources:   (n_angles, 2) — (x, y) per angle
        detectors: (n_angles, n_det, 2) — (x, y) per (angle, det_pixel)
    """
    scale = compute_scale()
    sod_s = SOD * scale
    sdd_minus_sod_s = (SDD - SOD) * scale
    det_pix_s = (2 * DET_PIX) * scale

    angles = np.asarray(angles, dtype=np.float64)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    # Source at (0, -SOD) rotated CCW by θ → (SOD sin θ, -SOD cos θ)
    sources = np.stack([sod_s * sin_t, -sod_s * cos_t], axis=-1)  # (n_a, 2)

    # Detector pixel offsets along +X at angle 0
    det_offsets = (np.arange(n_det) - (n_det - 1) / 2.0) * det_pix_s  # (n_det,)

    # Detector pixel d at (d, +(SDD-SOD)) at angle 0, rotate CCW by θ:
    #   x = d cos θ - (SDD-SOD) sin θ
    #   y = d sin θ + (SDD-SOD) cos θ
    d = det_offsets[None, :]  # (1, n_det)
    det_x = d * cos_t[:, None] - sdd_minus_sod_s * sin_t[:, None]
    det_y = d * sin_t[:, None] + sdd_minus_sod_s * cos_t[:, None]
    detectors = np.stack([det_x, det_y], axis=-1)  # (n_a, n_det, 2)

    return sources.astype(np.float32), detectors.astype(np.float32)


# ============================================================
#  Model: 2D INR (NeRF-style MLP with positional encoding)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, freq_bands=10):
        super().__init__()
        self.freq_bands = freq_bands
        self.out_dim = 2 + 4 * freq_bands  # 2 (raw) + 2 dims × 2 (sin/cos) × n_bands

    def forward(self, x):
        # x: (..., 2)
        out = [x]
        for i in range(self.freq_bands):
            freq = 2.0 ** i
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        return torch.cat(out, dim=-1)


class INR2D(nn.Module):
    """2D MLP with positional encoding, NeRF-style with skip connection.

    使用 exp 激活避免 Softplus 饱和导致的梯度消失。
    init bias 调到合适值，使初始 p_pred ≈ p_target average，避免"塌到 0"。
    """
    def __init__(self, hidden=256, n_layers=8, freq_bands=10, skip=4,
                 output_scale=0.02, init_bias=-3.0):
        super().__init__()
        self.pe = PositionalEncoding(freq_bands)
        in_dim = self.pe.out_dim
        self.skip = skip
        self.n_layers = n_layers
        self.output_scale = output_scale

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden))
            elif i == skip:
                layers.append(nn.Linear(hidden + in_dim, hidden))
            else:
                layers.append(nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden, 1)

        # Init: bias such that exp(bias) gives reasonable initial μ
        # exp(-4) ≈ 0.018 → μ_init ≈ 0.018 × output_scale = 1.8e-4
        # For ray length 4650, n_samples 128, step 36:
        #   p_init = 128 × 1.8e-4 × 36 ≈ 0.83 (close to target avg of ~1-2)
        nn.init.constant_(self.out.bias, init_bias)
        # Smaller weights on output layer to keep init stable
        nn.init.uniform_(self.out.weight, -0.01, 0.01)

    def forward(self, x):
        # x: (..., 2) in [-1, 1]
        feat = self.pe(x)
        h = feat
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, feat], dim=-1)
            h = torch.relu(layer(h))
        # exp activation: 不会像 softplus 那样在负无穷饱和
        # clamp 防止溢出 (exp(20) ≈ 5e8 已经很大)
        raw = self.out(h).squeeze(-1).clamp(max=20.0)
        mu = torch.exp(raw) * self.output_scale
        return mu


class Lineformer(nn.Module):
    """Simple Lineformer: 只看 μ values，学 per-sample integration weights。

    forward signature 跟 LineformerSAX 一致 (统一调用接口)，但不使用 points 参数。
    """
    def __init__(self, dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.proj_in = nn.Linear(1, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4*dim,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(dim, 1)

    def forward(self, mu_along_ray, points_along_ray, step_size):
        # mu_along_ray: (b, n)
        # points_along_ray: (b, n, 2) — IGNORED in simple version
        # step_size: (b,)
        x = mu_along_ray.unsqueeze(-1)  # (b, n, 1)
        x = self.proj_in(x)
        x = self.transformer(x)
        # Predict per-sample weight (multiplicative on uniform integration)
        weights = torch.sigmoid(self.proj_out(x)).squeeze(-1) * 2.0  # (b, n), in [0, 2]
        # Integrate: sum(mu * weight * step)
        p_pred = (mu_along_ray * weights).sum(dim=-1) * step_size
        return p_pred


class LineformerSAX(nn.Module):
    """SAX-NeRF 论文风格的 Lineformer (CVPR 2024)。

    跟 simple Lineformer 的区别：
      1. 输入除 μ 外，还包含每个 sample 的 (x, y) 位置 + 位置编码
      2. 多层 transformer encoder + GeLU + pre-norm（更稳定）
      3. 输出 per-sample 的 μ 残差修正（refined density），不是简单 weight
      4. residual 连接保留原 MLP 的 μ，让 transformer 只学修正项

    论文里这个 module 用来"沿射线建模结构信息"，对边缘和细节更敏感。
    """
    def __init__(self, dim=128, n_heads=4, n_layers=4, freq_bands=6,
                 residual_scale=1e-3):
        super().__init__()
        self.freq_bands = freq_bands
        self.residual_scale = residual_scale

        # 输入: μ (1) + raw xy (2) + sin/cos PE (4 × freq_bands)
        in_dim = 1 + 2 + 4 * freq_bands

        self.proj_in = nn.Linear(in_dim, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4*dim,
            batch_first=True, dropout=0.0,
            activation='gelu', norm_first=True,  # pre-norm 更稳
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(dim, 1)

        # Init output 接近 0，让初始 refinement ≈ 0（保留原 MLP μ）
        nn.init.uniform_(self.proj_out.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.proj_out.bias)

    def _positional_encoding(self, x):
        # x: (..., 2) in [-1, 1]
        out = []
        for i in range(self.freq_bands):
            freq = 2.0 ** i
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        return torch.cat(out, dim=-1)  # (..., 4 × freq_bands)

    def forward(self, mu_along_ray, points_along_ray, step_size):
        """
        Args:
            mu_along_ray:    (b, n) — NeRF MLP 输出的 μ
            points_along_ray:(b, n, 2) — sample 位置 in [-1, 1]
            step_size:       (b,) — per-sample 积分步长
        Returns:
            p_pred: (b,) — refined projection prediction
        """
        # 拼接特征: μ + xy + PE
        pe = self._positional_encoding(points_along_ray)            # (b, n, 4f)
        feat = torch.cat([
            mu_along_ray.unsqueeze(-1),                              # (b, n, 1)
            points_along_ray,                                         # (b, n, 2)
            pe,                                                       # (b, n, 4f)
        ], dim=-1)                                                    # (b, n, 1+2+4f)

        # Transformer encoder
        x = self.proj_in(feat)                                       # (b, n, dim)
        x = self.transformer(x)                                      # (b, n, dim)

        # Per-sample refinement (small residual on top of MLP μ)
        refinement = self.proj_out(x).squeeze(-1)                    # (b, n)
        mu_refined = torch.clamp(
            mu_along_ray + refinement * self.residual_scale,
            min=0,
        )                                                             # (b, n)

        # Integrate
        p_pred = mu_refined.sum(dim=-1) * step_size                  # (b,)
        return p_pred


# ============================================================
#  Training (per-slice optimization)
# ============================================================
def sample_along_rays(sources, detectors, ang_idx, det_idx, n_samples,
                      jitter=True, fov_radius=None, device='cuda'):
    """Sample n_samples points along each ray, optionally clipped to FOV.

    fov_radius: 如果指定，先求 ray 与该半径的圆的交点 [t_in, t_out]，
                只在物体范围内采样，**采样密度提升 4-5 倍**（核心优化！）

    Returns:
        points: (batch, n_samples, 2)
        step_size: (batch,) — per-sample integration step in scaled units
    """
    S = sources[ang_idx]  # (b, 2)
    D = detectors[ang_idx, det_idx]  # (b, 2)
    direction = D - S  # (b, 2)
    ray_length_full = torch.norm(direction, dim=-1)  # (b,)

    if fov_radius is not None:
        # 求 ray 与半径 fov_radius 圆的交点
        # ||S + t*d||^2 = R^2  =>  a t^2 + b t + c = 0
        # a = |d|^2, b = 2 S·d, c = |S|^2 - R^2
        a = (direction ** 2).sum(dim=-1)
        b_quad = 2 * (S * direction).sum(dim=-1)
        c = (S ** 2).sum(dim=-1) - fov_radius ** 2
        disc = b_quad ** 2 - 4 * a * c
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0))
        t_in = ((-b_quad - sqrt_disc) / (2 * a + 1e-12)).clamp(0, 1)
        t_out = ((-b_quad + sqrt_disc) / (2 * a + 1e-12)).clamp(0, 1)
        # 如果 disc < 0（ray 完全错过 FOV），t_in = t_out = 同一值，
        # 后续 step_size = 0 自动忽略
    else:
        t_in = torch.zeros(S.shape[0], device=device, dtype=torch.float32)
        t_out = torch.ones(S.shape[0], device=device, dtype=torch.float32)

    # Stratified sampling in [t_in, t_out]
    t_unit = (torch.arange(n_samples, device=device, dtype=torch.float32) + 0.5) / n_samples
    if jitter:
        jit = (torch.rand(n_samples, device=device) - 0.5) / n_samples
        t_unit = (t_unit + jit).clamp(0, 1)
    # Map [0,1] → [t_in, t_out] per ray
    t = t_in[:, None] + t_unit[None, :] * (t_out - t_in)[:, None]  # (b, n)

    # Points
    points = S[:, None, :] + t[:, :, None] * direction[:, None, :]  # (b, n, 2)

    # Step size in scaled units (only inside FOV)
    inside_length = (t_out - t_in) * ray_length_full
    step_size = inside_length / n_samples  # (b,)
    return points, step_size


def train_one_slice(p_target, sources, detectors, args, device='cuda',
                     log_prefix=""):
    """Train 2D INR for one slice.

    Args:
        p_target: (n_angles, n_det) numpy array, line integrals
        sources: (n_angles, 2) numpy array
        detectors: (n_angles, n_det, 2) numpy array

    Returns:
        recon: (rec_size, rec_size) numpy array
    """
    n_angles, n_det = p_target.shape

    # To tensor
    p_t = torch.tensor(p_target, dtype=torch.float32, device=device)  # (n_a, n_det)
    src_t = torch.tensor(sources, dtype=torch.float32, device=device)
    det_t = torch.tensor(detectors, dtype=torch.float32, device=device)

    # Coordinate normalization: divide by rec_size/2 to get [-1, 1]
    # The reconstruction grid is centered at origin with extent ±rec_size/2
    norm_scale = args.rec_size / 2.0

    # Model
    model = INR2D(
        hidden=args.hidden_dim,
        n_layers=args.n_layers,
        freq_bands=args.freq_bands,
        output_scale=args.output_scale,
        init_bias=args.init_bias,
    ).to(device)

    use_lineformer = args.use_lineformer
    if use_lineformer:
        if args.lineformer_type == "sax":
            lineformer = LineformerSAX(
                dim=args.lineformer_dim,
                n_heads=4,
                n_layers=args.lineformer_layers,
                freq_bands=args.lineformer_pe_bands,
            ).to(device)
        else:
            lineformer = Lineformer(dim=args.lineformer_dim).to(device)
        params = list(model.parameters()) + list(lineformer.parameters())
    else:
        params = list(model.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.n_iter // 3), gamma=0.5
    )

    pbar = tqdm(range(args.n_iter), desc=f"{log_prefix}NeRF train",
                dynamic_ncols=True, leave=False)
    last_loss = float('inf')
    for it in pbar:
        # Random batch of (angle, det) pairs
        ang_idx = torch.randint(0, n_angles, (args.batch_rays,), device=device)
        det_idx = torch.randint(0, n_det, (args.batch_rays,), device=device)

        # Sample points along rays (clipped to FOV for efficient sampling)
        # FOV radius = rec_size/2 in scaled coords
        points, step_size = sample_along_rays(
            src_t, det_t, ang_idx, det_idx, args.n_samples,
            jitter=True, fov_radius=norm_scale, device=device
        )

        # Normalize coords to [-1, 1] for MLP
        points_norm = points / norm_scale

        # MLP forward (batch × n_samples points)
        flat_points = points_norm.reshape(-1, 2)
        mu_flat = model(flat_points)
        mu = mu_flat.reshape(args.batch_rays, args.n_samples)

        # Integrate
        if use_lineformer:
            # Both Lineformer (simple) 和 LineformerSAX 现在统一接受 (mu, points_norm, step_size)
            # simple 内部会忽略 points_norm
            p_pred = lineformer(mu, points_norm, step_size)
        else:
            p_pred = (mu.sum(dim=-1)) * step_size

        # Target
        p_t_batch = p_t[ang_idx, det_idx]

        # Loss
        loss = ((p_pred - p_t_batch) ** 2).mean()
        if args.tv_weight > 0:
            # Simple L1 TV-like reg on neighboring sample diff
            tv = (mu[:, 1:] - mu[:, :-1]).abs().mean()
            loss = loss + args.tv_weight * tv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        last_loss = loss.item()
        if (it + 1) % 100 == 0:
            pbar.set_postfix(loss=f"{last_loss:.6f}")

    pbar.close()

    # ─── Final reconstruction: sample MLP on rec_size × rec_size grid ───
    # ASTRA convention: row 0 of vol = +Y_max (top of display), so we use
    # decreasing ys to make recon[0, :] correspond to top of physical space.
    model.eval()
    with torch.no_grad():
        xs = torch.linspace(-norm_scale, norm_scale, args.rec_size, device=device)
        ys = torch.linspace(norm_scale, -norm_scale, args.rec_size, device=device)  # +Y at top
        gx, gy = torch.meshgrid(xs, ys, indexing='xy')
        grid = torch.stack([gx, gy], dim=-1) / norm_scale  # in [-1, 1]

        # Process in tiles to save memory
        recon_flat = []
        flat_grid = grid.reshape(-1, 2)
        tile_size = 65536  # 64K points per tile
        for i in range(0, flat_grid.shape[0], tile_size):
            tile = flat_grid[i:i + tile_size]
            mu_tile = model(tile)
            recon_flat.append(mu_tile.cpu())
        recon = torch.cat(recon_flat).reshape(args.rec_size, args.rec_size).numpy()

    return recon, last_loss


# ============================================================
#  Evaluation (matches evaluate.py)
# ============================================================
def load_gt(gt_dir, slice_idx, gt_mode=2):
    gt_path = os.path.join(gt_dir, f"slice{slice_idx:05d}",
                            f"mode{gt_mode}", "reconstruction.tif")
    if not os.path.exists(gt_path):
        return None
    return imageio.imread(gt_path).astype(np.float32)


def match_sizes(gt, rec):
    """Center-crop the larger one to match the smaller one's size."""
    if gt.shape == rec.shape:
        return gt, rec
    gt_h, gt_w = gt.shape
    rec_h, rec_w = rec.shape
    if rec_h > gt_h or rec_w > gt_w:
        y0 = (rec_h - gt_h) // 2
        x0 = (rec_w - gt_w) // 2
        return gt, rec[y0:y0 + gt_h, x0:x0 + gt_w]
    else:
        y0 = (gt_h - rec_h) // 2
        x0 = (gt_w - rec_w) // 2
        return gt[y0:y0 + rec_h, x0:x0 + rec_w], rec


def normalize_for_metrics(gt, rec):
    """Normalize to [0, 1] using GT's range."""
    vmin, vmax = float(gt.min()), float(gt.max())
    if vmax - vmin < 1e-12:
        return gt, rec, 1.0
    gt_n = (gt - vmin) / (vmax - vmin)
    rec_n = np.clip((rec - vmin) / (vmax - vmin), 0, 1)
    return gt_n, rec_n, 1.0


def evaluate_one(gt, rec):
    """Compute all metrics for one (gt, rec) pair."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    gt, rec = match_sizes(gt, rec)
    gt_n, rec_n, dr = normalize_for_metrics(gt, rec)

    metrics = {}
    metrics['PSNR'] = float(peak_signal_noise_ratio(gt_n, rec_n, data_range=dr))
    metrics['SSIM'] = float(structural_similarity(gt_n, rec_n, data_range=dr))
    metrics['RMSE'] = float(np.sqrt(np.mean((gt_n - rec_n) ** 2)))
    metrics['MAE'] = float(np.mean(np.abs(gt_n - rec_n)))
    metrics['MaxErr'] = float(np.max(np.abs(gt_n - rec_n)))
    metrics['MSE'] = float(np.mean((gt_n - rec_n) ** 2))
    sig_p = float(np.sum(gt_n ** 2))
    nz_p = float(np.sum((gt_n - rec_n) ** 2))
    metrics['SNR'] = float(10 * np.log10(sig_p / nz_p)) if nz_p > 0 else float('inf')

    # MS-SSIM (optional)
    try:
        from pytorch_msssim import ms_ssim as compute_ms_ssim
        gt_t = torch.from_numpy(gt_n).float().unsqueeze(0).unsqueeze(0).clamp(0)
        rec_t = torch.from_numpy(rec_n).float().unsqueeze(0).unsqueeze(0).clamp(0)
        h = gt.shape[0]
        if h >= 160:
            metrics['MS-SSIM'] = float(compute_ms_ssim(gt_t, rec_t, data_range=1.0, win_size=7).item())
        else:
            n_scales = max(1, int(np.log2(h / 10)))
            weights = [1.0 / n_scales] * n_scales
            metrics['MS-SSIM'] = float(compute_ms_ssim(
                gt_t, rec_t, data_range=1.0, win_size=7, weights=weights
            ).item())
    except Exception:
        metrics['MS-SSIM'] = float('nan')

    return metrics


# ============================================================
#  Visualization
# ============================================================
def visualize_one(recon, gt, sino_degraded, slice_idx, save_dir, args):
    """Save a multi-panel comparison figure."""
    os.makedirs(save_dir, exist_ok=True)

    if gt is not None:
        gt_m, rec_m = match_sizes(gt, recon)
        diff = rec_m - gt_m
    else:
        gt_m, rec_m = None, recon
        diff = None

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Sinogram (degraded input)
    ax = axes[0]
    im = ax.imshow(sino_degraded, cmap='gray', aspect='auto')
    ax.set_title(f"Input sinogram (n_angles={sino_degraded.shape[0]})")
    ax.set_xlabel("detector")
    ax.set_ylabel("angle")
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # NeRF reconstruction
    ax = axes[1]
    vmin, vmax = float(rec_m.min()), float(np.percentile(rec_m, 99.5))
    im = ax.imshow(rec_m, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f"NeRF recon ({rec_m.shape[0]}×{rec_m.shape[1]})")
    ax.axis('off')
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # GT
    ax = axes[2]
    if gt_m is not None:
        im = ax.imshow(gt_m, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"GT (mode 2)")
    else:
        ax.text(0.5, 0.5, "GT not available", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("GT")
    ax.axis('off')
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    if gt_m is not None:
        plt.colorbar(im, cax=cax)

    # Error map
    ax = axes[3]
    if diff is not None:
        emax = float(np.percentile(np.abs(diff), 99.5))
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-emax, vmax=emax)
        ax.set_title(f"Error (recon - GT), max={emax:.4f}")
    else:
        ax.text(0.5, 0.5, "no diff", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("Error")
    ax.axis('off')
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    if diff is not None:
        plt.colorbar(im, cax=cax)

    plt.suptitle(f"Slice {slice_idx:05d} | NeRF | mode={args.mode} "
                 f"| views={sino_degraded.shape[0]} | iter={args.n_iter}")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"slice{slice_idx:05d}_compare.png"),
                dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================
#  Main pipeline
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="2D NeRF for fan-beam CT")

    # Data
    p.add_argument("--data_dir", type=str,
                   default="/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_slicesAll")
    p.add_argument("--mode3_noisy_dir", type=str,
                   default="/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_slicesAll_mode3_noisy")
    p.add_argument("--gt_dir", type=str,
                   default="/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_RecSeg/2DeteCT_slices_RecSeg_All")
    p.add_argument("--out_dir", type=str, default="./results")
    p.add_argument("--slices", type=int, nargs=2, default=[1, 100],
                   help="Slice range [start, end] inclusive")
    p.add_argument("--mode", type=str, default="3p",
                   help="1, 2, 3, or 3p")
    p.add_argument("--gt_mode", type=int, default=2,
                   help="Mode for GT (default mode 2 reconstruction.tif)")

    # Degradation
    p.add_argument("--ang_subsamp", type=int, default=6,
                   help="Sparse view subsampling factor (1=full)")
    p.add_argument("--max_ang", type=float, default=360.0,
                   help="Limited angle in degrees (360=full)")

    # NeRF training
    p.add_argument("--n_iter", type=int, default=2000,
                   help="Training iterations per slice")
    p.add_argument("--batch_rays", type=int, default=4096,
                   help="Number of rays per training batch")
    p.add_argument("--n_samples", type=int, default=128,
                   help="Number of samples along each ray")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--freq_bands", type=int, default=10,
                   help="Positional encoding frequency bands")
    p.add_argument("--output_scale", type=float, default=0.02,
                   help="Multiplicative scale on MLP output (matches expected μ range; tuned for FOV-clipped sampling)")
    p.add_argument("--init_bias", type=float, default=-3.0,
                   help="Init bias for output layer; -3 gives μ_init ≈ 0.001 → p_init ≈ 1 with FOV clipping")
    p.add_argument("--flip", type=str, default="none",
                   choices=["none", "calibrate", "rot180", "vflip", "hflip", "rot90", "rot270",
                            "transpose", "antitranspose"],
                   help="Orientation correction.\n"
                        "  'none' (default, safe): no post-processing flip\n"
                        "  'calibrate': use ONLY slice 1 + GT to find best flip, then APPLY SAME flip "
                        "to all subsequent slices (avoids per-slice GT leakage)\n"
                        "  'rot90'/'rot180'/etc: hardcoded flip applied to all slices\n"
                        "  WARNING: avoid 'auto' (per-slice GT-based selection — leaks GT into evaluation!)")
    p.add_argument("--tv_weight", type=float, default=0.0,
                   help="Total variation regularization weight (0 = off)")
    p.add_argument("--use_lineformer", action="store_true",
                   help="Use Lineformer for ray integration")
    p.add_argument("--lineformer_type", type=str, default="simple",
                   choices=["simple", "sax"],
                   help="Lineformer architecture: 'simple' (only μ) or 'sax' (paper-style: μ+xy+PE)")
    p.add_argument("--lineformer_dim", type=int, default=64,
                   help="Transformer hidden dim (simple default 64, sax recommend 128)")
    p.add_argument("--lineformer_layers", type=int, default=4,
                   help="Number of transformer layers (sax type only; simple uses 2)")
    p.add_argument("--lineformer_pe_bands", type=int, default=6,
                   help="Positional encoding bands inside SAX Lineformer")

    # Reconstruction grid
    p.add_argument("--rec_size", type=int, default=1024,
                   help="Reconstruction grid size")

    # System
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_vis", type=int, default=10,
                   help="Number of slices to visualize (evenly spaced)")
    p.add_argument("--method_tag", type=str, default="NeRF",
                   help="Method name for output dir (NeRF or NeRF_LF)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute slices even if .npy exists")
    p.add_argument("--no_eval", action="store_true",
                   help="Skip evaluation (no GT comparison)")
    return p.parse_args()


def build_exp_dir(args):
    """Build experiment directory name following reconstruct.py convention."""
    run_id = datetime.now().strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:4]
    method = args.method_tag
    if args.use_lineformer:
        # Differentiate simple vs SAX Lineformer in folder name
        suffix = "_LFsax" if args.lineformer_type == "sax" else "_LF"
        if suffix not in method:
            method = method + suffix
    n_eff = 3600 // args.ang_subsamp
    if args.max_ang < 360:
        n_eff = min(n_eff, int(3600 * args.max_ang / 360) // args.ang_subsamp)

    tag_parts = [
        run_id,
        f"slices{args.slices[0]}-{args.slices[1]}",
        f"mode{args.mode}",
        f"angles{n_eff}",
        f"subsamp{args.ang_subsamp}x",
        f"maxang{int(args.max_ang)}",
        f"method_{method}",
        f"iter{args.n_iter}",
        f"rec{args.rec_size}",
        f"hd{args.hidden_dim}_layers{args.n_layers}_pe{args.freq_bands}",
        f"lr{args.lr}",
    ]
    return "__".join(tag_parts), method


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(args.gpu_index)}")

    # Find slice folders
    if args.mode == "3p":
        data_root = args.mode3_noisy_dir
        slice_subfolder = "mode3"
    else:
        data_root = args.data_dir
        slice_subfolder = f"mode{args.mode}"
    all_folders = sorted(glob.glob(os.path.join(data_root, "slice?????")))
    slice_list = []
    for f in all_folders:
        try:
            idx = int(os.path.basename(f).replace("slice", ""))
        except ValueError:
            continue
        if args.slices[0] <= idx <= args.slices[1]:
            slice_list.append((idx, os.path.join(f, slice_subfolder)))

    if not slice_list:
        print(f"ERROR: No slice folders in {data_root} for range {args.slices}")
        sys.exit(1)
    print(f"Found {len(slice_list)} slices in range {args.slices}")

    # Build exp dir
    exp_tag, method_name = build_exp_dir(args)
    exp_dir = os.path.join(args.out_dir, exp_tag)
    out_subdir = os.path.join(exp_dir, mode_dir_name(args.mode), method_name)
    os.makedirs(out_subdir, exist_ok=True)
    vis_dir = os.path.join(exp_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Save config
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "n_slices": len(slice_list),
        "parameters": vars(args),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # CSV header
    csv_path = os.path.join(exp_dir, "metrics.csv")
    csv_cols = ["slice_idx", "PSNR", "SSIM", "MS-SSIM", "RMSE", "MAE",
                "MaxErr", "MSE", "SNR", "train_loss", "wallclock_sec"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(",".join(csv_cols) + "\n")

    # Visualization indices: evenly spaced
    n_total = len(slice_list)
    if args.n_vis > 0 and n_total > 0:
        vis_set = set(int(round(i * (n_total - 1) / max(1, args.n_vis - 1)))
                      for i in range(args.n_vis))
    else:
        vis_set = set()

    # Pre-compute angles
    all_angles_full = np.linspace(0, 2 * np.pi, TOTAL_PROJECTIONS)[:-1]  # 3600

    # Per-slice loop
    n_done = 0
    n_skip = 0
    n_fail = 0
    t_global = time.time()
    metric_collector = []

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_tag}")
    print(f"Output:     {exp_dir}")
    print(f"Mode:       {args.mode}")
    print(f"Sparse:     ang_subsamp={args.ang_subsamp}, max_ang={args.max_ang}")
    print(f"NeRF:       n_iter={args.n_iter}, hidden={args.hidden_dim}, "
          f"layers={args.n_layers}, lr={args.lr}")
    if args.use_lineformer:
        if args.lineformer_type == "sax":
            print(f"  + Lineformer-SAX (dim={args.lineformer_dim}, "
                  f"layers={args.lineformer_layers}, pe_bands={args.lineformer_pe_bands})")
        else:
            print(f"  + Lineformer-simple (dim={args.lineformer_dim})")
    print(f"{'='*60}\n")

    for ti, (slice_idx, slice_path) in enumerate(slice_list):
        out_path = os.path.join(out_subdir, f"slice{slice_idx:05d}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue

        t_slice = time.time()
        log_prefix = f"[{ti+1}/{n_total}] slice{slice_idx:05d} "

        # Load + preprocess
        try:
            sino = load_and_preprocess(slice_path, slice_idx, mode=args.mode)
        except Exception as e:
            print(f"{log_prefix}LOAD FAILED: {e}")
            n_fail += 1
            continue

        # Apply degradation
        sino_deg, angles_deg, dtag = apply_degradation(sino, all_angles_full, args)

        # Geometry
        sources, detectors = get_sources_and_detectors(angles_deg, n_det=N_DET_BINNED)

        # Train + reconstruct
        try:
            recon, last_loss = train_one_slice(
                sino_deg, sources, detectors, args,
                device=device, log_prefix=log_prefix
            )
        except Exception as e:
            print(f"{log_prefix}TRAIN FAILED: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1
            continue

        # Ensure non-negative
        recon = np.maximum(recon, 0).astype(np.float32)

        # ─── Apply orientation correction to align with GT ───
        # 8 dihedral group operations
        ORIENTATIONS = {
            "none":          lambda x: x,
            "vflip":         lambda x: x[::-1, :],
            "hflip":         lambda x: x[:, ::-1],
            "rot180":        lambda x: x[::-1, ::-1],
            "rot90":         lambda x: np.rot90(x, 1),
            "rot270":        lambda x: np.rot90(x, 3),
            "transpose":     lambda x: x.T,
            "antitranspose": lambda x: np.rot90(x, 1).T,
        }
        if args.flip == "calibrate":
            # 'calibrate' mode: use ONLY the first processed slice to determine
            # the best flip, then save it to a state variable and use SAME flip
            # for all subsequent slices. This avoids per-slice GT leakage.
            if not hasattr(main, '_calibrated_flip'):
                # First slice: probe with GT
                gt_for_orient = load_gt(args.gt_dir, slice_idx, args.gt_mode)
                if gt_for_orient is not None:
                    from skimage.metrics import peak_signal_noise_ratio
                    best_op, best_psnr = "none", -np.inf
                    for op_name, op_func in ORIENTATIONS.items():
                        try:
                            r_try = np.ascontiguousarray(op_func(recon))
                            gt_m, r_m = match_sizes(gt_for_orient, r_try)
                            gt_n, r_n, dr = normalize_for_metrics(gt_m, r_m)
                            p = peak_signal_noise_ratio(gt_n, r_n, data_range=dr)
                            if p > best_psnr:
                                best_psnr, best_op = p, op_name
                        except Exception:
                            continue
                    main._calibrated_flip = best_op
                    print(f"{log_prefix}calibration done: best flip = '{best_op}' "
                          f"(slice 1 PSNR={best_psnr:.2f})")
                    print(f"  → All subsequent slices will use '{best_op}' (no further GT peeking)")
                else:
                    main._calibrated_flip = "none"
                    print(f"{log_prefix}calibration FAILED (no GT for slice 1) → using 'none'")
            # Apply calibrated flip
            recon = np.ascontiguousarray(
                ORIENTATIONS[main._calibrated_flip](recon)
            ).astype(np.float32)
        elif args.flip in ORIENTATIONS:
            recon = np.ascontiguousarray(ORIENTATIONS[args.flip](recon)).astype(np.float32)
        # "none" → no change (default, safe)

        # Save
        np.save(out_path, recon)

        wallclock = time.time() - t_slice

        # Evaluate
        gt = None
        metrics = {k: float('nan') for k in
                   ['PSNR', 'SSIM', 'MS-SSIM', 'RMSE', 'MAE', 'MaxErr', 'MSE', 'SNR']}
        if not args.no_eval:
            gt = load_gt(args.gt_dir, slice_idx, args.gt_mode)
            if gt is not None:
                try:
                    metrics = evaluate_one(gt, recon)
                except Exception as e:
                    print(f"{log_prefix}EVAL FAILED: {e}")

        # Visualize sample slices
        if ti in vis_set:
            try:
                visualize_one(recon, gt, sino_deg, slice_idx, vis_dir, args)
                vis_path = os.path.join(vis_dir, f"slice{slice_idx:05d}_compare.png")
                print(f"{log_prefix}vis saved: {vis_path}", flush=True)
            except Exception as e:
                import traceback
                print(f"{log_prefix}VIS FAILED: {e}", flush=True)
                traceback.print_exc()

        # Append CSV
        row = [
            f"{slice_idx:05d}",
            f"{metrics['PSNR']:.4f}",
            f"{metrics['SSIM']:.4f}",
            f"{metrics['MS-SSIM']:.4f}",
            f"{metrics['RMSE']:.6f}",
            f"{metrics['MAE']:.6f}",
            f"{metrics['MaxErr']:.6f}",
            f"{metrics['MSE']:.8f}",
            f"{metrics['SNR']:.4f}",
            f"{last_loss:.6f}",
            f"{wallclock:.1f}",
        ]
        with open(csv_path, "a") as f:
            f.write(",".join(row) + "\n")

        metric_collector.append(metrics)

        eta = (time.time() - t_global) / (n_done + 1) * (n_total - n_done - 1)
        print(f"{log_prefix}done in {wallclock:.0f}s, "
              f"PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, "
              f"loss={last_loss:.4f}, ETA={eta/60:.1f} min", flush=True)
        n_done += 1

    total_time = time.time() - t_global

    # Aggregate summary
    print(f"\n{'='*60}")
    print(f"Done: {n_done} OK, {n_skip} skipped (already done), {n_fail} failed")
    print(f"Total time: {total_time/60:.1f} min")
    if metric_collector:
        for k in ['PSNR', 'SSIM', 'MS-SSIM', 'RMSE', 'MAE']:
            vals = [m[k] for m in metric_collector if not np.isnan(m[k])]
            if vals:
                print(f"  {k:8s}: mean={np.mean(vals):.4f}, "
                      f"std={np.std(vals):.4f}, "
                      f"median={np.median(vals):.4f}")
    print(f"Output: {exp_dir}")
    print(f"CSV:    {csv_path}")
    print(f"Vis:    {vis_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

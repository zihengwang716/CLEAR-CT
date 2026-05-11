#!/usr/bin/env python3
"""
2D Gaussian Splatting for fan-beam CT reconstruction.

模型：图像表示为 N 个 2D 各向异性高斯的叠加：
    μ(x) = Σ_i ρ_i × exp(-0.5 (x - μ_i)^T Σ_i^-1 (x - μ_i))

每个 Gaussian 沿 ray 的线积分 **有解析解**:
    ∫ G_i(S + t·d) dt = ρ_i √(2π/A_i) exp(-0.5(C_i - B_i²/A_i))
    其中 A = d^T Σ^-1 d, B = d^T Σ^-1 (S-μ), C = (S-μ)^T Σ^-1 (S-μ)

跟 NeRF 比的优势：
  1. 解析积分（无 Riemann 求和误差）
  2. 显式表示 → 没高斯的地方真 0（背景干净）
  3. 锐利边缘（高斯 σ 可以很小）
  4. 训练快（无 MLP forward 开销）

用法:
  python gauss_2d_ct.py --slices 4501 4501 --mode 3p --ang_subsamp 6 \\
      --n_gaussians 20000 --n_iter 3000 --n_vis 1
"""

import argparse
import glob
import json
import os
import sys
import time
import uuid
from datetime import datetime

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Reuse data loading + geometry from nerf module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nerf_2d_ct_bh import (
    load_and_preprocess, apply_degradation,
    get_sources_and_detectors,
    apply_bh_correction,
    load_gt, match_sizes, normalize_for_metrics, evaluate_one,
    mode_dir_name,
    TOTAL_PROJECTIONS, N_DET_BINNED, SOD, SDD, DET_PIX,
    compute_scale,
)

PHYSICAL_FOV = 1024.0


# ============================================================
#  Gaussian Field 2D
# ============================================================
class GaussianField2D(nn.Module):
    """N 个 2D 各向异性高斯的可微表示。

    每个 Gaussian:
      - position μ ∈ ℝ²            (in [-1, 1] 归一化坐标)
      - log_scales (s_x, s_y) ∈ ℝ²  (实际 scale = exp(log_scale))
      - rotation θ ∈ ℝ              (覆盖范围 [-π/2, π/2] via tan)
      - density ρ ∈ ℝ⁺              (use softplus, bound by max_density)
    """
    def __init__(self, n_gaussians=10000, init_scale=0.02, max_density=0.01,
                 init_density=0.001, device='cuda',
                 init_positions=None, init_densities=None):
        super().__init__()
        self.max_density = max_density

        if init_positions is not None:
            # AGD-init mode: 用 AGD 重建图采样 Gaussian 位置 + 初始密度
            n_gaussians = init_positions.shape[0]
            positions = torch.tensor(init_positions, dtype=torch.float32,
                                      device=device)
        else:
            # Random init: uniform in [-0.7, 0.7] (within FOV circle)
            positions = (torch.rand(n_gaussians, 2, device=device) - 0.5) * 1.4

        self.N = n_gaussians
        self.positions = nn.Parameter(positions)

        # Init log_scales: log(init_scale) → e.g. log(0.02) = -3.9
        log_scales = torch.full((n_gaussians, 2), float(np.log(init_scale)),
                                 device=device)
        self.log_scales = nn.Parameter(log_scales)

        # Init rotations: 0 (axis-aligned)
        self.rotations = nn.Parameter(torch.zeros(n_gaussians, device=device))

        # Init density:
        if init_densities is not None:
            # AGD-init: 从 image 取的密度
            ratios = np.clip(init_densities / max_density, 1e-6, 0.99)
            init_raw = np.log(np.exp(ratios) - 1.0)
            densities_raw = torch.tensor(init_raw.astype(np.float32),
                                          device=device)
        else:
            target_ratio = init_density / max_density
            target_ratio = max(min(target_ratio, 0.99), 1e-6)
            init_raw = float(np.log(np.exp(target_ratio) - 1.0))
            densities_raw = torch.full((n_gaussians,), init_raw, device=device)
        self.densities_raw = nn.Parameter(densities_raw)

    def get_density(self):
        # softplus to ensure positive, scale by max_density to bound
        return F.softplus(self.densities_raw) * self.max_density

    def get_covariance_inverse(self):
        """Return (Σ⁻¹_xx, Σ⁻¹_yy, Σ⁻¹_xy) per Gaussian, plus log|Σ| for normalization."""
        # scales: (N, 2), positive
        s = torch.exp(self.log_scales).clamp(min=1e-4, max=1.0)
        sx2 = s[:, 0] ** 2
        sy2 = s[:, 1] ** 2
        cos_t = torch.cos(self.rotations)
        sin_t = torch.sin(self.rotations)

        # Σ = R diag(sx², sy²) R^T, where R = [[c, -s], [s, c]]
        # Σ_xx = sx² c² + sy² s²
        # Σ_yy = sx² s² + sy² c²
        # Σ_xy = (sx² - sy²) c s
        Σ_xx = sx2 * cos_t**2 + sy2 * sin_t**2
        Σ_yy = sx2 * sin_t**2 + sy2 * cos_t**2
        Σ_xy = (sx2 - sy2) * cos_t * sin_t

        # det(Σ) = Σ_xx Σ_yy - Σ_xy²
        det = Σ_xx * Σ_yy - Σ_xy**2 + 1e-12

        # Σ⁻¹ = (1/det) [[Σ_yy, -Σ_xy], [-Σ_xy, Σ_xx]]
        inv_det = 1.0 / det
        Σi_xx = Σ_yy * inv_det
        Σi_yy = Σ_xx * inv_det
        Σi_xy = -Σ_xy * inv_det
        return Σi_xx, Σi_yy, Σi_xy

    def line_integral(self, sources, directions, ray_lengths=None,
                       chunk_size=2048):
        """对每条 ray 求 N 个高斯的线积分总和。

        Args:
            sources: (R, 2)  - ray 起点 (in [-1, 1])
            directions: (R, 2) - ray 方向 (unit vector)
            ray_lengths: (R,) optional - 用于截断到 ray 内部
            chunk_size: int - rays 分批处理（防 OOM）

        Returns:
            p_pred: (R,) projected line integrals
        """
        Σi_xx, Σi_yy, Σi_xy = self.get_covariance_inverse()  # each (N,)
        densities = self.get_density()                        # (N,)
        positions = self.positions                            # (N, 2)

        R = sources.shape[0]
        results = torch.zeros(R, device=sources.device)

        # Process rays in chunks to limit memory
        for i in range(0, R, chunk_size):
            S = sources[i:i+chunk_size]      # (r, 2)
            d = directions[i:i+chunk_size]   # (r, 2)
            r = S.shape[0]

            # diff = S - μ: (r, 1, 2) - (1, N, 2) = (r, N, 2)
            diff = S.unsqueeze(1) - positions.unsqueeze(0)
            diff_x = diff[..., 0]   # (r, N)
            diff_y = diff[..., 1]

            # d shape: (r, 2) → (r, 1)
            d_x = d[:, 0:1]    # (r, 1)
            d_y = d[:, 1:2]

            # A = d^T Σ⁻¹ d
            #   = d_x² Σi_xx + 2 d_x d_y Σi_xy + d_y² Σi_yy
            A = (d_x**2) * Σi_xx.unsqueeze(0) \
                + 2 * d_x * d_y * Σi_xy.unsqueeze(0) \
                + (d_y**2) * Σi_yy.unsqueeze(0)             # (r, N)

            # Σi · diff: shape (r, N, 2)
            Σi_diff_x = Σi_xx.unsqueeze(0) * diff_x + Σi_xy.unsqueeze(0) * diff_y
            Σi_diff_y = Σi_xy.unsqueeze(0) * diff_x + Σi_yy.unsqueeze(0) * diff_y

            # B = d^T (Σi · diff)
            B = d_x * Σi_diff_x + d_y * Σi_diff_y           # (r, N)

            # C = diff^T (Σi · diff)
            C = diff_x * Σi_diff_x + diff_y * Σi_diff_y     # (r, N)

            # Per-gaussian contribution
            # = ρ × √(2π/A) × exp(-0.5 (C - B²/A))
            inv_A = 1.0 / (A + 1e-12)
            quad = C - B**2 * inv_A
            contribution = densities.unsqueeze(0) \
                * torch.sqrt(2 * np.pi * inv_A) \
                * torch.exp(-0.5 * quad)                    # (r, N)

            results[i:i+r] = contribution.sum(dim=-1)

        return results

    def render_image(self, rec_size, fov_extent=1.0, chunk_size=4096):
        """在 rec_size × rec_size 网格上渲染高斯场。

        Returns:
            image: (rec_size, rec_size) numpy array
        """
        device = self.positions.device
        # ASTRA convention: row 0 = +Y_max
        xs = torch.linspace(-fov_extent, fov_extent, rec_size, device=device)
        ys = torch.linspace(fov_extent, -fov_extent, rec_size, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing='xy')
        grid_xy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (H*W, 2)

        Σi_xx, Σi_yy, Σi_xy = self.get_covariance_inverse()
        densities = self.get_density()
        positions = self.positions

        # Process pixels in chunks
        n_pix = grid_xy.shape[0]
        out = torch.zeros(n_pix, device=device)
        for i in range(0, n_pix, chunk_size):
            pts = grid_xy[i:i+chunk_size]                  # (k, 2)
            diff = pts.unsqueeze(1) - positions.unsqueeze(0)  # (k, N, 2)
            dx = diff[..., 0]
            dy = diff[..., 1]
            # quadratic = (Σi·diff)^T diff
            #           = dx² Σi_xx + 2 dx dy Σi_xy + dy² Σi_yy
            quad = (dx**2) * Σi_xx.unsqueeze(0) \
                 + 2 * dx * dy * Σi_xy.unsqueeze(0) \
                 + (dy**2) * Σi_yy.unsqueeze(0)             # (k, N)
            # Each Gaussian contribution: ρ × exp(-0.5 quad)
            contribution = densities.unsqueeze(0) * torch.exp(-0.5 * quad)
            out[i:i+chunk_size] = contribution.sum(dim=-1)

        return out.reshape(rec_size, rec_size).cpu().numpy()

    def prune(self, density_threshold=None, min_keep=100):
        """删除密度过低的 Gaussians。返回 (kept_indices, n_pruned)。

        Args:
            density_threshold: 绝对 density 阈值。None → 用 0.05 × max_density
            min_keep: 至少保留这么多 (避免全部被删)
        """
        if density_threshold is None:
            density_threshold = 0.05 * self.max_density
        with torch.no_grad():
            densities = self.get_density()
            keep = densities > density_threshold
            n_kept = keep.sum().item()
            if n_kept < min_keep:
                # 把 top-min_keep 强制保留
                top_idx = torch.topk(densities, min_keep).indices
                keep = torch.zeros_like(keep)
                keep[top_idx] = True
                n_kept = min_keep

            n_pruned = self.N - n_kept
            self.positions = nn.Parameter(self.positions.data[keep].clone())
            self.log_scales = nn.Parameter(self.log_scales.data[keep].clone())
            self.rotations = nn.Parameter(self.rotations.data[keep].clone())
            self.densities_raw = nn.Parameter(self.densities_raw.data[keep].clone())
            self.N = n_kept
        return n_kept, n_pruned

    def densify(self, scale_threshold=0.05, density_threshold_ratio=0.3,
                 max_total=100000):
        """Split 大尺寸高密度 Gaussian 成 2 个小的。

        Args:
            scale_threshold: 大于这个 scale 的 Gaussian 视为"太大"
            density_threshold_ratio: density > this × max_density 才考虑 split
            max_total: 防止无限增长，超过这个就不 densify

        Returns: (n_split, new_total)
        """
        if self.N >= max_total:
            return 0, self.N
        with torch.no_grad():
            scales = torch.exp(self.log_scales)            # (N, 2)
            max_scale_per_g = scales.max(dim=-1)[0]         # (N,)
            densities = self.get_density()                  # (N,)

            split_mask = (max_scale_per_g > scale_threshold) & \
                         (densities > density_threshold_ratio * self.max_density)
            n_split = split_mask.sum().item()
            if n_split == 0:
                return 0, self.N

            # Limit splits to stay under max_total
            allowed = max_total - self.N
            if n_split > allowed:
                # Randomly pick `allowed` of the splittable
                idx = torch.where(split_mask)[0]
                perm = torch.randperm(idx.shape[0], device=idx.device)[:allowed]
                new_mask = torch.zeros_like(split_mask)
                new_mask[idx[perm]] = True
                split_mask = new_mask
                n_split = allowed

            # Parents
            parent_pos = self.positions.data[split_mask]    # (S, 2)
            parent_scales = self.log_scales.data[split_mask]
            parent_rot = self.rotations.data[split_mask]
            parent_dens = self.densities_raw.data[split_mask]

            # Children: offset along major axis ± half-scale
            cos_t = torch.cos(parent_rot)
            sin_t = torch.sin(parent_rot)
            major_dir = torch.stack([cos_t, sin_t], dim=-1)         # (S, 2)
            offset_mag = torch.exp(parent_scales[:, 0:1]) * 0.5
            offset = major_dir * offset_mag

            child1_pos = parent_pos + offset
            child2_pos = parent_pos - offset
            # Halve scale: log(s/2) = log(s) - log(2)
            child_scales = parent_scales - float(np.log(2.0))

            keep_mask = ~split_mask
            self.positions = nn.Parameter(torch.cat([
                self.positions.data[keep_mask], child1_pos, child2_pos,
            ], dim=0).clone())
            self.log_scales = nn.Parameter(torch.cat([
                self.log_scales.data[keep_mask], child_scales, child_scales,
            ], dim=0).clone())
            self.rotations = nn.Parameter(torch.cat([
                self.rotations.data[keep_mask], parent_rot, parent_rot,
            ], dim=0).clone())
            self.densities_raw = nn.Parameter(torch.cat([
                self.densities_raw.data[keep_mask], parent_dens, parent_dens,
            ], dim=0).clone())
            self.N = self.positions.shape[0]
        return n_split, self.N


# ============================================================
#  AGD Initialization (option C)
# ============================================================
# Try to register AGD plugin once at module load
_AGD_PLUGIN_LOADED = False
def _try_load_agd_plugin():
    global _AGD_PLUGIN_LOADED
    if _AGD_PLUGIN_LOADED:
        return True
    try:
        import inspect
        if not hasattr(inspect, 'getargspec'):
            def getargspec(func):
                spec = inspect.getfullargspec(func)
                return inspect.ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
            inspect.getargspec = getargspec
        import astra
        from NesterovGradient import AcceleratedGradientPlugin
        astra.plugin.register(AcceleratedGradientPlugin)
        _AGD_PLUGIN_LOADED = True
        return True
    except Exception as e:
        print(f"  [warn] AGD plugin unavailable: {e}. Will fall back to SIRT.")
        return False


def agd_quick_reconstruction(sino_log, angles, rec_size=512, n_iter=30,
                              gpu_index=0, force_method=None,
                              pad_factor=2):
    """Quick AGD (or SIRT fallback) reconstruction using ASTRA — used as Gaussian init.

    Matches the official 2DeteCT recon (Reconstructions_2DeteCT.py) in 3 ways:
      1. MinConstraint=0 在每个 iter 强制 μ ≥ 0 (避免 negative ↔ positive 互相抵消
         导致 contrast 被压扁).
      2. Vol_geom 用 pad_factor × rec_size 大小 (官方 2×: recSz=2048 → crop 中心 1024),
         把 boundary artifacts 推到 FOV 外面再 crop 掉.
      3. crop 中心 rec_size × rec_size 输出.

    Args:
        force_method: 'AGD' / 'SIRT' / None (auto: try AGD, fallback SIRT)
        pad_factor: 重建在 pad_factor × rec_size 上 (default 2 = 官方做法).
                    设 1 = 不 pad (旧行为, 会有 boundary ring artifact).

    Returns:
        rec: (rec_size, rec_size) numpy array, μ ≥ 0
        method_used: str ('AGD' or 'SIRT')
    """
    import inspect
    if not hasattr(inspect, 'getargspec'):
        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return inspect.ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
        inspect.getargspec = getargspec
    import astra
    astra.set_gpu_index(gpu_index)

    # Try AGD first (Nesterov accelerated gradient — much better than SIRT)
    use_agd = (force_method != 'SIRT') and _try_load_agd_plugin()

    scale = compute_scale()
    proj_geom = astra.create_proj_geom(
        'fanflat', 2 * DET_PIX * scale, N_DET_BINNED, angles,
        SOD * scale, (SDD - SOD) * scale,
    )
    # ─── FIX: 2× padded vol_geom (官方做法, 官方代码 recSz=(2048,2048) crop 1024) ───
    # compute_scale() 把 PHYSICAL_FOV=1024 vox 校准为完整 FOV.
    # 这里我们用 pad_factor × rec_size 个 vox, 每 vox 物理大小 = PHYSICAL_FOV/rec_size,
    # 总 extent = pad_factor × PHYSICAL_FOV. 重建后裁中心 rec_size × rec_size = 完整 FOV.
    pad_size = int(rec_size * pad_factor)
    half_extent = (PHYSICAL_FOV / 2.0) * pad_factor
    vol_geom = astra.create_vol_geom(
        pad_size, pad_size,
        -half_extent, half_extent, -half_extent, half_extent,
    )

    rec_id = astra.data2d.create('-vol', vol_geom)
    sino_id = astra.data2d.create('-sino', proj_geom, sino_log)
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    if use_agd:
        # AGD plugin (Nesterov accelerated gradient)
        cfg = astra.astra_dict('AGD-PLUGIN')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = proj_id
        # FIX: 必须设 MinConstraint=0, 否则负值压扁 contrast (官方 line 225).
        # NesterovGradient.py:104 确认 plugin 会 respect MinConstraint 如果设了.
        cfg['option'] = {'MinConstraint': 0}
        method_used = 'AGD'
    else:
        # SIRT fallback
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = proj_id
        cfg['option'] = {'MinConstraint': 0}
        method_used = 'SIRT'

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, n_iter)
    rec_full = astra.data2d.get(rec_id)

    # ─── FIX: crop 中心 rec_size × rec_size (剥掉 boundary artifacts) ───
    if pad_factor > 1:
        c = pad_size // 2
        s = rec_size // 2
        rec = rec_full[c - s:c + s, c - s:c + s]
    else:
        rec = rec_full

    # 安全网: 即使 MinConstraint 被设了, AGD 第一次 update 前的 init 是 0, 应该 OK.
    # 如果 plugin 因任何原因没 enforce, 这里再夹一次.
    rec = np.maximum(rec, 0)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return rec.astype(np.float32), method_used


def init_gaussians_from_image(image, n_gaussians, fov_extent=1.0,
                               min_density=1e-5, max_density=0.005,
                               jitter=0.005, overlap_correction=True):
    """从重建图采样 Gaussian 位置 + 初始化密度。

    overlap_correction (FIX): 当多个 Gaussian 落在同一区域时，
        density 会因为渲染叠加而 overshoot。我们用核密度估计计算每个
        位置的 "local Gaussian count"，把 density 归一化掉。

    Returns:
        positions: (N, 2) numpy in [-fov_extent, +fov_extent]
        densities: (N,) initial density values (overlap-corrected)
    """
    H, W = image.shape
    image = np.maximum(image, 0).astype(np.float64)

    flat = image.flatten()
    if flat.sum() < 1e-12:
        positions = (np.random.rand(n_gaussians, 2) - 0.5) * 2 * fov_extent * 0.7
        densities = np.full(n_gaussians, 1e-4, dtype=np.float32)
        return positions.astype(np.float32), densities

    # Sample positions weighted by image intensity
    prob = flat / flat.sum()
    indices = np.random.choice(len(flat), size=n_gaussians, p=prob, replace=True)
    i = indices // W
    j = indices % W

    # Map to ASTRA-convention coords (row 0 = +Y_max)
    x = ((j / (W - 1)) - 0.5) * 2 * fov_extent
    y = (0.5 - (i / (H - 1))) * 2 * fov_extent
    x = x + np.random.randn(n_gaussians) * jitter
    y = y + np.random.randn(n_gaussians) * jitter
    positions = np.stack([x, y], axis=-1).astype(np.float32)

    # Init density from sampled image value
    densities = flat[indices].astype(np.float32)
    if densities.max() > 0:
        densities = densities / densities.max() * max_density
    densities = np.clip(densities, min_density, max_density).astype(np.float32)

    # ─── FIX: Overlap correction via KDE-style local count ───
    if overlap_correction:
        # For each position, count how many other Gaussians fall within radius ~init_scale
        # In [-1, 1] coords, init_scale ≈ 0.005 → use radius 0.01 (2× scale)
        try:
            from scipy.spatial import cKDTree
            radius = 0.01 * fov_extent
            tree = cKDTree(positions)
            n_within = np.array([
                len(tree.query_ball_point(p, radius)) for p in positions
            ], dtype=np.float32)
            n_within = np.clip(n_within, 1.0, None)  # avoid div by 0
            # Divide density by overlap count (heuristic)
            densities = (densities / n_within).astype(np.float32)
            densities = np.clip(densities, min_density, max_density).astype(np.float32)
        except ImportError:
            warnings.warn("scipy.spatial not available; skipping overlap correction")
    return positions, densities


def image_domain_fit(field, target_image, n_iter=200, batch_pixels=8192,
                      lr_position=5e-4, lr_scale=2e-3, lr_rotation=1e-3,
                      lr_density=1e-2, verbose=False):
    """直接在图像域 fit Gaussian field 到 target image (不通过 sinogram)。

    用 random pixel sampling (minibatch SGD over pixels) 避免 OOM —
    全图 1024² × N_gauss 的 autograd graph 在 80GB GPU 上也撑不住。

    用于:
      - 验证 representational capacity (Test 0.5: image Oracle)
      - 修复 init overshoot

    Returns:
        loss_history: list of per-iter losses
        rendered_final: (H, W) numpy 图 (no_grad full render at end)
    """
    device = field.positions.device
    target = torch.tensor(target_image, dtype=torch.float32, device=device)
    H, W = target.shape

    # Pre-build coordinate grid (static, on device)
    ys = torch.linspace(1.0, -1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing='xy')
    grid_xy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (H*W, 2)
    target_flat = target.reshape(-1)
    n_pix = grid_xy.shape[0]
    batch_pixels = min(batch_pixels, n_pix)

    optimizer = torch.optim.Adam([
        {'params': [field.positions],     'lr': lr_position},
        {'params': [field.log_scales],    'lr': lr_scale},
        {'params': [field.rotations],     'lr': lr_rotation},
        {'params': [field.densities_raw], 'lr': lr_density},
    ], betas=(0.9, 0.99), eps=1e-15)

    losses = []
    for it in range(n_iter):
        # ── Random pixel batch (minibatch SGD over pixels) ──
        pix_idx = torch.randint(0, n_pix, (batch_pixels,), device=device)
        pts = grid_xy[pix_idx]              # (B, 2)
        target_batch = target_flat[pix_idx] # (B,)

        Σi_xx, Σi_yy, Σi_xy = field.get_covariance_inverse()
        densities = field.get_density()

        diff = pts.unsqueeze(1) - field.positions.unsqueeze(0)  # (B, N, 2)
        dx = diff[..., 0]
        dy = diff[..., 1]
        quad = (dx ** 2) * Σi_xx.unsqueeze(0) \
             + 2 * dx * dy * Σi_xy.unsqueeze(0) \
             + (dy ** 2) * Σi_yy.unsqueeze(0)
        contribution = densities.unsqueeze(0) * torch.exp(-0.5 * quad)
        rendered_batch = contribution.sum(dim=-1)              # (B,)

        loss = ((rendered_batch - target_batch) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (it + 1) % max(1, n_iter // 10) == 0:
            with torch.no_grad():
                mse = float(loss.item())
                gt_range = float(target.max() - target.min())
                psnr_batch = -10 * np.log10(mse / (gt_range ** 2 + 1e-12))
                print(f"  iter {it+1:4d}: batch_loss={mse:.6e}, "
                      f"batch PSNR (GT range)={psnr_batch:.2f}")

    # ── Final full render (no_grad — chunked, no autograd memory) ──
    with torch.no_grad():
        Σi_xx, Σi_yy, Σi_xy = field.get_covariance_inverse()
        densities = field.get_density()
        rendered_full = torch.zeros(n_pix, device=device)
        chunk = 8192
        for i in range(0, n_pix, chunk):
            pts = grid_xy[i:i + chunk]
            diff = pts.unsqueeze(1) - field.positions.unsqueeze(0)
            dx = diff[..., 0]; dy = diff[..., 1]
            quad = (dx ** 2) * Σi_xx.unsqueeze(0) \
                 + 2 * dx * dy * Σi_xy.unsqueeze(0) \
                 + (dy ** 2) * Σi_yy.unsqueeze(0)
            rendered_full[i:i + chunk] = (
                densities.unsqueeze(0) * torch.exp(-0.5 * quad)
            ).sum(dim=-1)
        rendered_final = rendered_full.reshape(H, W).cpu().numpy()

    return losses, rendered_final


# ============================================================
#  Training
# ============================================================
def train_one_slice(p_target, sources, detectors, angles, args, device='cuda',
                     log_prefix=""):
    """Train Gaussian field for one slice.

    Args:
        p_target: (n_angles, n_det) numpy
        sources, detectors: from get_sources_and_detectors (in scaled coords)
        angles: (n_angles,) numpy — needed for AGD init
    """
    n_angles, n_det = p_target.shape

    # To tensor
    p_t = torch.tensor(p_target, dtype=torch.float32, device=device)
    src_t = torch.tensor(sources, dtype=torch.float32, device=device)
    det_t = torch.tensor(detectors, dtype=torch.float32, device=device)

    # Normalize coords from scaled units to [-1, 1]
    norm_scale = PHYSICAL_FOV / 2.0
    src_norm = src_t / norm_scale       # (n_angles, 2)
    det_norm = det_t / norm_scale       # (n_angles, n_det, 2)

    # ─── Optionally: AGD init (option C) ───
    init_positions = None
    init_densities = None
    if args.init_from_agd:
        if log_prefix:
            print(f"{log_prefix}AGD init: recon at {args.agd_init_size}², "
                  f"{args.agd_init_iter} iter ...", flush=True)
        try:
            agd_rec, method_used = agd_quick_reconstruction(
                p_target, angles,
                rec_size=args.agd_init_size, n_iter=args.agd_init_iter,
                gpu_index=args.gpu_index,
            )
            # Phase 3 init params (match diagnose_gauss.py test_2.6 which proved
            # 27-29 dB across slices). Legacy used max_density*0.5 + jitter 0.005;
            # Phase 3 uses full max_density + smaller jitter for sharper init.
            if args.early_stop == "tv_rise":
                init_max_density = args.max_density
                init_jitter = 0.002
            else:
                init_max_density = args.max_density * 0.5
                init_jitter = 0.005
            init_positions, init_densities = init_gaussians_from_image(
                agd_rec, args.n_gaussians, fov_extent=1.0,
                max_density=init_max_density,
                jitter=init_jitter,
            )
            if log_prefix:
                print(f"{log_prefix}AGD init: done with {method_used}. "
                      f"{args.n_gaussians} Gaussians placed by intensity prior.",
                      flush=True)
        except Exception as e:
            print(f"{log_prefix}AGD init FAILED: {e}, falling back to random init")
            import traceback; traceback.print_exc()

    # Initialize Gaussian field
    field = GaussianField2D(
        n_gaussians=args.n_gaussians,
        init_scale=args.init_scale,
        max_density=args.max_density,
        init_density=args.init_density,
        device=device,
        init_positions=init_positions,
        init_densities=init_densities,
    ).to(device)

    def _build_optimizer(field):
        return torch.optim.Adam([
            {'params': [field.positions],     'lr': args.lr_position},
            {'params': [field.log_scales],    'lr': args.lr_scale},
            {'params': [field.rotations],     'lr': args.lr_rotation},
            {'params': [field.densities_raw], 'lr': args.lr_density},
        ], betas=(0.9, 0.99), eps=1e-15)

    optimizer = _build_optimizer(field)

    # Phase 3: when TV early stop is on, no scheduler (LR is already small).
    use_tv_stop = (args.early_stop == "tv_rise")
    if not use_tv_stop:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.n_iter // 3), gamma=0.5
        )
    else:
        scheduler = None

    # ─── Phase 3: TV early-stop state ───
    # CRITICAL perf fix: TV check renders at LOW resolution (default 256×256) —
    # full 1024² render takes ~5s × 60 checks = 300s/slice (5× slower than train!).
    # 256² render takes ~80ms × 60 = 5s/slice. TV ratio is invariant to image size
    # at fixed Gaussian field, so the 1.05× threshold transfers cleanly.
    # Final output render (after early stop) still uses args.rec_size = 1024.
    tv_render_size = args.tv_render_size
    tv_render_chunk = args.tv_render_chunk_size

    if use_tv_stop:
        with torch.no_grad():
            recon_init_lo = field.render_image(
                tv_render_size, fov_extent=1.0, chunk_size=tv_render_chunk,
            )
        recon_init_lo = np.maximum(recon_init_lo, 0).astype(np.float32)
        tv_init = float(
            np.abs(recon_init_lo[1:] - recon_init_lo[:-1]).sum()
            + np.abs(recon_init_lo[:, 1:] - recon_init_lo[:, :-1]).sum()
        )
        min_tv_seen = tv_init
        # best_recon gets updated when TV-rise triggers — render at FULL res then.
        # Default to None; if no trigger, the post-loop branch renders final at full res.
        best_recon = None
        tv_stop_iter = None
        if log_prefix:
            tqdm.write(f"{log_prefix}TV-stop ON: init_TV={tv_init:.2f}, "
                        f"factor={args.tv_rise_factor}, "
                        f"check every {args.tv_log_every} iter @ "
                        f"{tv_render_size}² (perf)",)
    else:
        min_tv_seen = None
        best_recon = None
        tv_stop_iter = None

    pbar = tqdm(range(args.n_iter), desc=f"{log_prefix}Gauss train",
                dynamic_ncols=True, leave=False)
    last_loss = float('inf')
    for it in pbar:
        # Random batch of (angle, det) pairs
        ang_idx = torch.randint(0, n_angles, (args.batch_rays,), device=device)
        det_idx = torch.randint(0, n_det, (args.batch_rays,), device=device)

        # Get ray endpoints
        S = src_norm[ang_idx]                          # (R, 2)
        D = det_norm[ang_idx, det_idx]                 # (R, 2)
        direction = D - S                              # (R, 2)
        ray_len = torch.norm(direction, dim=-1, keepdim=True)
        directions = direction / (ray_len + 1e-12)     # (R, 2) unit

        # Forward project (analytical line integral!)
        p_pred = field.line_integral(S, directions, chunk_size=args.chunk_size)

        # Compensate for normalization scale: physical line integral
        # in original (un-normalized) coords is the same number of "units" of attenuation,
        # but our integral is in [-1,1] coords. Scale by norm_scale to recover units.
        p_pred = p_pred * norm_scale

        # Target
        p_t_batch = p_t[ang_idx, det_idx]

        # Loss
        residual = p_pred - p_t_batch
        if args.loss_type == "huber":
            loss = F.huber_loss(p_pred, p_t_batch, delta=args.huber_delta)
        elif args.loss_type == "l1":
            loss = residual.abs().mean()
        else:
            loss = (residual ** 2).mean()

        # Density sparsity reg
        if args.sparsity_weight > 0:
            sparsity = field.get_density().mean()
            loss = loss + args.sparsity_weight * sparsity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        last_loss = loss.item()
        if (it + 1) % 100 == 0:
            n_active = (field.get_density() > 0.01 * args.max_density).sum().item()
            pbar.set_postfix(loss=f"{last_loss:.6f}",
                              active=f"{n_active}/{field.N}")

        # ─── Phase 3: TV early-stop check (LOW RES for speed) ───
        if use_tv_stop and (it + 1) % args.tv_log_every == 0:
            with torch.no_grad():
                recon_chk = field.render_image(
                    tv_render_size, fov_extent=1.0,
                    chunk_size=tv_render_chunk,
                )
            recon_chk = np.maximum(recon_chk, 0).astype(np.float32)
            tv_chk = float(
                np.abs(recon_chk[1:] - recon_chk[:-1]).sum()
                + np.abs(recon_chk[:, 1:] - recon_chk[:, :-1]).sum()
            )
            if tv_chk < min_tv_seen:
                min_tv_seen = tv_chk

            # ── Trigger 1: TV-rise threshold (preferred) ──
            tv_triggered = (tv_chk > min_tv_seen * args.tv_rise_factor)

            # ── Trigger 2: force_stop_iter safety net ──
            # Some slices (~8% in slice 1-13 sample) have unreliable TV signal
            # at 256² and never trigger. Without this, they train all 1000 iter
            # and severely overfit (PSNR can drop to 13). Force stop at
            # force_stop_iter (typically 600) — empirically near peak.
            force_stop_triggered = (args.force_stop_iter > 0
                                     and (it + 1) >= args.force_stop_iter)

            if (tv_triggered or force_stop_triggered) and tv_stop_iter is None:
                # Render at FULL resolution for the saved snapshot, then stop.
                with torch.no_grad():
                    best_recon = field.render_image(
                        args.rec_size, fov_extent=1.0,
                        chunk_size=args.chunk_size,
                    )
                best_recon = np.maximum(best_recon, 0).astype(np.float32)
                tv_stop_iter = it + 1
                if tv_triggered:
                    tqdm.write(f"{log_prefix}  [iter {it+1}] TV early-stop: "
                                f"TV={tv_chk:.2f}, min={min_tv_seen:.2f}, "
                                f"ratio={tv_chk/min_tv_seen:.3f} "
                                f"(>{args.tv_rise_factor})")
                else:
                    tqdm.write(f"{log_prefix}  [iter {it+1}] FORCE STOP "
                                f"(TV={tv_chk:.2f} ratio={tv_chk/min_tv_seen:.3f} "
                                f"never reached {args.tv_rise_factor}, "
                                f"max iter={args.force_stop_iter})")
                break

        # ─── Periodic Prune (option B) ───
        if args.prune_every > 0 and (it + 1) % args.prune_every == 0 \
                and (it + 1) < args.n_iter * 0.9:  # 不在最后 10% prune
            n_kept, n_pruned = field.prune(
                density_threshold=args.prune_threshold_ratio * args.max_density,
                min_keep=100,
            )
            if n_pruned > 0:
                tqdm.write(f"{log_prefix}  [iter {it+1}] pruned {n_pruned}, "
                            f"kept {n_kept}")
                # Optimizer state mismatched after param replacement → rebuild
                optimizer = _build_optimizer(field)
                # Note: scheduler state lost, but acceptable

        # ─── Periodic Densify (option B) ───
        if args.densify_every > 0 and (it + 1) % args.densify_every == 0 \
                and (it + 1) < args.n_iter * 0.7:  # 不在最后 30% densify (避免 thrashing)
            n_split, new_total = field.densify(
                scale_threshold=args.densify_scale_threshold,
                density_threshold_ratio=args.densify_density_ratio,
                max_total=args.max_gaussians,
            )
            if n_split > 0:
                tqdm.write(f"{log_prefix}  [iter {it+1}] densified {n_split}, "
                            f"total {new_total}")
                optimizer = _build_optimizer(field)

    pbar.close()

    # ─── Phase 3: return TV-stop snapshot if it triggered ───
    if use_tv_stop and tv_stop_iter is not None:
        # best_recon was saved at the trigger time
        if log_prefix:
            print(f"{log_prefix}returned snapshot @ iter {tv_stop_iter} "
                  f"(TV early-stop)", flush=True)
        return best_recon, last_loss

    # ─── Otherwise (no early stop, or TV never triggered): render final ───
    field.eval()
    with torch.no_grad():
        recon = field.render_image(args.rec_size, fov_extent=1.0,
                                    chunk_size=args.chunk_size)
    if use_tv_stop and tv_stop_iter is None and log_prefix:
        print(f"{log_prefix}TV never triggered ({args.n_iter} iter), "
              f"using last snapshot", flush=True)
    return recon, last_loss


# ============================================================
#  Visualization
# ============================================================
def visualize_one(recon, gt, sino_degraded, slice_idx, save_dir, args):
    os.makedirs(save_dir, exist_ok=True)
    if gt is not None:
        gt_m, rec_m = match_sizes(gt, recon)
        diff = rec_m - gt_m
    else:
        gt_m, rec_m = None, recon
        diff = None

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Sinogram
    ax = axes[0]
    im = ax.imshow(sino_degraded, cmap='gray', aspect='auto')
    ax.set_title(f"Input sinogram (n_angles={sino_degraded.shape[0]})")
    ax.set_xlabel("detector"); ax.set_ylabel("angle")
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # vmin/vmax from GT (fair display)
    if gt_m is not None:
        vmin = float(gt_m.min())
        vmax = float(np.percentile(gt_m, 99.5))
    else:
        vmin = float(rec_m.min())
        vmax = float(np.percentile(rec_m, 99.5))

    ax = axes[1]
    im = ax.imshow(rec_m, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f"Gauss recon ({rec_m.shape[0]}×{rec_m.shape[1]})")
    ax.axis('off')
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)

    ax = axes[2]
    if gt_m is not None:
        im = ax.imshow(gt_m, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title("GT (mode 2)")
    else:
        ax.text(0.5, 0.5, "GT not available", ha='center', va='center',
                transform=ax.transAxes)
    ax.axis('off')
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    if gt_m is not None: plt.colorbar(im, cax=cax)

    ax = axes[3]
    if diff is not None:
        emax = float(np.percentile(np.abs(diff), 99.5))
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-emax, vmax=emax)
        ax.set_title(f"Error (recon - GT), max={emax:.4f}")
    else:
        ax.text(0.5, 0.5, "no diff", ha='center', va='center',
                transform=ax.transAxes)
    ax.axis('off')
    div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.1)
    if diff is not None: plt.colorbar(im, cax=cax)

    plt.suptitle(f"Slice {slice_idx:05d} | 2D Gaussian Splatting | "
                 f"mode={args.mode} | views={sino_degraded.shape[0]} "
                 f"| n_gauss={args.n_gaussians} | iter={args.n_iter}")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"slice{slice_idx:05d}_compare.png"),
                dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================
#  Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="2D Gaussian Splatting for fan-beam CT")
    # Data
    p.add_argument("--data_dir", type=str,
                   default="/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_slicesAll")
    p.add_argument("--mode3_noisy_dir", type=str,
                   default="/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_slicesAll_mode3_noisy")
    p.add_argument("--gt_dir", type=str,
                   default="/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_RecSeg/2DeteCT_slices_RecSeg_All")
    p.add_argument("--out_dir", type=str, default="./results")
    p.add_argument("--slices", type=int, nargs=2, default=[1, 100])
    p.add_argument("--mode", type=str, default="3p")
    p.add_argument("--gt_mode", type=int, default=2)

    # Degradation
    p.add_argument("--ang_subsamp", type=int, default=6)
    p.add_argument("--max_ang", type=float, default=360.0)
    p.add_argument("--bh_coeffs", type=str, default=None)

    # Gaussian field
    p.add_argument("--n_gaussians", type=int, default=20000,
                   help="Number of Gaussians (10K-100K typical)")
    p.add_argument("--init_scale", type=float, default=0.02,
                   help="Initial Gaussian scale (in [-1,1] coords; 0.02 ≈ 20 pixels in 1024)")
    p.add_argument("--max_density", type=float, default=0.01,
                   help="Bound on per-Gaussian density (matches typical CT μ range)")
    p.add_argument("--init_density", type=float, default=0.0005,
                   help="Initial density (small)")

    # Training
    p.add_argument("--n_iter", type=int, default=3000)
    p.add_argument("--batch_rays", type=int, default=4096)
    p.add_argument("--chunk_size", type=int, default=2048,
                   help="Process rays in chunks to limit memory")
    p.add_argument("--lr_position", type=float, default=1e-3)
    p.add_argument("--lr_scale", type=float, default=5e-3)
    p.add_argument("--lr_rotation", type=float, default=1e-3)
    p.add_argument("--lr_density", type=float, default=2e-2)
    p.add_argument("--loss_type", type=str, default="l2",
                   choices=["l2", "l1", "huber"])
    p.add_argument("--huber_delta", type=float, default=0.1)
    p.add_argument("--sparsity_weight", type=float, default=0.0,
                   help="L1 reg on density (push unused Gaussians → 0)")

    # Densification / Pruning (option B)
    p.add_argument("--prune_every", type=int, default=0,
                   help="Run pruning every N iters (0 = disable). Recommended: 200-500")
    p.add_argument("--prune_threshold_ratio", type=float, default=0.05,
                   help="Density < this × max_density → prune")
    p.add_argument("--densify_every", type=int, default=0,
                   help="Run densification every N iters (0 = disable). Recommended: 500")
    p.add_argument("--densify_scale_threshold", type=float, default=0.05,
                   help="Gaussians with scale > this get split (in [-1,1] coords)")
    p.add_argument("--densify_density_ratio", type=float, default=0.3,
                   help="Only split Gaussians with density > this × max_density")
    p.add_argument("--max_gaussians", type=int, default=100000,
                   help="Hard cap on total Gaussians (densify stops here)")

    # AGD initialization (option C)
    p.add_argument("--init_from_agd", action="store_true",
                   help="Initialize Gaussian positions from quick AGD reconstruction")
    p.add_argument("--agd_init_size", type=int, default=512,
                   help="Resolution of AGD init reconstruction (smaller = faster)")
    p.add_argument("--agd_init_iter", type=int, default=30,
                   help="Iterations for AGD init reconstruction")

    # ─── Phase 3: TV-rise early stop (no GT needed) ───
    p.add_argument("--early_stop", type=str, default="none",
                   choices=["none", "tv_rise"],
                   help="Phase 3: 'tv_rise' = stop when image TV exceeds "
                        "min × tv_rise_factor (no GT needed). "
                        "Recommended config: --early_stop tv_rise "
                        "--tv_rise_factor 1.05 --tv_log_every 10.")
    p.add_argument("--tv_log_every", type=int, default=10,
                   help="(Phase 3) Render image and check TV every N iter.")
    p.add_argument("--tv_rise_factor", type=float, default=1.05,
                   help="(Phase 3) Trigger early stop when TV / min(TV) "
                        "exceeds this. 1.05 = aggressive (catch peak fast), "
                        "1.10 = safer (more training).")
    p.add_argument("--tv_render_size", type=int, default=512,
                   help="(Phase 3) Resolution for TV-check renders. "
                        "512 is ~4× faster than 1024 with same TV signal "
                        "(TV ratio is scale-invariant). 256 is faster but "
                        "may alias small Gaussians (init_scale=0.005 → "
                        "0.64 pixel at 256 res).")
    p.add_argument("--tv_render_chunk_size", type=int, default=8192,
                   help="(Phase 3) Pixel chunk size for TV-check renders. "
                        "Larger = better GPU utilization (fewer Python loop "
                        "iterations). 8192 OK on 80GB GPU.")
    p.add_argument("--force_stop_iter", type=int, default=0,
                   help="(Phase 3) Safety net: if TV trigger never fires by "
                        "this iter, force stop and save snapshot. "
                        "0 = disabled (use full n_iter). "
                        "600 recommended (test 2.6 peaks at 400-650).")

    # Output
    p.add_argument("--rec_size", type=int, default=1024)
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_vis", type=int, default=10)
    p.add_argument("--method_tag", type=str, default="Gauss")
    p.add_argument("--run_id", type=str, default="",
                   help="Fixed run_id for resumable sbatch jobs. If empty, "
                        "auto-generates from date+uuid (NOT resumable). "
                        "Pass same string across re-submissions to resume.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_eval", action="store_true")
    return p.parse_args()


def build_exp_dir(args):
    # If --run_id is set, use it (for resumable sbatch). Otherwise auto-generate.
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = datetime.now().strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:4]
    method = args.method_tag
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
        f"ng{args.n_gaussians}",
        f"loss{args.loss_type}",
    ]
    if args.bh_coeffs:
        bh_name = os.path.splitext(os.path.basename(args.bh_coeffs))[0]
        tag_parts.append(f"BH_{bh_name}")
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
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "n_slices": len(slice_list),
            "parameters": vars(args),
        }, f, indent=2, default=str)

    # CSV
    csv_path = os.path.join(exp_dir, "metrics.csv")
    csv_cols = ["slice_idx", "PSNR", "SSIM", "MS-SSIM", "RMSE", "MAE",
                "MaxErr", "MSE", "SNR", "train_loss", "wallclock_sec"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(",".join(csv_cols) + "\n")

    # Visualization indices
    n_total = len(slice_list)
    if args.n_vis > 0 and n_total > 0:
        vis_set = set(int(round(i * (n_total - 1) / max(1, args.n_vis - 1)))
                      for i in range(args.n_vis))
    else:
        vis_set = set()

    all_angles_full = np.linspace(0, 2 * np.pi, TOTAL_PROJECTIONS)[:-1]

    # Load BH coeffs if provided
    bh_coeffs = None
    if args.bh_coeffs:
        if not os.path.exists(args.bh_coeffs):
            print(f"ERROR: BH coeffs file not found: {args.bh_coeffs}")
            sys.exit(1)
        bh_coeffs = np.load(args.bh_coeffs)
        print(f"\n📐 BH correction: {len(bh_coeffs)} coeffs loaded")

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_tag}")
    print(f"Output:     {exp_dir}")
    print(f"Mode:       {args.mode}, sparse={args.ang_subsamp}x, "
          f"max_ang={args.max_ang}")
    print(f"Gaussians:  N={args.n_gaussians}, init_scale={args.init_scale}, "
          f"max_density={args.max_density}")
    print(f"Train:      n_iter={args.n_iter}, loss={args.loss_type}, "
          f"sparsity={args.sparsity_weight}")
    print(f"{'='*60}\n")

    n_done, n_skip, n_fail = 0, 0, 0
    t_global = time.time()
    metric_collector = []

    for ti, (slice_idx, slice_path) in enumerate(slice_list):
        out_path = os.path.join(out_subdir, f"slice{slice_idx:05d}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue

        t_slice = time.time()
        log_prefix = f"[{ti+1}/{n_total}] slice{slice_idx:05d} "

        try:
            sino = load_and_preprocess(slice_path, slice_idx, mode=args.mode)
        except Exception as e:
            print(f"{log_prefix}LOAD FAILED: {e}")
            n_fail += 1
            continue

        if bh_coeffs is not None:
            sino = apply_bh_correction(sino, bh_coeffs)

        sino_deg, angles_deg, dtag = apply_degradation(sino, all_angles_full, args)
        sources, detectors = get_sources_and_detectors(angles_deg, n_det=N_DET_BINNED)

        try:
            recon, last_loss = train_one_slice(
                sino_deg, sources, detectors, angles_deg, args,
                device=device, log_prefix=log_prefix
            )
        except Exception as e:
            print(f"{log_prefix}TRAIN FAILED: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1
            continue

        recon = np.maximum(recon, 0).astype(np.float32)
        np.save(out_path, recon)

        wallclock = time.time() - t_slice

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

        if ti in vis_set:
            try:
                visualize_one(recon, gt, sino_deg, slice_idx, vis_dir, args)
                print(f"{log_prefix}vis saved", flush=True)
            except Exception as e:
                import traceback
                print(f"{log_prefix}VIS FAILED: {e}")
                traceback.print_exc()

        row = [
            f"{slice_idx:05d}",
            f"{metrics['PSNR']:.4f}", f"{metrics['SSIM']:.4f}",
            f"{metrics['MS-SSIM']:.4f}", f"{metrics['RMSE']:.6f}",
            f"{metrics['MAE']:.6f}", f"{metrics['MaxErr']:.6f}",
            f"{metrics['MSE']:.8f}", f"{metrics['SNR']:.4f}",
            f"{last_loss:.6f}", f"{wallclock:.1f}",
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
    print(f"\n{'='*60}")
    print(f"Done: {n_done} OK, {n_skip} skipped, {n_fail} failed")
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


if __name__ == "__main__":
    main()

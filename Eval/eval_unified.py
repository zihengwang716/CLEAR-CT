"""
==================================================================
  统一图像重建质量评估脚本 (Unified Reconstruction Eval) - Fixed
==================================================================

本版本相对原版的修复（保持接口与默认行为兼容）：
  [F1]  SSIM 现在严格按 skimage 的 win_size 裁掉边界再做 mask 加权平均，
        修复前景 SSIM 因边缘像素被错误纳入而系统性偏低的问题。
  [F2]  `minmax` 归一化下不再不对称地只 clip rec 而不 clip gt；
        默认改为两者都不 clip（保留过冲，避免人为抬高 PSNR）。
        新增 --normalize_clip 参数让用户显式控制。
  [F3]  CNR 的 seg 与 rec/gt 对齐：现在用与 gt/rec 完全一致的中心裁切
        逻辑把 seg 也对齐到相同尺寸，三者大小不一致也安全。
  [F4]  streak_polar_fft 现在使用 FOV 自动检测出的 (cx, cy) 作为极坐标
        变换中心，而非默认的图像几何中心。
  [F5]  create_circular_mask_auto 现在打印检测到的 (cx, cy, r)（在 debug
        模式下），并改用 (Δx + Δy) / 4 取平均半径，对椭圆/截断的 FOV
        更稳健；同时返回 (mask, cx, cy, r)，便于其他指标复用。
  [F6]  PSNR 在 mse=0 时返回 100.0 dB（一个有限的高值），避免 inf 把
        np.nanmean 拉成 inf。
  [F7]  所有 mask 内像素数为 0 时统一返回 np.nan，由 np.nanmean 自动忽略，
        不再用 0.0 当“坏值”混淆方向（RMSE/LPIPS 的 0 表示完美重建）。
  [F8]  GME 的 Sobel 命名修正为 gy, gx（语义正确，对幅值无影响）。
  [F9]  hf_rms 的 sigma 改为相对图像短边的比例（默认 sigma_ratio=0.012），
        跨分辨率可比；--hf_sigma_ratio 可调。
  [F10] CNR 启用但 --seg_dir 未提供时打印明确 warning，不再静默跳过。
  [F11] 新增 --psnr_data_range 让用户传入全数据集统一的 data_range
        (例如 CT μ 值的 [0, μ_max])，启用后跨数据集 PSNR/RMSE 才有可比性。
  [F12] 新增 --debug_index 指定可视化第几张图（默认 0 = 第一张）。
  [F13] 新增 fov_center / fov_radius 在 debug 标题中打印，便于核验。
  [F14] 新增 --seg_suffix（默认 '_segmentation'）：seg 文件名 = base_name +
        seg_suffix + .tif。若 seg 与 gt 同名，传 --seg_suffix '' 即可。
  [F15] 进度条 postfix 实时显示当前各指标的累计均值，跑完后打印汇总。
        默认开启；--no_verbose_per_image 关闭 postfix（仅末尾汇总）。

支持的指标 (--metrics):
  保真度类 (Fidelity):
    psnr   -- Peak Signal-to-Noise Ratio  (↑)
    ssim   -- Structural Similarity        (↑)
    rmse   -- Root Mean Square Error       (↓)
  感知质量类 (Perceptual):
    lpips  -- Learned Perceptual Image Patch Similarity (↓) [需 GPU]
    vif    -- Visual Information Fidelity                (↑) [仅全图]
    gme    -- Gradient Magnitude Error                   (↓)
  伪影/解耦类 (Artifact / Decoupled):
    gaps   -- Soft Gradient-Aware Perceptual Score       (↓)
    cnr    -- Contrast-to-Noise Ratio (需要分割图)         (↑)
    streak -- 极坐标 FFT 条形伪影能量                       (↓)
    hf_rms -- 高频 RMS 能量                                (↓)
  特殊:
    all    -- 全部启用

支持的区域 (--region):
    full / fg / bg / all  (默认 all)
==================================================================
"""

import os
os.environ.setdefault('TORCH_HOME', './cache/checkpoint')

import argparse
import glob
import warnings
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from scipy import ndimage
from skimage.metrics import structural_similarity
from skimage.util import crop as sk_crop

warnings.filterwarnings("ignore")

# ===== 可选的重型依赖（按需懒加载）=====
_torch = None
_F = None
_lpips_pkg = None
_vifp = None
_warp_polar = None
_gaussian_filter = None
_plt = None


def _lazy_import_torch():
    global _torch, _F
    if _torch is None:
        import torch
        import torch.nn.functional as F
        _torch, _F = torch, F
    return _torch, _F


def _lazy_import_lpips():
    global _lpips_pkg
    if _lpips_pkg is None:
        import lpips
        _lpips_pkg = lpips
    return _lpips_pkg


def _lazy_import_vif():
    global _vifp
    if _vifp is None:
        from sewar.full_ref import vifp
        _vifp = vifp
    return _vifp


def _lazy_import_warp_polar():
    global _warp_polar
    if _warp_polar is None:
        from skimage.transform import warp_polar
        _warp_polar = warp_polar
    return _warp_polar


def _lazy_import_gaussian():
    global _gaussian_filter
    if _gaussian_filter is None:
        from scipy.ndimage import gaussian_filter
        _gaussian_filter = gaussian_filter
    return _gaussian_filter


def _lazy_import_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# ============================================================
#  通用工具
# ============================================================
def center_crop_to(arr, target_h, target_w):
    """把 arr 中心裁到 (target_h, target_w)。若已小于目标，会返回原 arr 截到的部分。"""
    h, w = arr.shape[:2]
    sh = min(h, target_h)
    sw = min(w, target_w)
    y0 = (h - sh) // 2
    x0 = (w - sw) // 2
    return arr[y0:y0 + sh, x0:x0 + sw]


# ============================================================
#  Mask 生成
# ============================================================
def create_circular_mask_auto(img, verbose=False):
    """
    根据图像内容自动检测实际 FOV 的圆形 Mask。
    返回: (mask, cx, cy, radius)
    """
    h, w = img.shape
    bg_val = np.median([img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]])
    threshold = bg_val + 1e-4 if img.dtype in [np.float32, np.float64] else bg_val + 1
    valid_coords = np.argwhere(img > threshold)

    if len(valid_coords) == 0:
        center_x, center_y = w // 2, h // 2
        radius = min(w // 2, h // 2) - 1
    else:
        y_min, x_min = valid_coords.min(axis=0)
        y_max, x_max = valid_coords.max(axis=0)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        # [F5] 用 x/y 跨度的平均值作为半径，对椭圆/部分截断 FOV 更稳健
        radius = ((x_max - x_min) + (y_max - y_min)) / 4.0 - 3

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist_from_center <= radius

    if verbose:
        print(f"  [FOV] center=({center_x}, {center_y}), radius={radius:.1f}, "
              f"image=({w}x{h})")

    return mask, int(center_x), int(center_y), float(radius)


# ============================================================
#  归一化
# ============================================================
def normalize_for_metrics(gt, rec, mode='minmax', clip=False):
    """
    mode = 'minmax'      : 用 GT 的 min/max 线性归一化
    mode = 'percentile'  : 用 GT 的 0.1% / 99.9% 分位数归一化（去除噪点）
    mode = 'raw'         : 不做归一化（保留 μ 值物理量纲）
    clip                 : 是否对归一化后的 gt 与 rec 同步 clip 到 [0,1]。
                           [F2] 之前 minmax 模式下只 clip rec 不 clip gt 是错的。
    返回 (gt_proc, rec_proc, data_range)
    """
    if mode == 'raw':
        dr = float(gt.max() - gt.min())
        return gt.astype(np.float32), rec.astype(np.float32), (dr if dr > 0 else 1.0)

    if mode == 'percentile':
        vmin = np.percentile(gt, 0.1)
        vmax = np.percentile(gt, 99.9)
    else:  # minmax
        vmin, vmax = gt.min(), gt.max()

    if vmax - vmin == 0:
        return gt.copy().astype(np.float32), rec.copy().astype(np.float32), 1.0

    gt_norm = (gt - vmin) / (vmax - vmin)
    rec_norm = (rec - vmin) / (vmax - vmin)

    # [F2] percentile 模式总是 clip（gt 也可能因为分位数低于 max 而出界）；
    # minmax 模式默认不 clip（gt 必然在 [0,1] 内，rec 可能过冲，clip 会人为压低误差）；
    # 用户可通过 --normalize_clip 强制 clip。
    if mode == 'percentile' or clip:
        gt_norm = np.clip(gt_norm, 0, 1)
        rec_norm = np.clip(rec_norm, 0, 1)

    return gt_norm.astype(np.float32), rec_norm.astype(np.float32), 1.0


# ============================================================
#  基础保真度指标 (Mask-aware)
# ============================================================
def compute_psnr_masked(gt, rec, mask, data_range):
    valid = int(np.sum(mask))
    if valid == 0:
        return float('nan')  # [F7]
    mse = float(np.sum(((gt - rec) * mask) ** 2) / valid)
    if mse <= 0:
        return 100.0  # [F6] cap，避免 inf 污染均值
    return float(10 * np.log10((data_range ** 2) / mse))


def compute_ssim_masked(gt, rec, mask, data_range, win_size=7):
    """
    [F1] 修复：skimage 的 ssim_map 在边界 win//2 圈内不可靠。
    我们按 (win_size-1)//2 把 ssim_map 与 mask 同步裁掉，再做加权平均。
    """
    if int(np.sum(mask)) == 0:
        return float('nan')  # [F7]

    _, ssim_map = structural_similarity(
        gt, rec, data_range=data_range, full=True, win_size=win_size
    )
    pad = (win_size - 1) // 2
    if pad > 0:
        ssim_map_c = sk_crop(ssim_map, pad, copy=False)
        mask_c = sk_crop(mask.astype(np.float32), pad, copy=False)
    else:
        ssim_map_c = ssim_map
        mask_c = mask.astype(np.float32)

    valid = float(mask_c.sum())
    if valid == 0:
        return float('nan')
    return float(np.sum(ssim_map_c * mask_c) / valid)


def compute_rmse_masked(gt, rec, mask):
    valid = int(np.sum(mask))
    if valid == 0:
        return float('nan')  # [F7]
    mse = float(np.sum(((gt - rec) * mask) ** 2) / valid)
    return float(np.sqrt(mse))


# ============================================================
#  梯度幅值误差 GME
# ============================================================
def compute_gme(gt, rec, mask):
    # [F8] sobel(axis=0) 是沿行方向（即对 y 求导），命名修正为 gy/gx
    gt_gy, gt_gx = ndimage.sobel(gt, axis=0), ndimage.sobel(gt, axis=1)
    rec_gy, rec_gx = ndimage.sobel(rec, axis=0), ndimage.sobel(rec, axis=1)
    gt_grad = np.hypot(gt_gx, gt_gy)
    rec_grad = np.hypot(rec_gx, rec_gy)

    valid = int(np.sum(mask))
    if valid == 0:
        return float('nan'), gt_grad, rec_grad
    mae_grad = float(np.sum(np.abs(gt_grad - rec_grad) * mask) / valid)
    return mae_grad, gt_grad, rec_grad


# ============================================================
#  Streak (Polar FFT) & High-Freq RMS
# ============================================================
def streak_polar_fft(image, center=None, n_angular=360, n_radial=200,
                     freq_cutoff_ratio=0.25):
    """
    [F4] 现在接受 center=(cx, cy)；若为 None 则退回图像中心（旧行为）。
    """
    warp_polar = _lazy_import_warp_polar()
    img = image.astype(np.float32)
    H, W = img.shape

    if center is None:
        cx, cy = W / 2.0, H / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])

    # 半径取离 (cx,cy) 最远的图像角到中心的较小值，确保极坐标圆完全在图内
    max_r = min(cx, W - 1 - cx, cy, H - 1 - cy)
    radius = int(max_r) - 1
    if radius < 10:
        return float('nan')

    polar = warp_polar(img, center=(cy, cx),  # warp_polar 用 (row, col)
                       radius=radius, output_shape=(n_angular, n_radial))
    spec = np.abs(np.fft.rfft(polar, axis=0))
    n_freq = spec.shape[0]
    cutoff = max(1, int(n_freq * freq_cutoff_ratio))
    streak_band = spec[cutoff:, :]
    streak_energy = float((streak_band ** 2).sum())
    total_energy = float((img ** 2).sum()) + 1e-10
    return streak_energy / total_energy


def high_freq_rms(image, sigma_ratio=0.012):
    """
    [F9] sigma 改为相对图像短边的比例，跨分辨率可比。
    256 像素时 sigma≈3.07，与原默认 sigma=3.0 接近，向下兼容。
    """
    gauss = _lazy_import_gaussian()
    img = image.astype(np.float32)
    short_side = min(img.shape)
    sigma = max(0.5, sigma_ratio * short_side)
    smooth = gauss(img, sigma)
    high_freq = img - smooth
    img_rms = float(np.sqrt(np.mean(img ** 2))) + 1e-10
    hf_rms = float(np.sqrt(np.mean(high_freq ** 2)))
    return hf_rms / img_rms


# ============================================================
#  CNR (依赖分割图)
# ============================================================
def compute_cnr(image, seg_mask, min_pixels=100):
    """seg_mask: 1=air, 3=filler, 4=obj (2DeteCT 约定; 标签 2 一般为容器壁，被忽略)"""
    img = image.astype(np.float32)
    seg = seg_mask.astype(np.int32)
    air = (seg == 1)
    filler = (seg == 3)
    obj = (seg == 4)

    out = {
        'noise_std_air': float('nan'),
        'mean_air': float('nan'),
        'mean_filler': float('nan'),
        'mean_obj': float('nan'),
        'CNR_obj_air': float('nan'),
        'CNR_obj_filler': float('nan'),
    }

    if air.sum() >= min_pixels:
        out['noise_std_air'] = float(img[air].std())
        out['mean_air'] = float(img[air].mean())
    if filler.sum() >= min_pixels:
        out['mean_filler'] = float(img[filler].mean())
    if obj.sum() >= min_pixels:
        out['mean_obj'] = float(img[obj].mean())

    sn = out['noise_std_air']
    if not np.isnan(sn) and sn > 1e-10:
        if not np.isnan(out['mean_obj']) and not np.isnan(out['mean_air']):
            out['CNR_obj_air'] = float(abs(out['mean_obj'] - out['mean_air']) / sn)
        if not np.isnan(out['mean_obj']) and not np.isnan(out['mean_filler']):
            out['CNR_obj_filler'] = float(abs(out['mean_obj'] - out['mean_filler']) / sn)
    return out


# ============================================================
#  LPIPS (空间图)
# ============================================================
class LPIPSEvaluator:
    def __init__(self, device):
        torch, F = _lazy_import_torch()
        lpips = _lazy_import_lpips()
        self.torch, self.F = torch, F
        self.device = device
        print("正在加载 LPIPS (VGG) 模型...")
        self.lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device)
        self.lpips_fn.eval()

    def img_to_tensor(self, img_norm):
        t = self.torch.tensor(img_norm, dtype=self.torch.float32).unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 3, 1, 1)
        t = t * 2.0 - 1.0
        return t.to(self.device)

    def get_spatial(self, gt_norm, rec_norm):
        h, w = gt_norm.shape
        gt_t = self.img_to_tensor(gt_norm)
        rec_t = self.img_to_tensor(rec_norm)
        with self.torch.no_grad():
            dist_map = self.lpips_fn.forward(gt_t, rec_t)
        dist_map = self.F.interpolate(dist_map, size=(h, w),
                                      mode='bilinear', align_corners=False)
        return dist_map.squeeze().cpu().numpy()


def lpips_masked(dist_map_np, mask):
    valid = int(np.sum(mask))
    if valid == 0:
        return float('nan')  # [F7]
    return float(np.sum(dist_map_np * mask) / valid)


# ============================================================
#  Soft GAPS
# ============================================================
def compute_soft_gaps(gt_norm, rec_norm, dist_map_np, fg_mask, k=15.0,
                      edge_percentile=85.0, lambda_flat=5.0):
    """返回: gaps_total, edge_lpips, flat_mae, W, W_inv  (后两个用于可视化)"""
    # [F8] 命名修正
    gt_gy, gt_gx = ndimage.sobel(gt_norm, axis=0), ndimage.sobel(gt_norm, axis=1)
    gt_grad = np.hypot(gt_gx, gt_gy)

    grad_in_fg = gt_grad[fg_mask]
    if len(grad_in_fg) > 0:
        edge_thresh = np.percentile(grad_in_fg, edge_percentile)
    else:
        edge_thresh = 0.1

    z = np.clip(-k * (gt_grad - edge_thresh), -80, 80)
    W = 1.0 / (1.0 + np.exp(z))

    W_fg = W * fg_mask
    W_inv_fg = (1.0 - W) * fg_mask
    sum_W = float(W_fg.sum())
    sum_W_inv = float(W_inv_fg.sum())

    edge_lpips = float((dist_map_np * W_fg).sum() / sum_W) if sum_W > 0 else 0.0
    flat_mae = float((np.abs(gt_norm - rec_norm) * W_inv_fg).sum() / sum_W_inv) \
        if sum_W_inv > 0 else 0.0
    gaps_total = edge_lpips + lambda_flat * flat_mae

    return gaps_total, edge_lpips, flat_mae, W_fg, W_inv_fg


# ============================================================
#  可视化
# ============================================================
def make_debug_visualization(out_path, base_name, gt_norm, rec_norm,
                             masks_to_show, dist_map_np=None,
                             gt_grad=None, rec_grad=None,
                             gaps_W_fg=None, gaps_W_inv_fg=None,
                             vis_norm_label='minmax',
                             fov_info=None):
    """
    一张大图，包含：
      Row 1: GT | Pred | |GT - Pred| 误差图
      Row 2: 区域 Mask 叠加 | LPIPS 热力图(若有) | 梯度差异图(若有)
      Row 3: GAPS 权重叠加(若有)
    """
    plt = _lazy_import_plt()

    panels = []
    panels.append(('Ground Truth', gt_norm, 'gray', None))
    panels.append(('Prediction', rec_norm, 'gray', None))
    panels.append(('|GT - Pred|', np.abs(gt_norm - rec_norm), 'hot', 'Abs Error'))

    overlay = np.stack([gt_norm, gt_norm, gt_norm], axis=-1).astype(np.float32)
    overlay = np.clip(overlay, 0, 1)
    palette = {
        'fg': np.array([1.0, 0.2, 0.2]),
        'bg': np.array([0.2, 0.5, 1.0]),
        'full': np.array([0.3, 1.0, 0.3]),
    }
    legend_lines = []
    for name, m in masks_to_show.items():
        color = palette.get(name, np.array([1.0, 1.0, 0.0]))
        alpha = 0.30
        for c in range(3):
            overlay[..., c] = np.where(m, overlay[..., c] * (1 - alpha) + color[c] * alpha,
                                       overlay[..., c])
        legend_lines.append(f"{name}")
    panels.append(('Region Mask Overlay\n[' + ' / '.join(legend_lines) + ']',
                   overlay, None, None))

    if dist_map_np is not None:
        panels.append(('LPIPS Spatial Map', dist_map_np, 'jet', 'LPIPS'))

    if gt_grad is not None and rec_grad is not None:
        grad_diff = np.abs(gt_grad - rec_grad)
        panels.append(('|∇GT| - |∇Pred|  (GME)', grad_diff, 'magma', 'Δ|∇|'))

    if gaps_W_fg is not None and gaps_W_inv_fg is not None:
        gaps_overlay = np.stack([gt_norm, gt_norm, gt_norm], axis=-1) * 0.4
        color_edge = np.array([0.9, 0.9, 0.0])
        color_flat = np.array([0.0, 0.3, 0.9])
        gaps_overlay += gaps_W_fg[..., None] * color_edge \
                      + gaps_W_inv_fg[..., None] * color_flat
        gaps_overlay = np.clip(gaps_overlay, 0, 1)
        panels.append(('Soft GAPS Weight\n(Yellow=edge LPIPS, Blue=flat MAE)',
                       gaps_overlay, None, None))

    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, data, cmap, cbar_label) in zip(axes, panels):
        if cmap is None:
            im = ax.imshow(data)
        else:
            im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        if cbar_label is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    for ax in axes[n:]:
        ax.axis('off')

    fov_str = ""
    if fov_info is not None:
        cx, cy, r = fov_info
        fov_str = f"   FOV center=({cx},{cy}), r={r:.1f}"
    plt.suptitle(f"Debug — {base_name}  (display norm: {vis_norm_label}){fov_str}",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ============================================================
#  主流程
# ============================================================
ALL_METRICS = ['psnr', 'ssim', 'rmse', 'lpips', 'gme', 'vif',
               'gaps', 'cnr', 'streak', 'hf_rms']
PERCEPTUAL_METRICS = {'lpips', 'gaps'}

# 每个指标使用的归一化方式（与其他指标是否启用无关）
METRIC_NORM = {
    'psnr':   'minmax',
    'ssim':   'minmax',
    'rmse':   'minmax',
    'lpips':  'percentile',
    'gme':    'percentile',
    'vif':    'percentile',
    'gaps':   'percentile',
    'cnr':    'raw',
    'streak': 'raw',
    'hf_rms': 'raw',
}


def resolve_metrics(arg_metrics):
    if 'all' in arg_metrics:
        return list(ALL_METRICS)
    invalid = [m for m in arg_metrics if m not in ALL_METRICS]
    if invalid:
        raise ValueError(f"未知指标: {invalid}. 可选: {ALL_METRICS + ['all']}")
    return list(arg_metrics)


def resolve_regions(arg_region):
    if arg_region == 'all':
        return ['full', 'fg', 'bg']
    if arg_region in ['full', 'fg', 'bg']:
        return [arg_region]
    raise ValueError(f"未知 region: {arg_region}")


def evaluate_files(args):
    metrics = resolve_metrics(args.metrics)
    regions = resolve_regions(args.region)
    print(f"\n>>> 启用指标: {metrics}")
    print(f">>> 计算区域: {regions}")
    print(">>> 每指标归一化:")
    for m in metrics:
        print(f"      {m:7s} -> {METRIC_NORM[m]}")
    if args.psnr_data_range is not None:
        print(f">>> PSNR/RMSE 使用全局 data_range = {args.psnr_data_range} (raw 物理量纲)")
    if args.normalize_clip:
        print(">>> [WARN] --normalize_clip 已启用：minmax 模式下将对 gt/rec 都 clip 到 [0,1]")

    # [F10] CNR + seg 缺失检查
    if 'cnr' in metrics and not args.seg_dir:
        print("\033[93m[WARN] 启用了 cnr 但未提供 --seg_dir，CNR 将被跳过。\033[0m")

    pred_files = []
    for ext in ('*.npy', '*.tif', '*.tiff'):
        pred_files.extend(glob.glob(os.path.join(args.results_dir, ext)))
    pred_files = sorted(pred_files)
    if not pred_files:
        print(f"未在 {args.results_dir} 中找到 .npy / .tif 文件。")
        return

    # 初始化 LPIPS（如需要）
    evaluator = None
    if any(m in PERCEPTUAL_METRICS for m in metrics):
        torch, _ = _lazy_import_torch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f">>> LPIPS 计算设备: {device}")
        evaluator = LPIPSEvaluator(device)

    results = {r: {} for r in regions}
    global_results = {}

    paired_count = 0
    debug_target_idx = max(0, int(args.debug_index))
    seen_paired_idx = -1

    pbar = tqdm(pred_files, desc="Evaluating", unit="file")
    for pred_path in pbar:
        base_name, ext = os.path.splitext(os.path.basename(pred_path))
        gt_path = os.path.join(args.gt_dir, base_name + '.tif')
        if not os.path.exists(gt_path):
            continue

        if ext.lower() == '.npy':
            rec = np.load(pred_path).astype(np.float32)
        else:
            rec = imageio.imread(pred_path).astype(np.float32)
        gt = imageio.imread(gt_path).astype(np.float32)

        # GT/rec 中心裁剪对齐
        if gt.shape != rec.shape:
            h_min = min(gt.shape[0], rec.shape[0])
            w_min = min(gt.shape[1], rec.shape[1])
            gt = center_crop_to(gt, h_min, w_min)
            rec = center_crop_to(rec, h_min, w_min)

        h, w = gt.shape
        seen_paired_idx += 1
        is_debug_target = (args.debug and seen_paired_idx == debug_target_idx)

        # ----- 按需归一化：每种 mode 只算一次并缓存 -----
        norm_cache = {}

        def get_norm(mode):
            if mode not in norm_cache:
                norm_cache[mode] = normalize_for_metrics(
                    gt, rec, mode=mode, clip=args.normalize_clip
                )
            return norm_cache[mode]

        # FOV mask 用 minmax 归一化结果检测
        gt_for_mask, _, _ = get_norm('minmax')
        full_mask = np.ones((h, w), dtype=bool)
        fg_mask, fov_cx, fov_cy, fov_r = create_circular_mask_auto(
            gt_for_mask, verbose=is_debug_target
        )
        bg_mask = ~fg_mask
        all_masks = {'full': full_mask, 'fg': fg_mask, 'bg': bg_mask}
        active_masks = {r: all_masks[r] for r in regions}

        # ====== 提前算可能复用的中间量 ======
        dist_map_np = None
        if 'lpips' in metrics or 'gaps' in metrics:
            gt_p, rec_p, _ = get_norm('percentile')
            dist_map_np = evaluator.get_spatial(gt_p, rec_p)

        gt_grad = rec_grad = None
        if 'gme' in metrics or args.debug:
            gt_p, rec_p, _ = get_norm('percentile')
            # [F8] gy, gx 命名修正
            gy_g, gx_g = ndimage.sobel(gt_p, axis=0), ndimage.sobel(gt_p, axis=1)
            gy_r, gx_r = ndimage.sobel(rec_p, axis=0), ndimage.sobel(rec_p, axis=1)
            gt_grad = np.hypot(gx_g, gy_g)
            rec_grad = np.hypot(gx_r, gy_r)

        # ====== Per-region 指标 ======
        for r in regions:
            m = active_masks[r]
            store = results[r]

            if 'psnr' in metrics:
                gt_n, rec_n, dr = get_norm(METRIC_NORM['psnr'])
                # [F11] 若用户提供了全局 data_range（基于物理量纲），则用 raw 数据 + 该 dr
                if args.psnr_data_range is not None:
                    gt_n, rec_n, _ = get_norm('raw')
                    dr = float(args.psnr_data_range)
                store.setdefault('psnr', []).append(
                    compute_psnr_masked(gt_n, rec_n, m, dr))
            if 'ssim' in metrics:
                gt_n, rec_n, dr = get_norm(METRIC_NORM['ssim'])
                store.setdefault('ssim', []).append(
                    compute_ssim_masked(gt_n, rec_n, m, dr))
            if 'rmse' in metrics:
                if args.psnr_data_range is not None:
                    gt_n, rec_n, _ = get_norm('raw')
                else:
                    gt_n, rec_n, _ = get_norm(METRIC_NORM['rmse'])
                store.setdefault('rmse', []).append(
                    compute_rmse_masked(gt_n, rec_n, m))
            if 'gme' in metrics:
                valid = int(np.sum(m))
                if valid > 0:
                    val = float(np.sum(np.abs(gt_grad - rec_grad) * m) / valid)
                else:
                    val = float('nan')  # [F7]
                store.setdefault('gme', []).append(val)
            if 'lpips' in metrics:
                store.setdefault('lpips', []).append(
                    lpips_masked(dist_map_np, m))

        # ====== 全图标量指标 ======
        if 'vif' in metrics:
            vifp = _lazy_import_vif()
            gt_n, rec_n, _ = get_norm(METRIC_NORM['vif'])
            global_results.setdefault('vif', []).append(float(vifp(gt_n, rec_n)))

        if 'streak' in metrics:
            _, rec_raw, _ = get_norm(METRIC_NORM['streak'])
            # [F4] 把 FOV 中心传给极坐标变换
            global_results.setdefault('streak', []).append(
                streak_polar_fft(rec_raw, center=(fov_cx, fov_cy)))

        if 'hf_rms' in metrics:
            _, rec_raw, _ = get_norm(METRIC_NORM['hf_rms'])
            global_results.setdefault('hf_rms', []).append(
                high_freq_rms(rec_raw, sigma_ratio=args.hf_sigma_ratio))

        # ====== GAPS ======
        gaps_W_fg = gaps_W_inv_fg = None
        if 'gaps' in metrics:
            gt_p, rec_p, _ = get_norm(METRIC_NORM['gaps'])
            gaps_total, edge_lpips, flat_mae, gaps_W_fg, gaps_W_inv_fg = \
                compute_soft_gaps(gt_p, rec_p, dist_map_np, fg_mask)
            global_results.setdefault('gaps_total', []).append(gaps_total)
            global_results.setdefault('gaps_edge_lpips', []).append(edge_lpips)
            global_results.setdefault('gaps_flat_mae', []).append(flat_mae)

        # ====== CNR ======
        per_image_cnr = None
        if 'cnr' in metrics and args.seg_dir:
            # [F14] seg 文件名 = base_name + seg_suffix + .tif
            seg_filename = f"{base_name}{args.seg_suffix}.tif"
            seg_path = os.path.join(args.seg_dir, seg_filename)
            if os.path.exists(seg_path):
                seg = imageio.imread(seg_path)
                # [F3] 与 gt/rec 用相同的中心裁切对齐
                if seg.shape != rec.shape:
                    seg = center_crop_to(seg, h, w)
                if seg.shape == rec.shape:
                    cnr = compute_cnr(rec, seg, min_pixels=args.cnr_min_pixels)
                    for k, v in cnr.items():
                        global_results.setdefault(k, []).append(v)
                    per_image_cnr = cnr
                else:
                    print(f"\033[93m[WARN] {base_name}: seg 与 rec 尺寸无法对齐 "
                          f"({seg.shape} vs {rec.shape})，跳过 CNR\033[0m")
            else:
                if not getattr(args, '_seg_missing_warned', False):
                    print(f"\033[93m[WARN] {base_name}: 找不到 seg 文件 "
                          f"{seg_path}（之后类似的缺失将不再提示）\033[0m")
                    args._seg_missing_warned = True

        # ====== Debug 可视化 ======
        if is_debug_target:
            debug_dir = args.debug_dir
            os.makedirs(debug_dir, exist_ok=True)
            out_png = os.path.join(debug_dir, f"{base_name}_debug.png")
            if 'percentile' in norm_cache:
                gt_vis, rec_vis, _ = norm_cache['percentile']
                vis_norm_label = 'percentile'
            else:
                gt_vis, rec_vis, _ = norm_cache['minmax']
                vis_norm_label = 'minmax'
            make_debug_visualization(
                out_path=out_png,
                base_name=base_name,
                gt_norm=gt_vis,
                rec_norm=rec_vis,
                masks_to_show=active_masks,
                dist_map_np=dist_map_np,
                gt_grad=gt_grad if 'gme' in metrics else None,
                rec_grad=rec_grad if 'gme' in metrics else None,
                gaps_W_fg=gaps_W_fg,
                gaps_W_inv_fg=gaps_W_inv_fg,
                vis_norm_label=vis_norm_label,
                fov_info=(fov_cx, fov_cy, fov_r),
            )
            print(f"\n\033[92m[Debug] 可视化已保存至: {out_png}\033[0m")

        # ====== [F15] 进度条实时更新各指标的累计均值 ======
        if args.verbose_per_image:
            postfix = {}

            # Per-region 指标：选一个代表性 region 显示在 postfix（避免太长）
            # 优先 fg（前景，最常用），没 fg 就 full
            preferred = 'fg' if 'fg' in regions else regions[0]
            store = results[preferred]
            for mname in ['psnr', 'ssim', 'rmse', 'gme', 'lpips']:
                if mname in store and len(store[mname]) > 0:
                    vals = [v for v in store[mname]
                            if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    if vals:
                        postfix[f"{mname}({preferred})"] = f"{np.mean(vals):.4f}"

            # Global 指标
            for mname in ['vif', 'streak', 'hf_rms', 'gaps_total']:
                if mname in global_results and len(global_results[mname]) > 0:
                    vals = [v for v in global_results[mname]
                            if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    if vals:
                        postfix[mname] = f"{np.mean(vals):.4f}"

            # CNR 关键项
            for mname in ['CNR_obj_air', 'CNR_obj_filler']:
                if mname in global_results and len(global_results[mname]) > 0:
                    vals = [v for v in global_results[mname]
                            if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    if vals:
                        # 缩短显示名
                        short = mname.replace('CNR_obj_', 'CNR.')
                        postfix[short] = f"{np.mean(vals):.3f}"

            if postfix:
                pbar.set_postfix(postfix)

        paired_count += 1

    if paired_count == 0:
        print("未找到任何配对的 GT 文件。")
        return

    # ====== 汇总打印 ======
    arrows = {
        'psnr': '↑', 'ssim': '↑', 'rmse': '↓',
        'lpips': '↓', 'gme': '↓', 'vif': '↑',
        'gaps_total': '↓', 'gaps_edge_lpips': '↓', 'gaps_flat_mae': '↓',
        'streak': '↓', 'hf_rms': '↓',
        'CNR_obj_air': '↑', 'CNR_obj_filler': '↑',
        'noise_std_air': '↓',
        'mean_air': '-', 'mean_filler': '-', 'mean_obj': '-',
    }

    lines = []
    lines.append("\n" + "=" * 55)
    lines.append(f"   评估结果 (共 {paired_count} 对图像)")
    lines.append("=" * 55)

    region_label = {'full': 'Full Image 全图',
                    'fg': 'Foreground 前景 (FOV圆内)',
                    'bg': 'Background 背景 (FOV圆外)'}

    for r in regions:
        if not results[r]:
            continue
        lines.append(f"\n[{region_label[r]}]")
        for mname, vals in results[r].items():
            arrow = arrows.get(mname, '')
            mean_val = np.nanmean(vals) if len(vals) > 0 else float('nan')
            lines.append(f"  {mname.upper():6s} ({arrow}): {mean_val:.4f}")

    if global_results:
        lines.append("\n" + "-" * 55)
        lines.append("[Global / Region-independent 指标]")
        for mname, vals in global_results.items():
            arrow = arrows.get(mname, '')
            valid = [v for v in vals
                     if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if valid:
                lines.append(f"  {mname:18s} ({arrow}): {np.nanmean(valid):.4f}")
            else:
                lines.append(f"  {mname:18s}: (无有效值)")

    lines.append("=" * 55 + "\n")
    result_str = "\n".join(lines)
    print(result_str)

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or '.',
                    exist_ok=True)
        with open(args.save_path, 'w', encoding='utf-8') as f:
            f.write(result_str)
        print(f"结果已保存至: {args.save_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--results_dir", type=str, required=True,
                   help="预测结果 (.npy / .tif) 所在目录")
    p.add_argument("--gt_dir", type=str, required=True,
                   help="真值图 (.tif) 所在目录")
    p.add_argument("--save_path", type=str, default=None,
                   help="结果保存的 txt 文件路径")

    p.add_argument("--metrics", type=str, nargs='+', default=['psnr', 'ssim', 'rmse'],
                   help=f"要计算的指标，可多选。可选: {ALL_METRICS + ['all']}. "
                        f"默认: psnr ssim rmse")
    p.add_argument("--region", type=str, default='all',
                   choices=['full', 'fg', 'bg', 'all'],
                   help="计算区域：full=全图, fg=前景(FOV圆内), bg=背景, all=三者都算")

    p.add_argument("--debug", action='store_true',
                   help="启用后，对指定索引的图生成可视化")
    p.add_argument("--debug_dir", type=str, default='./cache/result/eval_debug',
                   help="Debug 可视化保存目录")
    p.add_argument("--debug_index", type=int, default=0,
                   help="[F12] 可视化第几对配对图像（0-based，默认 0=第一张）")

    # CNR 用
    p.add_argument("--seg_dir", type=str, default=None,
                   help="分割图目录 (与 GT 同名 .tif，1=air, 3=filler, 4=obj)，仅 cnr 指标需要")
    p.add_argument("--seg_suffix", type=str, default='_segmentation',
                   help="[F14] seg 文件名后缀。完整文件名 = base_name + seg_suffix + .tif。"
                        "例如 GT='slice04501.tif', seg='slice04501_segmentation.tif' 时"
                        "用默认值即可。如 seg 与 GT 完全同名，传 '' 空字符串。")
    p.add_argument("--cnr_min_pixels", type=int, default=100,
                   help="CNR 计算时每个类别最少需要的像素数")

    # [F15] 进度条 postfix 显示累计均值
    p.add_argument("--verbose_per_image", dest='verbose_per_image',
                   action='store_true', default=True,
                   help="[F15] 进度条后实时显示当前各指标的累计均值（默认开启）")
    p.add_argument("--no_verbose_per_image", dest='verbose_per_image',
                   action='store_false',
                   help="关闭进度条 postfix（仅在末尾打印汇总）")

    # [F2/F11] 归一化与 PSNR data range 控制
    p.add_argument("--normalize_clip", action='store_true',
                   help="[F2] minmax 模式下也对 gt/rec 同步 clip 到 [0,1]。默认关闭，"
                        "保留 rec 的过冲，避免人为抬高 PSNR。")
    p.add_argument("--psnr_data_range", type=float, default=None,
                   help="[F11] 若给定，PSNR/RMSE 使用此值作为全局 data_range（在 raw "
                        "物理量纲上计算），跨数据集才有可比性。例: CT μ 值 1.0 / HU "
                        "范围对应的归一化值。默认 None = 沿用 minmax 逐图归一化。")

    # [F9] hf_rms 的 sigma 比例
    p.add_argument("--hf_sigma_ratio", type=float, default=0.012,
                   help="[F9] high_freq_rms 的 sigma = sigma_ratio * min(H,W)，"
                        "默认 0.012（256 像素时 sigma≈3，向后兼容）。")

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    evaluate_files(args)


'''
# 1) 基础保真度 (PSNR/SSIM/RMSE)，全图+前景+背景
python eval_unified.py \
    --results_dir /path/to/pred \
    --gt_dir /path/to/gt \
    --metrics psnr ssim rmse \
    --region all \
    --save_path /path/to/metrics.txt

# 2) 感知质量套餐 (LPIPS+GME+VIF+GAPS)，开启可视化
python eval_unified.py \
    --results_dir /path/to/pred \
    --gt_dir /path/to/gt \
    --metrics lpips gme vif gaps \
    --region all \
    --debug --debug_index 0 \
    --save_path /path/to/perceptual.txt

# 3) 跨数据集可比的 PSNR（基于固定物理 data_range）
python eval_unified.py \
    --results_dir /path/to/pred --gt_dir /path/to/gt \
    --metrics psnr rmse --region fg \
    --psnr_data_range 1.0

# 4) 全套指标 + CNR + Streak（需要 seg_dir）
python /ibex/user/wangz0r/CS_300_Final_Project/CS_300/Eval/eval_unified.py \
    --results_dir /ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/result/LD_comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner/test \
    --gt_dir /ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/data/mode1_agd_ds6__20260502_044401/comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner/Test/gt \
    --seg_dir /ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/data/00_SEG_GT \
    --metrics all \
    --region all \
    --normalize_clip \
    --debug \
    --debug_index 1 \
    --save_path /ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/result/LD_comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner/metrics.txt
'''
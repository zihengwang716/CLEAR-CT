#!/usr/bin/env python3
"""
三合一问题下的 BH 校正重建增强版。

Pipeline:
  mode3/mode3_noisy -> optional BH 多项式校正 -> 下采样 -> AGD/FISTA 重建 -> 与 mode2 GT 比较

新增功能：
  1. 更稳健的默认 step 配置：power iteration=25，step_scale=0.8
  2. 可选 inverse-variance weighting：min_x 0.5 * ||sqrt(W)(Ax-b)||^2
  3. 可选逐迭代 TV prox：FISTA/AGD 每轮 gradient step 后执行 TV proximal
  4. weight 模式会自动按 weight_clip_max 缩小 step，避免 weighted objective 步长过大
  5. BH 改为可选：不传 --bh_coeffs 时读取普通文件名；传入时默认读取 noisy/scaled 文件名
  6. 支持 --input_mode 选择 sliceXXXXX/mode? 输入目录，并写入同名输出子目录
  7. 支持 --experiment_name/--timestamp 管理输出目录
  8. 输出 tag 自描述：支持的写具体配置，不支持的写 noXXX
"""

import argparse
import csv
import inspect
import os
import time

import numpy as np
from scipy.interpolate import interp1d

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from skimage.metrics import structural_similarity

if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return inspect.ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

import astra

DET_PIX = 0.0748
SOD = 431.019989
SDD = 529.000488
CORR = np.array([1.00, 0.0])
N_DET_BINNED = 956
ASTRA_MAX_ANGLES = 2560
EPS = 1e-6


def sanitize_run_part(value):
    """Make a string safe for use as one output path component."""
    safe = []
    for ch in str(value).strip():
        if ch.isalnum() or ch in ("_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def format_float_tag(value):
    """Format float values for directory names, e.g. 1.0 -> 1, 0.6 -> 0p6."""
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def prepare_output_dir(base_out_dir, experiment_name="", timestamp=""):
    """Return OUT[/experiment_name__timestamp], creating it if needed."""
    if timestamp.lower() == "auto":
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    parts = []
    if experiment_name:
        parts.append(sanitize_run_part(experiment_name))
    if timestamp:
        parts.append(sanitize_run_part(timestamp))

    out_dir = os.path.join(base_out_dir, "__".join(parts)) if parts else base_out_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_tag(args, coeffs, layout):
    """Self-describing experiment tag.

    Supported features are written explicitly; disabled features use noXXX.
    Examples:
      comp__BH_deg3__W__TV_lam1__ds6__mode1_plain__iter200__rec2048
      comp__noBH__noW__noTV__ds1__mode3_plain__iter100__rec2048
    """
    parts = ["comp"]

    if args.bh_coeffs:
        parts.append(f"BH_deg{len(coeffs)}")
    else:
        parts.append("noBH")

    parts.append("W" if args.use_weight else "noW")

    if args.lambda_tv > 0:
        parts.append(f"TV_lam{format_float_tag(args.lambda_tv)}")
    else:
        parts.append("noTV")

    parts.append(f"ds{args.downsample}")
    parts.append(f"{sanitize_run_part(args.input_mode)}_{layout['variant']}")
    parts.append(f"iter{args.agd_iter}")
    parts.append(f"rec{args.rec_size}")
    parts.append(f"step{format_float_tag(args.step_scale)}")
    parts.append(f"pwr{args.power_iter}")

    if args.use_weight:
        parts.append(
            f"wclip{format_float_tag(args.weight_clip_min)}-{format_float_tag(args.weight_clip_max)}"
        )
        parts.append(f"wstep_{args.weight_step_mode}")
    else:
        parts.append("noWclip")
        parts.append("noWstep")

    if args.lambda_tv > 0:
        parts.append(f"tvinner{args.tv_inner}")
    else:
        parts.append("noTVinner")

    return "__".join(parts)


def first_frame(frame):
    return frame[0] if frame.ndim > 1 else frame


def input_layout(input_mode="mode3", use_bh=False, input_variant="auto"):
    """Return directory and file names for the requested input layout.

    input_mode:
      mode1/mode2/mode3/... : read from sliceXXXXX/<input_mode> and save to <input_mode>

    input_variant:
      auto   : noisy/scaled when BH is enabled, plain otherwise
      plain  : sinogram.tif + flat1.tif/flat2.tif
      noisy  : sinogram_with_poisson.tif + flat1_scaled.tif/flat2_scaled.tif
    """
    if input_variant == "auto":
        input_variant = "noisy" if use_bh else "plain"
    if input_variant == "noisy":
        return {
            "mode": input_mode,
            "sinogram": "sinogram_with_poisson.tif",
            "flat1": "flat1_scaled.tif",
            "flat2": "flat2_scaled.tif",
            "variant": "noisy",
            "tag": input_mode,
        }
    if input_variant == "plain":
        return {
            "mode": input_mode,
            "sinogram": "sinogram.tif",
            "flat1": "flat1.tif",
            "flat2": "flat2.tif",
            "variant": "plain",
            "tag": input_mode,
        }
    raise ValueError(f"unsupported input_variant: {input_variant}")


def load_sinogram(data_dir, slice_idx, input_mode="mode3", use_bh=False, input_variant="auto",
                  return_weights=False, weight_clip=(0.2, 5.0),
                  renormalize_after_clip=True):
    """Load sinogram from sliceXXXXX/<input_mode> and optionally inverse log-variance weights.

    Default naming behavior:
      - use_bh=True  -> sinogram_with_poisson.tif, flat1_scaled.tif, flat2_scaled.tif
      - use_bh=False -> sinogram.tif, flat1.tif, flat2.tif

    You can override this with --input_variant plain/noisy.

    sino_log = -log((I-D)/(I0-D))
    Var(sino_log) ≈ 1/(I-D) + 1/(I0-D)
    weight = 1 / Var(sino_log)
    """
    layout = input_layout(input_mode=input_mode, use_bh=use_bh, input_variant=input_variant)
    slice_dir = os.path.join(data_dir, f"slice{slice_idx:05d}", input_mode)
    dark = imageio.imread(os.path.join(slice_dir, 'dark.tif')).astype('float32')
    flat1 = imageio.imread(os.path.join(slice_dir, layout['flat1'])).astype('float32')
    flat2 = imageio.imread(os.path.join(slice_dir, layout['flat2'])).astype('float32')
    sinogram = imageio.imread(os.path.join(slice_dir,
                              layout['sinogram'])).astype('float32')
    flat = np.mean(np.array([flat1, flat2]), axis=0)

    sino_binned = sinogram[:, 0::2] + sinogram[:, 1::2]
    dark_frame = first_frame(dark)
    flat_frame = first_frame(flat)
    dark_binned = dark_frame[0::2] + dark_frame[1::2]
    flat_binned = flat_frame[0::2] + flat_frame[1::2]

    signal = sino_binned - dark_binned
    flat_signal = flat_binned - dark_binned
    signal_safe = np.clip(signal, EPS, None)
    flat_safe = np.clip(flat_signal, EPS, None)

    ratio = signal_safe / flat_safe
    ratio = ratio[:-1, :]

    weights = None
    if return_weights:
        weights = 1.0 / (1.0 / signal_safe + 1.0 / flat_safe)
        weights = weights[:-1, :]

    if slice_idx <= 2830 or (5521 <= slice_idx <= 5870):
        det_shift = CORR[0] * DET_PIX
    else:
        det_shift = CORR[1] * DET_PIX

    det_grid = np.arange(0, N_DET_BINNED) * DET_PIX
    ratio = interp1d(det_grid, ratio, kind='linear',
                     fill_value='extrapolate')(det_grid + det_shift)
    sino_log = -np.log(np.clip(ratio, EPS, None)).astype(np.float32)

    if not return_weights:
        return sino_log

    weights = interp1d(det_grid, weights, kind='linear',
                       fill_value='extrapolate')(det_grid + det_shift)
    weights = np.clip(weights, EPS, None)
    weights = weights / max(float(np.mean(weights)), EPS)
    weights = np.clip(weights, weight_clip[0], weight_clip[1])
    if renormalize_after_clip:
        weights = weights / max(float(np.mean(weights)), EPS)
    return sino_log, weights.astype(np.float32)

def apply_correction(sino_log, coeffs):
    result = np.zeros_like(sino_log)
    for k in range(len(coeffs)):
        result += coeffs[k] * (sino_log ** (k + 1))
    return result.astype(np.float32)


def load_gt(gt_dir, slice_idx, mode=2):
    path = os.path.join(gt_dir, f"slice{slice_idx:05d}", f"mode{mode}",
                        "reconstruction.tif")
    if not os.path.exists(path):
        return None
    return imageio.imread(path).astype(np.float32)


def cap_and_downsample(n_full, downsample_factor, max_angles=ASTRA_MAX_ANGLES):
    ds_idx = np.arange(0, n_full, downsample_factor)
    if len(ds_idx) > max_angles:
        sub = np.round(np.linspace(0, len(ds_idx) - 1, max_angles)).astype(int)
        ds_idx = ds_idx[sub]
    return ds_idx


def setup_geometry(angles, rec_size=2048):
    det_pix_sz = 2 * DET_PIX
    scale = 1.0 / ((det_pix_sz * N_DET_BINNED * SOD / SDD) / 1024)
    proj_geo = astra.create_proj_geom(
        'fanflat', det_pix_sz * scale, N_DET_BINNED, angles,
        SOD * scale, (SDD - SOD) * scale)
    vol_geo = astra.create_vol_geom(rec_size, rec_size)
    return proj_geo, vol_geo


def center_crop(img, target_h, target_w):
    h, w = img.shape
    if h <= target_h and w <= target_w:
        return img
    y0 = max((h - target_h) // 2, 0)
    x0 = max((w - target_w) // 2, 0)
    return img[y0:y0 + target_h, x0:x0 + target_w]


def compute_metrics(rec, gt):
    dr = max(float(gt.max() - gt.min()), 1e-10)
    mse = float(np.mean((gt - rec) ** 2))
    psnr = 10 * np.log10(dr ** 2 / max(mse, 1e-10))
    ssim = structural_similarity(gt, rec, data_range=dr)
    rmse = np.sqrt(mse)
    mae = float(np.mean(np.abs(gt - rec)))
    return {"PSNR": psnr, "SSIM": ssim, "RMSE": rmse, "MAE": mae}


def prox_tv(x, tau, n_inner=30):
    """Isotropic TV proximal operator via Chambolle projection.

    Solves approximately: min_z 0.5 * ||z - x||^2 + tau * TV(z)
    """
    if tau <= 0:
        return x

    ny, nx = x.shape
    p = np.zeros((2, ny, nx), dtype=np.float32)

    for _ in range(n_inner):
        div = np.zeros_like(x)
        div[:-1, :] += p[0, :-1, :]
        div[1:, :] -= p[0, :-1, :]
        div[:, :-1] += p[1, :, :-1]
        div[:, 1:] -= p[1, :, :-1]

        u = x + tau * div
        g = np.zeros((2, ny, nx), dtype=np.float32)
        g[0, :-1, :] = u[1:, :] - u[:-1, :]
        g[1, :, :-1] = u[:, 1:] - u[:, :-1]

        norm_g = np.sqrt(g[0] ** 2 + g[1] ** 2 + 1e-10)
        p = (p + (1.0 / (4 * tau + 1e-10)) * g) / (1 + norm_g / (tau + 1e-10))

    div = np.zeros_like(x)
    div[:-1, :] += p[0, :-1, :]
    div[1:, :] -= p[0, :-1, :]
    div[:, :-1] += p[1, :, :-1]
    div[:, 1:] -= p[1, :, :-1]
    return (x + tau * div).astype(np.float32)


class AGDReconstructor:
    def __init__(self, proj_geo, vol_geo, n_iter=200,
                 step_scale=0.8, power_iter=25, weight_step_mode="clip_max",
                 weight_clip_max=5.0):
        self.proj_geo = proj_geo
        self.vol_geo = vol_geo
        self.n_iter = n_iter
        self.step_scale = step_scale
        self.power_iter = power_iter
        self.weight_step_mode = weight_step_mode
        self.weight_clip_max = weight_clip_max
        self.n_rows = vol_geo['GridRowCount']
        self.n_cols = vol_geo['GridColCount']
        self.shape = (self.n_rows, self.n_cols)

        self.proj_id = astra.create_projector('cuda', proj_geo, vol_geo)
        self.vol_id = astra.data2d.create('-vol', vol_geo)
        self.sino_id = astra.data2d.create('-sino', proj_geo)
        self.grad_id = astra.data2d.create('-vol', vol_geo)

        fp_cfg = astra.astra_dict('FP_CUDA')
        fp_cfg['ProjectorId'] = self.proj_id
        fp_cfg['VolumeDataId'] = self.vol_id
        fp_cfg['ProjectionDataId'] = self.sino_id
        self.fp_alg = astra.algorithm.create(fp_cfg)

        bp_cfg = astra.astra_dict('BP_CUDA')
        bp_cfg['ProjectorId'] = self.proj_id
        bp_cfg['ReconstructionDataId'] = self.grad_id
        bp_cfg['ProjectionDataId'] = self.sino_id
        self.bp_alg = astra.algorithm.create(bp_cfg)

        self.lipschitz = self._estimate_lipschitz(power_iter)
        self.base_step = self.step_scale / self.lipschitz

    def _estimate_lipschitz(self, n_power_iter=25):
        x = np.random.randn(*self.shape).astype(np.float32)
        x /= max(np.linalg.norm(x.ravel()), EPS)
        for _ in range(n_power_iter):
            astra.data2d.store(self.vol_id, x)
            astra.algorithm.run(self.fp_alg)
            astra.data2d.store(self.grad_id,
                               np.zeros(self.shape, dtype=np.float32))
            astra.algorithm.run(self.bp_alg)
            ATAx = astra.data2d.get(self.grad_id)
            norm = np.linalg.norm(ATAx.ravel())
            if norm < 1e-10:
                return 1.0
            x = (ATAx / norm).astype(np.float32)
        astra.data2d.store(self.vol_id, x)
        astra.algorithm.run(self.fp_alg)
        Ax = astra.data2d.get(self.sino_id)
        L = float(np.sum(Ax ** 2) / max(np.sum(x ** 2), 1e-10))
        return max(L, EPS)

    def _effective_step(self, weights=None):
        if weights is None:
            return self.base_step
        if self.weight_step_mode == "none":
            return self.base_step
        if self.weight_step_mode == "actual_max":
            denom = max(float(np.max(weights)), EPS)
        else:  # clip_max: conservative and stable across slices
            denom = max(float(self.weight_clip_max), EPS)
        return self.base_step / denom

    def reconstruct(self, sino, weights=None, lambda_tv=0.0, tv_inner=30):
        x = np.zeros(self.shape, dtype=np.float32)
        y = x.copy()
        t = 1.0
        step = self._effective_step(weights)

        for _ in range(self.n_iter):
            astra.data2d.store(self.vol_id, y)
            astra.algorithm.run(self.fp_alg)
            Ay = astra.data2d.get(self.sino_id)
            residual = Ay - sino
            if weights is not None:
                residual = residual * weights
            astra.data2d.store(self.sino_id, residual.astype(np.float32))
            astra.data2d.store(self.grad_id,
                               np.zeros(self.shape, dtype=np.float32))
            astra.algorithm.run(self.bp_alg)
            grad = astra.data2d.get(self.grad_id)

            z = y - step * grad
            if lambda_tv > 0:
                z = prox_tv(z, step * lambda_tv, tv_inner)

            x_new = np.maximum(z, 0)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y = x_new + (t - 1) / t_new * (x_new - x)
            x = x_new.astype(np.float32)
            t = t_new
        return x

    def cleanup(self):
        astra.algorithm.delete(self.fp_alg)
        astra.algorithm.delete(self.bp_alg)
        astra.data2d.delete(self.vol_id)
        astra.data2d.delete(self.sino_id)
        astra.data2d.delete(self.grad_id)
        astra.projector.delete(self.proj_id)


def main():
    p = argparse.ArgumentParser(
        description="BH correction eval on compound degradation with optional weighting and TV")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--experiment_name", default="",
                   help="optional experiment name added as an output subdirectory")
    p.add_argument("--timestamp", default="",
                   help="optional timestamp added as an output subdirectory; use auto to generate one")
    p.add_argument("--bh_coeffs", default=None,
                   help="optional BH polynomial coefficients .npy; if omitted, BH correction is disabled")
    p.add_argument("--input_mode", type=str, default="mode3",
                   help="input subdirectory under each slice, e.g. mode1/mode2/mode3; output is saved under the same subdirectory name")
    p.add_argument("--input_variant", choices=["auto", "plain", "noisy"], default="auto",
                   help="input file naming: auto=noisy when BH is enabled, plain otherwise; plain uses sinogram.tif/flat*.tif; noisy uses sinogram_with_poisson.tif/flat*_scaled.tif")
    p.add_argument("--downsample", type=int, default=1,
                   help="angular downsample factor; default 1 means no intentional downsampling, then cap to ASTRA_MAX_ANGLES if needed")
    p.add_argument("--agd_iter", type=int, default=None,
                   help="AGD/FISTA outer iterations; default: 100 without weight, 200 with weight")
    p.add_argument("--rec_size", type=int, default=2048)
    p.add_argument("--slice_start", type=int, default=1)
    p.add_argument("--slice_end", type=int, default=5000)

    # Step options
    p.add_argument("--step_scale", type=float, default=0.8,
                   help="base step = step_scale / estimated_L; default 0.8")
    p.add_argument("--power_iter", type=int, default=25,
                   help="power iterations for Lipschitz estimate; default 25")

    # Weight options
    p.add_argument("--use_weight", action="store_true",
                   help="enable inverse log-variance weighted AGD")
    p.add_argument("--weight_clip_min", type=float, default=0.2)
    p.add_argument("--weight_clip_max", type=float, default=5.0)
    p.add_argument("--no_weight_renorm_after_clip", action="store_true",
                   help="do not renormalize weights to mean=1 after clipping")
    p.add_argument("--weight_step_mode", choices=["clip_max", "actual_max", "none"],
                   default="clip_max",
                   help="weighted step rule: clip_max uses step/weight_clip_max; actual_max uses step/max(weights); none keeps base step")

    # TV options
    p.add_argument("--lambda_tv", type=float, default=0.0,
                   help="TV regularization lambda; 0 disables TV")
    p.add_argument("--tv_inner", type=int, default=30,
                   help="inner iterations for Chambolle TV prox")

    args = p.parse_args()

    if args.agd_iter is None:
        args.agd_iter = 200 if args.use_weight else 100

    use_bh = args.bh_coeffs is not None
    coeffs = np.load(args.bh_coeffs) if use_bh else None
    bh_deg = len(coeffs) if use_bh else 0
    layout = input_layout(input_mode=args.input_mode, use_bh=use_bh, input_variant=args.input_variant)

    method_parts = (["BH", f"deg{bh_deg}"] if use_bh else ["noBH"]) + ["AGD"]
    method_parts.append("WInvVar" if args.use_weight else "noW")
    method_parts.append(f"TV{args.lambda_tv:g}" if args.lambda_tv > 0 else "noTV")
    method_name = "_".join(method_parts)

    tag = build_tag(args, coeffs, layout)
    run_out_dir = prepare_output_dir(args.out_dir, args.experiment_name, args.timestamp)
    out_dir = os.path.join(run_out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    n_full = 3600
    all_angles = np.linspace(0, 2 * np.pi, n_full, endpoint=False)
    ang_idx = cap_and_downsample(n_full, args.downsample)
    n_angles = len(ang_idx)
    angles = all_angles[ang_idx]

    print("=" * 60)
    if use_bh:
        print(f"Compound: BH correction (deg {bh_deg}) + AGD/FISTA")
        print(f"  BH coeffs: {args.bh_coeffs}")
    else:
        print("Compound: no BH correction + AGD/FISTA")
    print(f"  Input mode: {args.input_mode}")
    print(f"  Input variant: {args.input_variant} -> {layout['variant']} "
          f"({layout['sinogram']}, {layout['flat1']}, {layout['flat2']})")
    print(f"  Weighted: {args.use_weight}")
    print(f"  TV lambda: {args.lambda_tv}  TV inner: {args.tv_inner}")
    print(f"  Angles: {n_full} -> ds x{args.downsample} -> {n_angles}")
    print(f"  AGD iter: {args.agd_iter}, rec_size: {args.rec_size}")
    print(f"  Step scale: {args.step_scale}, power_iter: {args.power_iter}")
    if args.use_weight:
        print(f"  Weight clip: [{args.weight_clip_min}, {args.weight_clip_max}], "
              f"step mode: {args.weight_step_mode}")
    print(f"  Slices: {args.slice_start} ~ {args.slice_end}")
    if args.experiment_name or args.timestamp:
        print(f"  Run output root: {run_out_dir}")
    print(f"  Tag: {tag}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    proj_geo, vol_geo = setup_geometry(angles, args.rec_size)
    recon = AGDReconstructor(
        proj_geo, vol_geo, args.agd_iter,
        step_scale=args.step_scale,
        power_iter=args.power_iter,
        weight_step_mode=args.weight_step_mode,
        weight_clip_max=args.weight_clip_max)
    print(f"Estimated L = {recon.lipschitz:.6e}")
    print(f"Base step = {recon.base_step:.6e}")
    if args.use_weight:
        demo_step = recon._effective_step(np.ones((n_angles, N_DET_BINNED), dtype=np.float32))
        print(f"Weighted effective step example = {demo_step:.6e}\n")
    else:
        print()

    rows = []
    slices = list(range(args.slice_start, args.slice_end + 1))
    n_total = len(slices)
    t_start = time.time()

    for i, sid in enumerate(slices):
        t0 = time.time()
        try:
            if args.use_weight:
                sino_full, weights_full = load_sinogram(
                    args.data_dir, sid,
                    input_mode=args.input_mode,
                    use_bh=use_bh,
                    input_variant=args.input_variant,
                    return_weights=True,
                    weight_clip=(args.weight_clip_min, args.weight_clip_max),
                    renormalize_after_clip=not args.no_weight_renorm_after_clip)
            else:
                sino_full = load_sinogram(
                    args.data_dir, sid,
                    input_mode=args.input_mode,
                    use_bh=use_bh,
                    input_variant=args.input_variant)
                weights_full = None
        except Exception as e:
            print(f"  [SKIP] slice {sid}: {e}")
            continue

        gt = load_gt(args.gt_dir, sid)
        if gt is None:
            print(f"  [SKIP] slice {sid}: no GT")
            continue

        sino_corr = apply_correction(sino_full, coeffs) if use_bh else sino_full
        sino_ds = sino_corr[ang_idx, :]
        weights_ds = weights_full[ang_idx, :] if weights_full is not None else None

        rec = recon.reconstruct(sino_ds, weights=weights_ds,
                                lambda_tv=args.lambda_tv,
                                tv_inner=args.tv_inner)
        rec_crop = center_crop(rec, gt.shape[0], gt.shape[1])
        m = compute_metrics(rec_crop, gt)

        slice_out = os.path.join(out_dir, f"slice{sid:05d}", layout['tag'])
        os.makedirs(slice_out, exist_ok=True)
        np.save(os.path.join(slice_out, f"slice{sid:05d}.npy"), rec_crop)

        dt = time.time() - t0
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (n_total - i - 1)

        w_info = ""
        if weights_ds is not None:
            eff_step = recon._effective_step(weights_ds)
            w_info = (f" | w=[{weights_ds.min():.3f},{weights_ds.max():.3f}]"
                      f" mean={weights_ds.mean():.3f} step={eff_step:.2e}")

        print(f"  [{i+1:4d}/{n_total}] slice {sid:05d} | "
              f"PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.4f}{w_info} | "
              f"{dt:.1f}s  ETA {eta/60:.0f}min")

        rows.append({"slice": sid, "method": method_name,
                      "PSNR": m['PSNR'], "SSIM": m['SSIM'],
                      "RMSE": m['RMSE'], "MAE": m['MAE']})

    if rows:
        rows.append({"slice": "AVG", "method": rows[0]["method"],
                      "PSNR": np.mean([r["PSNR"] for r in rows]),
                      "SSIM": np.mean([r["SSIM"] for r in rows]),
                      "RMSE": np.mean([r["RMSE"] for r in rows]),
                      "MAE": np.mean([r["MAE"] for r in rows])})

    csv_path = os.path.join(run_out_dir, f"{tag}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["slice", "method",
                                          "PSNR", "SSIM", "RMSE", "MAE"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n结果: {csv_path}")
    avg = [r for r in rows if r["slice"] == "AVG"]
    if avg:
        r = avg[0]
        print(f"  AVG  PSNR={r['PSNR']:.4f}  SSIM={r['SSIM']:.4f}")

    recon.cleanup()
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
网格搜索法选择 λ：AGD+TV 在 sparse-view 下的正则化参数调优。

在训练 slice 上对一组候选 λ 分别跑 AGD+TV 重建，用 mode2 GT 的 PSNR 选最优 λ，
然后在验证 slice 上评估 plain AGD 和 AGD+TV(λ*) 的对比效果。

用法:
  python lambda_grid_search.py \
      --data_dir ... --gt_dir ... --out_dir ... \
      --train_slices 1 500 1000 1500 2000 2500 3000 3500 4000 4500 \
      --val_slices 250 750 1250 1750 2250 2750 3250 3750 4250 4750 \
      --downsample 6 --agd_iter 100 --rec_size 2048
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === Fix Python 3.11 + ASTRA compatibility ===
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return inspect.ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

import astra

# ── 2DeteCT constants ──
DET_PIX = 0.0748
SOD = 431.019989
SDD = 529.000488
CORR = np.array([1.00, 0.0])
N_DET_BINNED = 956
ASTRA_MAX_ANGLES = 2560


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def load_sinogram(data_dir, slice_idx, mode):
    slice_dir = os.path.join(data_dir, f"slice{slice_idx:05d}", f"mode{mode}")
    dark = imageio.imread(os.path.join(slice_dir, 'dark.tif')).astype('float32')
    flat1 = imageio.imread(os.path.join(slice_dir, 'flat1.tif')).astype('float32')
    flat2 = imageio.imread(os.path.join(slice_dir, 'flat2.tif')).astype('float32')
    sinogram = imageio.imread(os.path.join(slice_dir, 'sinogram.tif')).astype('float32')
    flat = np.mean(np.array([flat1, flat2]), axis=0)
    sino_binned = sinogram[:, 0::2] + sinogram[:, 1::2]
    dark_binned = dark[0, 0::2] + dark[0, 1::2]
    flat_binned = flat[0, 0::2] + flat[0, 1::2]
    ratio = (sino_binned - dark_binned) / (flat_binned - dark_binned)
    ratio = ratio[:-1, :]
    if slice_idx <= 2830 or (5521 <= slice_idx <= 5870):
        det_shift = CORR[0] * DET_PIX
    else:
        det_shift = CORR[1] * DET_PIX
    det_grid = np.arange(0, N_DET_BINNED) * DET_PIX
    ratio = interp1d(det_grid, ratio, kind='linear',
                     fill_value='extrapolate')(det_grid + det_shift)
    sino_log = -np.log(np.clip(ratio, 1e-6, None))
    return sino_log


def load_sinogram_noisy(data_dir, slice_idx):
    """Load low-dose mode3 noisy data and return log-corrected sinogram."""
    slice_dir = os.path.join(data_dir, f"slice{slice_idx:05d}", "mode3")
    dark = imageio.imread(os.path.join(slice_dir, 'dark.tif')).astype('float32')
    flat1 = imageio.imread(os.path.join(slice_dir, 'flat1_scaled.tif')).astype('float32')
    flat2 = imageio.imread(os.path.join(slice_dir, 'flat2_scaled.tif')).astype('float32')
    sinogram = imageio.imread(os.path.join(slice_dir,
                              'sinogram_with_poisson.tif')).astype('float32')
    flat = np.mean(np.array([flat1, flat2]), axis=0)
    sino_binned = sinogram[:, 0::2] + sinogram[:, 1::2]
    dark_binned = dark[0, 0::2] + dark[0, 1::2]
    flat_binned = flat[0, 0::2] + flat[0, 1::2]
    ratio = (sino_binned - dark_binned) / (flat_binned - dark_binned)
    ratio = ratio[:-1, :]
    if slice_idx <= 2830 or (5521 <= slice_idx <= 5870):
        det_shift = CORR[0] * DET_PIX
    else:
        det_shift = CORR[1] * DET_PIX
    det_grid = np.arange(0, N_DET_BINNED) * DET_PIX
    ratio = interp1d(det_grid, ratio, kind='linear',
                     fill_value='extrapolate')(det_grid + det_shift)
    sino_log = -np.log(np.clip(ratio, 1e-6, None))
    return sino_log


def apply_correction(sino_log, coeffs):
    result = np.zeros_like(sino_log)
    for k in range(len(coeffs)):
        result += coeffs[k] * (sino_log ** (k + 1))
    return result


def load_experiment_sinogram(data_dir, slice_idx, mode, compound_noisy, coeffs):
    if compound_noisy:
        sino_full = load_sinogram_noisy(data_dir, slice_idx)
    else:
        sino_full = load_sinogram(data_dir, slice_idx, mode=mode)
    if coeffs is not None:
        sino_full = apply_correction(sino_full, coeffs)
    return sino_full


def sanitize_run_part(value):
    safe = []
    for ch in str(value).strip():
        if ch.isalnum() or ch in ("_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def prepare_output_dir(base_out_dir, experiment_name="", timestamp=""):
    if timestamp.lower() == "auto":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    if experiment_name and not timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    parts = []
    if experiment_name:
        parts.append(sanitize_run_part(experiment_name))
    if timestamp:
        parts.append(sanitize_run_part(timestamp))

    out_dir = os.path.join(base_out_dir, "__".join(parts)) if parts else base_out_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_gt(gt_dir, slice_idx, mode=2):
    path = os.path.join(gt_dir, f"slice{slice_idx:05d}", f"mode{mode}",
                        "reconstruction.tif")
    if not os.path.exists(path):
        return None
    return imageio.imread(path).astype(np.float32)


def downsample_sinogram(sino_full, n_full, factor):
    """均匀下采样角度。3600 角 downsample=6 → 600 角。"""
    idx = np.arange(0, n_full, factor)
    return sino_full[idx, :], idx


def cap_and_downsample(n_full, downsample_factor, max_angles=ASTRA_MAX_ANGLES):
    """先下采样，再 cap 到 ASTRA 限制。返回从 n_full 中选取的最终索引。"""
    ds_idx = np.arange(0, n_full, downsample_factor)
    n_ds = len(ds_idx)
    if n_ds > max_angles:
        sub = np.round(np.linspace(0, n_ds - 1, max_angles)).astype(int)
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


# ═══════════════════════════════════════════════════════════
# TV 近端算子 (Chambolle projection)
# ═══════════════════════════════════════════════════════════

def prox_tv(x, tau, n_inner=50):
    """Isotropic TV proximal operator via Chambolle's projection algorithm.
    Solves: min_z  0.5*||z - x||^2 + tau * TV(z)
    """
    ny, nx = x.shape
    p = np.zeros((2, ny, nx), dtype=np.float32)

    for _ in range(n_inner):
        # div(p)
        div = np.zeros_like(x)
        div[:-1, :] += p[0, :-1, :]
        div[1:, :] -= p[0, :-1, :]
        div[:, :-1] += p[1, :, :-1]
        div[:, 1:] -= p[1, :, :-1]

        # gradient of (x - tau * div(p))
        u = x + tau * div
        g = np.zeros((2, ny, nx), dtype=np.float32)
        g[0, :-1, :] = u[1:, :] - u[:-1, :]
        g[1, :, :-1] = u[:, 1:] - u[:, :-1]

        # update p
        norm_g = np.sqrt(g[0] ** 2 + g[1] ** 2 + 1e-10)
        p = (p + (1.0 / (4 * tau + 1e-10)) * g) / (1 + norm_g / (tau + 1e-10))

    # final div(p)
    div = np.zeros_like(x)
    div[:-1, :] += p[0, :-1, :]
    div[1:, :] -= p[0, :-1, :]
    div[:, :-1] += p[1, :, :-1]
    div[:, 1:] -= p[1, :, :-1]

    return x + tau * div


# ═══════════════════════════════════════════════════════════
# AGD+TV 重建器
# ═══════════════════════════════════════════════════════════

class AGDTVReconstructor:
    """FISTA with TV regularization and non-negativity.
    min ||Ax - b||^2 + lambda_tv * TV(x)  s.t. x >= 0
    """

    def __init__(self, proj_geo, vol_geo, n_iter=100, tv_prox_inner=50):
        self.proj_geo = proj_geo
        self.vol_geo = vol_geo
        self.n_iter = n_iter
        self.tv_prox_inner = tv_prox_inner
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

        self.step = 1.0 / self._estimate_lipschitz()

    def _estimate_lipschitz(self, n_power_iter=8):
        x = np.random.randn(*self.shape).astype(np.float32)
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
            x = ATAx / norm
        astra.data2d.store(self.vol_id, x)
        astra.algorithm.run(self.fp_alg)
        Ax = astra.data2d.get(self.sino_id)
        L = float(np.sum(Ax ** 2) / max(np.sum(x ** 2), 1e-10))
        return L

    def reconstruct(self, sino, lambda_tv=0.0):
        """FISTA + TV prox + non-negativity."""
        x = np.zeros(self.shape, dtype=np.float32)
        y = x.copy()
        t = 1.0

        for _ in range(self.n_iter):
            # gradient of ||Ax - b||^2
            astra.data2d.store(self.vol_id, y)
            astra.algorithm.run(self.fp_alg)
            Ay = astra.data2d.get(self.sino_id)
            astra.data2d.store(self.sino_id, Ay - sino)
            astra.data2d.store(self.grad_id,
                               np.zeros(self.shape, dtype=np.float32))
            astra.algorithm.run(self.bp_alg)
            grad = astra.data2d.get(self.grad_id)

            # gradient step
            z = y - self.step * grad

            # TV proximal operator (if lambda_tv > 0)
            if lambda_tv > 0:
                z = prox_tv(z, self.step * lambda_tv, self.tv_prox_inner)

            # non-negativity
            x_new = np.maximum(z, 0)

            # Nesterov momentum
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y = x_new + (t - 1) / t_new * (x_new - x)
            x = x_new
            t = t_new

        return x

    def compute_data_fidelity(self, x, sino):
        """||Ax - b||^2"""
        astra.data2d.store(self.vol_id, x)
        astra.algorithm.run(self.fp_alg)
        Ax = astra.data2d.get(self.sino_id)
        return float(np.sum((Ax - sino) ** 2))

    def cleanup(self):
        astra.algorithm.delete(self.fp_alg)
        astra.algorithm.delete(self.bp_alg)
        astra.data2d.delete(self.vol_id)
        astra.data2d.delete(self.sino_id)
        astra.data2d.delete(self.grad_id)
        astra.projector.delete(self.proj_id)


def tv_norm(x):
    """Isotropic TV norm."""
    dy = x[1:, :] - x[:-1, :]
    dx = x[:, 1:] - x[:, :-1]
    # pad to same shape
    dy_pad = np.zeros_like(x); dy_pad[:-1, :] = dy
    dx_pad = np.zeros_like(x); dx_pad[:, :-1] = dx
    return float(np.sum(np.sqrt(dy_pad ** 2 + dx_pad ** 2 + 1e-10)))


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Grid search for lambda (AGD+TV)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", type=int, default=2,
                   help="sinogram mode (default: 2, no BH)")
    p.add_argument("--compound_noisy", action="store_true",
                   help="read low-dose noisy mode3 files with scaled flats")
    p.add_argument("--bh_coeffs", default=None, help="BH correction coefficients .npy")
    p.add_argument("--train_slices", type=int, nargs='+', required=True)
    p.add_argument("--val_slices", type=int, nargs='+', required=True)
    p.add_argument("--downsample", type=int, default=6,
                   help="角度下采样倍率 (6 = 3600→600)")
    p.add_argument("--lambdas", type=float, nargs='+',
                   default=[0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
                   help="候选 λ 值列表")
    p.add_argument("--agd_iter", type=int, default=100)
    p.add_argument("--tv_inner", type=int, default=50)
    p.add_argument("--rec_size", type=int, default=2048)
    p.add_argument("--experiment_name", default="",
                   help="optional experiment name added as an output subdirectory")
    p.add_argument("--timestamp", default="",
                   help="optional timestamp added as an output subdirectory; use 'auto' to generate")
    args = p.parse_args()

    args.out_dir = prepare_output_dir(
        args.out_dir, args.experiment_name, args.timestamp)

    # ── BH 系数 ──
    coeffs = np.load(args.bh_coeffs) if args.bh_coeffs else None
    data_mode = "compound noisy mode3" if args.compound_noisy else f"mode{args.mode}"
    bh_status = f"BH coeffs: {args.bh_coeffs}" if coeffs is not None else "BH coeffs: none"
    print(f"Data mode: {data_mode}")
    print(bh_status)
    print(f"Output dir: {args.out_dir}")

    # ── 角度下采样 + cap ──
    n_full = 3600
    all_angles_full = np.linspace(0, 2 * np.pi, n_full, endpoint=False)
    ang_idx = cap_and_downsample(n_full, args.downsample)
    n_angles = len(ang_idx)
    angles = all_angles_full[ang_idx]
    print(f"角度: {n_full} → downsample ×{args.downsample} → {n_full // args.downsample}"
          f" → cap → {n_angles}")

    # ── Geometry + Reconstructor ──
    proj_geo, vol_geo = setup_geometry(angles, args.rec_size)
    recon = AGDTVReconstructor(proj_geo, vol_geo, args.agd_iter, args.tv_inner)
    print(f"Step size = {recon.step:.6e}")

    # ═══════════════════════════════════════════════════════
    # Phase 1: 在训练 slice 上搜索最优 λ
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 1: Grid search on training slices")
    print("=" * 60)

    lambda_scores = {lam: [] for lam in args.lambdas}
    lambda_scores[0.0] = []  # plain AGD baseline

    all_lambdas = [0.0] + list(args.lambdas)

    for sid in args.train_slices:
        sino_full = load_experiment_sinogram(
            args.data_dir, sid, args.mode, args.compound_noisy, coeffs)
        sino = sino_full[ang_idx, :]

        gt = load_gt(args.gt_dir, sid)
        if gt is None:
            print(f"  [SKIP] slice {sid}: no GT")
            continue

        for lam in all_lambdas:
            t0 = time.time()
            rec = recon.reconstruct(sino, lambda_tv=lam)
            rec_crop = center_crop(rec, gt.shape[0], gt.shape[1])
            m = compute_metrics(rec_crop, gt)
            dt = time.time() - t0
            lambda_scores[lam].append(m['PSNR'])
            tag = "plain AGD" if lam == 0.0 else f"λ={lam}"
            print(f"  slice {sid:05d} | {tag:15s} | "
                  f"PSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.4f}  ({dt:.1f}s)")

    # ── 汇总训练结果 ──
    print("\n--- Training summary ---")
    print(f"{'λ':>12s}  {'Avg PSNR':>10s}")
    best_lam = 0.0
    best_psnr = -999
    for lam in all_lambdas:
        scores = lambda_scores[lam]
        if len(scores) == 0:
            continue
        avg = np.mean(scores)
        print(f"  {lam:12.6f}  {avg:10.4f}")
        if avg > best_psnr:
            best_psnr = avg
            best_lam = lam

    print(f"\n>>> Best λ = {best_lam}  (avg PSNR = {best_psnr:.4f})")

    # 保存训练结果
    train_csv = os.path.join(args.out_dir, "grid_search_train.csv")
    with open(train_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["lambda", "avg_psnr", "per_slice_psnr"])
        for lam in all_lambdas:
            scores = lambda_scores[lam]
            if len(scores) == 0:
                continue
            w.writerow([lam, np.mean(scores),
                        ";".join(f"{s:.4f}" for s in scores)])
    print(f"训练结果: {train_csv}")

    # ═══════════════════════════════════════════════════════
    # Phase 2: 在验证 slice 上评估 plain AGD vs AGD+TV(λ*)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Phase 2: Validation — plain AGD vs AGD+TV(λ={best_lam})")
    print("=" * 60)

    val_rows = []
    for sid in args.val_slices:
        sino_full = load_experiment_sinogram(
            args.data_dir, sid, args.mode, args.compound_noisy, coeffs)
        sino = sino_full[ang_idx, :]

        gt = load_gt(args.gt_dir, sid)
        if gt is None:
            print(f"  [SKIP] slice {sid}: no GT")
            continue

        # plain AGD (λ=0)
        t0 = time.time()
        rec_plain = recon.reconstruct(sino, lambda_tv=0.0)
        rec_plain_crop = center_crop(rec_plain, gt.shape[0], gt.shape[1])
        m_plain = compute_metrics(rec_plain_crop, gt)
        dt_plain = time.time() - t0

        # AGD + TV (λ*)
        t0 = time.time()
        rec_tv = recon.reconstruct(sino, lambda_tv=best_lam)
        rec_tv_crop = center_crop(rec_tv, gt.shape[0], gt.shape[1])
        m_tv = compute_metrics(rec_tv_crop, gt)
        dt_tv = time.time() - t0

        print(f"  slice {sid:05d} | plain: PSNR={m_plain['PSNR']:.2f} SSIM={m_plain['SSIM']:.4f} ({dt_plain:.1f}s)"
              f" | TV(λ={best_lam}): PSNR={m_tv['PSNR']:.2f} SSIM={m_tv['SSIM']:.4f} ({dt_tv:.1f}s)")

        val_rows.append({
            "slice": sid,
            "method": "plain_AGD",
            "lambda": 0.0,
            "PSNR": m_plain['PSNR'],
            "SSIM": m_plain['SSIM'],
            "RMSE": m_plain['RMSE'],
            "MAE": m_plain['MAE'],
        })
        val_rows.append({
            "slice": sid,
            "method": f"AGD_TV",
            "lambda": best_lam,
            "PSNR": m_tv['PSNR'],
            "SSIM": m_tv['SSIM'],
            "RMSE": m_tv['RMSE'],
            "MAE": m_tv['MAE'],
        })

    # ── 计算平均值 ──
    for method_key in ["plain_AGD", "AGD_TV"]:
        rows = [r for r in val_rows if r["method"] == method_key]
        if rows:
            val_rows.append({
                "slice": "AVG",
                "method": method_key,
                "lambda": rows[0]["lambda"],
                "PSNR": np.mean([r["PSNR"] for r in rows]),
                "SSIM": np.mean([r["SSIM"] for r in rows]),
                "RMSE": np.mean([r["RMSE"] for r in rows]),
                "MAE": np.mean([r["MAE"] for r in rows]),
            })

    # ── 写 CSV ──
    val_csv = os.path.join(args.out_dir,
                           f"grid_search_val_ds{args.downsample}.csv")
    with open(val_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["slice", "method", "lambda",
                                          "PSNR", "SSIM", "RMSE", "MAE"])
        w.writeheader()
        w.writerows(val_rows)

    print(f"\n验证结果: {val_csv}")

    # ── 打印汇总 ──
    print("\n--- Validation summary ---")
    for method_key in ["plain_AGD", "AGD_TV"]:
        avg_row = [r for r in val_rows if r["slice"] == "AVG" and r["method"] == method_key]
        if avg_row:
            r = avg_row[0]
            print(f"  {method_key:12s}  λ={r['lambda']:<10}  "
                  f"PSNR={r['PSNR']:.4f}  SSIM={r['SSIM']:.4f}")

    recon.cleanup()
    print("\nDone.")


if __name__ == "__main__":
    main()

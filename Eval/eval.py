#!/usr/bin/env python3
"""
Simplified CT Reconstruction Quality Evaluation with File Export and Progress Bar.
"""

import argparse
import os
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imageio.v2 as imageio
from tqdm import tqdm  # 引入进度条库

def compute_psnr(gt, rec, data_range):
    return peak_signal_noise_ratio(gt, rec, data_range=data_range)

def compute_ssim(gt, rec, data_range):
    return structural_similarity(gt, rec, data_range=data_range)

def compute_rmse(gt, rec):
    return float(np.sqrt(np.mean((gt - rec) ** 2)))

def normalize_for_metrics(gt, rec):
    vmin = gt.min()
    vmax = gt.max()
    if vmax - vmin == 0:
        return gt, rec, 1.0
    gt_norm = (gt - vmin) / (vmax - vmin)
    rec_norm = (rec - vmin) / (vmax - vmin)
    rec_norm = np.clip(rec_norm, 0, 1)
    return gt_norm, rec_norm, 1.0

def evaluate_files(results_dir, gt_dir, save_path=None):
    pred_files = sorted(glob.glob(os.path.join(results_dir, "*.npy")))
    
    psnr_list, ssim_list, rmse_list = [], [], []

    if not pred_files:
        print("未找到任何 .npy 预测文件，请检查 results_dir 路径。")
        return
    
    # 用 tqdm 包裹 pred_files，添加 desc 参数作为进度条前的文字提示
    for pred_path in tqdm(pred_files, desc="Evaluating", unit="file"):
        filename = os.path.basename(pred_path).replace('.npy', '.tif')
        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            continue

        rec = np.load(pred_path).astype(np.float32)
        gt = imageio.imread(gt_path).astype(np.float32)
        
        # 尺寸对齐 (Center Crop)
        if gt.shape != rec.shape:
            h_min, w_min = min(gt.shape[0], rec.shape[0]), min(gt.shape[1], rec.shape[1])
            y0_gt, x0_gt = (gt.shape[0] - h_min) // 2, (gt.shape[1] - w_min) // 2
            y0_rec, x0_rec = (rec.shape[0] - h_min) // 2, (rec.shape[1] - w_min) // 2
            gt = gt[y0_gt:y0_gt+h_min, x0_gt:x0_gt+w_min]
            rec = rec[y0_rec:y0_rec+h_min, x0_rec:x0_rec+w_min]

        gt_norm, rec_norm, data_range = normalize_for_metrics(gt, rec)

        psnr_list.append(compute_psnr(gt_norm, rec_norm, data_range))
        ssim_list.append(compute_ssim(gt_norm, rec_norm, data_range))
        rmse_list.append(compute_rmse(gt_norm, rec_norm))

    if not psnr_list:
        print("未找到配对的 GT 文件，请检查 gt_dir 路径。")
        return

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    mean_rmse = np.mean(rmse_list)

    # 打印到屏幕
    result_str = (
        f"\n--- 评估结果 (均值) ---\n"
        f"PSNR: {mean_psnr:.4f}\n"
        f"SSIM: {mean_ssim:.4f}\n"
        f"RMSE: {mean_rmse:.4f}\n"
    )
    print(result_str)

    # 保存到文件
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(result_str)
        print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="预测结果(.npy)所在目录")
    parser.add_argument("--gt_dir", type=str, required=True, help="真值图(.tif)所在目录")
    parser.add_argument("--save_path", type=str, default=None, help="结果保存的txt文件路径")
    args = parser.parse_args()

    evaluate_files(args.results_dir, args.gt_dir, args.save_path)


# python /ibex/user/liuj0s/CS_300/DBF-UNet/eval.py \
#     --results_dir /ibex/user/liuj0s/CS_300/cache/data/3p-60x-AGD \
#     --gt_dir /ibex/user/liuj0s/CS_300/cache/data/260424-3p-6x-1024/Test/gt \
#     --save_path /ibex/user/liuj0s/CS_300/cache/data/3p-60x-AGD/metrics.txt

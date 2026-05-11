import argparse
import os
import glob
import numpy as np
from skimage.metrics import structural_similarity
import imageio.v2 as imageio
from tqdm import tqdm

def create_circular_mask_auto(img):
    """根据图像内容自动检测实际FOV的圆形Mask"""
    h, w = img.shape
    
    # 提取图像四个角的像素值，取均值或中位数作为死黑背景的参考值
    bg_val = np.median([img[0,0], img[0,w-1], img[h-1,0], img[h-1,w-1]])
    
    # 设定一个微小的阈值来剔除背景 (考虑到浮点数精度或极微小的噪声)
    threshold = bg_val + 1e-4 if img.dtype in [np.float32, np.float64] else bg_val + 1
    
    # 找到所有大于背景阈值的“有效像素”的坐标
    valid_coords = np.argwhere(img > threshold)
    
    if len(valid_coords) == 0:
        # 兜底：如果图全黑，退回默认逻辑
        center_x, center_y = w // 2, h // 2
        radius = min(w // 2, h // 2) - 1
    else:
        # 计算有效像素的上下左右边界
        y_min, x_min = valid_coords.min(axis=0)
        y_max, x_max = valid_coords.max(axis=0)
        
        # 真实圆心等于边界的中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # 真实半径等于边界宽高跨度的最小值的一半
        # 往内收缩 2-3 个像素，确保不会因为插值伪影吃到边界外的黑边
        radius = min(x_max - x_min, y_max - y_min) / 2.0 - 3

    # 生成 Mask
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius
    
    return mask

def compute_masked_metrics(gt, rec, mask, data_range):
    """计算基于 Mask 区域的 PSNR, SSIM, RMSE"""
    valid_pixels = np.sum(mask)
    if valid_pixels == 0:
        return 0.0, 0.0, 0.0

    # 1. 计算 Mask 区域的 MSE 与 RMSE
    mse = np.sum(((gt - rec) * mask) ** 2) / valid_pixels
    rmse = float(np.sqrt(mse))

    # 2. 计算 Mask 区域的 PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = float(10 * np.log10((data_range ** 2) / mse))

    # 3. 计算 Mask 区域的 SSIM
    _, ssim_map = structural_similarity(gt, rec, data_range=data_range, full=True)
    ssim = float(np.sum(ssim_map * mask) / valid_pixels)

    return psnr, ssim, rmse

def normalize_for_metrics(gt, rec):
    vmin = gt.min()
    vmax = gt.max()
    if vmax - vmin == 0:
        return gt, rec, 1.0
    gt_norm = (gt - vmin) / (vmax - vmin)
    rec_norm = (rec - vmin) / (vmax - vmin)
    rec_norm = np.clip(rec_norm, 0, 1)
    return gt_norm, rec_norm, 1.0

def evaluate_files(results_dir, gt_dir, save_path=None, debug=False):
    # 支持多种预测文件格式
    pred_files = []
    for ext in ['*.npy', '*.tif', '*.tiff']:
        pred_files.extend(glob.glob(os.path.join(results_dir, ext)))
    pred_files = sorted(pred_files)
    
    # 分别建立全图、前景和背景的列表
    full_psnr_list, full_ssim_list, full_rmse_list = [], [], []
    fg_psnr_list, fg_ssim_list, fg_rmse_list = [], [], []
    bg_psnr_list, bg_ssim_list, bg_rmse_list = [], [], []

    if not pred_files:
        print("未找到任何 .npy 或 .tif 预测文件，请检查 results_dir 路径。")
        return
    
    is_first_image = True  # 用于标记是否是第一张图

    for pred_path in tqdm(pred_files, desc="Evaluating", unit="file"):
        filename = os.path.basename(pred_path)
        base_name, ext = os.path.splitext(filename)
        
        # 假设 GT 文件夹中始终是 .tif 格式
        gt_path = os.path.join(gt_dir, base_name + '.tif')

        if not os.path.exists(gt_path):
            continue

        # 根据预测图的文件后缀选择不同的读取方式
        if ext.lower() == '.npy':
            rec = np.load(pred_path).astype(np.float32)
        elif ext.lower() in ['.tif', '.tiff']:
            rec = imageio.imread(pred_path).astype(np.float32)
        else:
            continue

        gt = imageio.imread(gt_path).astype(np.float32)
        
        # 尺寸对齐 (Center Crop)
        if gt.shape != rec.shape:
            h_min, w_min = min(gt.shape[0], rec.shape[0]), min(gt.shape[1], rec.shape[1])
            y0_gt, x0_gt = (gt.shape[0] - h_min) // 2, (gt.shape[1] - w_min) // 2
            y0_rec, x0_rec = (rec.shape[0] - h_min) // 2, (rec.shape[1] - w_min) // 2
            gt = gt[y0_gt:y0_gt+h_min, x0_gt:x0_gt+w_min]
            rec = rec[y0_rec:y0_rec+h_min, x0_rec:x0_rec+w_min]

        # 归一化
        gt_norm, rec_norm, data_range = normalize_for_metrics(gt, rec)

        # 创建 Mask
        h, w = gt_norm.shape
        full_mask = np.ones_like(gt_norm, dtype=bool) # 全图 Mask (全为 True)
        fg_mask = create_circular_mask_auto(gt_norm)  # 前景 Mask
        bg_mask = ~fg_mask                            # 取反得到背景 Mask

        # ================= Debug 可视化模块 =================
        if debug and is_first_image:
            debug_out_dir = './cache/result/eval_debug'
            os.makedirs(debug_out_dir, exist_ok=True)
            
            # 将归一化后的预测图转为 0-255 的 8-bit 图像
            vis_img = (rec_norm * 255).astype(np.uint8)
            # 扩展为 RGB 三通道图像，并转为 float32 以便进行通道计算
            vis_rgb = np.stack([vis_img, vis_img, vis_img], axis=-1).astype(np.float32)
            
            # 配置半透明颜色 Mask (Alpha Blending)
            color_mask = np.array([255, 0, 0], dtype=np.float32) # RGB: 纯红
            alpha = 0.3 # 透明度 30%
            
            # 将半透明红色叠加到前景区域 
            vis_rgb[fg_mask] = vis_rgb[fg_mask] * (1 - alpha) + color_mask * alpha
            
            # 限制数值范围并转回图片格式
            vis_rgb = np.clip(vis_rgb, 0, 255).astype(np.uint8)

            debug_save_path = os.path.join(debug_out_dir, f"{base_name}_debug_mask.png")
            imageio.imwrite(debug_save_path, vis_rgb)
            print(f"\n\033[92m[Debug] 前景半透明红色 Mask 已高亮并保存至: {debug_save_path}\033[0m")
            
            is_first_image = False # 确保只执行一次
        # ====================================================

        # 计算全图指标
        full_psnr, full_ssim, full_rmse = compute_masked_metrics(gt_norm, rec_norm, full_mask, data_range)
        full_psnr_list.append(full_psnr)
        full_ssim_list.append(full_ssim)
        full_rmse_list.append(full_rmse)

        # 计算前景指标
        fg_psnr, fg_ssim, fg_rmse = compute_masked_metrics(gt_norm, rec_norm, fg_mask, data_range)
        fg_psnr_list.append(fg_psnr)
        fg_ssim_list.append(fg_ssim)
        fg_rmse_list.append(fg_rmse)

        # 计算背景指标
        bg_psnr, bg_ssim, bg_rmse = compute_masked_metrics(gt_norm, rec_norm, bg_mask, data_range)
        bg_psnr_list.append(bg_psnr)
        bg_ssim_list.append(bg_ssim)
        bg_rmse_list.append(bg_rmse)

    if not fg_psnr_list:
        print("未找到配对的 GT 文件，请检查 gt_dir 路径。")
        return

    # 打印到屏幕
    result_str = (
        f"\n=====================================\n"
        f"        评估结果 (均值)              \n"
        f"=====================================\n"
        f"[Full Image 全图]\n"
        f"  PSNR: {np.mean(full_psnr_list):.4f}\n"
        f"  SSIM: {np.mean(full_ssim_list):.4f}\n"
        f"  RMSE: {np.mean(full_rmse_list):.4f}\n\n"
        f"[Foreground 前景 (FOV圆内)]\n"
        f"  PSNR: {np.mean(fg_psnr_list):.4f}\n"
        f"  SSIM: {np.mean(fg_ssim_list):.4f}\n"
        f"  RMSE: {np.mean(fg_rmse_list):.4f}\n\n"
        f"[Background 背景 (FOV圆外)]\n"
        f"  PSNR: {np.mean(bg_psnr_list):.4f}\n"
        f"  SSIM: {np.mean(bg_ssim_list):.4f}\n"
        f"  RMSE: {np.mean(bg_rmse_list):.4f}\n"
        f"=====================================\n"
    )
    print(result_str)

    # 保存到文件
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(result_str)
        print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="预测结果(.npy 或 .tif)所在目录")
    parser.add_argument("--gt_dir", type=str, required=True, help="真值图(.tif)所在目录")
    parser.add_argument("--save_path", type=str, default=None, help="结果保存的txt文件路径")
    parser.add_argument("--debug", action="store_true", help="启用后，将第一张图的前景叠加红色半透明Mask保存到 ./cache/result/eval_debug")
    args = parser.parse_args()

    evaluate_files(args.results_dir, args.gt_dir, args.save_path, args.debug)




'''
python /ibex/user/liuj0s/CS_300/Eval/eval_bgfg.py \
    --results_dir /ibex/user/liuj0s/CS_300/cache/data/NeRF \
    --gt_dir /ibex/user/liuj0s/CS_300/cache/data/260424-3p-6x-1024/Train/gt \
    --save_path /ibex/user/liuj0s/CS_300/cache/data/NeRF/metrics.txt \
    --debug


python /ibex/user/liuj0s/CS_300/Eval/eval_bgfg.py \
    --results_dir /ibex/user/liuj0s/CS_300/cache/result/exp3_3p6x_AGD_1048_channel32/test \
    --gt_dir /ibex/user/liuj0s/CS_300/cache/data/260424-3p-6x-1024/Test/gt \
    --save_path /ibex/user/liuj0s/CS_300/cache/result/exp3_3p6x_AGD_1048_channel32/metrics.txt
'''

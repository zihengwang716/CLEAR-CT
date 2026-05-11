import os
os.environ['TORCH_HOME'] = '/ibex/user/liuj0s/CS_300/cache/checkpoint'

import argparse
import glob
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt

# 深度学习相关的包
import torch
import lpips

# 引入 sewar 库计算 VIF
from sewar.full_ref import vifp

def create_circular_mask_auto(img):
    """根据图像内容自动检测实际FOV的圆形Mask"""
    h, w = img.shape
    bg_val = np.median([img[0,0], img[0,w-1], img[h-1,0], img[h-1,w-1]])
    threshold = bg_val + 1e-4 if img.dtype in [np.float32, np.float64] else bg_val + 1
    valid_coords = np.argwhere(img > threshold)
    
    if len(valid_coords) == 0:
        center_x, center_y = w // 2, h // 2
        radius = min(w // 2, h // 2) - 1
    else:
        y_min, x_min = valid_coords.min(axis=0)
        y_max, x_max = valid_coords.max(axis=0)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        radius = min(x_max - x_min, y_max - y_min) / 2.0 - 3

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    return dist_from_center <= radius

def normalize_for_metrics(gt, rec):
    """使用 0.1% 和 99.9% 分位数归一化，排除极亮噪点导致的归一化陷阱"""
    vmin = np.percentile(gt, 0.1)
    vmax = np.percentile(gt, 99.9)
    if vmax - vmin == 0:
        return gt, rec
    gt_norm = (gt - vmin) / (vmax - vmin)
    rec_norm = (rec - vmin) / (vmax - vmin)
    return np.clip(gt_norm, 0, 1), np.clip(rec_norm, 0, 1)

def compute_gme(gt, rec, mask):
    """计算 Gradient Magnitude Error (梯度幅值误差)"""
    gt_gx, gt_gy = ndimage.sobel(gt, axis=0), ndimage.sobel(gt, axis=1)
    rec_gx, rec_gy = ndimage.sobel(rec, axis=0), ndimage.sobel(rec, axis=1)
    
    gt_grad = np.hypot(gt_gx, gt_gy)
    rec_grad = np.hypot(rec_gx, rec_gy)
    
    valid_pixels = np.sum(mask)
    if valid_pixels == 0: return 0.0
    
    mae_grad = np.sum(np.abs(gt_grad - rec_grad) * mask) / valid_pixels
    return float(mae_grad)

class PerceptualEvaluator:
    def __init__(self, device):
        self.device = device
        print("正在加载 LPIPS (VGG) 模型...")
        self.lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device)
        self.lpips_fn.eval()

    def img_to_tensor(self, img_norm):
        t = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 3, 1, 1) 
        t = t * 2.0 - 1.0 
        return t.to(self.device)

    def get_lpips_spatial(self, gt_t, rec_t):
        with torch.no_grad():
            dist_map = self.lpips_fn.forward(gt_t, rec_t)
        return dist_map

def evaluate_files(results_dir, gt_dir, save_path=None, debug=False):
    pred_files = []
    for ext in ['*.npy', '*.tif', '*.tiff']:
        pred_files.extend(glob.glob(os.path.join(results_dir, ext)))
    pred_files = sorted(pred_files)

    if not pred_files:
        print("未找到任何预测文件。")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的计算设备: {device}")
    evaluator = PerceptualEvaluator(device)

    metrics = {
        'full': {'lpips': [], 'gme': [], 'vif': []},
        'fg':   {'lpips': [], 'gme': []},
        'bg':   {'lpips': [], 'gme': []}
    }
    
    gaps_metrics = {'edge_lpips': [], 'flat_mae': [], 'gaps_total': []}

    is_first_image = True

    for pred_path in tqdm(pred_files, desc="Evaluating", unit="file"):
        base_name, ext = os.path.splitext(os.path.basename(pred_path))
        gt_path = os.path.join(gt_dir, base_name + '.tif')
        if not os.path.exists(gt_path): continue

        rec = np.load(pred_path).astype(np.float32) if ext.lower() == '.npy' else imageio.imread(pred_path).astype(np.float32)
        gt = imageio.imread(gt_path).astype(np.float32)
        
        if gt.shape != rec.shape:
            h_min, w_min = min(gt.shape[0], rec.shape[0]), min(gt.shape[1], rec.shape[1])
            y0_gt, x0_gt = (gt.shape[0] - h_min) // 2, (gt.shape[1] - w_min) // 2
            y0_rec, x0_rec = (rec.shape[0] - h_min) // 2, (rec.shape[1] - w_min) // 2
            gt = gt[y0_gt:y0_gt+h_min, x0_gt:x0_gt+w_min]
            rec = rec[y0_rec:y0_rec+h_min, x0_rec:x0_rec+w_min]

        gt_norm, rec_norm = normalize_for_metrics(gt, rec)

        h, w = gt_norm.shape
        full_mask = np.ones_like(gt_norm, dtype=bool)
        fg_mask = create_circular_mask_auto(gt_norm)
        bg_mask = ~fg_mask
        masks = {'full': full_mask, 'fg': fg_mask, 'bg': bg_mask}

        vif_val = vifp(gt_norm, rec_norm)
        metrics['full']['vif'].append(vif_val)

        for key, m in masks.items():
            metrics[key]['gme'].append(compute_gme(gt_norm, rec_norm, m))

        gt_t = evaluator.img_to_tensor(gt_norm)
        rec_t = evaluator.img_to_tensor(rec_norm)
        dist_map_t = evaluator.get_lpips_spatial(gt_t, rec_t)
        dist_map_resized = torch.nn.functional.interpolate(dist_map_t, size=(h, w), mode='bilinear', align_corners=False)
        dist_map_np = dist_map_resized.squeeze().cpu().numpy()

        for key, m in masks.items():
            valid_pixels = np.sum(m)
            lpips_val = float(np.sum(dist_map_np * m) / valid_pixels) if valid_pixels > 0 else 0.0
            metrics[key]['lpips'].append(lpips_val)

        # ==========================================================
        # ====== Soft GAPS (引入 Sigmoid 的连续可导解耦评估) =======
        # ==========================================================
        gt_gx, gt_gy = ndimage.sobel(gt_norm, axis=0), ndimage.sobel(gt_norm, axis=1)
        gt_grad = np.hypot(gt_gx, gt_gy)
        
        grad_vals_in_fg = gt_grad[fg_mask]
        if len(grad_vals_in_fg) > 0:
            edge_thresh = np.percentile(grad_vals_in_fg, 85) 
        else:
            edge_thresh = 0.1

        # Sigmoid 缩放系数 k。控制过渡带的陡峭程度。
        # k 越大越接近硬切分，k 越小过渡越平缓。可根据梯度数值范围微调。
        k = 15.0 
        
        # 为了防止指数溢出，对输入做截断限制
        z = np.clip(-k * (gt_grad - edge_thresh), -80, 80)
        
        # 计算空间权重图 W (0~1连续值)
        # W 越接近 1，说明该像素越具备高频边缘特征
        # W 越接近 0，说明该像素越倾向于平坦区域
        W = 1.0 / (1.0 + np.exp(z))
        
        # 仅保留有效前景区域的权重
        W_fg = W * fg_mask
        W_inv_fg = (1.0 - W) * fg_mask
        
        # 计算归一化分母
        sum_W = np.sum(W_fg)
        sum_W_inv = np.sum(W_inv_fg)

        # 计算加权平均误差
        edge_lpips = float(np.sum(dist_map_np * W_fg) / sum_W) if sum_W > 0 else 0.0
        flat_mae = float(np.sum(np.abs(gt_norm - rec_norm) * W_inv_fg) / sum_W_inv) if sum_W_inv > 0 else 0.0
        
        # 融合计算 Total GAPS
        gaps_total = edge_lpips + 5.0 * flat_mae
        
        gaps_metrics['edge_lpips'].append(edge_lpips)
        gaps_metrics['flat_mae'].append(flat_mae)
        gaps_metrics['gaps_total'].append(gaps_total)

        # ================= Soft GAPS Debug 可视化模块 =================
        if debug and is_first_image:
            debug_out_dir = './cache/result/eval_debug'
            os.makedirs(debug_out_dir, exist_ok=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 1. 真值图
            axes[0].imshow(gt_norm, cmap='gray')
            axes[0].set_title("1. Ground Truth", fontsize=14)
            axes[0].axis('off')

            # 2. 梯度热力图
            im1 = axes[1].imshow(gt_grad, cmap='magma')
            axes[1].set_title("2. Gradient Magnitude", fontsize=14)
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # 3. Soft GAPS 连续权重叠加图
            vis_overlay = np.stack([gt_norm, gt_norm, gt_norm], axis=-1) * 0.4
            
            # 使用连续权重 W 进行颜色混合
            color_edge = np.array([0.6, 0.6, 0.0]) # 黄色代表倾向 LPIPS
            color_flat = np.array([0.0, 0.2, 0.5]) # 蓝色代表倾向 MAE
            
            # 扩展 W_fg 和 W_inv_fg 到 RGB 三个通道进行广播
            overlay_colors = W_fg[..., None] * color_edge + W_inv_fg[..., None] * color_flat
            vis_overlay += overlay_colors
            vis_overlay = np.clip(vis_overlay, 0, 1)

            axes[2].imshow(vis_overlay)
            axes[2].set_title("3. Soft GAPS Weight Overlay\n(Bright Yellow: ~100% LPIPS | Deep Blue: ~100% MAE)", fontsize=14)
            axes[2].axis('off')

            plt.tight_layout()
            save_fig_path = os.path.join(debug_out_dir, f"{base_name}_Soft_GAPS_Analysis.png")
            plt.savefig(save_fig_path, dpi=150, bbox_inches='tight')
            plt.imsave(os.path.join(debug_out_dir, f"{base_name}_LPIPS_Heatmap.png"), dist_map_np, cmap='jet')
            plt.close()

            print(f"\n\033[92m[Debug] Soft GAPS 连续权重分析图已保存至: {save_fig_path}\033[0m")
            is_first_image = False 
        # ====================================================

    result_str = (
        f"\n=====================================\n"
        f"        多维度感知质量评估结果         \n"
        f"=====================================\n"
        f"[Full Image 全图]\n"
        f"  VIF   (↑): {np.mean(metrics['full']['vif']):.4f}  (越高越好)\n"
        f"  LPIPS (↓): {np.mean(metrics['full']['lpips']):.4f}  (越低越好)\n"
        f"  GME   (↓): {np.mean(metrics['full']['gme']):.4f}  (越低越好)\n\n"
        
        f"-------------------------------------\n"
        f" ★ 全新指标: Soft GAPS (平滑可导评估) \n"
        f"-------------------------------------\n"
        f"  GAPS Total (↓): {np.mean(gaps_metrics['gaps_total']):.4f}\n"
        f"    ├─ Edge LPIPS (惩罚模糊): {np.mean(gaps_metrics['edge_lpips']):.4f}\n"
        f"    └─ Flat MAE   (惩罚噪声): {np.mean(gaps_metrics['flat_mae']):.4f}\n"
        f"=====================================\n"
    )
    print(result_str)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(result_str)
        print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="预测结果所在目录")
    parser.add_argument("--gt_dir", type=str, required=True, help="真值图所在目录")
    parser.add_argument("--save_path", type=str, default=None, help="结果保存路径")
    parser.add_argument("--debug", action="store_true", help="启用后，将保存可视化分析图")
    args = parser.parse_args()

    evaluate_files(args.results_dir, args.gt_dir, args.save_path, args.debug)



'''
python /ibex/user/liuj0s/CS_300/Eval/eval_VIF_GAPS.py \
    --results_dir /ibex/user/liuj0s/CS_300/cache/data/refined_output \
    --gt_dir /ibex/user/liuj0s/CS_300/cache/data/260424-3p-6x-1024/Test/gt \
    --save_path /ibex/user/liuj0s/CS_300/cache/data/refined_output/metrics.txt \
    --debug


python /ibex/user/liuj0s/CS_300/Eval/eval_VIF_GAPS.py \
    --results_dir /ibex/user/liuj0s/CS_300/cache/result/exp3_3p6x_AGD_1048_channel32/test \
    --gt_dir /ibex/user/liuj0s/CS_300/cache/data/260424-3p-6x-1024/Test/gt \
    --save_path /ibex/user/liuj0s/CS_300/cache/result/exp3_3p6x_AGD_1048_channel32/metrics.txt
'''

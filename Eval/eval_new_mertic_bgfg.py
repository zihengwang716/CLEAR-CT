import os
os.environ['TORCH_HOME'] = '/ibex/user/liuj0s/CS_300/cache/checkpoint'

import argparse
import glob
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from scipy import ndimage
import torch
import torch.nn.functional as F
import lpips

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
    # 使用 Sobel 算子提取 X 和 Y 方向的梯度
    gt_gx, gt_gy = ndimage.sobel(gt, axis=0), ndimage.sobel(gt, axis=1)
    rec_gx, rec_gy = ndimage.sobel(rec, axis=0), ndimage.sobel(rec, axis=1)
    
    # 计算梯度幅值
    gt_grad = np.hypot(gt_gx, gt_gy)
    rec_grad = np.hypot(rec_gx, rec_gy)
    
    valid_pixels = np.sum(mask)
    if valid_pixels == 0: return 0.0
    
    # 计算 Mask 内的平均绝对梯度误差 (MAE)
    mae_grad = np.sum(np.abs(gt_grad - rec_grad) * mask) / valid_pixels
    return float(mae_grad)

class PerceptualEvaluator:
    def __init__(self, device):
        self.device = device
        # 加载 LPIPS 模型 (VGG)
        print("正在加载 LPIPS (VGG) 模型...")
        self.lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device)
        self.lpips_fn.eval()

    def img_to_tensor(self, img_norm):
        """将 [0, 1] 的 1通道图像转为 [-1, 1] 的 3通道 Tensor"""
        t = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 3, 1, 1) # 1通道复制为3通道
        t = t * 2.0 - 1.0 # 映射到 [-1, 1]
        return t.to(self.device)

    def get_lpips_spatial(self, gt_t, rec_t):
        """获取空间像素级的 LPIPS 误差图"""
        with torch.no_grad():
            dist_map = self.lpips_fn.forward(gt_t, rec_t) # shape: (1, 1, H', W')
        return dist_map

def evaluate_files(results_dir, gt_dir, save_path=None, debug=False):
    pred_files = []
    for ext in ['*.npy', '*.tif', '*.tiff']:
        pred_files.extend(glob.glob(os.path.join(results_dir, ext)))
    pred_files = sorted(pred_files)

    if not pred_files:
        print("未找到任何预测文件。")
        return

    # 初始化 GPU/CPU 和评估器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的计算设备: {device}")
    evaluator = PerceptualEvaluator(device)

    # 存储指标的列表
    metrics = {
        'full': {'lpips': [], 'gme': []},
        'fg':   {'lpips': [], 'gme': []},
        'bg':   {'lpips': [], 'gme': []}
    }

    is_first_image = True

    for pred_path in tqdm(pred_files, desc="Evaluating", unit="file"):
        base_name, ext = os.path.splitext(os.path.basename(pred_path))
        gt_path = os.path.join(gt_dir, base_name + '.tif')
        if not os.path.exists(gt_path): continue

        # 读取并裁剪对齐
        rec = np.load(pred_path).astype(np.float32) if ext.lower() == '.npy' else imageio.imread(pred_path).astype(np.float32)
        gt = imageio.imread(gt_path).astype(np.float32)
        
        if gt.shape != rec.shape:
            h_min, w_min = min(gt.shape[0], rec.shape[0]), min(gt.shape[1], rec.shape[1])
            y0_gt, x0_gt = (gt.shape[0] - h_min) // 2, (gt.shape[1] - w_min) // 2
            y0_rec, x0_rec = (rec.shape[0] - h_min) // 2, (rec.shape[1] - w_min) // 2
            gt = gt[y0_gt:y0_gt+h_min, x0_gt:x0_gt+w_min]
            rec = rec[y0_rec:y0_rec+h_min, x0_rec:x0_rec+w_min]

        # 归一化
        gt_norm, rec_norm = normalize_for_metrics(gt, rec)

        # 生成 Masks
        h, w = gt_norm.shape
        full_mask = np.ones_like(gt_norm, dtype=bool)
        fg_mask = create_circular_mask_auto(gt_norm)
        bg_mask = ~fg_mask
        
        masks = {'full': full_mask, 'fg': fg_mask, 'bg': bg_mask}

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

        # ====== 1. 计算 Gradient Magnitude Error (GME) ======
        for key, m in masks.items():
            metrics[key]['gme'].append(compute_gme(gt_norm, rec_norm, m))

        # 转为 Tensor
        gt_t = evaluator.img_to_tensor(gt_norm)
        rec_t = evaluator.img_to_tensor(rec_norm)

        # ====== 2. 计算空间 LPIPS ======
        dist_map_t = evaluator.get_lpips_spatial(gt_t, rec_t)
        # 将较小的特征图放大回原图尺寸，以便应用 Mask
        dist_map_resized = F.interpolate(dist_map_t, size=(h, w), mode='bilinear', align_corners=False)
        dist_map_np = dist_map_resized.squeeze().cpu().numpy()

        for key, m in masks.items():
            valid_pixels = np.sum(m)
            lpips_val = float(np.sum(dist_map_np * m) / valid_pixels) if valid_pixels > 0 else 0.0
            metrics[key]['lpips'].append(lpips_val)

    # ====== 打印结果 ======
    result_str = (
        f"\n=====================================\n"
        f"        感知质量评估结果 (均值)        \n"
        f"=====================================\n"
        f"↓ (向下箭头代表该指标越低越好, Lower is better)\n\n"
        f"[Full Image 全图]\n"
        f"  LPIPS (↓): {np.mean(metrics['full']['lpips']):.4f}\n"
        f"  GME   (↓): {np.mean(metrics['full']['gme']):.4f}\n\n"
        f"[Foreground 前景 (FOV圆内)]\n"
        f"  LPIPS (↓): {np.mean(metrics['fg']['lpips']):.4f}\n"
        f"  GME   (↓): {np.mean(metrics['fg']['gme']):.4f}\n\n"
        f"[Background 背景 (FOV圆外)]\n"
        f"  LPIPS (↓): {np.mean(metrics['bg']['lpips']):.4f}\n"
        f"  GME   (↓): {np.mean(metrics['bg']['gme']):.4f}\n"
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
    parser.add_argument("--debug", action="store_true", help="启用后，将第一张图的前景叠加红色半透明Mask保存到 ./cache/result/eval_debug")
    args = parser.parse_args()

    evaluate_files(args.results_dir, args.gt_dir, args.save_path, args.debug)


'''
python /ibex/user/liuj0s/CS_300/Eval/eval_new_mertic_bgfg.py \
    --results_dir /ibex/user/liuj0s/CS_300/cache/data/3p6x-unet \
    --gt_dir /ibex/user/liuj0s/CS_300/cache/data/260424-3p-6x-1024/Test/gt \
    --save_path /ibex/user/liuj0s/CS_300/cache/data/3p6x-unet/metrics.txt \
    --debug


python /ibex/user/liuj0s/CS_300/Eval/eval_new_mertic_bgfg.py \
    --results_dir /ibex/user/liuj0s/CS_300/cache/result/exp3_3p6x_AGD_1048_channel32/test \
    --gt_dir /ibex/user/liuj0s/CS_300/cache/data/260424-3p-6x-1024/Test/gt \
    --save_path /ibex/user/liuj0s/CS_300/cache/result/exp3_3p6x_AGD_1048_channel32/metrics.txt
'''

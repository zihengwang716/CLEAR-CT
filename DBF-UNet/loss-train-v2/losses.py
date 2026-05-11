"""
Composite loss: Soft GAPS (LPIPS + MAE decoupled by gradient) + FFT magnitude L1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


# ============================================================
# Soft GAPS Loss (Continuous Edge/Flat Decoupling)
# ============================================================
class SoftGAPSLoss(nn.Module):
    def __init__(self, device, k=15.0, val_range=1.0):
        super().__init__()
        self.device = device
        self.k = k
        self.val_range = val_range
        
        # 1. 加载并冻结 LPIPS 模型 (VGG)
        # 必须设为 eval 且无需梯度，防止显存爆炸及错误更新
        self.lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(device)
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

        # 2. 定义 PyTorch 版本的 Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # 使用 register_buffer 确保算子自动移至对应 GPU，且不参与梯度更新
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W], assumed in [0, val_range].
        """
        B, C, H, W = target.shape

        # ==========================================
        # A. 在【真值 target】上计算梯度掩码 (无梯度追踪)
        # ==========================================
        with torch.no_grad():
            gt_gx = F.conv2d(target, self.sobel_x, padding=1)
            gt_gy = F.conv2d(target, self.sobel_y, padding=1)
            gt_grad = torch.hypot(gt_gx, gt_gy)

            # 动态计算每张图的 85% 分位数作为边缘阈值
            gt_grad_flat = gt_grad.view(B, -1)
            edge_thresh = torch.quantile(gt_grad_flat, 0.85, dim=1).view(B, 1, 1, 1)

            # 计算 Sigmoid 连续权重 W (趋近 1 为边缘，趋近 0 为平滑)
            z = torch.clamp(-self.k * (gt_grad - edge_thresh), min=-80.0, max=80.0)
            W_weight = torch.sigmoid(-z) 
            W_inv = 1.0 - W_weight

        # ==========================================
        # B. 计算空间 LPIPS (惩罚边缘模糊)
        # ==========================================
        # LPIPS 接受 [-1, 1] 范围的 3 通道 RGB 图像
        pred_scaled = (pred / self.val_range) * 2.0 - 1.0
        target_scaled = (target / self.val_range) * 2.0 - 1.0
        
        # 将单通道扩展为 3 通道
        pred_3c = pred_scaled.repeat(1, 3, 1, 1)
        target_3c = target_scaled.repeat(1, 3, 1, 1)

        dist_map = self.lpips_fn(pred_3c, target_3c)
        # 尺寸对齐
        dist_map = F.interpolate(dist_map, size=(H, W), mode='bilinear', align_corners=False)

        # ==========================================
        # C. 计算 L1/MAE (惩罚平滑区域噪声)
        # ==========================================
        l1_map = torch.abs(pred - target)

        # ==========================================
        # D. 加权融合
        # ==========================================
        sum_W = torch.sum(W_weight) + 1e-8
        sum_W_inv = torch.sum(W_inv) + 1e-8

        edge_lpips = torch.sum(dist_map * W_weight) / sum_W
        flat_mae = torch.sum(l1_map * W_inv) / sum_W_inv

        # 根据 Soft GAPS 公式，平滑区域 MAE 权重设为 5.0
        gaps_total = edge_lpips + 5.0 * flat_mae

        return gaps_total, edge_lpips, flat_mae


# ============================================================
# FFT Loss (magnitude L1 in frequency domain)
# ============================================================
class FFTLoss(nn.Module):
    """
    L1 loss on FFT magnitude. Helps suppress sparse-view streaks
    which have a strong frequency signature.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        X = torch.fft.rfft2(x, norm='ortho')
        Y = torch.fft.rfft2(y, norm='ortho')
        # L1 on real and imaginary parts separately
        return (X.real - Y.real).abs().mean() + (X.imag - Y.imag).abs().mean()


# ============================================================
# Composite Loss
# ============================================================
class CompositeLoss(nn.Module):
    def __init__(self, device, w_gaps=1.0, w_fft=0.05, val_range=1.0):
        super().__init__()
        self.w_gaps = w_gaps
        self.w_fft = w_fft
        self.gaps_loss = SoftGAPSLoss(device, val_range=val_range)
        self.fft_loss = FFTLoss()

    def forward(self, pred, target):
        gaps_total, edge_lpips, flat_mae = self.gaps_loss(pred, target)
        f = self.fft_loss(pred, target)
        
        total = self.w_gaps * gaps_total + self.w_fft * f
        
        return total, {
            "gaps_total": gaps_total.item(), 
            "edge_lpips": edge_lpips.item(),
            "flat_mae": flat_mae.item(),
            "fft": f.item()
        }
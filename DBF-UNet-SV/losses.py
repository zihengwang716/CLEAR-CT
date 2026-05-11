"""
Composite loss: L1 + SSIM + FFT magnitude L1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SSIM Loss (simple window-based SSIM)
# ============================================================
def _gaussian_window(window_size, sigma):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(x, y, window_size=11, sigma=1.5, val_range=1.0):
    """
    x, y: [B, 1, H, W], assumed in [0, val_range].
    Returns scalar mean SSIM in [-1, 1].
    """
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    device = x.device
    g = _gaussian_window(window_size, sigma).to(device)
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]

    mu_x = F.conv2d(x, window, padding=window_size // 2)
    mu_y = F.conv2d(y, window, padding=window_size // 2)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, val_range=1.0):
        super().__init__()
        self.ws = window_size
        self.sigma = sigma
        self.val_range = val_range

    def forward(self, x, y):
        return 1.0 - ssim(x, y, self.ws, self.sigma, self.val_range)


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
        # L1 on real and imaginary parts separately (equivalent to complex L1 up to constant)
        return (X.real - Y.real).abs().mean() + (X.imag - Y.imag).abs().mean()


# ============================================================
# Composite Loss
# ============================================================
class CompositeLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.1, w_fft=0.05, val_range=1.0):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_fft = w_fft
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss(val_range=val_range)
        self.fft_loss = FFTLoss()

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        s = self.ssim_loss(pred, target)
        f = self.fft_loss(pred, target)
        total = self.w_l1 * l1 + self.w_ssim * s + self.w_fft * f
        return total, {"l1": l1.item(), "ssim_loss": s.item(), "fft": f.item()}
"""
Utility functions: metrics, checkpoint, logging, visualization.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def psnr(x, y, val_range=1.0, eps=1e-10):
    """x, y: [B, 1, H, W], expected in [0, val_range]. Returns mean PSNR."""
    mse = F.mse_loss(x, y, reduction='none').mean(dim=(1, 2, 3))
    val = 10.0 * torch.log10((val_range ** 2) / (mse + eps))
    return val.mean().item()


@torch.no_grad()
def ssim_metric(x, y, val_range=1.0):
    from losses import ssim
    return ssim(x, y, val_range=val_range).item()


@torch.no_grad()
def rmse(x, y):
    return torch.sqrt(F.mse_loss(x, y)).item()


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_psnr, config):
    state = {
        "model": model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_psnr": best_psnr,
        "config": config,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    state = torch.load(path, map_location=map_location)
    msg = model.load_state_dict(state["model"], strict=True) if not hasattr(model, 'module') \
        else model.module.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", 0), state.get("best_psnr", 0.0)


# ============================================================
# Visualization (save triplets as PNG for inspection)
# ============================================================
@torch.no_grad()
def save_triplet(x_agd, x_pred, x_gt, out_path, clip=(0, 1)):
    """
    Save (input, prediction, gt) as a single PNG for visual inspection.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def to_np(t):
        return t.detach().cpu().squeeze().numpy()

    a = to_np(x_agd)
    p = to_np(x_pred)
    g = to_np(x_gt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [a, p, g], ["AGD (input)", "Pred", "GT"]):
        im = ax.imshow(img, cmap='gray', vmin=clip[0], vmax=clip[1])
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Misc
# ============================================================
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
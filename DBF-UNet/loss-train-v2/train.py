"""
Usage:
    python /ibex/user/liuj0s/CS_300/DBF-UNet/train.py
    python /ibex/user/liuj0s/CS_300/DBF-UNet/train.py --overfit 5 --config /ibex/user/liuj0s/CS_300/DBF-UNet/config.yaml
    python /ibex/user/liuj0s/CS_300/DBF-UNet/train.py --config /ibex/user/liuj0s/CS_300/DBF-UNet/config.yaml
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from data.dataset import CTRestorationDataset
from losses import CompositeLoss
from models.dbf_unet import DBFUNet
from utils import (
    psnr, ssim_metric, rmse,
    save_checkpoint, load_checkpoint,
    save_triplet, set_seed, count_params, ensure_dir,
)


def get_scheduler(optimizer, epochs, warmup_epochs, lr_min, lr_max):
    """Warmup + cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return (lr_min + (lr_max - lr_min) * cosine) / lr_max
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch, log_every, writer, global_step):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0

    for it, batch in enumerate(loader):
        x_agd = batch["x_agd"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(x_agd)
            loss, loss_dict = loss_fn(pred, x_gt)

        # NaN defense: skip this batch if loss is not finite
        if not torch.isfinite(loss):
            print(f"  [WARN] Non-finite loss at epoch {epoch} iter {it+1}: "
                  f"loss={loss.item()}, skipping batch.")
            if not hasattr(train_one_epoch, '_nan_reported'):
                train_one_epoch._nan_reported = True
                with torch.no_grad():
                    print(f"    x_agd stats: min={x_agd.min():.4f}, max={x_agd.max():.4f}, "
                          f"has_nan={torch.isnan(x_agd).any().item()}")
                    print(f"    x_gt  stats: min={x_gt.min():.4f}, max={x_gt.max():.4f}, "
                          f"has_nan={torch.isnan(x_gt).any().item()}")
                    print(f"    pred  stats: min={pred.min():.4f}, max={pred.max():.4f}, "
                          f"has_nan={torch.isnan(pred).any().item()}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = x_agd.size(0)
        total_loss += loss.item() * bs
        with torch.no_grad():
            p = psnr(pred.clamp(0, 1), x_gt.clamp(0, 1))
        total_psnr += p * bs
        n += bs
        global_step += 1

        if (it + 1) % log_every == 0:
            elapsed = time.time() - t0
            ips = n / elapsed
            print(f"  [epoch {epoch} iter {it+1}/{len(loader)}] "
                  f"loss={loss.item():.4f} "
                  f"gaps={loss_dict['gaps_total']:.4f} "
                  f"e_lpips={loss_dict['edge_lpips']:.4f} "
                  f"f_mae={loss_dict['flat_mae']:.4f} "
                  f"fft={loss_dict['fft']:.4f} "
                  f"psnr={p:.2f}dB "
                  f"({ips:.2f} img/s)")
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/gaps_total', loss_dict['gaps_total'], global_step)
            writer.add_scalar('train/edge_lpips', loss_dict['edge_lpips'], global_step)
            writer.add_scalar('train/flat_mae', loss_dict['flat_mae'], global_step)
            writer.add_scalar('train/fft', loss_dict['fft'], global_step)
            writer.add_scalar('train/psnr', p, global_step)

    return total_loss / max(n, 1), total_psnr / max(n, 1), global_step


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    n = 0
    for batch in loader:
        x_agd = batch["x_agd"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)
        pred = model(x_agd)
        loss, _ = loss_fn(pred, x_gt)

        pred_c = pred.clamp(0, 1)
        gt_c = x_gt.clamp(0, 1)
        bs = x_agd.size(0)
        total_loss += loss.item() * bs
        total_psnr += psnr(pred_c, gt_c) * bs
        total_ssim += ssim_metric(pred_c, gt_c) * bs
        total_rmse += rmse(pred_c, gt_c) * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "rmse": total_rmse / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--overfit", type=int, default=0,
                        help="If > 0, overfit on N samples for sanity check.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['train']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, GPU count: {torch.cuda.device_count()}")

    # ---- Dataset ----
    def build_ds(split_dir, augment):
        return CTRestorationDataset(
            split_dir=split_dir,
            agd_subdir=cfg['data']['agd_subdir'],
            gt_subdir=cfg['data']['gt_subdir'],
            file_prefix=cfg['data']['file_prefix'],
            index_digits=cfg['data']['index_digits'],
            normalize_mode=cfg['data']['normalize_mode'],
            global_gt_max=cfg['data']['global_gt_max'],
            global_agd_max=cfg['data']['global_agd_max'],
            augment=augment,
        )

    if args.overfit > 0:
        print(f"\n*** OVERFIT MODE: using {args.overfit} samples from train split ***\n")
        full_train = build_ds(cfg['data']['train_dir'], augment=False)
        train_ds = Subset(full_train, list(range(min(args.overfit, len(full_train)))))
        val_ds = train_ds 
    else:
        train_ds = build_ds(cfg['data']['train_dir'], augment=True)
        val_ds = build_ds(cfg['data']['val_dir'], augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
    )

    # ---- Model ----
    model = DBFUNet(
        in_channels=1,
        out_channels=1,
        base_channels=cfg['model']['base_channels'],
        num_levels=cfg['model']['num_levels'],
        freq_channels=cfg['model']['freq_channels'],
        freq_blocks=cfg['model']['freq_blocks'],
        use_freq_branch=cfg['model']['use_freq_branch'],
        dropout=cfg['model']['dropout'],
    ).to(device)
    print(f"Model params: {count_params(model) / 1e6:.2f} M")

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # ---- Loss / Optimizer / Scheduler ----
    # 引入了 device 并使用 w_gaps (如果在 yaml 中没配，默认 fallback 到 1.0)
    loss_fn = CompositeLoss(
        device=device,
        w_gaps=cfg['loss'].get('w_gaps', 1.0),
        w_fft=cfg['loss'].get('w_fft', 0.05),
        val_range=1.0,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )
    scheduler = get_scheduler(
        optimizer,
        epochs=cfg['train']['epochs'],
        warmup_epochs=cfg['train']['warmup_epochs'],
        lr_min=cfg['train']['lr_min'],
        lr_max=cfg['train']['lr'],
    )
    scaler = torch.cuda.amp.GradScaler() if cfg['train']['use_amp'] else None

    # ---- Directories ----
    ensure_dir(cfg['log']['log_dir'])
    ensure_dir(cfg['log']['ckpt_dir'])
    ensure_dir(Path(cfg['log']['log_dir']) / "viz")
    writer = SummaryWriter(log_dir=cfg['log']['log_dir'])

    # ---- Resume ----
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer, scheduler,
                                                 map_location=device)
        print(f"Resumed from epoch {start_epoch}, best PSNR {best_psnr:.2f}")

    # ---- Training loop ----
    global_step = 0
    for epoch in range(start_epoch, cfg['train']['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{cfg['train']['epochs']}  lr={optimizer.param_groups[0]['lr']:.2e} ===")
        train_loss, train_psnr, global_step = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            epoch + 1, cfg['log']['log_every'], writer, global_step,
        )
        scheduler.step()

        if (epoch + 1) % cfg['log']['val_every'] == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            print(f"[val] loss={val_metrics['loss']:.4f} "
                  f"psnr={val_metrics['psnr']:.2f}dB "
                  f"ssim={val_metrics['ssim']:.4f} "
                  f"rmse={val_metrics['rmse']:.5f}")
            writer.add_scalar('val/loss', val_metrics['loss'], epoch + 1)
            writer.add_scalar('val/psnr', val_metrics['psnr'], epoch + 1)
            writer.add_scalar('val/ssim', val_metrics['ssim'], epoch + 1)
            writer.add_scalar('val/rmse', val_metrics['rmse'], epoch + 1)

            # Save best
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    Path(cfg['log']['ckpt_dir']) / "best.pth",
                    model, optimizer, scheduler, epoch + 1, best_psnr, cfg,
                )
                print(f"  >>> New best PSNR: {best_psnr:.2f}dB, saved.")

        # Periodic save + viz
        if (epoch + 1) % cfg['log']['save_every'] == 0:
            save_checkpoint(
                Path(cfg['log']['ckpt_dir']) / f"epoch_{epoch+1:04d}.pth",
                model, optimizer, scheduler, epoch + 1, best_psnr, cfg,
            )

        if (epoch + 1) % cfg['log']['visualize_every'] == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(val_loader))
                x_agd = batch["x_agd"].to(device)
                x_gt = batch["x_gt"].to(device)
                pred = model(x_agd)
                save_triplet(
                    x_agd[0], pred[0].clamp(0, 1), x_gt[0].clamp(0, 1),
                    Path(cfg['log']['log_dir']) / "viz" / f"epoch_{epoch+1:04d}.png",
                )

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
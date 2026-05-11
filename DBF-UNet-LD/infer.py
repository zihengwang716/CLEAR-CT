import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from data.dataset import CTRestorationDataset
# 【修改点 1】：将 DBFUNet 改为和 train 相同的 SpatialUNet_LowDose
from models.dbf_unet import SpatialUNet_LowDose
from utils import ensure_dir


def build_model(cfg, device):
    # 【修改点 2】：使用 SpatialUNet_LowDose 并移除 freq 相关参数，与 train 代码保持完全一致
    model = SpatialUNet_LowDose(
        in_channels=1,
        out_channels=1,
        base_channels=cfg['model']['base_channels'],
        num_levels=cfg['model']['num_levels'],
        dropout=0.0, # 推理阶段通常不使用 dropout
    ).to(device)
    return model


def load_weights(model, ckpt_path, device):
    # 修改点：添加了 weights_only=False 以兼容 PyTorch 2.6+ 针对包含 numpy 标量等额外信息的 checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Checkpoint might be saved with or without DataParallel wrapping
    sd = state["model"] if "model" in state else state
    # Strip "module." prefix if present
    new_sd = {}
    for k, v in sd.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new_sd, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")
    if "epoch" in state:
        print(f"  epoch: {state['epoch']}, best_psnr: {state.get('best_psnr', 'N/A')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to run inference on")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Directory to save predictions")
    parser.add_argument("--denormalize", action="store_true",
                        help="Save predictions in original (un-normalized) scale")
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, only run on first N samples (for debugging)")
    parser.add_argument("--save_format", type=str, default="npy",
                        choices=["npy", "tif"],
                        help="Format to save predictions")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Pick the split dir ----
    split_dir_key = f"{args.split}_dir"
    split_dir = cfg['data'][split_dir_key]
    print(f"Running inference on split '{args.split}' at: {split_dir}")

    # ---- Dataset ----
    dataset = CTRestorationDataset(
        split_dir=split_dir,
        agd_subdir=cfg['data']['agd_subdir'],
        gt_subdir=cfg['data']['gt_subdir'],
        file_prefix=cfg['data']['file_prefix'],
        index_digits=cfg['data']['index_digits'],
        normalize_mode=cfg['data']['normalize_mode'],
        global_gt_max=cfg['data']['global_gt_max'],
        global_agd_max=cfg['data']['global_agd_max'],
        augment=False,
    )

    if args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
        print(f"Limited to first {len(dataset)} samples.")

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=cfg['train']['num_workers'], pin_memory=True,
    )

    # ---- Model ----
    model = build_model(cfg, device)
    load_weights(model, args.ckpt, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f} M")

    # ---- Output dir ----
    out_dir = Path(args.output_dir) / args.split
    ensure_dir(out_dir)
    print(f"Saving predictions to: {out_dir}")

    # ---- Inference loop ----
    t0 = time.time()
    prefix = cfg['data']['file_prefix']
    digits = cfg['data']['index_digits']

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_agd = batch["x_agd"].to(device, non_blocking=True)
            scale = batch["scale"].item()
            idx = batch["idx"].item()

            pred = model(x_agd)  # [1, 1, H, W]

            # Optionally restore original scale
            if args.denormalize:
                pred_np = (pred[0, 0].cpu().numpy() * scale).astype(np.float32)
            else:
                pred_np = pred[0, 0].cpu().numpy().astype(np.float32)

            # Save
            fname = f"{prefix}{idx:0{digits}d}"
            if args.save_format == "npy":
                np.save(out_dir / f"{fname}.npy", pred_np)
            else:
                import tifffile
                tifffile.imwrite(out_dir / f"{fname}.tif", pred_np)

            if (i + 1) % 50 == 0 or (i + 1) == len(loader):
                elapsed = time.time() - t0
                ips = (i + 1) / elapsed
                print(f"  [{i+1}/{len(loader)}] {ips:.2f} img/s, "
                      f"ETA: {(len(loader) - i - 1) / ips:.1f}s")

    total = time.time() - t0
    print(f"\nDone. Processed {len(loader)} samples in {total:.1f}s ({len(loader)/total:.2f} img/s)")
    print(f"Predictions saved to: {out_dir}")
    print(f"  Format: .{args.save_format}")
    print(f"  Scale:  {'ORIGINAL (un-normalized)' if args.denormalize else 'NORMALIZED (as-trained, [0,1])'}")


if __name__ == "__main__":
    main()


'''
python /ibex/user/wangz0r/CS_300_Final_Project/CS_300/DBF-UNet-LD/infer.py \
--config /ibex/user/wangz0r/CS_300_Final_Project/CS_300/DBF-UNet-LD/config/config_exp2_ablation_low_dose_only.yaml \
--ckpt /ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/checkpoint/LD_comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner/best.pth \
--output_dir /ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/result/LD_comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner
'''
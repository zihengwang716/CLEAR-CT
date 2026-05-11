"""
Dataset for paired (AGD reconstruction, GT) images.
Handles scale unification between AGD and GT.
Train/val/test splits are pre-organized by directory.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile


class CTRestorationDataset(Dataset):
    """
    Loads paired (x_agd, x_gt) samples from a single split directory.

    File layout expected:
        split_dir/
            agd_subdir/slice00001.npy  (float, [H, W])
            gt_subdir/slice00001.tif   (float, [H, W])

    Scale unification:
        - 'per_sample':
              scale = max(x_gt)
              x_gt_norm  = x_gt  / scale
              x_agd_norm = x_agd / scale     # use GT's max for both
        - 'global':
              x_gt_norm  = x_gt  / global_gt_max
              x_agd_norm = x_agd / global_gt_max
    """
    def __init__(
        self,
        split_dir,
        agd_subdir="agd_recon",
        gt_subdir="gt",
        file_prefix="slice",
        index_digits=5,
        normalize_mode="per_sample",
        global_gt_max=0.014,
        global_agd_max=0.036,
        augment=False,
    ):
        self.root = Path(split_dir)
        self.agd_dir = self.root / agd_subdir
        self.gt_dir = self.root / gt_subdir
        self.prefix = file_prefix
        self.index_digits = index_digits
        self.normalize_mode = normalize_mode
        self.global_gt_max = float(global_gt_max)
        self.global_agd_max = float(global_agd_max)
        self.augment = augment

        # Discover all paired samples in this split
        agd_files = sorted(self.agd_dir.glob(f"{file_prefix}*.npy"))
        self.indices = []
        for f in agd_files:
            stem = f.stem  # e.g. "slice00001"
            idx_str = stem[len(file_prefix):]
            try:
                idx = int(idx_str)
                gt_path = self.gt_dir / f"{file_prefix}{idx_str}.tif"
                if gt_path.exists():
                    self.indices.append(idx)
            except ValueError:
                continue
        self.indices.sort()

        if len(self.indices) == 0:
            raise RuntimeError(
                f"No paired samples found in split. Check paths:\n"
                f"  agd_dir: {self.agd_dir}\n"
                f"  gt_dir:  {self.gt_dir}"
            )

        print(f"[Dataset] Loaded {len(self.indices)} samples from {self.root}")
        print(f"[Dataset] Normalize mode: {normalize_mode}, augment: {augment}")

    def __len__(self):
        return len(self.indices)

    def _fname(self, idx, ext):
        return f"{self.prefix}{idx:0{self.index_digits}d}.{ext}"

    def _load_one(self, idx):
        agd_path = self.agd_dir / self._fname(idx, "npy")
        gt_path = self.gt_dir / self._fname(idx, "tif")
        x_agd = np.load(agd_path).astype(np.float32)
        x_gt = tifffile.imread(gt_path).astype(np.float32)
        return x_agd, x_gt

    def _normalize(self, x_agd, x_gt):
        """Force scale unification between x_agd and x_gt."""
        if self.normalize_mode == "per_sample":
            scale = float(x_gt.max())
            if scale < 1e-8:
                scale = 1.0
            x_gt = x_gt / scale
            x_agd = x_agd / scale
            x_agd = np.clip(x_agd, -0.5, 3.0)
        elif self.normalize_mode == "global":
            scale = self.global_gt_max
            x_gt = x_gt / scale
            x_agd = x_agd / scale
            x_agd = np.clip(x_agd, -0.5, 3.0)
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        return x_agd, x_gt, scale

    def _augment(self, x_agd, x_gt):
        if np.random.rand() < 0.5:
            x_agd = np.flip(x_agd, axis=-1).copy()
            x_gt = np.flip(x_gt, axis=-1).copy()
        if np.random.rand() < 0.5:
            x_agd = np.flip(x_agd, axis=-2).copy()
            x_gt = np.flip(x_gt, axis=-2).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            x_agd = np.rot90(x_agd, k=k, axes=(-2, -1)).copy()
            x_gt = np.rot90(x_gt, k=k, axes=(-2, -1)).copy()
        return x_agd, x_gt

    def __getitem__(self, i):
        idx = self.indices[i]
        x_agd, x_gt = self._load_one(idx)

        if x_agd.ndim == 3:
            x_agd = x_agd.squeeze()
        if x_gt.ndim == 3:
            x_gt = x_gt.squeeze()

        x_agd, x_gt, scale = self._normalize(x_agd, x_gt)

        if self.augment:
            x_agd, x_gt = self._augment(x_agd, x_gt)

        x_agd = torch.from_numpy(x_agd).unsqueeze(0).float()
        x_gt = torch.from_numpy(x_gt).unsqueeze(0).float()

        # ========================================================
        # [新增代码] 强制对齐输入和GT的空间分辨率
        # ========================================================
        import torch.nn.functional as F
        if x_agd.shape[-2:] != x_gt.shape[-2:]:
            # interpolate 需要 4D 张量 [B, C, H, W]，所以先 unsqueeze 升维，插值完再 squeeze 降维
            x_agd = F.interpolate(
                x_agd.unsqueeze(0), 
                size=x_gt.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        # ========================================================

        return {
            "x_agd": x_agd,
            "x_gt": x_gt,
            "scale": float(scale),
            "idx": int(idx),
        }
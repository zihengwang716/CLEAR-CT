# CLEAR-CT

CLEAR-CT is a research-oriented CT reconstruction and restoration framework built on the 2DeteCT dataset. The project combines physics-based iterative reconstruction and learning-based image restoration to address several common CT degradation problems:

* Low-dose noise
* Sparse-view sampling
* Beam-hardening artifacts

The repository integrates accelerated iterative reconstruction methods with deep neural image restoration pipelines based on DBF-UNet.

---

# Features

## Physics-Based Reconstruction

* Fan-beam CT reconstruction
* ASTRA CUDA projector backend
* Accelerated Gradient Descent (AGD)
* FISTA-style momentum acceleration
* Optional inverse-variance weighting
* Optional beam-hardening polynomial correction
* Optional TV regularization
* Sparse-view angular downsampling
* Configurable reconstruction size and iteration count

## Learning-Based Restoration

* DBF-UNet restoration pipeline
* Support for train / validation / test splits
* Composite loss functions
* Validation image saving
* TensorBoard logging
* Checkpoint saving and resume training
* Inference and evaluation scripts

---

# Repository Structure

```text
CLEAR-CT/
├── recon/                  # AGD/FISTA CT reconstruction
├── DBF-UNet/               # Main DBF-UNet restoration framework
├── DBF-UNet-LD/            # Low-dose variant
├── DBF-UNet-SV/            # Sparse-view variant
├── Eval/                   # Evaluation utilities
├── dcgm/                   # Dataset/cache generation tools
├── gen_mode3p/             # Utilities for generating degraded data
├── cache/                  # Cached data and outputs
└── DBF_Unet_envs_create.txt
```

---

# Environment Setup

## Create Conda Environment

```bash
conda create -n cs300 python=3.10.20 -c defaults
conda activate cs300
```

## Install PyTorch

```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

## Install Remaining Dependencies

```bash
pip install astra-toolbox scikit-image scipy tifffile imageio matplotlib tqdm pyyaml tensorboard lpips sewar
```

For the exact environment used during development, see:

```text
DBF_Unet_envs_create.txt
```

---

# Dataset Structure

## 2DeteCT Input Data

The reconstruction code expects a 2DeteCT-style dataset layout.

Example:

```text
DATA_DIR/
└── slice00001/
    └── mode3/
        ├── dark.tif
        ├── sinogram.tif
        ├── flat1.tif
        ├── flat2.tif
        ├── sinogram_with_poisson.tif
        ├── flat1_scaled.tif
        └── flat2_scaled.tif
```

Ground-truth reconstruction layout:

```text
GT_DIR/
└── slice00001/
    └── mode2/
        └── reconstruction.tif
```

---

# DBF-UNet Dataset Layout

DBF-UNet expects the following structure:

```text
DATASET_ROOT/
├── Train/
│   ├── agd_recon/
│   └── gt/
├── Val/
│   ├── agd_recon/
│   └── gt/
└── Test/
    ├── agd_recon/
    └── gt/
```

Example split:

```text
Training Set:     1 - 4000
Validation Set:   4001 - 4500
Testing Set:      4501 - 5000
```

---

# Quick Start

## Step 1: Run AGD Reconstruction

```bash
cd recon

python reconstruction_agd_enhanced.py \
  --data_dir /path/to/2DeteCT_slicesAll \
  --gt_dir /path/to/2DeteCT_slices_RecSeg_All \
  --out_dir /path/to/result \
  --input_mode mode3 \
  --input_variant noisy \
  --use_weight \
  --bh_coeffs paired_deg3_coeffs.npy \
  --lambda_tv 3 \
  --tv_inner 30 \
  --downsample 6 \
  --experiment_name weight_bh_tv_mode3p_ds6_lam3 \
  --timestamp auto \
  --slice_start 1 \
  --slice_end 5000
```

This step generates AGD reconstructions used as DBF-UNet inputs.

---

## Step 2: Prepare DBF-UNet Dataset

Organize the reconstructed AGD images and ground-truth images into:

```text
Train/
Val/
Test/
```

following the dataset structure described above.

---

## Step 3: Train DBF-UNet

```bash
cd DBF-UNet

python train.py \
  --config config/config_exp1_all.yaml
```

---

## Step 4: Run Validation / Inference

```bash
python infer.py \
  --config config/config_exp1_all.yaml \
  --ckpt /path/to/best.pth \
  --split test \
  --output_dir predictions
```

---

# Physics-Based Reconstruction

Main reconstruction script:

```text
recon/reconstruction_agd_enhanced.py
```

---

## Reconstruction Options

| Parameter           | Description                                   |
| ------------------- | --------------------------------------------- |
| `--input_mode`      | Input mode folder such as mode1, mode2, mode3 |
| `--input_variant`   | plain / noisy / auto                          |
| `--downsample`      | Angular downsampling factor                   |
| `--agd_iter`        | Number of AGD iterations                      |
| `--rec_size`        | Reconstruction resolution                     |
| `--use_weight`      | Enable inverse-variance weighting             |
| `--bh_coeffs`       | Beam-hardening polynomial coefficients        |
| `--lambda_tv`       | TV regularization weight                      |
| `--tv_inner`        | TV proximal inner iterations                  |
| `--slice_start`     | Starting slice index                          |
| `--slice_end`       | Ending slice index                            |
| `--experiment_name` | Experiment identifier                         |
| `--timestamp`       | Append timestamp to output directory          |

---

## Input Variants

### Plain Variant

Uses:

```text
sinogram.tif
flat1.tif
flat2.tif
```

### Noisy Variant

Uses:

```text
sinogram_with_poisson.tif
flat1_scaled.tif
flat2_scaled.tif
```

---

# Reconstruction Output Structure

Typical reconstruction output:

```text
result/
└── weight_bh_tv_mode3p_ds6_lam3__20260511_210049/
    ├── recon/
    ├── metrics.csv
    ├── config.yaml
    └── logs/
```

Per-slice reconstructions are saved individually.

---

# DBF-UNet Training

Main training script:

```text
DBF-UNet/train.py
```

---

## Configure Dataset Paths

Edit:

```text
config/config_exp1_all.yaml
```

Example:

```yaml
data:
  train_dir: "/path/to/Train"
  val_dir: "/path/to/Val"
  test_dir: "/path/to/Test"
  agd_subdir: "agd_recon"
  gt_subdir: "gt"
```

---

## Start Training

```bash
python train.py \
  --config config/config_exp1_all.yaml
```

---

## Resume Training

```bash
python train.py \
  --config config/config_exp1_all.yaml \
  --resume /path/to/checkpoint.pth
```

---

## Overfit Sanity Check

```bash
python train.py \
  --config config/config_exp1_all.yaml \
  --overfit 8
```

---

# Validation and Image Saving

Validation reconstructions are automatically saved during training.

Typical output:

```text
logs/lightning_logs/version_0/val_recons/
└── epoch_0000/
    ├── val_batch0000_item00.tif
    ├── val_batch0000_item01.tif
    └── ...
```

Example visualization:

```bash
python -c "
import tifffile
import matplotlib.pyplot as plt

img = tifffile.imread('val_batch0000_item00.tif')

plt.imshow(img, cmap='gray')
plt.colorbar()
plt.savefig('preview.png', dpi=200, bbox_inches='tight')
"
```

---

# DBF-UNet Inference

Run inference:

```bash
python infer.py \
  --config config/config_exp1_all.yaml \
  --ckpt /path/to/best.pth \
  --split test \
  --output_dir predictions \
  --save_format tif
```

Supported splits:

```text
train
val
test
```

---

# Inference Output Structure

```text
predictions/
└── test/
    ├── slice00001.tif
    ├── slice00002.tif
    └── ...
```

---

# Loss Function

DBF-UNet uses a composite restoration loss:

```text
Loss = w_l1 * L1 + w_ssim * SSIM + w_fft * FFTLoss
```

Typical configuration:

```yaml
loss:
  w_l1: 1.0
  w_ssim: 0.1
  w_fft: 0.05
```

---

# Evaluation

Reported metrics include:

* PSNR
* SSIM
* RMSE
* MAE

TensorBoard logging:

```bash
tensorboard --logdir logs
```

---

# Experimental Settings

## Low-Dose Setting

* Noisy sinogram input
* Weighting enabled
* Full-view reconstruction

---

## Sparse-View Setting

* Angular downsampling enabled
* Typical setting:

```text
--downsample 6
```

---

## Beam-Hardening Setting

* Polynomial correction enabled
* Typical degree:

```text
degree 3
```

---

# SLURM Usage

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=agd_mode3p
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=96:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source $(conda info --base)/etc/profile.d/conda.sh
conda activate cs300

cd /path/to/CLEAR-CT/recon

python reconstruction_agd_enhanced.py \
  --data_dir /path/to/data \
  --gt_dir /path/to/gt \
  --out_dir /path/to/result \
  --input_mode mode3 \
  --input_variant noisy \
  --downsample 6
```

Submit:

```bash
sbatch run_recon.sh
```

---

# Notes

* CUDA-capable GPUs are strongly recommended.
* ASTRA CUDA backend is required for fast reconstruction.
* Reconstruction resolution is typically 2048.
* Outputs may be center-cropped to 1024 for evaluation or visualization.
* If `--bh_coeffs` is omitted, beam-hardening correction is disabled.
* If `--use_weight` is enabled and iteration count is not specified, more AGD iterations may be automatically used.

---

# Project Status

This repository is intended for research and educational purposes.

Paths in configuration files are machine-specific and should be updated before running on a new system.

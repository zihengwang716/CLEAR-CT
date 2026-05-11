#!/bin/bash -l
#SBATCH --job-name=nerf_full5k
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/new_nerf_full5k_%j.out
#SBATCH --error=logs/new_nerf_full5k_%j.err

# ============================================================
# 2D NeRF Full 5000 slices of Mode 3p
#   Sparse view: 6x (3600 → 600 angles)
#   Per-slice: 2000 iters, ~30-60s
#   Total estimated: 40-80 hours
#
# 支持断点续传：如果 job 被打断，重新提交会自动跳过已完成的 slice。
# 想加速可以同时提交多个 job 跑不同 slice 范围（修改 --slices）。
# ============================================================

set -e
mkdir -p logs
cd path/to/nerf
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xxx

echo "============================================================"
echo "NeRF FULL 5000 slices - Mode 3p, sparse 6x"
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

python nerf_2d_ct.py \
    --data_dir path/to/2DeteCT_slicesAll \
    --mode3_noisy_dir path/to/2DeteCT_slicesAll_mode3_noisy \
    --gt_dir path/to/2DeteCT_slices_RecSeg_All \
    --slices 1 5000 \
    --mode 3p \
    --gt_mode 2 \
    --ang_subsamp 6 \
    --max_ang 360 \
    --n_iter 4000 \
    --batch_rays 4096 \
    --n_samples 128 \
    --lr 5e-4 \
    --hidden_dim 256 \
    --n_layers 8 \
    --freq_bands 10 \
    --output_scale 0.02 \
    --init_bias -3.0 \
    --rec_size 1024 \
    --n_vis 20 \
    --method_tag NeRF \
    --flip none \
    --out_dir ./results
    # --output_scale 0.05 \
    # --rec_size 1024 \
    # --n_vis 20 \
    # --method_tag NeRF \
    # --out_dir ./results

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

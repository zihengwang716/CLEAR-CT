#!/bin/bash -l
#SBATCH --job-name=gauss_p3_full5k
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --mem=48G
#SBATCH --output=logs/new_gauss_p3_full5k_%j.out
#SBATCH --error=logs/new_gauss_p3_full5k_%j.err

# ============================================================
# Phase 3 Gaussian Splatting FULL 5000 slices (mode 3p, sparse 6×).
#
# Phase 3 config (validated on slices 1, 2500, 4501):
#   - Init: AGD (30 iter, 1024×1024)
#   - Loss: Huber (δ=0.05)
#   - LR:   density 1e-4, position 1e-5, scale/rot 5e-4
#   - Early stop: TV rise 1.05× (no GT needed at deployment)
#
# Perf optimizations:
#   - chunk_size 8192 (was 2048) — 4× fewer chunks → less Python overhead
#   - tv_render_size 256 (was 1024) — 16× fewer pixels for TV check
#   Expected ~50-90s/slice → 5000 × 70s ≈ 100 hours (2-3 sbatch resumes).
#
# Resumable: re-run automatically skips done slices.
# If 48h not enough, re-submit until done.
# ============================================================

set -e
mkdir -p logs
cd path/to/2dgs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cs300_fp

echo "============================================================"
echo "Phase 3 Gauss FULL 5000 slices — Mode 3p, sparse 6×"
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

python gauss_2d_ct.py \
    --data_dir path/to/2DeteCT_slicesAll \
    --mode3_noisy_dir path/to/2DeteCT_slicesAll_mode3_noisy \
    --gt_dir path/to/2DeteCT_slices_RecSeg_All \
    --slices 1 5000 \
    --mode 3p \
    --gt_mode 2 \
    --ang_subsamp 6 \
    --max_ang 360 \
    --n_gaussians 30000 \
    --init_scale 0.005 \
    --max_density 0.01 \
    --init_density 0.0005 \
    --n_iter 1000 \
    --batch_rays 4096 \
    --chunk_size 8192 \
    --lr_position 1e-5 \
    --lr_scale 5e-4 \
    --lr_rotation 5e-4 \
    --lr_density 1e-4 \
    --loss_type huber \
    --huber_delta 0.05 \
    --sparsity_weight 0.0 \
    --init_from_agd \
    --agd_init_size 1024 \
    --agd_init_iter 30 \
    --early_stop tv_rise \
    --tv_log_every 10 \
    --tv_rise_factor 1.05 \
    --tv_render_size 256 \
    --tv_render_chunk_size 8192 \
    --force_stop_iter 600 \
    --rec_size 1024 \
    --gpu_index 0 \
    --seed 42 \
    --n_vis 20 \
    --method_tag GaussP3 \
    --run_id phase3_v1 \
    --out_dir ./results

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

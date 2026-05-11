#!/bin/bash
#SBATCH --job-name=w_bh_tv_m3p_ds6_lam3
#SBATCH --output=logs/run_weight_bh_tv_mode3p_ds6_lam3_%j.out
#SBATCH --error=logs/run_weight_bh_tv_mode3p_ds6_lam3_%j.err
#SBATCH --time=96:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e

cd path/to/recon
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xxx

mkdir -p logs

DATA_DIR="path/to/2DeteCT_slicesAll_mode3_noisy"
GT_DIR="path/to/2DeteCT_slices_RecSeg_All"
OUT_DIR="path/to//result"
BH_COEFFS="path/to/paired_deg3_coeffs.npy"

EXP_NAME="${EXP_NAME:-weight_bh_tv_mode3p_ds6_lam3_4501_5000}"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
LAMBDA_TV="${LAMBDA_TV:-3}"

echo "=== mode3p weighted AGD ds6 + BH + TV(lambda=3) ==="
echo "Start: $(date)"
echo "Experiment: $EXP_NAME"
echo "Timestamp:  $TIMESTAMP"
echo "lambda_tv:  $LAMBDA_TV"
echo "Slices:     4501-5000"

python reconstruction_agd_enhanced.py \
    --data_dir "$DATA_DIR" \
    --gt_dir "$GT_DIR" \
    --out_dir "$OUT_DIR" \
    --input_mode mode3 \
    --input_variant noisy \
    --use_weight \
    --bh_coeffs "$BH_COEFFS" \
    --lambda_tv "$LAMBDA_TV" \
    --tv_inner 30 \
    --downsample 6 \
    --experiment_name "$EXP_NAME" \
    --timestamp "$TIMESTAMP" \
    --slice_start 4501 \
    --slice_end 5000

echo ""
echo "=== Done: $(date) ==="
echo "Results: $OUT_DIR/${EXP_NAME}__${TIMESTAMP}/"

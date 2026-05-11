#!/bin/bash
#!/bin/bash
#SBATCH --job-name=xxx
#SBATCH --output=logs/run_grid_search_%j.out
#SBATCH --error=logs/run_grid_search_%j.err
#SBATCH --time=96:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e
cd path/to/CLEAR-TV/tv
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cs300

DATA_DIR="path/to/2DeteCT_slicesAll_mode3_noisy"
GT_DIR="path/to/2DeteCT_RecSeg/2DeteCT_slices_RecSeg_All"
OUT_DIR="path/to/compound_grid_search"
EXP_NAME="${EXP_NAME:-compound_lambda_highrange_10_20_30_50_75_100_150_200}"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"

echo "=== Grid Search: AGD+TV lambda tuning on compound noisy mode3 + BH (downsample ×6) ==="
echo "Start: $(date)"
echo "Timestamp:  $TIMESTAMP"
echo "Start: $(date)"

python lambda_grid_search.py \
    --data_dir "$DATA_DIR" \
    --gt_dir "$GT_DIR" \
    --out_dir "$OUT_DIR" \
    --experiment_name "$EXP_NAME" \
    --timestamp "$TIMESTAMP" \
    --mode 3 \
    --compound_noisy \
    --train_slices 1 500 1000 1500 2000 2500 3000 3500 4000 4500 \
    --val_slices 250 750 1250 1750 2250 2750 3250 3750 4250 4750 \
    --downsample 6 \
    --lambdas 10 20 30 50 75 100 150 200 \
    --agd_iter 100 \
    --rec_size 2048

echo ""
echo "=== Done: $(date) ==="
echo "Results: $OUT_DIR/${EXP_NAME}__${TIMESTAMP}/"

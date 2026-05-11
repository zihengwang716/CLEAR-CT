#!/bin/bash -l
#SBATCH --job-name=300_SV_EXP1          # 任务名称
#SBATCH --account=conf-neurips-2026.05.15-mengy
#SBATCH --time=24:00:00                   # 训练通常较长，建议给 24 小时或根据需要调整
#SBATCH --nodes=1                         # 1 个节点
#SBATCH --gpus=1                          # 1 张显卡
#SBATCH --constraint=a100                 # 确保分配到 A100 (80GB 显存版本)
#SBATCH --cpus-per-task=8                 # CPU 核心数，建议与 num_workers 匹配或略多
#SBATCH --mem=80G                         # 系统内存
#SBATCH --output=/ibex/user/liuj0s/CS_300/cache/logs/train_%j.out # 标准输出日志


source /ibex/user/liuj0s/miniconda3/etc/profile.d/conda.sh
conda activate cs300

CONFIG_FILE="config_exp1_ablation_sparse_view_only.yaml"
CONFIG_PATH="/ibex/user/liuj0s/CS_300/DBF-UNet-SV/config/${CONFIG_FILE}"


python /ibex/user/liuj0s/CS_300/DBF-UNet-SV/train.py --config ${CONFIG_PATH}

echo "Job finished at $(date)"
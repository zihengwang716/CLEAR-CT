#!/bin/bash -l
#SBATCH --job-name=300_All_EXP1          # 任务名称
#SBATCH --account=conf-neurips-2026.05.15-mengy
#SBATCH --time=24:00:00                   # 训练通常较长，建议给 24 小时或根据需要调整
#SBATCH --nodes=1                         # 1 个节点
#SBATCH --gpus=2                          # 2 张显卡
#SBATCH --constraint=a100                 # 确保分配到 A100 (80GB 显存版本)
#SBATCH --cpus-per-task=16                 # CPU 核心数，建议与 num_workers 匹配或略多
#SBATCH --mem=160G                         # 系统内存
#SBATCH --output=/ibex/user/liuj0s/CS_300/cache/logs/train_%j.out # 标准输出日志


source /ibex/user/liuj0s/miniconda3/etc/profile.d/conda.sh
conda activate cs300

CONFIG_FILE="config_exp1_all.yaml"
CONFIG_PATH="/ibex/user/liuj0s/CS_300/DBF-UNet/config/${CONFIG_FILE}"


python /ibex/user/liuj0s/CS_300/DBF-UNet/train.py --config ${CONFIG_PATH}

echo "Job finished at $(date)"
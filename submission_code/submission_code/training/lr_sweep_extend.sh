#!/bin/bash
#SBATCH --job-name=dentsim-lr-sweep-ext
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/lr_sweep_ext_%A_%a.out

# Extension to Stage 2: original sweep (5e-5, 1e-4, 2e-4, 3e-4) found 3e-4 -
# the highest value tested - nominal best for both models, which risks
# meaning the true optimum sits above the tested range rather than at 3e-4
# itself. Testing further out to check whether the trend keeps improving
# or has started to level off/reverse.
#
# 6 tasks total, no %-throttle - fits within the partition's 8 GPUs without
# needing to queue in waves.

module load container_env python3/2025.1-py312
cd ~/DentalHygieneLLM/training

CONFIGS=(
  "../config/training_sweep/granite-4.1-8B_stage2_lr.yaml 5e-4"
  "../config/training_sweep/granite-4.1-8B_stage2_lr.yaml 7e-4"
  "../config/training_sweep/granite-4.1-8B_stage2_lr.yaml 1e-3"
  "../config/training_sweep/qwen3-8B_stage2_lr.yaml 5e-4"
  "../config/training_sweep/qwen3-8B_stage2_lr.yaml 7e-4"
  "../config/training_sweep/qwen3-8B_stage2_lr.yaml 1e-3"
)

read -r CONFIG_PATH LR <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH" --lr "$LR"
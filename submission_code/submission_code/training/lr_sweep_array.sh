#!/bin/bash
#SBATCH --job-name=dentsim-lr-sweep
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/lr_sweep_%A_%a.out

module load container_env python3/2025.1-py312
cd ~/DentalHygieneLLM/training

# rank fixed at 16 per Stage 1's decision (baked into each stage2 config
# file already, so no --lora_r override needed here)
CONFIGS=(
  "../config/training_sweep/granite-4.1-8B_stage2_lr.yaml 5e-5"
  "../config/training_sweep/granite-4.1-8B_stage2_lr.yaml 1e-4"
  "../config/training_sweep/granite-4.1-8B_stage2_lr.yaml 2e-4"
  "../config/training_sweep/granite-4.1-8B_stage2_lr.yaml 3e-4"
  "../config/training_sweep/qwen3-8B_stage2_lr.yaml 5e-5"
  "../config/training_sweep/qwen3-8B_stage2_lr.yaml 1e-4"
  "../config/training_sweep/qwen3-8B_stage2_lr.yaml 2e-4"
  "../config/training_sweep/qwen3-8B_stage2_lr.yaml 3e-4"
)

read -r CONFIG_PATH LR <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH" --lr "$LR"
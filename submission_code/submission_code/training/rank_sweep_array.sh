#!/bin/bash
#SBATCH --job-name=dentsim-rank-sweep
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:30:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/rank_sweep_%A_%a.out

module load container_env python3/2025.1-py312
cd ~/DentalHygieneLLM/training

CONFIGS=(
  "../config/training_sweep/granite-4.1-8B_baseline.yaml 8"
  "../config/training_sweep/granite-4.1-8B_baseline.yaml 16"
  "../config/training_sweep/granite-4.1-8B_baseline.yaml 32"
  "../config/training_sweep/granite-4.1-8B_baseline.yaml 64"
  "../config/training_sweep/qwen3-8B_baseline.yaml 8"
  "../config/training_sweep/qwen3-8B_baseline.yaml 16"
  "../config/training_sweep/qwen3-8B_baseline.yaml 32"
  "../config/training_sweep/qwen3-8B_baseline.yaml 64"
)

read -r CONFIG_PATH LORA_R <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH" --lora_r "$LORA_R"
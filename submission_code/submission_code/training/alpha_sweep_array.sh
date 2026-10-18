#!/bin/bash
#SBATCH --job-name=dentsim-alpha-sweep
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/alpha_sweep_%A_%a.out

module load container_env python3/2025.1-py312
cd ~/DentalHygieneLLM/training

# alpha=32 (2x rank) intentionally omitted - it's already exactly Stage 3's
# attn+MLP baseline result. Copy that run into this stage's folder instead
# of retraining it (see the cp commands alongside this script).
CONFIGS=(
  "../config/training_sweep/granite-4.1-8B_stage4_alpha_ratio.yaml 16"
  "../config/training_sweep/granite-4.1-8B_stage4_alpha_ratio.yaml 64"
  "../config/training_sweep/qwen3-8B_stage4_alpha_ratio.yaml 16"
  "../config/training_sweep/qwen3-8B_stage4_alpha_ratio.yaml 64"
)

read -r CONFIG_PATH ALPHA <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH" --lora_alpha "$ALPHA"

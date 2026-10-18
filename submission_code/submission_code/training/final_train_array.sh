#!/bin/bash
#SBATCH --job-name=dentsim-final-train
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --output=logs/final_train_%A_%a.out

# Final production training on the full ~4000-example dataset, settled
# hyperparameters from Stages 1-5 (rank=16, alpha=32, lr=3e-4, epochs=3,
# attn+MLP). No sweeping - one run per model.
#
# Time estimate: ~4000 examples is ~2.1x the sweep's 1,880-example corpus,
# so expect roughly 2.1x the sweep's ~85-92 min (Granite) / ~80-83 min
# (Qwen) per-candidate runtime - i.e. ~3-3.25 hr for Granite, ~2.8-3 hr for
# Qwen. -t 04:00:00 leaves real margin for first-run model download time
# and any queue variance.

module load container_env python3/2025.1-py312
cd ~/DentalHygieneLLM/training

CONFIGS=(
  "../config/training_sweep/granite-4.1-8B_final.yaml"
  "../config/training_sweep/qwen3-8B_final.yaml"
)

CONFIG_PATH="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH"

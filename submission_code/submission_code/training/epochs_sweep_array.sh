#!/bin/bash
#SBATCH --job-name=dentsim-epochs-sweep
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/epochs_sweep_%A_%a.out

module load container_env python3/2025.1-py312
cd ~/DentalHygieneLLM/training

# epochs=3 intentionally omitted - it's already exactly Stage 4's alpha=32
# result. Copy that run into this stage's folder instead of retraining it
# (see the cp commands alongside this script). epochs=4 will run longer
# than prior stages (~633 steps -> ~844 steps), hence the extended -t.
CONFIGS=(
  "../config/training_sweep/granite-4.1-8B_stage5_epochs.yaml 2"
  "../config/training_sweep/granite-4.1-8B_stage5_epochs.yaml 4"
  "../config/training_sweep/qwen3-8B_stage5_epochs.yaml 2"
  "../config/training_sweep/qwen3-8B_stage5_epochs.yaml 4"
)

read -r CONFIG_PATH EPOCHS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH" --epochs "$EPOCHS"

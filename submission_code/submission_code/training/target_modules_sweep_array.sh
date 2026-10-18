#!/bin/bash
#SBATCH --job-name=dentsim-target-modules-sweep
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/target_modules_sweep_%A_%a.out

module load container_env python3/2025.1-py312
cd ~/DentalHygieneLLM/training

# lr=3e-4, lora_r=16 carried forward from Stage 1-2 decisions (baked into
# each stage3 config file already). "full" = default attention+MLP
# (no override needed); "attn_only" adds the --attn_only flag.
CONFIGS=(
  "../config/training_sweep/granite-4.1-8B_stage3_target_modules.yaml full"
  "../config/training_sweep/granite-4.1-8B_stage3_target_modules.yaml attn_only"
  "../config/training_sweep/qwen3-8B_stage3_target_modules.yaml full"
  "../config/training_sweep/qwen3-8B_stage3_target_modules.yaml attn_only"
)

read -r CONFIG_PATH MODE <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

if [ "$MODE" = "attn_only" ]; then
    crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH" --attn_only
else
    crun -p ~/envs/dh-lora-py312 python lora_finetune.py --config "$CONFIG_PATH"
fi

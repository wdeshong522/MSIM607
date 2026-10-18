#!/bin/bash
# submit_paper_arms.sh  (DRAFT - review paths before running)
# Chains the extra hyperparameter-contrast arms for the class paper.
# Each arm trains on the full pool (3,800) with the production config and ONE changed
# hyperparameter. Uses train_one_condition.sbatch, so the CLI override goes through
# EXTRA_ARGS (see the 3-line edit to the sbatch below).
#
# REQUIRED EDIT to train_one_condition.sbatch (so overrides can be passed):
#   1) after OUTPUT_NAME="$3" add:   EXTRA_ARGS=( ${4:-} )
#   2) in the crun python line add:  "${EXTRA_ARGS[@]}"  before --num_gpus
# lora_finetune_ray.py already accepts --lr and --attn_only.
set -euo pipefail
REPO=/home/wdesh001/DentalHygieneLLM
JOB="$REPO/training/train_one_condition.sbatch"
G="$REPO/config/training_sweep/granite-4.1-8B_final.yaml"
Q="$REPO/config/training_sweep/qwen3-8B_final.yaml"

# config  subset  output_name  extra_args
ARMS=(
 "$G|full|granite-4.1-8b__nfull_lr2e-4|--lr 2e-4"
 "$G|full|granite-4.1-8b__nfull_lr5e-5|--lr 5e-5"
 "$Q|full|qwen3-8b__nfull_lr2e-4|--lr 2e-4"
 "$Q|full|qwen3-8b__nfull_lr5e-5|--lr 5e-5"
)
PREV=""
for a in "${ARMS[@]}"; do
  IFS='|' read -r cfg subset name extra <<< "$a"
  dep=(); [ -n "$PREV" ] && dep=(--dependency="afterany:$PREV")
  out=$(sbatch --job-name="$name" "${dep[@]}" "$JOB" "$cfg" "$subset" "$name" "$extra")
  PREV=$(echo "$out" | grep -oE '[0-9]+$')
  echo "[$name] -> job $PREV"
done
# Conditional (only if checks fail): Granite n1000 rerun (~12 min), Qwen n1000/n2000/full reruns (~84 min).

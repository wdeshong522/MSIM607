#!/bin/bash
# submit_paper_generation.sh  (DRAFT)
# Generates the 200 holdout outputs for the five new adapters.
#  - Granite n1000 (4-GPU, batch 32) already exists -> starts immediately.
#  - The four hyperparameter arms wait for their training job (afterok).
# Run from the repo root AFTER add_vllm_entries.py. Training job IDs are the ones
# submitted on 2026-10-14: 10814288..10814291. Edit below if yours differ.
set -euo pipefail
REPO=/home/wdesh001/DentalHygieneLLM
JOB="$REPO/generation/generate_one_condition.sbatch"
BASEJOB="$REPO/generation/generate_base_condition.sbatch"
G=ibm-granite/granite-4.1-8b
Q=Qwen/Qwen3-8B-Base

# model_key | served_name | base_model | adapter_dir_name | training_job_to_wait_for ("-" = none)
CONDITIONS=(
 "granite-4.1-8B-n1000-b32|granite-n1000-b32|$G|granite-4.1-8b__n1000_wahab_gpu4|-"
 "granite-4.1-8B-nfull-lr2e-4|granite-nfull-lr2e4|$G|granite-4.1-8b__nfull_lr2e-4|10814288"
 "granite-4.1-8B-nfull-lr5e-5|granite-nfull-lr5e5|$G|granite-4.1-8b__nfull_lr5e-5|10814289"
 "qwen3-8B-nfull-lr2e-4|qwen-nfull-lr2e4|$Q|qwen3-8b__nfull_lr2e-4|10814290"
 "qwen3-8B-nfull-lr5e-5|qwen-nfull-lr5e5|$Q|qwen3-8b__nfull_lr5e-5|10814291"
)
for c in "${CONDITIONS[@]}"; do
  IFS='|' read -r key served base adir dep <<< "$c"
  depargs=(); [ "$dep" != "-" ] && depargs=(--dependency="afterok:$dep")
  out=$(sbatch --job-name="gen-$key" "${depargs[@]}" "$JOB" "$key" "$served" "$base" "$adir")
  echo "[$key] -> $out"
done
# Untuned base models (tokenizer/chat template taken from a tuned adapter so the prompt format matches)
for b in "granite-4.1-8B-base|granite-base|$G|granite-4.1-8b__nfull" "qwen3-8B-base|qwen-base|$Q|qwen3-8b__nfull"; do
  IFS='|' read -r key served base tokdir <<< "$b"
  out=$(sbatch --job-name="gen-$key" "$BASEJOB" "$key" "$served" "$base" "$tokdir")
  echo "[$key] -> $out"
done
echo "Note: afterok means a failed training job leaves its generation job pending forever (scancel it)."

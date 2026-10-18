#!/bin/bash
# submit_generation_runs.sh
# ============================
# Submits all 6 generation conditions (Granite + Qwen x n1000/n2000/nfull)
# as INDEPENDENT SLURM jobs — deliberately NO --dependency chaining, unlike
# the training driver. These conditions don't depend on each other at all,
# so they're submitted all at once and SLURM's own scheduler runs as many
# simultaneously as GPU availability allows: each job needs 2 GPUs, Turing's
# MSIM715 partition has 8 total (2 nodes x 4), so up to 4 of the 6 run at
# once, the remaining 2 queue automatically and start as slots free up.
#
# Run this ONCE from the login node — sbatch submission itself doesn't
# need GPUs. Safe to walk away from immediately after.
#
# Usage:
#   bash submit_generation_runs.sh

set -euo pipefail

REPO=/home/wdesh001/DentalHygieneLLM
GEN_DIR="$REPO/generation"
LOG_DIR="$GEN_DIR/logs"
JOB_SCRIPT="$GEN_DIR/generate_one_condition.sbatch"
VLLM_CONFIG="$REPO/config/vllm_models.yaml"

echo "Running preflight checks..."

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "ERROR: job script not found at $JOB_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$VLLM_CONFIG" ]; then
    echo "ERROR: vllm_models.yaml not found at $VLLM_CONFIG" >&2
    exit 1
fi

if [ ! -f "$GEN_DIR/generate_judge_examples.py" ]; then
    echo "ERROR: generate_judge_examples.py not found under $GEN_DIR" >&2
    exit 1
fi

if [ ! -f "$REPO/data/old_examples/eval_holdout_ids.json" ]; then
    echo "ERROR: frozen split not found under data/old_examples — run" >&2
    echo "       split_utils.py first (should already be done)." >&2
    exit 1
fi

for adapter_dir in \
    granite-4.1-8b__n1000 granite-4.1-8b__n2000 granite-4.1-8b__nfull \
    qwen3-8b__n1000 qwen3-8b__n2000 qwen3-8b__nfull
do
    if [ ! -d "$REPO/training/outputs/final/$adapter_dir" ]; then
        echo "ERROR: adapter directory not found: training/outputs/final/$adapter_dir" >&2
        echo "       (all 6 training conditions should already be complete)" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"
echo "Preflight checks passed."
echo

# model_key | served_name | base_model | adapter_dir_name
# Must match config/vllm_models.yaml exactly.
CONDITIONS=(
    "granite-4.1-8B-n1000|granite-n1000|ibm-granite/granite-4.1-8b|granite-4.1-8b__n1000"
    "granite-4.1-8B-n2000|granite-n2000|ibm-granite/granite-4.1-8b|granite-4.1-8b__n2000"
    "granite-4.1-8B-nfull|granite-nfull|ibm-granite/granite-4.1-8b|granite-4.1-8b__nfull"
    "qwen3-8B-n1000|qwen-n1000|Qwen/Qwen3-8B-Base|qwen3-8b__n1000"
    "qwen3-8B-n2000|qwen-n2000|Qwen/Qwen3-8B-Base|qwen3-8b__n2000"
    "qwen3-8B-nfull|qwen-nfull|Qwen/Qwen3-8B-Base|qwen3-8b__nfull"
)

echo "Submitting ${#CONDITIONS[@]} INDEPENDENT jobs (no chaining — SLURM runs up to 4 at once)..."
echo

for condition in "${CONDITIONS[@]}"; do
    IFS='|' read -r model_key served_name base_model adapter_dir_name <<< "$condition"

    SUBMIT_OUTPUT=$(sbatch \
        --job-name="gen-$model_key" \
        "$JOB_SCRIPT" "$model_key" "$served_name" "$base_model" "$adapter_dir_name")

    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oE '[0-9]+$')

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: could not parse job ID from sbatch output: $SUBMIT_OUTPUT" >&2
        echo "       Stopping — check 'squeue -u $USER' before resubmitting." >&2
        exit 1
    fi

    echo "  [$model_key] -> job $JOB_ID"
done

echo
echo "All jobs submitted. Check status with:"
echo "  squeue -u $USER"
echo
echo "Expect up to 4 running (R) simultaneously, the rest pending (PD) on"
echo "resources — that's the task parallelism working as intended, not stuck."
echo
echo "When done, check what actually completed:"
echo "  cat $LOG_DIR/generation_summary.log"
echo
echo "Per-job logs (including each vLLM server's own log) are under:"
echo "  $LOG_DIR/gen-<model_key>_<job-id>.out / .err"
echo "  $LOG_DIR/gen-<model_key>_<job-id>_vllm_server.log"
echo
echo "Results (judge_examples_<timestamp>.json, ~200 examples each) land under:"
echo "  $GEN_DIR/outputs/judge_examples/<model_key>/"

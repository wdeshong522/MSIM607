#!/bin/bash
# submit_overnight_runs.sh
# ==========================
# Submits the five remaining training conditions (Granite n=2000/full,
# Qwen n=1000/2000/full) as separate SLURM jobs, chained with
# --dependency=afterany so each starts as soon as the previous one ENDS
# (success or failure — afterany, not afterok, so one failed condition
# doesn't silently block the rest overnight; check overnight_summary.log
# in the morning to see which ones actually succeeded).
#
# Run this ONCE from the login node (no srun/interactive allocation
# needed — sbatch submission itself doesn't need GPUs). Safe to walk
# away from immediately after; each job runs independently of this
# shell, so closing the terminal or losing the SSH connection does not
# affect anything once submission is done.
#
# Usage:
#   bash submit_overnight_runs.sh

set -euo pipefail

REPO=/home/wdesh001/DentalHygieneLLM
TRAIN_DIR="$REPO/training"
LOG_DIR="$TRAIN_DIR/logs"
JOB_SCRIPT="$TRAIN_DIR/train_one_condition.sbatch"

GRANITE_CONFIG="$REPO/config/training_sweep/granite-4.1-8B_final.yaml"
QWEN_CONFIG="$REPO/config/training_sweep/qwen3-8B_final.yaml"

# ---------------------------------------------------------------------------
# Preflight checks — fail LOUDLY now, before submitting anything, rather
# than having a wrong path silently break a job at 2am with nobody watching.
# ---------------------------------------------------------------------------
echo "Running preflight checks..."

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "ERROR: job script not found at $JOB_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$GRANITE_CONFIG" ]; then
    echo "ERROR: Granite config not found at $GRANITE_CONFIG" >&2
    exit 1
fi

if [ ! -f "$QWEN_CONFIG" ]; then
    echo "ERROR: Qwen config not found at $QWEN_CONFIG" >&2
    echo "       (this script assumed the filename mirrors Granite's exactly —" >&2
    echo "       granite-4.1-8B_final.yaml -> qwen3-8B_final.yaml. If the real" >&2
    echo "       filename differs, edit QWEN_CONFIG at the top of this script.)" >&2
    exit 1
fi

if [ ! -f "$REPO/data/old_examples/eval_holdout_ids.json" ]; then
    echo "ERROR: frozen split not found under data/old_examples — run" >&2
    echo "       split_utils.py first (should already be done from tonight)." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Preflight checks passed."
echo

# ---------------------------------------------------------------------------
# The five remaining conditions, in the order they'll run.
# Format: "config_path  subset_size  output_dir_name"
#   subset_size is a number, or "full" to train on the whole available pool.
# ---------------------------------------------------------------------------
CONDITIONS=(
    "$GRANITE_CONFIG 2000 granite-4.1-8b__n2000"
    "$GRANITE_CONFIG full granite-4.1-8b__nfull"
    "$QWEN_CONFIG    1000 qwen3-8b__n1000"
    "$QWEN_CONFIG    2000 qwen3-8b__n2000"
    "$QWEN_CONFIG    full qwen3-8b__nfull"
)

echo "Submitting ${#CONDITIONS[@]} chained jobs..."
echo

PREV_JOB_ID=""
for condition in "${CONDITIONS[@]}"; do
    read -r cfg subset outname <<< "$condition"

    if [ -z "$PREV_JOB_ID" ]; then
        DEP_ARGS=()
    else
        DEP_ARGS=(--dependency="afterany:$PREV_JOB_ID")
    fi

    SUBMIT_OUTPUT=$(sbatch \
        --job-name="$outname" \
        "${DEP_ARGS[@]}" \
        "$JOB_SCRIPT" "$cfg" "$subset" "$outname")

    # sbatch prints: "Submitted batch job 123456"
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oE '[0-9]+$')

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: could not parse job ID from sbatch output: $SUBMIT_OUTPUT" >&2
        echo "       Stopping — check what was actually submitted with 'squeue -u $USER' before resubmitting." >&2
        exit 1
    fi

    echo "  [$outname] -> job $JOB_ID${PREV_JOB_ID:+ (after job $PREV_JOB_ID)}"
    PREV_JOB_ID="$JOB_ID"
done

echo
echo "All jobs submitted. Check status with:"
echo "  squeue -u $USER"
echo
echo "In the morning, check what actually completed:"
echo "  cat $LOG_DIR/overnight_summary.log"
echo
echo "Per-job logs (stdout/stderr, including the nproc/GPU sanity check"
echo "at the top of each) are under:"
echo "  $LOG_DIR/<job-name>_<job-id>.out"
echo "  $LOG_DIR/<job-name>_<job-id>.err"

#!/bin/bash
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --time=02:45:00
#SBATCH --output=/home/wdesh001/DentalHygieneLLM/training/logs/%x_%j.out
#SBATCH --error=/home/wdesh001/DentalHygieneLLM/training/logs/%x_%j.err
#
# train_one_condition.sbatch
# ===========================
# Runs exactly ONE training condition, then exits. Not meant to be run
# directly — submitted by submit_overnight_runs.sh, which passes the
# three positional args below and chains multiple submissions of this
# same script together so the whole overnight set survives any single
# job hitting the 02:45:00 walltime cap (each condition gets its OWN
# fresh budget, rather than all five sharing one job's clock).
#
# Args:
#   $1 = path to the --config YAML (e.g. config/training_sweep/granite-4.1-8B_final.yaml)
#   $2 = --train_subset_size value, or the literal string "full" to omit
#        that flag entirely and train on the full available pool
#   $3 = output directory name under training/outputs/final/
#        (e.g. granite-4.1-8b__n2000) — always passed explicitly so every
#        condition's result lands somewhere unambiguous and nothing risks
#        overwriting the old, already-discarded pre-frozen-split adapters
#        sitting at training/outputs/final/<model>  (no suffix).

set -uo pipefail  # NOT -e: we want to reach the summary-log line even
                   # if the training command itself fails, so SLURM's own
                   # accounting and this repo's own log both record the
                   # failure rather than the script just vanishing.

# Load explicitly rather than relying on the submitting shell's inherited
# environment happening to already have this — that fragility is exactly
# what broke every job in the first parallel GENERATION batch (2026-09-23,
# identical "crun: command not found" on all 6 — see generation_summary.log
# history). This script happened to work without it by luck (the
# submitting shell had the module loaded from earlier interactive work);
# don't rely on that luck holding for a future run.
module load python3/2025.1-py312

CONFIG="$1"
SUBSET_SIZE="$2"
OUTPUT_NAME="$3"

REPO=/home/wdesh001/DentalHygieneLLM
LOG_DIR="$REPO/training/logs"
SUMMARY_LOG="$LOG_DIR/overnight_summary.log"
OUTPUT_DIR="$REPO/training/outputs/final/$OUTPUT_NAME"

mkdir -p "$LOG_DIR"

echo "=== $(date) : starting job $SLURM_JOB_ID ($SLURM_JOB_NAME) ==="
echo "config=$CONFIG  subset_size=$SUBSET_SIZE  output_dir=$OUTPUT_DIR"
echo "--- environment sanity check (this is what broke everything earlier tonight) ---"
echo "nproc: $(nproc)"
crun -p ~/envs/dh-lora-py312 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "--------------------------------------------------------------------------------"

cd "$REPO/training"

if [ "$SUBSET_SIZE" = "full" ]; then
    SUBSET_ARGS=()
else
    SUBSET_ARGS=(--train_subset_size "$SUBSET_SIZE")
fi

crun -p ~/envs/dh-lora-py312 python lora_finetune_ray.py \
    --config "$CONFIG" \
    --data_dir "$REPO/data/old_examples" \
    "${SUBSET_ARGS[@]}" \
    --num_gpus 4 \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  SUCCESS  $OUTPUT_NAME  (job $SLURM_JOB_ID)" >> "$SUMMARY_LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S')  FAILED(exit=$EXIT_CODE)  $OUTPUT_NAME  (job $SLURM_JOB_ID) - check $LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" >> "$SUMMARY_LOG"
fi

echo "=== $(date) : job $SLURM_JOB_ID finished, exit code $EXIT_CODE ==="
exit "$EXIT_CODE"

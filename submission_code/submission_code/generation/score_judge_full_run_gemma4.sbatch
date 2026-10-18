#!/bin/bash
#SBATCH -A odu
#SBATCH --qos normal
#SBATCH -p high-gpu-mem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --output=/home/wdesh001/DentalHygieneLLM/generation/logs/%x_%j.out
#SBATCH --error=/home/wdesh001/DentalHygieneLLM/generation/logs/%x_%j.err
#
# score_judge_full_run_gemma4.sbatch
# =====================================
# Third-judge Phase 5 pass: google/gemma-4-31B-it (out-of-family judge,
# neither Granite nor Qwen lineage) scoring all 6 conditions x 200
# examples x N judge metrics (see score_judge_examples.py's PROMPT-FILE
# ARCHITECTURE docstring). Reasoning capture is ON (see the note in
# score_judge_full_run.sbatch for why --no_reasoning was reversed on
# 2026-09-27: a single-example check found reasoning-OFF gave a false
# hallucination=1 across the board that reasoning-ON correctly resolved
# to 0, plus real correctness differences reasoning-OFF was missing).
# Time budget bumped to near the qos max for the same reason - this job
# is resumable via --run_id if it doesn't finish in one submission (see
# RESUMABILITY CAVEAT in score_judge_examples.py). Meant to run IN PARALLEL with
# score_judge_full_run_granite42.sbatch overnight - each is its own
# independent sbatch job, so submit both and let Wahab's queue handle GPU
# allocation; no manual coordination needed between them.
#
# DB_FILE is the SAME shared judge_scores_all.db as the Qwen and Granite
# full runs (COMBINED-DB / RUN-ID ARCHITECTURE) - every row is stamped
# with judge_model="gemma4-31b-judge" and this job's run_id, so it
# coexists with the other judges' rows in one table instead of a
# separate file.
#
# BERTScore: confirmed working as of 2026-09-27 - no --skip_bertscore here
# any more. compute_bertscore() auto-copies a cached value from any other
# judge's row for the same (example_id, model_key) before computing
# anything fresh, so if an earlier run already populated BERTScore for
# these examples, this run reuses it for free instead of recomputing.
#
# Prereq: the gemma-4-31B-it smoke test (score_judge_smoke_test_
# gemma4.sbatch) already ran clean, confirming the model loads (not
# gated), scores parse, and the checkpoint is already in the shared HF
# cache.
#
# run_id defaults to $SLURM_JOB_ID - save it (printed at the top of this
# job's log) if you need to resume this run after a timeout/crash via
# --run_id <that id> on resubmit (see RESUMABILITY CAVEAT in
# score_judge_examples.py).

set -uo pipefail

module load container_env/0.1
module load python3/2025.1-py312

REPO=/home/wdesh001/DentalHygieneLLM
LOG_DIR="$REPO/generation/logs"
mkdir -p "$LOG_DIR"

DB_FILE="$REPO/generation/outputs/judge_scores_all.db"

INPUT_GLOB="$REPO/generation/outputs/judge_examples/*/*.json"
BASE_MODEL="google/gemma-4-31B-it"
SERVED_NAME="gemma4-31b-judge"
PORT=$((8000 + SLURM_JOB_ID % 1000))
SERVER_URL="http://localhost:$PORT"
SERVER_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vllm_server.log"
SCORE_SCRIPT="$SLURM_SUBMIT_DIR/score_judge_examples.py"

VLLM_ENV=~/envs/vllm-test
if ! crun -p "$VLLM_ENV" python -c "import vllm" 2>/dev/null; then
    echo "FAILED: $VLLM_ENV does not have a working vLLM install"
    exit 1
fi
echo "Confirmed: $VLLM_ENV has a working vLLM install"

if ! crun -p "$VLLM_ENV" python -c "import bert_score" 2>/dev/null; then
    echo "WARNING: $VLLM_ENV does not have bert-score installed - BERTScore "
    echo "will be silently skipped this run. Run: crun -p $VLLM_ENV pip install bert-score"
fi

if [ ! -f "$SCORE_SCRIPT" ]; then
    echo "FAILED: score_judge_examples.py not found at $SCORE_SCRIPT"
    echo "(expected alongside this sbatch script in the same submit dir --"
    echo " copy score_judge_examples.py there, or edit SCORE_SCRIPT above)"
    exit 1
fi
if [ ! -d "$SLURM_SUBMIT_DIR/judge_prompts" ]; then
    echo "FAILED: judge_prompts/ not found next to this sbatch script"
    echo "(unzip judge_prompts.zip into $SLURM_SUBMIT_DIR first)"
    exit 1
fi

echo "--- Starting vLLM server for gemma-4-31B-it (single GPU, no LoRA, no PP/TP) ---"
crun -p "$VLLM_ENV" env PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --enforce-eager \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Server starting (PID $SERVER_PID)... waiting for readiness (up to ~35 min)"
READY=0
for _ in $(seq 1 420); do
    if curl -s "$SERVER_URL/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process died during startup -- check $SERVER_LOG"
        break
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "FAILED: server never became ready. Tail of $SERVER_LOG:"
    tail -n 40 "$SERVER_LOG"
    kill -9 "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo "--- Server ready. Running FULL judge pass (gemma-4-31B-it): all models, all 200 examples, reasoning ON ---"
crun -p "$VLLM_ENV" env PYTHONUNBUFFERED=1 JUDGE_MODEL="$SERVED_NAME" \
    python "$SCORE_SCRIPT" \
    --input "$INPUT_GLOB" \
    --backend vllm \
    --vllm_url "$SERVER_URL" \
    --db_file "$DB_FILE"
SCORE_EXIT_CODE=$?

echo "--- Tearing down server ---"
kill -9 "$SERVER_PID" 2>/dev/null || true
for pid in $(crun -p "$VLLM_ENV" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
done

echo "Combined DB: $DB_FILE"
echo "=== $(date) : job $SLURM_JOB_ID finished, score exit code $SCORE_EXIT_CODE ==="
exit "$SCORE_EXIT_CODE"

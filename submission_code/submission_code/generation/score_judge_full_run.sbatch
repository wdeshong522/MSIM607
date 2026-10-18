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
# score_judge_full_run.sbatch
# ==============================
# The real Phase 5 scoring pass: all 6 conditions x 200 examples x N judge
# metrics (metrics now come from judge_prompts/*.txt - see score_judge_
# examples.py's PROMPT-FILE ARCHITECTURE docstring) + BERTScore, against
# the shared combined DB (judge_scores_all.db - see COMBINED-DB / RUN-ID
# ARCHITECTURE docstring: every row is stamped with judge_model/run_id/
# run_created_at, so this and the Granite/Gemma full runs can all write
# into the SAME file without overwriting each other).
#
# REASONING CAPTURE IS ON (changed 2026-09-27, reversing the prior
# --no_reasoning decision from 2026-09-25). A single-example reasoning-on
# re-check (eval-009756bb80b678fb, all 6 model_keys) found that the
# reasoning-OFF binary hallucination decode path was giving hallucination=1
# across the board with no actual issue present, while the reasoning-ON
# pass correctly gave hallucination=0 for all 6 AND changed 3 of 6
# correctness scores (catching real technique-terminology errors the
# reasoning-OFF pass missed entirely). That's a large enough, systematic
# enough swing that the reasoning-OFF fast path is no longer trusted for
# this study - see claude_diagnostic_grades.md and the smoke-test/
# single-example-reasoning follow-up for the evidence.
#
# TIME BUDGET: bumped from 4h to the qos's near-max (23:59:00) because
# reasoning capture roughly 6x's the token budget per call (max_tokens
# 1200 vs 200) and score_all() is strictly sequential (one call at a
# time, no concurrency) across ~6,000-9,600+ calls. This may still not be
# enough for one job to finish in a single submission - that's fine and
# expected: this script is resumable (see RESUMABILITY CAVEAT in
# score_judge_examples.py). If it times out, note the run_id this job
# printed at startup and resubmit with --run_id <that id> added to the
# python invocation below so the resumed job only scores what's still
# unscored under that same run_id, rather than starting over. Adjust
# --time upward here first if your qos/partition allows a longer limit.
#
# BERTScore: confirmed working as of 2026-09-27 (bert-score package
# installed into ~/envs/vllm-test; model download from Hugging Face
# succeeded from this environment). No --skip_bertscore here - let it run.
# compute_bertscore() auto-copies a cached value from any other judge's
# run for the same (example_id, model_key) before computing anything
# fresh, so this only pays the real compute cost once across however many
# judges eventually share this DB.
#
# run_id defaults to $SLURM_JOB_ID (printed at the top of this job's log) -
# save it if this job times out and you want to RESUME it via
# --run_id <that id> on resubmit, rather than starting a new run from
# scratch. See score_judge_examples.py's RESUMABILITY CAVEAT.

set -uo pipefail

module load container_env/0.1
module load python3/2025.1-py312

REPO=/home/wdesh001/DentalHygieneLLM
LOG_DIR="$REPO/generation/logs"
mkdir -p "$LOG_DIR"

DB_FILE="$REPO/generation/outputs/judge_scores_all.db"

INPUT_GLOB="$REPO/generation/outputs/judge_examples/*/*.json"
BASE_MODEL="Qwen/Qwen3.8-27B"
SERVED_NAME="qwen3-27b-judge"
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

echo "--- Starting vLLM server (single GPU, no LoRA, no PP/TP) ---"
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

echo "--- Server ready. Running FULL judge pass: all models, all 200 examples, reasoning ON ---"
# No --limit, no --example_id, no --models: scores everything ingested.
# No --no_reasoning: reasoning capture is ON for this study (see note at
# top of file for why the earlier --no_reasoning decision was reversed).
# No RESCORE: resumable -- if this job times out or dies, resubmitting
# with --run_id <this job's run_id> picks up only the still-unscored rows
# for THAT run rather than starting over (see RESUMABILITY CAVEAT).
crun -p "$VLLM_ENV" env PYTHONUNBUFFERED=1 JUDGE_MODEL="$SERVED_NAME" \
    python "$SCORE_SCRIPT" \
    --input "$INPUT_GLOB" \
    --backend vllm \
    --vllm_url "$SERVER_URL" \
    --db_file "$DB_FILE" \
    --run_id 6849544
SCORE_EXIT_CODE=$?

echo "--- Tearing down server ---"
kill -9 "$SERVER_PID" 2>/dev/null || true
for pid in $(crun -p "$VLLM_ENV" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
done

echo "Combined DB: $DB_FILE"
echo "=== $(date) : job $SLURM_JOB_ID finished, score exit code $SCORE_EXIT_CODE ==="
exit "$SCORE_EXIT_CODE"

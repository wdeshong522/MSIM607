#!/bin/bash
#SBATCH -A msim715
#SBATCH --qos msim715
#SBATCH -p MSIM715
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/home/wdesh001/DentalHygieneLLM/generation/logs/%x_%j.out
#SBATCH --error=/home/wdesh001/DentalHygieneLLM/generation/logs/%x_%j.err
#
# generate_one_condition.sbatch
# ================================
# Starts a vLLM PP2 server for ONE condition's adapter, waits for
# readiness, runs generate_judge_examples.py against it with concurrent
# requests (--max_concurrent), tears the server down, then exits.
#
# Sized deliberately light on GPUs (2, not 4) and CPUs (4, not the 28
# training needed) so SLURM can pack TWO of these per 4-GPU Turing node —
# submit_generation_runs.sh submits all 6 conditions with NO dependency
# chaining, so up to 4 run simultaneously across the partition's 8 GPUs
# (2 nodes x 4 GPUs), the rest queue automatically as slots free up.
#
# Args:
#   $1 = model_key       (matches a key in config/vllm_models.yaml, passed
#                          to generate_judge_examples.py's --models)
#   $2 = served_name      (must match exactly what --lora-modules registers
#                          below, or the generation script's registered-
#                          model check will skip this condition as "not
#                          currently being served")
#   $3 = base_model       (HF model id, e.g. ibm-granite/granite-4.1-8b)
#   $4 = adapter_dir_name (relative to training/outputs/final/)

set -uo pipefail

module load python3/2025.1-py312

CONFIG_KEY="$1"
SERVED_NAME="$2"
BASE_MODEL="$3"
ADAPTER_DIR_NAME="$4"

REPO=/home/wdesh001/DentalHygieneLLM
VLLM_ENV=~/envs/vllm-test
ADAPTER_DIR="$REPO/training/outputs/final/$ADAPTER_DIR_NAME"
LOG_DIR="$REPO/generation/logs"
SUMMARY_LOG="$LOG_DIR/generation_summary.log"
OUTPUT_DIR="$REPO/generation/outputs/judge_examples/$CONFIG_KEY"
SERVER_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vllm_server.log"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Unique port per job -- second, independent layer of defense alongside
# the PID-scoped kill fix below. With 2 jobs packed per node by design,
# a hardcoded port 8000 risks a genuine bind collision if both jobs'
# startup windows ever overlap exactly; this closes that off entirely
# rather than relying only on the kill fix to prevent interference.
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "Using port: $PORT"

echo "=== $(date) : starting job $SLURM_JOB_ID ($SLURM_JOB_NAME) ==="
echo "model_key=$CONFIG_KEY  served_name=$SERVED_NAME  base_model=$BASE_MODEL  adapter_dir=$ADAPTER_DIR"
echo "nproc: $(nproc)"
crun -p "$VLLM_ENV" nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

sum_gpu_memory_used() {
    crun -p "$VLLM_ENV" nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{sum += $1} END {print sum + 0}'
}

kill_process_tree() {
    # Recursively kill a PID and all its descendants -- unlike
    # `pkill -f "vllm.entrypoints..."`, this is scoped ONLY to processes
    # actually descended from the one WE launched. Confirmed necessary
    # 2026-09-24: with 2 generation jobs packed per node (by design), a
    # plain pattern-matched pkill searches the WHOLE node's process table
    # regardless of which job owns what -- SLURM's GPU/CPU cgroups don't
    # restrict process NAME visibility, only resource usage. One job
    # finishing and tearing down its own server was killing a co-located
    # sibling job's still-running/still-starting server too (two workers
    # died 4 seconds apart across two unrelated jobs -- the sibling never
    # served a single request before being killed this way).
    local pid="$1"
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_process_tree "$child"
    done
    kill -9 "$pid" 2>/dev/null || true
}

kill_vllm_and_wait() {
    kill_process_tree "$SERVER_PID"
    sleep 3
    # This part IS safe to leave scoped by GPU rather than by PID tree --
    # SLURM's GRES/cgroup allocation already restricts CUDA_VISIBLE_DEVICES
    # (and therefore what nvidia-smi can even see) to just this job's own
    # allocated GPUs, regardless of what else shares the physical node.
    for pid in $(crun -p "$VLLM_ENV" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 30); do
        USED=$(sum_gpu_memory_used)
        [ "$USED" -lt 500 ] && return 0
        sleep 2
    done
    echo "  WARNING: GPU memory still not clear after 60s (last: ${USED}MiB)"
}

record_result() {
    local status="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $status  $CONFIG_KEY  (job $SLURM_JOB_ID)" >> "$SUMMARY_LOG"
}

echo "--- Starting vLLM server ---"
crun -p "$VLLM_ENV" env PYTHONUNBUFFERED=1 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --tokenizer "$ADAPTER_DIR" \
    --enable-lora \
    --lora-modules "${SERVED_NAME}=${ADAPTER_DIR}" \
    --pipeline-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096 \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Server starting (PID $SERVER_PID)... waiting for readiness (up to ~10 min)"
READY=0
for _ in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process died during startup — check $SERVER_LOG"
        break
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "FAILED: server never became ready. See $SERVER_LOG"
    record_result "FAILED(server-not-ready)"
    kill -9 "$SERVER_PID" 2>/dev/null || true
    kill_vllm_and_wait
    exit 1
fi

echo "--- Server ready. Running generation ---"
crun -p "$VLLM_ENV" python "$REPO/generation/generate_judge_examples.py" \
    --backend vllm \
    --vllm_config "$REPO/config/vllm_models.yaml" \
    --vllm_url "http://localhost:$PORT" \
    --models "$CONFIG_KEY" \
    --data_dir "$REPO/data/old_examples" \
    --max_examples 200 \
    --max_concurrent 12 \
    --output_dir "$OUTPUT_DIR"

GEN_EXIT_CODE=$?

if [ "$GEN_EXIT_CODE" -eq 0 ]; then
    record_result "SUCCESS"
else
    record_result "FAILED(exit=$GEN_EXIT_CODE)"
fi

echo "--- Tearing down server ---"
kill -9 "$SERVER_PID" 2>/dev/null || true
kill_vllm_and_wait

echo "=== $(date) : job $SLURM_JOB_ID finished, generation exit code $GEN_EXIT_CODE ==="
exit "$GEN_EXIT_CODE"

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
# generate_base_condition.sbatch  (DRAFT)
# Same as generate_one_condition.sbatch, but serves the UNTUNED base model (no LoRA).
# The tokenizer / chat template comes from a fine-tuned adapter dir so the base model
# sees exactly the same prompt format as the tuned models.
# Args: $1 model_key  $2 served_name  $3 base_model  $4 tokenizer_dir_name (under training/outputs/final/)
set -uo pipefail
module load python3/2025.1-py312

CONFIG_KEY="$1"; SERVED_NAME="$2"; BASE_MODEL="$3"; TOKENIZER_DIR_NAME="$4"
REPO=/home/wdesh001/DentalHygieneLLM
VLLM_ENV=~/envs/vllm-test
TOKENIZER_DIR="$REPO/training/outputs/final/$TOKENIZER_DIR_NAME"
LOG_DIR="$REPO/generation/logs"
SUMMARY_LOG="$LOG_DIR/generation_summary.log"
OUTPUT_DIR="$REPO/generation/outputs/judge_examples/$CONFIG_KEY"
SERVER_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vllm_server.log"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "=== $(date) : starting job $SLURM_JOB_ID ($SLURM_JOB_NAME) ==="
echo "model_key=$CONFIG_KEY served_name=$SERVED_NAME base_model=$BASE_MODEL tokenizer_dir=$TOKENIZER_DIR port=$PORT"
crun -p "$VLLM_ENV" nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

sum_gpu_memory_used() {
    crun -p "$VLLM_ENV" nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}'
}
kill_process_tree() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do kill_process_tree "$child"; done
    kill -9 "$pid" 2>/dev/null || true
}
kill_vllm_and_wait() {
    kill_process_tree "$SERVER_PID"; sleep 3
    for pid in $(crun -p "$VLLM_ENV" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done
    for _ in $(seq 1 30); do
        USED=$(sum_gpu_memory_used); [ "$USED" -lt 500 ] && return 0; sleep 2
    done
    echo "  WARNING: GPU memory not clear after 60s (last: ${USED}MiB)"
}
record_result() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1  $CONFIG_KEY  (job $SLURM_JOB_ID)" >> "$SUMMARY_LOG"; }

echo "--- Starting vLLM server (base model, no LoRA) ---"
crun -p "$VLLM_ENV" env PYTHONUNBUFFERED=1 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --tokenizer "$TOKENIZER_DIR" \
    --served-model-name "$SERVED_NAME" \
    --pipeline-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096 \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then READY=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "Server died during startup - see $SERVER_LOG"; break; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "FAILED: server never ready. See $SERVER_LOG"; record_result "FAILED(server-not-ready)"
    kill -9 "$SERVER_PID" 2>/dev/null || true; kill_vllm_and_wait; exit 1
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
if [ "$GEN_EXIT_CODE" -eq 0 ]; then record_result "SUCCESS"; else record_result "FAILED(exit=$GEN_EXIT_CODE)"; fi
kill -9 "$SERVER_PID" 2>/dev/null || true
kill_vllm_and_wait
echo "=== $(date) : job $SLURM_JOB_ID finished, exit code $GEN_EXIT_CODE ==="
exit "$GEN_EXIT_CODE"

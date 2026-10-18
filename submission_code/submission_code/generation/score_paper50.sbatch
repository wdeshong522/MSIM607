#!/bin/bash
#SBATCH -A odu
#SBATCH --qos normal
#SBATCH -p high-gpu-mem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/wdesh001/DentalHygieneLLM/generation/logs/%x_%j.out
#SBATCH --error=/home/wdesh001/DentalHygieneLLM/generation/logs/%x_%j.err
#
# score_paper50.sbatch  (DRAFT)
# Judges the fixed 50-prompt subset for the NEW paper conditions with ONE judge.
# Submit FROM the generation/ directory (it needs score_judge_examples.py and judge_prompts/ there).
# Usage: sbatch --job-name=judge50-qwen score_paper50.sbatch qwen <model_key> [<model_key> ...]
#        judge = qwen | gemma
# Writes to generation/outputs/judge_scores_paper50.db (separate from judge_scores_all.db).
# Reasoning capture stays ON (same as the full runs). BERTScore is skipped (dropped from the study).
set -uo pipefail
module load container_env/0.1
module load python3/2025.1-py312

JUDGE="$1"; shift
KEYS=("$@")
case "$JUDGE" in
  qwen)  BASE_MODEL="Qwen/Qwen3.8-27B";      SERVED_NAME="qwen3-27b-judge" ;;
  gemma) BASE_MODEL="google/gemma-4-31B-it"; SERVED_NAME="gemma4-31b-judge" ;;
  *) echo "judge must be qwen or gemma"; exit 1 ;;
esac

REPO=/home/wdesh001/DentalHygieneLLM
LOG_DIR="$REPO/generation/logs"; mkdir -p "$LOG_DIR"
DB_FILE="${DB_FILE:-$REPO/generation/outputs/judge_scores_paper50.db}"
PROMPTS_DIR="${PROMPTS_DIR:-judge_prompts}"   # relative to the submit dir (generation/)
PORT=$((8000 + SLURM_JOB_ID % 1000))
SERVER_URL="http://localhost:$PORT"
SERVER_LOG="$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vllm_server.log"
SCORE_SCRIPT="$SLURM_SUBMIT_DIR/score_judge_examples.py"
VLLM_ENV=~/envs/vllm-test

[ -f "$SCORE_SCRIPT" ] || { echo "FAILED: $SCORE_SCRIPT not found (submit from generation/)"; exit 1; }
[ -d "$SLURM_SUBMIT_DIR/$PROMPTS_DIR" ] || { echo "FAILED: $PROMPTS_DIR/ not found in $SLURM_SUBMIT_DIR"; exit 1; }
echo "judge=$JUDGE  keys=${KEYS[*]}  db=$DB_FILE  submit_dir=$SLURM_SUBMIT_DIR"
echo "prompts_dir=$PROMPTS_DIR"
echo "metrics (prompt files): $(ls $SLURM_SUBMIT_DIR/$PROMPTS_DIR/*.txt | xargs -n1 basename | tr '\n' ' ')"

# 1. Build / reuse the fixed 50-prompt subset for these keys (fails fast if a generation is missing)
cd "$REPO"
crun -p "$VLLM_ENV" python generation/paper50_select.py --keys "${KEYS[@]}" || { echo "FAILED: subset selection"; exit 1; }
cd "$SLURM_SUBMIT_DIR"

INPUT_FILES=()
for k in "${KEYS[@]}"; do INPUT_FILES+=("$REPO/generation/outputs/paper50/$k/judge_examples_paper50.json"); done

# 2. Judge server
crun -p "$VLLM_ENV" env PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" --served-model-name "$SERVED_NAME" \
    --dtype bfloat16 --max-model-len 8192 --enforce-eager --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
READY=0
for _ in $(seq 1 420); do
    if curl -s "$SERVER_URL/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then READY=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "Server died during startup - see $SERVER_LOG"; break; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "FAILED: server never ready"; tail -n 40 "$SERVER_LOG"; kill -9 "$SERVER_PID" 2>/dev/null || true; exit 1; fi

# 3. Score (run_id defaults to $SLURM_JOB_ID; to resume after a timeout add --run_id <that id>)
crun -p "$VLLM_ENV" env PYTHONUNBUFFERED=1 JUDGE_MODEL="$SERVED_NAME" \
    python "$SCORE_SCRIPT" \
    --input "${INPUT_FILES[@]}" \
    --models "${KEYS[@]}" \
    --backend vllm --vllm_url "$SERVER_URL" \
    --db_file "$DB_FILE" \
    --prompts_dir "$PROMPTS_DIR" \
    --skip_bertscore
SCORE_EXIT_CODE=$?

kill -9 "$SERVER_PID" 2>/dev/null || true
for pid in $(crun -p "$VLLM_ENV" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done
echo "=== $(date) : job $SLURM_JOB_ID finished, exit $SCORE_EXIT_CODE ==="
exit "$SCORE_EXIT_CODE"

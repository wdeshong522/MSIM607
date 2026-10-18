#!/bin/bash
# submit_paper_judging.sh  (DRAFT) - run from DentalHygieneLLM/generation/
# wave1 (already done): base models + Granite n1000-b32, all 8 prompt files, DB judge_scores_paper50.db
# wave2: the four lr arms, split into 2 jobs per judge (Granite arms / Qwen arms), 7 metrics (no clinical_accuracy),
#        one DB per job to avoid SQLite lock contention.
# Setup once:  cp -r judge_prompts judge_prompts_paper7 && rm judge_prompts_paper7/clinical_accuracy.txt
set -euo pipefail
case "${1:-}" in
  wave1) GROUPS_=("w1|granite-4.1-8B-base qwen3-8B-base granite-4.1-8B-n1000-b32") ; PD=judge_prompts ;;
  wave2) GROUPS_=("w2g|granite-4.1-8B-nfull-lr2e-4 granite-4.1-8B-nfull-lr5e-5" "w2q|qwen3-8B-nfull-lr2e-4 qwen3-8B-nfull-lr5e-5") ; PD=judge_prompts_paper7 ;;
  *) echo "usage: bash submit_paper_judging.sh wave1|wave2"; exit 1 ;;
esac
[ -d "$PD" ] || { echo "ERROR: $PD/ missing. Run: cp -r judge_prompts judge_prompts_paper7 && rm judge_prompts_paper7/clinical_accuracy.txt"; exit 1; }
[ ! -f "$PD/clinical_accuracy.txt" ] || [ "$PD" = judge_prompts ] || { echo "ERROR: $PD still contains clinical_accuracy.txt"; exit 1; }
for g in "${GROUPS_[@]}"; do
  tag="${g%%|*}"; read -r -a KEYS <<< "${g#*|}"
  for j in qwen gemma; do
    db="/home/wdesh001/DentalHygieneLLM/generation/outputs/judge_scores_paper50_${tag}_${j}.db"
    out=$(sbatch --job-name="judge50-$tag-$j" --export=ALL,PROMPTS_DIR="$PD",DB_FILE="$db" score_paper50.sbatch "$j" "${KEYS[@]}")
    echo "[$tag $j] ${KEYS[*]} -> $out  (db: $(basename $db))"
  done
done
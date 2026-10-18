# DentSim-AI LoRA hyperparameter tuning: code for submission

Pipeline order (run from the repo root on the cluster; Slurm + vLLM + crun environment):

1. Data: `data/` (4,000 synthetic ShareGPT records, ~11 MB). `split_utils.py` builds the frozen 200-example holdout and 3,800 train pool; the saved split is in `split/`.
2. Sweep (1 GPU): `training/{rank,lr,lr_extend,target_modules,alpha,epochs}_sweep*.sh` call `training/lora_finetune.py` with `config/training_sweep/*_baseline|stage2..5*.yaml`; `training/sweep_summary.py` summarizes held-out loss.
3. Production (4 GPU, Ray DDP): `training/submit_overnight_runs.sh` and `training/submit_paper_arms.sh` submit `training/train_one_condition.sbatch`, which runs `training/lora_finetune_ray.py` with `config/training_sweep/*_final.yaml`.
4. Generation (vLLM): `generation/submit_paper_generation.sh` -> `generate_one_condition.sbatch` / `generate_base_condition.sbatch` -> `generate_judge_examples.py` (settings in `config/vllm_models.yaml`, `config/study_config.yaml`, `generation/study_config.py`).
5. Judging: `generation/submit_paper_judging.sh` -> `score_paper50.sbatch` -> `paper50_select.py` (50-prompt sample, seed 12345) and `score_judge_examples.py` with `generation/judge_prompts/*.txt` (the paper uses 7 metrics; `clinical_accuracy.txt` was dropped, see submit_paper_judging.sh). The earlier 200-prompt runs used `score_judge_full_run*.sbatch`.
6. Tables: `analysis/build_paper_tables.py` (means, bootstrap CIs, paired deltas), `analysis/top_level_counts.py`, `analysis/truncated_counts.py`.
7. Figures: `figures_code/make_fig*.py`.

Not included: model weights/adapters, Slurm logs, judge score databases (`judge_scores_*.db`), generated responses.

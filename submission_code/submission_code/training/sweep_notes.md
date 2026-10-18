# DentSim LoRA Hyperparameter Sweep — Results & Notes

## Setup

- **Models:** Granite-4.1-8B (`ibm-granite/granite-4.1-8b`), Qwen3-8B (`Qwen/Qwen3-8B-Base`)
- **Hardware:** ODU HPC Turing cluster, `MSIM715` partition, V100 16GB GPUs (environment also validated on Wahab V100 nodes)
- **Data:** 1,880 examples total (1,692 train / 188 held-out eval, seed=42 split)
- **Screening method:** staged single-variable sweep — each stage varies one hyperparameter, holding the rest at the current baseline, screened via held-out per-example loss (mean ± SEM) rather than the full 200-example gen+judge pipeline
- **Tie rule:** a candidate is treated as tying the best when `candidate.mean_loss - candidate.sem <= best.mean_loss + best.sem` (±1 SEM band overlap)
- **Fixed baseline (unless a stage is testing it):** epochs=3, batch_size=1, grad_accum=8, lr=2e-4, lora_alpha=32, lora_dropout=0.05, max_seq_len=2048, target_modules = attention+MLP (q,k,v,o,gate,up,down)
- **Output convention:** each stage gets its own config file per model (`config/training_sweep/<model>_stageN_<param>.yaml`) with `output_dir` scoped to `outputs/sweep/stageN_<param>/` — keeps `sweep_summary.py --root` clean per stage instead of mixing every stage's swept parameter into one flat listing. Stage 1's results were moved into `outputs/sweep/stage1_rank/` after the fact, once this convention was adopted.

---

## Stage 1 — LoRA Rank

Candidates: r = 8, 16, 32, 64 (alpha held fixed at 32)

### Granite-4.1-8B

| rank | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 8 | 1.0661 | 0.0113 | 188 | |
| 16 | 1.0616 | 0.0114 | 188 | tie w/ best |
| 32 | 1.0603 | 0.0115 | 188 | BEST (nominal) |
| 64 | 1.0610 | 0.0114 | 188 | tie w/ best |

### Qwen3-8B

| rank | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 8 | 1.1553 | 0.0106 | 188 | tie w/ best |
| 16 | 1.1537 | 0.0106 | 188 | tie w/ best |
| 32 | 1.1525 | 0.0106 | 188 | BEST (nominal) |
| 64 | — | — | — | **OOM on V100 16GB** — 174.6M trainable params (2.09%), too large to fit; not retried |

### Decision

**Rank = 16** (alpha = 32) selected as the shared setting going into Stage 2. All ranks tie within SEM for both models — rank has little measurable effect on held-out loss at this data scale. 16 ties the nominal best for both models, keeps clear margin under Qwen's OOM ceiling, and keeps adapter size/VRAM lower for later stages at no measured cost.

### Notes

- Flat rank sensitivity across 8→64 suggests LoRA capacity isn't the current bottleneck — data quality/quantity may matter more than capacity at this scale.
- **Environment fix along the way:** V100 lacks native BF16 tensor cores, but `torch.cuda.is_bf16_supported()` returns `True` anyway by default (it permits a slow fp32-emulated fallback). Switched precision selection in `lora_finetune.py` to check compute capability directly (`_gpu_supports_fast_bf16()`, requires CC≥8/Ampere for real bf16 speed). Fixed a ~4-5x training slowdown (33 s/it → 7 s/it on V100).
- Qwen3-8B's LoRA layers are larger per unit of rank than Granite's — rank 64 hit 174.6M trainable params on Qwen vs. Granite's presumably much smaller footprint at the same rank. Worth remembering if later stages also sweep rank-adjacent settings.

---

## Stage 2 — Learning Rate

Candidates: lr = 5e-5, 1e-4, 2e-4, 3e-4 (original range); extended to 5e-4, 7e-4, 1e-3 once 3e-4 (the highest originally tested) came out nominal best for both models, to check whether the true optimum sat above the tested range.

### Qwen3-8B

| lr | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 5e-05 | 1.2235 | 0.0107 | 188 | |
| 1e-04 | 1.1906 | 0.0108 | 188 | |
| 2e-04 | 1.1530 | 0.0106 | 188 | |
| 3e-04 | 1.1353 | 0.0107 | 188 | tie w/ best |
| 5e-04 | 1.1296 | 0.0109 | 188 | BEST (nominal) |
| 7e-04 | 1.1315 | 0.0112 | 188 | tie w/ best |
| 1e-03 | 1.1470 | 0.0115 | 188 | tie w/ best |

### Granite-4.1-8B

| lr | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 5e-05 | 1.1309 | 0.0112 | 188 | |
| 1e-04 | 1.0951 | 0.0111 | 188 | |
| 2e-04 | 1.0612 | 0.0115 | 188 | tie w/ best |
| 3e-04 | 1.0548 | 0.0117 | 188 | BEST (nominal) |
| 5e-04 | 1.0611 | 0.0120 | 188 | tie w/ best |
| 7e-04 | 1.3395 | 0.0122 | 188 | clearly worse, not a tie |
| 1e-03 | — | — | — | **diverged** — loss exploded into the thousands/millions, `grad_norm: nan`, `eval_loss: nan`; excluded as unstable, not a valid comparison point |

### Decision

**lr = 3e-4** for both models. Ties as best (or near-best) for both, and — importantly — sits comfortably inside Granite's confirmed-stable range rather than near the 5e-4/7e-4 boundary where Granite starts to degrade and eventually diverge. Picking a value near an instability edge just because it nominally wins a statistically-insignificant tie would be a weaker choice than one both models handle without issue.

### Notes

- Lower learning rates (5e-5, 1e-4) show real, non-tied separation from the rest for both models — unlike Stage 1's rank sweep, LR clearly matters at this data scale.
- **Granite-specific instability above ~5e-4**: 7e-4 is clearly worse (not a SEM tie) and 1e-3 diverges outright. Qwen tolerates the same values fine (7e-4 ties with best, 1e-3 ties with best) — this looks like a genuine per-model difference in LR tolerance, not a property of the LR value itself. Worth remembering if any later stage revisits LR-adjacent settings.
- **Bug found and fixed along the way:** `compute_per_example_eval_loss()` crashed on NaN losses (Python's `statistics.stdev()` can't handle NaN — its exact-fraction internals raise `AttributeError`), which destroyed the whole run's eval file rather than recording the divergence. Rewrote the calc using plain float arithmetic (NaN propagates harmlessly through ordinary math) so a diverged candidate now shows up as a visible NaN result instead of vanishing via a crash. Not re-run for Granite's lr=1e-3, since the training log alone (exploding loss, NaN grad_norm) was already conclusive — re-running just to get a formally-recorded NaN wasn't worth ~86 more minutes of GPU time.

---

## Training Timing Reference

Per-candidate wall-clock time on Turing's MSIM715 V100s, 633 steps each (1,692 examples, epochs=3, batch_size=1, grad_accum=8). Useful for estimating how long later stages will take and how aggressively to parallelize.

| Stage | Granite runtime | Qwen runtime |
|---|---|---|
| Stage 1 (rank) | 87.5–92.4 min (avg 89.5) | 80.4–83.3 min (avg 82.3) |
| Stage 2 (lr, original 4) | 87.5–89.7 min (avg 88.3) | 80.4–81.4 min (avg 80.8) |
| Stage 2 (lr, extended) | 86.0–87.4 min (avg 86.6)* | 80.4–95.8 min (avg 88.5) |

*excludes lr=1e-3's divergence run itself (86.0 min) — comparable runtime despite diverging, since step count/cost don't depend on loss behavior.

**Consistent pattern:** Qwen trains ~8-10% faster than Granite per candidate across every stage so far (80-83 min vs. 87-92 min) — likely architectural (Qwen3's layer structure being cheaper per forward/backward pass), not anything setup-specific, given the gap held steady across two independent stages.

**One outlier:** Qwen's lr=1e-3 (extended sweep) ran 95.8 min, well outside Qwen's usual band, despite scoring normally (no divergence). Most likely GPU contention from another job sharing the node at that moment, not a property of that hyperparameter.

---

## Stage 3 — Target Modules

Candidates: attention-only (q,k,v,o) vs. attention+MLP (q,k,v,o,gate,up,down) — lr=3e-4, rank=16 carried forward

### Qwen3-8B

| target modules | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| attn+MLP (default) | 1.1359 | 0.0108 | 188 | BEST |
| attn only | 1.2074 | 0.0107 | 188 | clearly worse, not a tie |

### Granite-4.1-8B

| target modules | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| attn+MLP (default) | 1.0534 | 0.0117 | 188 | BEST |
| attn only | 1.1217 | 0.0113 | 188 | clearly worse, not a tie |

### Decision

**Attention+MLP** (the default already used in every prior stage) — no change. Clear, non-tied win for both models; attention-only measurably hurts performance. Confirms the default used throughout Stages 1-2 was the right call rather than an untested assumption.

### Notes

- `lora_finetune.py` had no way to select attention-only before this stage — added a `--attn_only` CLI/config flag (defaults to `False`, i.e. attention+MLP, matching every prior stage's implicit setting).
- Unlike Stage 1 (rank), this stage showed real, non-tied separation — target-module selection clearly matters more than rank did at this data scale.

---

## Stage 4 — Alpha Ratio

Candidates: alpha = 16 (1×rank), 32 (2×rank), 64 (4×rank) — rank=16, lr=3e-4, attn+MLP carried forward

### Qwen3-8B

| alpha | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 16 | 1.1537 | 0.0106 | 188 | clearly worse, not a tie |
| 32 | 1.1359 | 0.0108 | 188 | tie w/ best |
| 64 | 1.1271 | 0.0109 | 188 | BEST (nominal) |

### Granite-4.1-8B

| alpha | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 16 | 1.0642 | 0.0114 | 188 | tie w/ best |
| 32 | 1.0534 | 0.0117 | 188 | BEST (nominal) |
| 64 | 1.0589 | 0.0120 | 188 | tie w/ best |

### Decision

**alpha = 32** (2×rank, the standard heuristic ratio, already used throughout every prior stage) — no change. Ties as effectively equivalent to the nominal best for both models. Unlike Stage 2's LR sweep, this doesn't read as an under-tested boundary case: alpha=16 is clearly worse for both models (not a tie), but 32 and 64 sit close together and tied — a plateau above 16 rather than a trend still climbing, so no need to test alpha=128.

### Notes

- alpha=32 not retrained for this stage — reused directly from Stage 3's attn+MLP baseline run (identical config), saving ~85-90 min of GPU time per model.

---

## Stage 5 — Epochs

Candidates: 2, 3, 4 — rank=16, lr=3e-4, alpha=32, attn+MLP carried forward

### Qwen3-8B

| epochs | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 2 | 1.1528 | 0.0107 | 188 | tie w/ best |
| 3 | 1.1359 | 0.0108 | 188 | BEST (nominal) |
| 4 | 1.1408 | 0.0107 | 188 | tie w/ best |

### Granite-4.1-8B

| epochs | mean loss | SEM | n | verdict |
|---|---|---|---|---|
| 2 | 1.0649 | 0.0115 | 188 | tie w/ best |
| 3 | 1.0534 | 0.0117 | 188 | BEST (nominal) |
| 4 | 1.0608 | 0.0117 | 188 | tie w/ best |

### Decision

**epochs = 3** — no change from baseline. Ties as best for both models across the whole 2-4 range; no meaningful improvement from training longer, and no overfitting penalty from epochs=4 either (a real possibility flagged going in, but not observed).

### Notes

- epochs=3 not retrained — reused directly from Stage 4's alpha=32 run.
- Like Stage 1 (rank), this stage showed no meaningful separation across candidates — epochs, like rank, doesn't appear to be the bottleneck at this data scale.

---

## Stage 6 — Trust-Tier λ (not run)

**Concept:** the training corpus mixes validated examples (confirmed correct) with unvalidated ones (not yet reviewed). λ down-weights the loss contribution of unvalidated examples relative to validated ones (validated=1.0, unvalidated=λ), so the model leans less heavily on content that hasn't been confirmed — aimed at reducing confident fabrication on exactly the kind of content most likely to contain it.

**Why it's skipped for now:** unlike Stages 1-5, this can't be screened via held-out loss alone — the effect λ is meant to produce (less confident fabrication) isn't reliably visible in perplexity. It needs real generation + judge scoring against a probe set (the automated LLM-judge pipeline already used for model comparison, not human expert review) — infrastructure this sweep didn't build. Revisit once that pipeline is otherwise in active use.

---

## Comparison to Original Baseline

The only hyperparameter that actually changed across this entire 5-stage sweep is **learning rate: 2e-4 → 3e-4**. Rank (16), alpha (32), epochs (3), dropout (0.05), and target modules (attention+MLP) all came back exactly matching what `run_training.py`'s original defaults already used — the sweep mostly *validated* the original choices rather than overturning them.

**Expectation for judge scores:** modest, not dramatic. The loss improvement from the LR change alone was small in absolute terms — roughly 0.7% relative for Granite (1.0612 → 1.0534), about 1.5% for Qwen (1.1530 → 1.1359). Held-out loss and downstream judge-scored metrics (correctness, hallucination rate, etc.) aren't linearly related, and judge scores carry their own measurement noise — a change this size in perplexity could translate to a barely-detectable shift in judge scores, not a clear jump. Worth treating this as "the LR was already close to right" rather than "we found a lever that meaningfully moves the needle."

**Granite vs. Qwen — no winner declared, and this sweep doesn't produce one.** Granite's held-out loss is consistently lower than Qwen's at every matched configuration (~1.05 vs. ~1.14), but loss values aren't directly comparable across models with different tokenizers/vocabularies in any strict sense — lower loss for Granite doesn't guarantee better judge-scored generation quality. The original SLM comparison table (correctness/completeness/relevance/faithfulness/hallucination-rate/BERTScore, from the very start of this conversation) is the actual signal for that question, not this sweep's per-example loss numbers. Since the study's stated purpose is comparing models' judge-validity rather than picking one winner, both stay in the roster.

**The bigger lever is probably still data, not hyperparameters.** Both Stage 1 (rank) and Stage 5 (epochs) showed flat, no-meaningful-difference results — the two hyperparameters most directly tied to model *capacity* and training *duration* didn't matter much. That's consistent with the concern raised back when this sweep started: data quality and quantity (the flagged fabrication risks, the underweighted domains like Oral Cancer Screening and Radiographic Interpretation) are more likely to move judge scores than further hyperparameter tuning at this same 1,880-example corpus size. If the original plan to scale up to ~4,000 examples across the rebalanced domain targets goes forward, that's the change more likely to produce a real, visible improvement in judge scores than this LR tweak.

### Gap to Target (Sept 11 baseline, pre-sweep fine-tuned scores)

| Metric | Target | Granite (FT) | Qwen (FT) | Gap |
|---|---|---|---|---|
| Correctness | ≥0.8 | 0.690 | 0.680 | ~0.11-0.12 needed |
| Completeness | ≥0.8 | 0.676 | 0.682 | ~0.12 needed |
| Relevance | ≥0.8 | 0.965 | 0.956 | met |
| Faithfulness | ≥0.8 | 0.775 | 0.798 | ~met (Granite close, Qwen essentially there) |
| Hallucination rate | ≤0.2 | 0.335 | 0.293 | ~0.09-0.14 reduction needed |
| BERTScore F1 | ≥0.8 | 0.892 | 0.887 | met |

3 of 6 metrics already clear target. The real gap is correctness, completeness, and hallucination rate — all tied to factual accuracy/knowledge coverage rather than fluency or relevance, which is why data (quantity + quality) is expected to matter more here than further hyperparameter tuning.

---

## Final Selected Config (Stages 1-5)

Both models converged on **identical** hyperparameters — no per-model divergence needed:

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | attention + MLP (q,k,v,o,gate,up,down) |
| Learning rate | 3e-4 |
| Epochs | 3 |
| LoRA dropout | 0.05 (never swept — inherited from original baseline) |
| Batch size / grad accum | 1 / 8 (never swept) |

---

## Remaining Steps

1. ~~Generate the expanded/rebalanced training set~~ **Done** — ~4,000 examples (3,995 valid), rebalanced per-domain targets.
2. ~~Retrain both models~~ **Done** — see "Final Training Run" above. Held-out loss improved ~10-12% relative for both models.
3. **Generate + judge a 30-50 example probe** against the new held-out set, with `think=True` (matching the mode that produced the original 0.8/0.2 targets, so the comparison is apples-to-apples — see reasoning below). **In progress on laptop (Sept 16), 30 examples, started before the split-consistency fix below landed — treat this run's results as a rough first look, not the trustworthy final answer, until that's fixed.**
4. **Check against targets** (0.8 per metric, 0.2 for hallucination). Already-met metrics (relevance, BERTScore F1, faithfulness) just need to hold; correctness, completeness, and hallucination rate are the real gap (see gap analysis from Sept 11 baseline — Granite/Qwen both ~0.68-0.69 correctness, ~0.68 completeness, ~0.29-0.34 hallucination rate).
5. **If targets are met:** done, current config + new data is the answer.
6. **If not met:** implement the Stage 6 trust-tier λ sweep (needs the gen+judge pipeline from step 3 anyway, so this reuses that infrastructure rather than starting fresh).

**Why `think=True` for the step-3 probe specifically:** the original baseline that set the 0.8/0.2 targets was scored with `think=True`. Comparing a `think=False` probe against those targets would be comparing two different measurement instruments — if reasoning mode changes score distributions and not just speed, a near-threshold result could read differently depending on which mode judged it. `think=False` remains the right choice for *relative* comparisons (ranking Stage 6 λ candidates against each other), just not for this *absolute* comparison against an externally-set target.

**Longer-term vision (not now):** an iterative generate→judge→retrain loop. Feasible as an orchestration pattern, but only safely if judged model outputs don't get fed back into training as trusted data automatically — that risks model collapse, compounding hallucinations the judge fails to catch rather than reducing them. The safer version ties back to Stage 6's own mechanism: model-generated examples can re-enter training, but as unvalidated (low λ weight) content requiring expert review before being trusted, not equal-weight retraining data. Worth building only once Stage 6 itself is in place, since it's the same infrastructure either way.

**Dependency worth checking before assuming parallelization at the generate/judge step:** the pipeline is built around Ollama, which hasn't been tested on Turing/Wahab's containerized environment yet. Given everything else about this HPC setup has surprised us at least once (Python version, precision handling, `crun`'s sandboxing), a quick smoke test — does Ollama even start and serve a model on a compute node here — is worth doing before planning around example-level parallelization of generation/judging.

---

## Generation & Judging Timing Reference

Placeholder — fill in once generation/judging runs begin. Per your original estimates: ~1.5 hr/model to generate (30-45s/example), ~7-8 hr/model to judge with `think=True` (90-120s/example), roughly half-to-third that with `think=False` for screening passes. Track actual per-run times here the same way as the training table above once real numbers are in hand.

| Run | Model | think mode | n examples | Generate time | Judge time |
|---|---|---|---|---|---|
| | | | | | |

---

## Final Training Run (Full ~4,000-Example Dataset)

Settled Stage 1-5 hyperparameters, no sweeping — one production run per model on the expanded/rebalanced corpus (3,995 valid examples after 5 malformed lines were skipped automatically; see "Data Quality Check" below).

| Model | Sweep baseline (1,880 ex, n=188) | Final (≈4,000 ex, n=400) | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| Granite-4.1-8B | 1.0534 (SEM 0.0117) | 0.9262 (SEM 0.0089) | −0.1272 | −12.1% |
| Qwen3-8B | 1.1359 (SEM 0.0106) | 1.0212 (SEM 0.0083) | −0.1147 | −10.1% |

Both improvements are many multiples of either measurement's SEM — a real effect, not noise. This confirms the hypothesis raised throughout Stages 1 and 5 (flat rank/epoch sensitivity): the added and rebalanced training data was the actual lever, not further hyperparameter tuning — a ~10-12% relative loss reduction here dwarfs the ~1-2% the LR change alone produced in Stage 2.

**Caveat carried over from earlier:** this compares two different held-out sets (188 examples from the old 1,880-example corpus vs. 400 from the new ~4,000), not one fixed eval set — the sweep's own SEM-overlap tie rule wasn't designed for this comparison. Treat this as "clearly better, likely by a wide margin," not a formally rigorous statistical claim.

**Next:** the 30-50 example generate+judge probe (with `think=True`) against the actual 0.8/0.2 correctness/completeness/hallucination targets — held-out loss improving doesn't guarantee those judge-scored metrics move by a proportional amount.

### Data Quality Check (pre-training)

`domain_summary.py` found 5/4005 lines (0.1%) malformed across the full corpus (Domain 1: 1, Domain 6: 1, Domain 10: 3) — `repair_jsonl.py` was run first but its rejoin-based repair logic doesn't apply to what was actually broken (unescaped inner quotes/apostrophes in string values, not the raw-newline-split case it was built for), and has its own bug: when a broken line's rejoin attempt never succeeds within 15 lines, it silently discards the *entire* accumulated buffer as "1" unrecoverable record — which can eat genuinely valid lines caught in the same failed buffer. Confirmed happening once (`dental_hygiene_domain10_batch15.jsonl`: 2 real broken lines 15 apart triggered a 16-line swallow). Not fixed in `repair_jsonl.py` itself — not worth the effort at this data scale. Training itself was never at risk either way: `lora_finetune.py`'s loader parses lines independently with no buffering, so it correctly skipped only the 5 genuinely malformed lines and used the rest.

### Checkpoint Resume — Learnings

`MSIM715`'s `MaxTime=03:00:00` is below the ~3.0-3.25 hr the full-dataset runs need (1,350 steps vs. the sweep's 633) — Granite's first attempt hit this and got killed. Added auto-resume to `lora_finetune.py` (finds the latest `checkpoint-*` in `output_dir`, passes it to `trainer.train(resume_from_checkpoint=...)`), plus `--requeue` on the array script for automatic resubmission — though `--requeue` didn't actually fire when tested (likely disabled at the `msim715` QoS level; manual resubmission via a plain `sbatch` of the same config is the reliable path here).

**Resume itself needed two follow-up fixes**, both stemming from PyTorch's `weights_only=True` default (a security measure) blocking `Trainer`'s internal RNG-state restore:
1. First attempt: allowlisting the specific missing global (`numpy.core.multiarray._reconstruct`) just revealed a second one (`_codecs.encode`) on retry — allowlisting individual globals for a legacy numpy pickle is whack-a-mole.
2. Fix: monkeypatch `torch.load` to force `weights_only=False` for the rest of the process, since the file being loaded is always our own just-generated checkpoint (not an untrusted source — the actual scenario `weights_only` protects against). First attempt at this used `kwargs.setdefault(...)`, which silently failed to override `Trainer`'s explicit `weights_only=True` call — needed `kwargs["weights_only"] = False` (force-override) instead.

Worth remembering for any future long run on this partition that needs a resume — this will recur.

---

## Known Issues / Pending Fixes

### Eval-set split consistency (NOT YET FIXED — affects all judge scoring, including the Sept 11 baseline)

`lora_finetune.py` splits data via `dataset.train_test_split(test_size=0.10, seed=42)` (the `datasets` library's own numpy-backed shuffle). `generate_judge_examples.py`'s `load_eval_examples()` does its own independent split instead — `random.Random(seed).shuffle(indices)`, Python's built-in `random` module. Same seed value, same sorted file order, but **different PRNG implementations produce different permutations** — there's no guarantee (and good reason to doubt) these two scripts select the same 10% of records as "held out." `load_eval_examples()`'s record filtering is also looser than `build_dataset()`'s (no ≥2-message / chat-template-validity check), so even the total count feeding each split likely differs.

**Real consequence:** some fraction of what `generate_judge_examples.py` treats as "held-out" for judging may actually have been in the model's training set — meaning the model already saw the ground-truth answer for those examples. This would inflate correctness/completeness scores in a way unrelated to real generalization. This isn't new to the current 30-example probe — it's been true since the original Sept 11 baseline scoring.

**Fix needed:** make `generate_judge_examples.py` derive its held-out set via the exact same mechanism `lora_finetune.py` uses (same `datasets.Dataset.train_test_split(seed=42)` call, same pre-split filtering), rather than its own hand-rolled shuffle.

**Status:** blocked on getting the complete `generate_judge_examples.py` file (only fragments available via project-knowledge search so far — this is exactly the kind of precise, correctness-critical edit not worth guessing at from partial context).

---

## Data Parallelism (Future Work — Not Yet Run)

Distinct from the task parallelism used throughout this sweep (independent jobs, each processing the full dataset on its own single GPU). Data parallelism splits ONE training run's batches across multiple GPUs simultaneously (each GPU holds a full model copy, processes a slice of the batch, gradients sync via DDP) — reduces wall-clock time of a single run rather than running more candidates side-by-side. Directly relevant to the checkpoint-resume pain from the final training run: DDP across 4 GPUs would very plausibly bring a ~3.2 hr single-GPU run under an hour, likely avoiding `MSIM715`'s 3-hour cap entirely for any future full-dataset retrain.

**Code changes made to `lora_finetune.py` (untested on real multi-GPU hardware so far):**
- `load_base_model()`: replaced `device_map="auto"` (shards one model across all visible GPUs within a process — model/pipeline parallelism, conflicts with DDP) with `device_map={"": PartialState().process_index}` (pins each process to its own single GPU — HF's documented QLoRA+DDP pattern). No-op on single-GPU runs.
- `SFTConfig`: added `ddp_find_unused_parameters=False` (required for LoRA under DDP — most base-model params are frozen, and DDP's default unused-parameter detection is slow/broken with mostly-frozen PEFT models) and `gradient_checkpointing_kwargs={"use_reentrant": False}` (avoids a known gradient-checkpointing + DDP deadlock). Both no-ops on single-GPU runs.

**Critical methodological point — effective batch size must be preserved:** effective batch = `num_gpus × batch_size × grad_accum`. Stages 1-5 validated lr=3e-4 at effective batch=8 (1×1×8, single GPU). Scaling GPU count without compensating `grad_accum` down changes the effective batch silently — e.g. 4 GPUs with grad_accum still at 8 = effective batch 32, no longer the batch size the LR was tuned for. Must scale `grad_accum` inversely with GPU count to keep effective batch at 8:

| GPUs | batch_size | grad_accum | effective batch |
|---|---|---|---|
| 1 (current) | 1 | 8 | 8 |
| 2 | 1 | 4 | 8 |
| 4 | 1 | 2 | 8 |

**Smoke test written but not yet run:** `run_ddp_smoke_test.sh` — 2-GPU `torchrun` launch against the same tiny placeholder model/synthetic data as `run_smoke_sweep.py`, to validate the DDP mechanics (device pinning, no hangs from gradient-checkpointing+DDP interaction) before trusting a real production run to it. **Deliberately held off running this** to prioritize the generate+judge probe first — DDP is a future training-speed improvement, not something blocking the current research question.

---

## Open Items

- Qwen3-8B rank 64 never retried after OOM — treated as excluded for hardware reasons, not a performance finding.
- Once Stages 1–5 converge on a config per model, run that config through the full 200-example gen+judge pass for the numbers that actually get reported (per the original Phase C plan) — the held-out loss screening here is for ranking candidates against each other, not for final reported metrics.
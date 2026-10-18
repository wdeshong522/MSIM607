"""
sweep_summary.py
=================
Walks a directory tree looking for run folders that contain both
effective_config.json (written by lora_finetune.py's save_effective_config())
and eval_loss_per_example.json (written by compute_per_example_eval_loss()),
and builds a comparison table across all of them — one row per sweep
candidate, grouped by model.

Which hyperparameter is "being swept" isn't hardcoded — for each model
group, the script finds whichever config keys actually differ across that
group's runs and uses those as the candidate label. That means the same
script works for the rank sweep, the LR sweep, the target-modules sweep,
etc. without changes.

Candidates are sorted by mean held-out loss (lowest first) within each
model group and flagged as a statistical tie with the current best when
their +/-1 SEM bands overlap, rather than picking a "winner" on a gap
that's within noise:
    tie  if  candidate.mean_loss - candidate.sem <= best.mean_loss + best.sem

Usage:
    python sweep_summary.py --root ./outputs/sweep
    python sweep_summary.py --root ./outputs/sweep --save_csv
"""

import argparse
import csv
import json
from pathlib import Path

# Keys that always differ (bookkeeping) or never matter for labeling a
# candidate — excluded when detecting which hyperparameter is being swept.
IGNORE_KEYS = {
    "run_dir", "output_dir", "config_file", "data_dir", "model_name",
    "mean_loss", "sem", "n_examples", "chat_template",
}


def find_runs(root: Path) -> tuple[list[dict], list[dict]]:
    """
    Find every subdirectory under root containing effective_config.json.
    Returns (completed_runs, skipped_runs) — skipped_runs are candidates
    that started (have a config) but never produced eval_loss_per_example.json,
    e.g. an OOM or other crash mid-training. Config is included for skipped
    runs too, so you can see which candidate needs attention.
    """
    completed = []
    skipped = []
    for config_path in sorted(root.rglob("effective_config.json")):
        run_dir = config_path.parent
        loss_path = run_dir / "eval_loss_per_example.json"

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        if not loss_path.exists():
            skipped.append({"run_dir": str(run_dir), **config})
            continue

        with open(loss_path, encoding="utf-8") as f:
            loss = json.load(f)

        completed.append({
            "run_dir": str(run_dir),
            **config,
            "mean_loss": loss["mean_loss"],
            "sem": loss["sem"],
            "n_examples": loss["n_examples"],
        })
    return completed, skipped


def detect_swept_keys(rows: list[dict]) -> list[str]:
    """Within one model's group of rows, find which config keys differ."""
    if not rows:
        return []
    keys = [k for k in rows[0].keys() if k not in IGNORE_KEYS]
    swept = []
    for k in keys:
        values = {r.get(k) for r in rows}
        if len(values) > 1:
            swept.append(k)
    return swept


def build_skipped_section(skipped: list[dict]) -> str:
    if not skipped:
        return ""
    lines = ["", "=" * 78, f"INCOMPLETE — {len(skipped)} candidate(s) started but have no eval_loss_per_example.json", "(check the corresponding job log for a crash/OOM — this doesn't mean the run is still in progress)", "=" * 78]
    for r in skipped:
        model = r.get("model_name", "?")
        detail_keys = [k for k in r if k not in {"run_dir", "output_dir", "config_file", "data_dir", "model_name", "chat_template"}]
        detail = ", ".join(f"{k}={r[k]}" for k in detail_keys)
        lines.append(f"  {model} ({detail}) — {r['run_dir']}")
    return "\n".join(lines)


def build_table(runs: list[dict]) -> str:
    if not runs:
        return "No completed runs found."

    by_model: dict[str, list[dict]] = {}
    for r in runs:
        by_model.setdefault(r["model_name"], []).append(r)

    lines = []
    for model_name, rows in sorted(by_model.items()):
        rows = sorted(rows, key=lambda r: r["mean_loss"])
        best = rows[0]
        swept_keys = detect_swept_keys(rows) or ["(no differing hyperparameters — check output_dir uniqueness)"]

        lines.append("=" * 78)
        lines.append(f"MODEL: {model_name}")
        lines.append(f"Swept parameter(s): {', '.join(swept_keys)}")
        lines.append("=" * 78)
        header = f"{'Candidate':<42} {'Mean Loss':>10} {'SEM':>8} {'n':>5}  {'Verdict'}"
        lines.append(header)
        lines.append("-" * len(header))

        for r in rows:
            candidate_label = ", ".join(
                f"{k}={r.get(k)}" for k in swept_keys if k in r
            ) or Path(r["run_dir"]).name

            if r is best:
                verdict = "BEST"
            else:
                overlaps = (r["mean_loss"] - r["sem"]) <= (best["mean_loss"] + best["sem"])
                verdict = "tie w/ best (SEM overlap)" if overlaps else ""

            lines.append(
                f"{candidate_label:<42} {r['mean_loss']:>10.4f} {r['sem']:>8.4f} "
                f"{r['n_examples']:>5}  {verdict}"
            )
        lines.append("")

    return "\n".join(lines)


def save_csv(runs: list[dict], path: Path):
    if not runs:
        print("No runs to write to CSV.")
        return
    keys = sorted({k for r in runs for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(runs)
    print(f"CSV written to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Summarize lora_finetune.py sweep runs")
    p.add_argument("--root", default="./outputs/sweep",
                   help="Directory to search recursively for run outputs")
    p.add_argument("--save_csv", action="store_true",
                   help="Also write a full CSV of every run's config + loss")
    p.add_argument("--csv_path", default="./outputs/sweep/sweep_summary.csv",
                   help="Where to write the CSV if --save_csv is set")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    completed, skipped = find_runs(root)
    print(f"\nFound {len(completed)} completed run(s) under {root}\n")
    print(build_table(completed))
    print(build_skipped_section(skipped))

    if args.save_csv:
        save_csv(completed, Path(args.csv_path))
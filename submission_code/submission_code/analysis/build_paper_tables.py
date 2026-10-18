"""
build_paper_tables.py  (DRAFT)
Merges judge scores for the paper's conditions on the fixed 50-prompt subset and writes tables
with bootstrap 95% CIs (resampling PROMPTS) and paired differences.

Sources (read-only):
  generation/outputs/judge_scores_all.db            old 6 conditions x 200 prompts (judges: qwen3-27b-judge, gemma4-31b-judge)
  generation/outputs/judge_scores_paper50*.db       new conditions (base, n1000-b32, lr arms), 50 prompts
Prompt ids: generation/outputs/paper50/paper50_ids.json  (seed 12345)
Metrics used (7): correctness, completeness, faithfulness, relevance, instruction_following, safety, hallucination
  (clinical_accuracy and BERTScore are excluded). hallucination is a 0/1 flag rate: LOWER is better.
Missing scores (judge reply truncated before the score) are skipped, never treated as 0.

Usage (repo root):  python3 build_paper_tables.py [--out generation/outputs/paper_tables]
"""
import argparse, collections, csv, glob, json, os, random, sqlite3, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="generation/outputs")
ap.add_argument("--out", default="generation/outputs/paper_tables")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=12345)
a = ap.parse_args()

JUDGES = {"qwen3-27b-judge": "Qwen3.8-27B", "gemma4-31b-judge": "Gemma-4-31B-it"}
QUALITY = ["correctness", "completeness", "faithfulness", "relevance", "instruction_following", "safety"]
METRICS = QUALITY + ["hallucination"]
COLS = METRICS

# (source label) -> display name, in display order
ORDER = [
 ("granite-4.1-8B-base",        "Granite / base (untuned)"),
 ("granite-4.1-8B-n1000-b32",   "Granite / tuned n=1,000"),
 ("granite-4.1-8B-n2000",       "Granite / tuned n=2,000"),
 ("granite-4.1-8B-nfull",       "Granite / tuned n=3,800 (lr 3e-4)"),
 ("granite-4.1-8B-nfull-lr2e-4","Granite / n=3,800, lr 2e-4"),
 ("granite-4.1-8B-nfull-lr5e-5","Granite / n=3,800, lr 5e-5"),
 ("qwen3-8B-base",              "Qwen3 / base (untuned)"),
 ("qwen3-8B-n1000",             "Qwen3 / tuned n=1,000"),
 ("qwen3-8B-n2000",             "Qwen3 / tuned n=2,000"),
 ("qwen3-8B-nfull",             "Qwen3 / tuned n=3,800 (lr 3e-4)"),
 ("qwen3-8B-nfull-lr2e-4",      "Qwen3 / n=3,800, lr 2e-4"),
 ("qwen3-8B-nfull-lr5e-5",      "Qwen3 / n=3,800, lr 5e-5"),
]
NAME = dict(ORDER)

ids = set(json.load(open(os.path.join(a.root, "paper50", "paper50_ids.json")))["example_ids"])
assert len(ids) == 50, len(ids)

# data[(judge, cond)][example_id] = {metric: value or None}
data = collections.defaultdict(dict)
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
# audit[judge] = counters over every (condition, prompt, metric) cell actually used
audit = collections.defaultdict(lambda: collections.Counter())
shift = collections.defaultdict(lambda: [0.0, 0.0, 0])  # (judge, metric) -> [sum_raw, sum_rounded, n]
def snap(v, binary):
    """Nearest allowed level; ties go to the higher value (same rule as the arXiv audit)."""
    if binary: return 1.0 if v >= 0.5 else 0.0
    return min(1.0, max(0.0, int(v * 4 + 0.5) / 4.0))
raw = {}            # (judge, cond, example_id) -> (order, cols, values, run_id)
dups = collections.Counter(); dup_disagree = collections.Counter(); dup_runs = collections.defaultdict(set)
_order = [0]
pairs = []   # duplicate (same judge, condition, prompt) pairs, for the test-retest summary
def run_rank(run, order):
    try: return (1, int(run))          # numeric job id: higher = later
    except (TypeError, ValueError): return (0, order)
def load(db, only_conds=None):
    """Collect rows; if a (judge, condition, prompt) appears more than once, keep the LAST row read
    (later database file, then higher rowid) and count the duplicate. Scoring happens in finalize()."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    have = {r[1] for r in con.execute("pragma table_info(results)")}
    cols = [m for m in METRICS if m in have]
    rid = "run_id" if "run_id" in have else "'-'"
    q = f"select judge_model, model_key, example_id, {rid}, {', '.join(cols)} from results order by rowid"
    n = 0
    for row in con.execute(q):
        j, c, ex, run = row[:4]
        if j not in JUDGES or ex not in ids or c not in NAME: continue
        if only_conds and c not in only_conds: continue
        vals = tuple(row[4:]); _order[0] += 1
        key = (j, c, ex)
        if key in raw:
            prev = raw[key]
            dups[(j, c)] += 1; dup_runs[(j, c)].update([prev[3], run])
            if prev[2] != vals: dup_disagree[(j, c)] += 1
            pairs.append((j, c, ex, cols, prev[2], vals, prev[3], run))
            if run_rank(run, _order[0]) < run_rank(prev[3], prev[0]):   # keep the later job (higher run_id)
                continue
        raw[key] = (_order[0], cols, vals, run); n += 1
    con.close(); return n

def finalize():
    for (j, c, ex), (_, cols, vals, run) in raw.items():
        rec = {}
        for m, v in zip(cols, vals):
            audit[j]["cells_expected"] += 1
            if v is None:
                audit[j]["missing"] += 1; rec[m] = None; continue
            binary = (m == "hallucination")
            ok = (v in (0, 1)) if binary else any(abs(v - L) < 1e-9 for L in LEVELS)
            w = v if ok else snap(v, binary)
            if not ok:
                audit[j]["offscale"] += 1; audit[j]["offscale_" + m] += 1
            sh = shift[(j, m)]; sh[0] += v; sh[1] += w; sh[2] += 1
            rec[m] = w
        if any(rec.get(m) is None for m in cols): audit[j]["responses_with_missing"] += 1
        audit[j]["responses"] += 1
        data[(j, c)][ex] = rec

old = os.path.join(a.root, "judge_scores_all.db")
OLD_CONDS = {"granite-4.1-8B-n2000","granite-4.1-8B-nfull","qwen3-8B-n1000","qwen3-8B-n2000","qwen3-8B-nfull"}
# NOTE: old granite-4.1-8B-n1000 (batch 8, best-checkpoint adapter) is intentionally NOT used; the batch-32 run replaces it.
print("old DB rows used:", load(old, OLD_CONDS))
for db in sorted(glob.glob(os.path.join(a.root, "judge_scores_paper50*.db"))):
    print(os.path.basename(db), "rows read:", load(db))
finalize()
if dups:
    print("DUPLICATE rows (same judge, condition, prompt) found; kept the later job (higher run_id):")
    for k, v in sorted(dups.items()):
        print(f"  {JUDGES[k[0]]} / {NAME[k[1]]}: {v} duplicate rows, {dup_disagree[k]} with different scores, run_ids {sorted(map(str, dup_runs[k]))}")

def per_prompt(j, c):
    d = data[(j, c)]
    out = {}
    for ex, r in d.items():
        q = [r[m] for m in QUALITY if r.get(m) is not None]
        row = dict(r); row["overall_quality"] = (sum(q) / len(q)) if q else None
        out[ex] = row
    return out
ALLM = COLS + ["overall_quality"]
rng = random.Random(a.seed)

def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def boot_ci(vals):
    if len(vals) < 2: return (float("nan"),) * 2
    n = len(vals); ms = []
    for _ in range(a.boot):
        ms.append(mean([vals[rng.randrange(n)] for _ in range(n)]))
    ms.sort(); return ms[int(0.025 * a.boot)], ms[int(0.975 * a.boot) - 1]

os.makedirs(a.out, exist_ok=True)
long_rows, wide = [], collections.defaultdict(dict)
cache = {}
for j in JUDGES:
    for c, _ in ORDER:
        if (j, c) not in data: continue
        pp = per_prompt(j, c); cache[(j, c)] = pp
        for m in ALLM:
            vals = [r[m] for r in pp.values() if r.get(m) is not None]
            lo, hi = boot_ci(vals)
            long_rows.append([JUDGES[j], NAME[c], m, len(vals), len(pp) - len(vals), round(mean(vals), 4), round(lo, 4), round(hi, 4)])
            wide[(j, c)][m] = (mean(vals), lo, hi, len(pp) - len(vals))
with open(os.path.join(a.out, "paper50_results_long.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["judge", "condition", "metric", "n_scored", "n_missing", "mean", "ci_lo", "ci_hi"]); w.writerows(long_rows)

# paired deltas: (B - A) on prompts where both scored; CI by resampling prompts
PAIRS = [
 ("Granite / tuning effect (n=3,800 lr3e-4 - base)",   "granite-4.1-8B-nfull", "granite-4.1-8B-base"),
 ("Qwen3 / tuning effect (n=3,800 lr3e-4 - base)",     "qwen3-8B-nfull", "qwen3-8B-base"),
 ("Granite / data 1,000 -> 2,000", "granite-4.1-8B-n2000", "granite-4.1-8B-n1000-b32"),
 ("Granite / data 2,000 -> 3,800", "granite-4.1-8B-nfull", "granite-4.1-8B-n2000"),
 ("Qwen3 / data 1,000 -> 2,000",   "qwen3-8B-n2000", "qwen3-8B-n1000"),
 ("Qwen3 / data 2,000 -> 3,800",   "qwen3-8B-nfull", "qwen3-8B-n2000"),
 ("Granite / lr 2e-4 vs 3e-4",     "granite-4.1-8B-nfull-lr2e-4", "granite-4.1-8B-nfull"),
 ("Granite / lr 5e-5 vs 3e-4",     "granite-4.1-8B-nfull-lr5e-5", "granite-4.1-8B-nfull"),
 ("Qwen3 / lr 2e-4 vs 3e-4",       "qwen3-8B-nfull-lr2e-4", "qwen3-8B-nfull"),
 ("Qwen3 / lr 5e-5 vs 3e-4",       "qwen3-8B-nfull-lr5e-5", "qwen3-8B-nfull"),
]
drows = []
for j in JUDGES:
    for label, b, ref in PAIRS:
        if (j, b) not in cache or (j, ref) not in cache: continue
        for m in ALLM:
            diffs = [cache[(j, b)][ex][m] - cache[(j, ref)][ex][m]
                     for ex in ids if ex in cache[(j, b)] and ex in cache[(j, ref)]
                     and cache[(j, b)][ex].get(m) is not None and cache[(j, ref)][ex].get(m) is not None]
            lo, hi = boot_ci(diffs)
            drows.append([JUDGES[j], label, m, len(diffs), round(mean(diffs), 4), round(lo, 4), round(hi, 4),
                          "excludes 0" if (lo > 0 or hi < 0) else "includes 0"])
with open(os.path.join(a.out, "paper50_paired_deltas.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["judge", "comparison (B - A)", "metric", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "ci_vs_zero"]); w.writerows(drows)

# markdown tables
with open(os.path.join(a.out, "paper50_results.md"), "w") as f:
    f.write("# Judge scores on the fixed 50-prompt subset\n\nMean [95% bootstrap CI over prompts]. Hallucination is a flag rate (lower is better). "
            "`(k)` = k scores missing (judge reply cut off before the score).\n")
    for j, jn in JUDGES.items():
        f.write(f"\n## {jn}\n\n| Condition | " + " | ".join(ALLM) + " |\n|---|" + "---|" * len(ALLM) + "\n")
        for c, nm in ORDER:
            if (j, c) not in wide: continue
            cells = []
            for m in ALLM:
                mu, lo, hi, miss = wide[(j, c)][m]
                cells.append(f"{mu:.3f} [{lo:.2f}, {hi:.2f}]" + (f" ({miss})" if miss else ""))
            f.write(f"| {nm} | " + " | ".join(cells) + " |\n")
    f.write("\n## Paired differences (B - A), same prompts\n\n| Judge | Comparison | Metric | n | Diff | 95% CI | vs 0 |\n|---|---|---|---|---|---|---|\n")
    for r in drows:
        if r[2] in ("overall_quality", "correctness", "completeness", "hallucination"):
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:+.3f} | [{r[5]:+.3f}, {r[6]:+.3f}] | {r[7]} |\n")
# ---- test-retest summary of duplicate runs ----
if pairs:
    agree = collections.defaultdict(lambda: [0, 0])     # (judge, metric) -> [identical, compared]
    diffs = collections.defaultdict(list)
    for j, c, ex, cols, v1, v2, r1, r2 in pairs:
        for m, a1, a2 in zip(cols, v1, v2):
            if a1 is None or a2 is None: continue
            agree[(j, m)][1] += 1; agree[(j, m)][0] += (a1 == a2)
            diffs[(j, m)].append(abs(a1 - a2))
    with open(os.path.join(a.out, "paper50_duplicate_runs.md"), "w") as f:
        f.write("# Test-retest of duplicate judge runs\n\nSame judge, same responses, scored twice (temperature 0). "
                "The later job is the one used in the tables. Share of cells with identical scores, and mean absolute difference.\n\n"
                "| Judge | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS) + "\n")
        for j, jn in JUDGES.items():
            cells = []
            for m in METRICS:
                i, t = agree[(j, m)]
                cells.append(f"{i}/{t} ({100*i/t:.0f}%), {sum(diffs[(j, m)])/t:.3f}" if t else "-")
            f.write(f"| {jn} | " + " | ".join(cells) + " |\n")
    print(open(os.path.join(a.out, "paper50_duplicate_runs.md")).read())

# ---- judge audit (Stage-2 style) ----
const_cells = collections.Counter(); total_cells = collections.Counter()
for (j, c), pp in cache.items():
    for m in METRICS:
        vals = [r[m] for r in pp.values() if r.get(m) is not None]
        if len(vals) >= 2:
            total_cells[j] += 1
            if len(set(vals)) == 1: const_cells[j] += 1
with open(os.path.join(a.out, "paper50_judge_audit.md"), "w") as f:
    f.write("# Judge output audit, 50-prompt sample\n\nCells are judge x metric x response over the conditions in the tables. "
            "Off-scale = value not in {0, .25, .5, .75, 1} (hallucination: not 0/1); remapped to the nearest level, ties up. "
            "Missing = reply cut off before the score. Constant cell = one judge x metric x condition where every scored response got the same value.\n\n"
            "| Judge | Responses | Cells expected | Missing | Responses with >=1 missing | Off-scale remapped | Constant cells |\n|---|---|---|---|---|---|---|\n")
    for j, jn in JUDGES.items():
        A = audit[j]
        pct = lambda k: f"{A[k]} ({100*A[k]/A['cells_expected']:.1f}%)" if A["cells_expected"] else "0"
        f.write(f"| {jn} | {A['responses']} | {A['cells_expected']} | {pct('missing')} | {A['responses_with_missing']} | {pct('offscale')} | {const_cells[j]} of {total_cells[j]} |\n")
    f.write("\nOff-scale remaps by metric:\n\n| Judge | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS) + "\n")
    for j, jn in JUDGES.items():
        f.write(f"| {jn} | " + " | ".join(str(audit[j]['offscale_' + m]) for m in METRICS) + " |\n")
    f.write("\nLargest change of a metric mean caused by remapping (per judge):\n\n")
    for j, jn in JUDGES.items():
        best = max(((abs(shift[(j, m)][1] - shift[(j, m)][0]) / shift[(j, m)][2], m) for m in METRICS if shift[(j, m)][2]), default=(0, "-"))
        f.write(f"- {jn}: {best[0]:+.4f} ({best[1]})\n")
print(open(os.path.join(a.out, "paper50_judge_audit.md")).read())
missing = [(JUDGES[j], NAME[c]) for j in JUDGES for c, _ in ORDER if (j, c) not in data]
print("conditions with NO data yet:", missing or "none")
print("wrote", a.out)
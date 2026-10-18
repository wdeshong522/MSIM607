"""
truncated_counts.py: how many of the 50 responses per condition were cut off at the token limit
(done_reason == "length", max 1,536 new tokens), and how many were left unscored by each judge.

Run from the repo root on the cluster, next to build_paper_tables.py:
  python3 truncated_counts.py [--gen_root generation/outputs/paper50] [--out generation/outputs/paper_tables]
  Optional, for conditions stored elsewhere (the older n=2,000 / n=3,800 / Qwen n=1,000 runs):
  python3 truncated_counts.py --extra "Granite / tuned n=2,000=path/to/judge_examples_xxx.json" ...

Reads every <gen_root>/<condition>/judge_examples_paper50.json (these hold done_reason), keeps the 50 ids
in paper50_ids.json, and writes paper50_truncated_counts.md/.csv. Judge-missing counts use the same
cleaned data as build_paper_tables.py.
"""
import sys, os, csv, json, glob, argparse, collections
ap = argparse.ArgumentParser()
ap.add_argument("--gen_root", default="generation/outputs/paper50")
ap.add_argument("--out", default="generation/outputs/paper_tables")
ap.add_argument("--extra", nargs="*", default=[], help='"display name=path.json"')
b, rest = ap.parse_known_args()   # unknown args (e.g. --root) pass through to build_paper_tables.py

ids = set(json.load(open(os.path.join(b.gen_root, "paper50_ids.json")))["example_ids"])
files = {os.path.basename(os.path.dirname(f)): f for f in glob.glob(os.path.join(b.gen_root, "*", "judge_examples_paper50.json"))}
for e in b.extra:
    n, p = e.split("=", 1); files[n] = p

# judge-missing counts: reuse build_paper_tables.py (its own argparse reads sys.argv, so hide ours)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_paper_tables.py")).read()
head = src.split('ALLM = COLS + ["overall_quality"]')[0]
argv, sys.argv = sys.argv, [sys.argv[0]] + rest
g = {"__name__": "btp"}
exec(compile(head, "build_paper_tables.py", "exec"), g)
sys.argv = argv
data, ORDER, JUDGES, QUALITY = g["data"], g["ORDER"], g["JUDGES"], g["QUALITY"]

rows = []
for c, name in ORDER:
    f = files.get(c) or files.get(name)
    rec = {"condition": name, "n": None, "trunc": None, "reasons": ""}
    if f and os.path.exists(f):
        rs = [r for r in json.load(open(f)) if r["example_id"] in ids]
        cnt = collections.Counter(r.get("done_reason") for r in rs)
        rec.update(n=len(rs), trunc=cnt.get("length", 0), reasons=", ".join(f"{k}:{v}" for k, v in cnt.items()))
    for j, jn in JUDGES.items():
        d = data.get((j, c), {})
        rec[jn] = sum(1 for r in d.values() if any(r.get(m) is None for m in QUALITY + ["hallucination"])) if d else None
    rows.append(rec)

os.makedirs(b.out, exist_ok=True)
jn = list(JUDGES.values())
with open(os.path.join(b.out, "paper50_truncated_counts.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition", "n_responses", "truncated", "done_reasons"] + [f"unscored_{x}" for x in jn])
    for r in rows: w.writerow([r["condition"], r["n"], r["trunc"], r["reasons"]] + [r[x] for x in jn])
fmt = lambda v: "n/a" if v is None else str(v)
with open(os.path.join(b.out, "paper50_truncated_counts.md"), "w") as f:
    f.write("# Truncated and unscored responses, 50-prompt sample\n\n"
            "Truncated = generation stopped at the 1,536-token limit (done_reason = length). "
            "Unscored = responses with at least one missing metric from that judge. n/a = file not found (use --extra).\n\n"
            f"| Condition | Truncated (of 50) | Unscored, {jn[0]} | Unscored, {jn[1]} |\n|---|---|---|---|\n")
    for r in rows: f.write(f"| {r['condition']} | {fmt(r['trunc'])} | {fmt(r[jn[0]])} | {fmt(r[jn[1]])} |\n")
print(open(os.path.join(b.out, "paper50_truncated_counts.md")).read())

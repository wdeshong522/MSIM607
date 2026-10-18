"""
top_level_counts.py: for each judge and condition, how many of the 50 responses got the highest score
on each metric (1.0 on the five-level scales). Uses exactly the same data as build_paper_tables.py
(same databases, same duplicate rule, same off-scale remapping), so the counts agree with the tables.

Put this file next to build_paper_tables.py and run from the repo root:
  python3 top_level_counts.py [--out generation/outputs/paper_tables]
Writes paper50_top_level_counts.md and .csv in the --out folder.

Cell format: "k/n" = k responses at 1.0 out of n scored (a missing score is not counted in n).
Hallucination is a 0/1 flag where 1 means a hallucination was flagged, so its column reports the
number of responses flagged ("k/n flagged"); lower is better, and 0 is the best level.
"""
import sys, os, csv
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_paper_tables.py")).read()
head = src.split('ALLM = COLS + ["overall_quality"]')[0]
exec(compile(head, "build_paper_tables.py", "exec"))     # loads and cleans the scores: defines data, ORDER, JUDGES, QUALITY

rows = []
for j, jn in JUDGES.items():
    for c, cname in ORDER:
        if (j, c) not in data: continue
        d = data[(j, c)]
        rec = {"judge": jn, "condition": cname}
        for m in QUALITY + ["hallucination"]:
            vals = [r[m] for r in d.values() if r.get(m) is not None]
            top = sum(1 for v in vals if v == (1.0)) if m != "hallucination" else sum(1 for v in vals if v == 1.0)
            rec[m] = (top, len(vals))
        rows.append(rec)

os.makedirs(a.out, exist_ok=True)
short = {"correctness": "Correct.", "completeness": "Complete.", "faithfulness": "Faithful.", "relevance": "Relevance",
         "instruction_following": "Instr. following", "safety": "Safety", "hallucination": "Halluc. flagged"}
with open(os.path.join(a.out, "paper50_top_level_counts.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["judge", "condition", "metric", "n_at_top", "n_scored"])
    for r in rows:
        for m in QUALITY + ["hallucination"]:
            w.writerow([r["judge"], r["condition"], m, r[m][0], r[m][1]])
with open(os.path.join(a.out, "paper50_top_level_counts.md"), "w") as f:
    f.write("# Responses at the highest score, 50-prompt sample\n\n"
            "k/n = k responses at 1.0 out of n scored. Halluc. flagged = responses flagged for hallucination (lower is better).\n")
    for jn in JUDGES.values():
        f.write(f"\n## {jn}\n\n| Condition | " + " | ".join(short[m] for m in QUALITY + ["hallucination"]) + " |\n")
        f.write("|---|" + "---|" * 7 + "\n")
        for r in rows:
            if r["judge"] != jn: continue
            f.write(f"| {r['condition']} | " + " | ".join(f"{r[m][0]}/{r[m][1]}" for m in QUALITY + ["hallucination"]) + " |\n")
print(open(os.path.join(a.out, "paper50_top_level_counts.md")).read())

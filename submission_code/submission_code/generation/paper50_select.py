"""
paper50_select.py  (DRAFT)
Creates a fixed 50-prompt subset of the frozen 200-example holdout and writes a filtered
generation JSON for each requested model key, so judging only scores those 50.

  - The 50 ids are chosen ONCE (seeded, stratified by difficulty) and saved to
    <out_root>/paper50_ids.json; every later call reuses that file, so all conditions and
    both judges see exactly the same prompts.
  - For each key it takes the NEWEST judge_examples_*.json under <src_root>/<key>/ and keeps
    only the 50 ids. It fails loudly if any of the 50 is missing from a condition.

Usage (from repo root):
  python3 generation/paper50_select.py --keys granite-4.1-8B-base qwen3-8B-base ...
"""
import argparse, glob, json, os, random, sys, collections

p = argparse.ArgumentParser()
p.add_argument("--src_root", default="generation/outputs/judge_examples")
p.add_argument("--out_root", default="generation/outputs/paper50")
p.add_argument("--keys", nargs="+", required=True)
p.add_argument("--n", type=int, default=50)
p.add_argument("--seed", type=int, default=12345)
a = p.parse_args()

def newest(key):
    files = sorted(glob.glob(os.path.join(a.src_root, key, "judge_examples_*.json")))
    if not files:
        sys.exit(f"ERROR: no generation JSON for {key} under {a.src_root}/{key}")
    return files[-1]

os.makedirs(a.out_root, exist_ok=True)
ids_path = os.path.join(a.out_root, "paper50_ids.json")
if os.path.exists(ids_path):
    ids = json.load(open(ids_path))["example_ids"]
    print(f"Reusing {len(ids)} saved ids from {ids_path}")
else:
    recs = json.load(open(newest(a.keys[0])))
    by_diff = collections.defaultdict(list)
    for r in recs:
        by_diff[r.get("difficulty", "UNKNOWN")].append(r["example_id"])
    total = sum(len(v) for v in by_diff.values())
    # largest-remainder proportional allocation
    quota = {d: a.n * len(v) / total for d, v in by_diff.items()}
    alloc = {d: int(q) for d, q in quota.items()}
    rem = a.n - sum(alloc.values())
    for d in sorted(quota, key=lambda d: quota[d] - alloc[d], reverse=True)[:rem]:
        alloc[d] += 1
    rng = random.Random(a.seed)
    ids = []
    for d in sorted(by_diff):
        ids += rng.sample(sorted(by_diff[d]), alloc[d])
    ids = sorted(ids)
    json.dump({"seed": a.seed, "n": a.n, "allocation": alloc, "source_key": a.keys[0],
               "example_ids": ids}, open(ids_path, "w"), indent=2)
    print(f"Chose {len(ids)} ids, allocation by difficulty: {alloc} -> {ids_path}")

idset = set(ids)
for key in a.keys:
    src = newest(key)
    recs = [r for r in json.load(open(src)) if r["example_id"] in idset]
    got = {r["example_id"] for r in recs}
    missing = idset - got
    if missing:
        sys.exit(f"ERROR: {key}: {len(missing)} of the {len(idset)} ids missing from {src}")
    out_dir = os.path.join(a.out_root, key); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "judge_examples_paper50.json")
    json.dump(recs, open(out, "w"), indent=2, ensure_ascii=False)
    trunc = sum(1 for r in recs if r.get("done_reason") == "length")
    print(f"{key}: {len(recs)} records from {os.path.basename(src)} -> {out}  (truncated at max tokens: {trunc})")

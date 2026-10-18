"""
show_example.py: print one training scenario and one held-out prompt with the generated
response(s), as markdown you can paste into a slide or the paper.

Run from the repo root on the cluster (the data is not in this workspace).

  # 1. see what is there
  python3 show_example.py --list

  # 2. one held-out example, base vs tuned, plus a training example
  python3 show_example.py --keys <base_key> <tuned_key> --train <path/to/train.jsonl> --train_index 0

Options:
  --example_id  pick a specific held-out id (default: the first of the 50 in paper50_ids.json)
  --maxchars    truncate each long field (default 1500; 0 = no truncation)
The script prints EVERY field of each record, so you can see what is available and
decide which fields to show. Nothing is changed on disk.
"""
import argparse, glob, json, os, sys

p = argparse.ArgumentParser()
p.add_argument("--root", default="generation/outputs/paper50")
p.add_argument("--keys", nargs="*", default=[])
p.add_argument("--example_id", default=None)
p.add_argument("--train", default=None, help="training file (.json or .jsonl)")
p.add_argument("--train_index", type=int, default=0)
p.add_argument("--maxchars", type=int, default=1500)
p.add_argument("--list", action="store_true")
a = p.parse_args()

def clip(s):
    s = s if isinstance(s, str) else json.dumps(s, indent=2, ensure_ascii=False)
    return s if not a.maxchars or len(s) <= a.maxchars else s[: a.maxchars] + f"\n[... {len(s) - a.maxchars} more characters]"

def show(title, rec, skip=()):
    print(f"\n### {title}\n")
    for k, v in rec.items():
        if k in skip:
            continue
        print(f"**{k}**\n\n```\n{clip(v)}\n```\n")

if a.list or not a.keys:
    print("Conditions under", a.root)
    for d in sorted(glob.glob(os.path.join(a.root, "*", "judge_examples_paper50.json"))):
        print("  ", os.path.basename(os.path.dirname(d)))
    if not a.keys:
        sys.exit(0)

ids = json.load(open(os.path.join(a.root, "paper50_ids.json")))["example_ids"]
eid = a.example_id or ids[0]
if eid not in ids:
    eid = next((i for i in ids if i.endswith(eid) or eid.endswith(i)), None) if a.example_id else None
    if eid is None:
        sys.exit(f"{a.example_id} is not one of the 50 ids")

for key in a.keys:
    f = key if key.endswith(".json") else os.path.join(a.root, key, "judge_examples_paper50.json")
    if not os.path.exists(f):
        sys.exit(f"missing {f}; run with --list")
    rec = next(r for r in json.load(open(f)) if r["example_id"] == eid)
    show(f"Held-out example {eid}: {key}", rec)

if a.train:
    if a.train.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(a.train) if l.strip()]
    else:
        rows = json.load(open(a.train))
    show(f"Training example #{a.train_index} ({a.train})", rows[a.train_index])

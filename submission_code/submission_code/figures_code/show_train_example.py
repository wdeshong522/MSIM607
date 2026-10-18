"""
show_train_example.py: print one record from the 3,800-example TRAIN pool, in the same
format the models were trained on (uses your split_utils.py, so it is the real split).

Run from the repo root on the cluster.
  python3 show_train_example.py --data_dir <data_dir> --grep "developmental groove"
  python3 show_train_example.py --data_dir <data_dir> --index 0

--grep  keeps only train records whose text contains this string (case-insensitive),
        so you can pick a scenario on the same topic as the held-out example.
Prints how many records matched and the first --show of them.
"""
import argparse, json, sys
sys.path.insert(0, ".")
from split_utils import load_all_records, load_frozen_split, stable_id

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--grep", nargs="*", default=None, help="one or more strings; a record must contain ALL of them")
p.add_argument("--index", type=int, default=0)
p.add_argument("--show", type=int, default=1)
a = p.parse_args()

records = load_all_records(a.data_dir)
eval_records, train_pool = load_frozen_split(a.data_dir, records)
print(f"train pool: {len(train_pool)}   frozen holdout: {len(eval_records)}")
rows = train_pool
if a.grep:
    gs = [g.lower() for g in a.grep]
    rows = [r for r in train_pool if all(g in json.dumps(r, ensure_ascii=False).lower() for g in gs)]
    print(f"{len(rows)} train records contain {a.grep!r}")
for r in rows[a.index : a.index + a.show]:
    print(f"\n=== train record id {stable_id(r)} ===")
    print(json.dumps(r, indent=2, ensure_ascii=False))

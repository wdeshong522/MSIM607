"""
split_utils.py
===============
Canonical, frozen train/eval split for the ShareGPT .jsonl corpus.

WHY THIS EXISTS
----------------
Two different split implementations existed for what was assumed to be
"the same seed=42, 90/10 split":
  - generate_judge_examples.py: random.Random(42); rng.shuffle(indices)
  - lora_finetune.py:           dataset.train_test_split(test_size=0.10, seed=42)
                                 (HuggingFace datasets — a different shuffle
                                 algorithm under the hood, not Python's
                                 random.shuffle)
Same seed VALUE does not mean same shuffle RESULT across two different
algorithms. These two splits were very likely diverging — meaning
"held-out" examples used for judging were not reliably excluded from
training. This module replaces both with one split, computed once,
frozen to disk, and read by every script that needs it.

It also replaces positional indexing (fragile: adding a new file whose
name sorts earlier shifts every index after it) with a stable,
content-hashed ID per record, so the frozen split survives the corpus
growing over time without needing to be recomputed.

DESIGN — two separate splits, two separate jobs
------------------------------------------------
1. A FIXED, GLOBAL judge reservation (n_judge examples, default 200):
   permanently excluded from every training condition, at every size,
   forever. This is what lets "did more training data improve the judge
   metrics" be a clean comparison — every condition is judged against the
   exact same, never-seen scenarios. Selected ONCE, proportionally by
   domain (see extract_domain), and never changed again — not even as the
   corpus grows toward ~10000. If it changed later, every training run
   already completed against the old reservation would become
   incomparable to anything judged after the change.

2. Per-condition internal 90/10 train/val splits (e.g. "the 1000
   condition" trains on 900, validates on 100, to watch for overfitting)
   are NOT this module's concern — those are condition-local and drawn
   fresh from whatever N-sized slice of the training pool that condition
   uses. That logic lives in lora_finetune.py, not here.

USAGE
-----
One-time setup (or "corpus grew, add new training records" — safe to
re-run; see below):
    python split_utils.py --data_dir data --n_judge 200 --seed 42

This writes:
    eval_holdout_ids.json          — the n_judge stable IDs permanently
                                      reserved for judging. Domain-
                                      stratified and interleaved (any
                                      prefix of this list stays roughly
                                      proportional too). FIXED forever
                                      once created.
    train_pool_order.json          — a FIXED shuffle of every OTHER
                                      record's stable ID (everything not
                                      in the judge reservation). A given
                                      training condition's pool is just a
                                      prefix of this list — train_pool[:1000],
                                      train_pool[:2000], etc. — which is
                                      what makes them nest automatically
                                      (1000 ⊆ 2000 ⊆ 4000 ⊆ ...).
    eval_holdout_domain_summary.json — per-domain corpus counts vs. judge
                                      allocation, for methods-section
                                      reporting / sanity-checking the
                                      stratification.

Re-running this command is SAFE when the corpus grows: the judge
reservation never changes (any --n_judge value passed on a later run is
ignored once the split already exists), and train_pool_order.json only
ever has new records APPENDED at the end — its existing prefixes, and
therefore every training condition's example set, never change. This is
what lets a 1000/2000/~3800 study extend cleanly to ~10000 later without
invalidating any subset boundary already used.

Library usage (both generate_judge_examples.py and lora_finetune.py
should import this rather than compute their own split):
    from split_utils import load_all_records, load_frozen_split

    records = load_all_records(data_dir)          # [(filepath, record, id), ...]
    judge_records, train_pool = load_frozen_split(data_dir, records)
    train_subset = train_pool[:1000]               # nested prefix
"""

import argparse
import glob
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stable per-record IDs
# ---------------------------------------------------------------------------

def stable_id(record: dict) -> str:
    """
    Content-hashed ID for a ShareGPT record — stable across file additions,
    reordering, or re-globbing, unlike a positional index. Hashes the
    concatenated turn values (from + value for every turn), so two records
    would only collide if their entire conversation content were identical
    (in which case they're effectively duplicates anyway).
    """
    convs = record.get("conversations", [])
    fingerprint = "||".join(
        f"{turn.get('from', '')}:{turn.get('value', '')}" for turn in convs
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


_DOMAIN_PATTERN = re.compile(r"domain(\d+)", re.IGNORECASE)


def extract_domain(filepath: str) -> str:
    """
    Domain isn't a field in the record itself — it's encoded in the
    filename (e.g. "dental_hygiene_domain5_batch1.jsonl" -> "domain5"),
    matching the corpus's existing file-naming convention. Records under
    a filename that doesn't match get bucketed as "unknown" rather than
    dropped or crashing — worth checking the logs if that bucket is
    non-empty, since it usually means a mis-named or misplaced file.
    """
    m = _DOMAIN_PATTERN.search(Path(filepath).name)
    return f"domain{m.group(1)}" if m else "unknown"


def _largest_remainder_allocation(domain_counts: dict, target_total: int) -> dict:
    """
    Proportionally allocate target_total items across domains by their
    share of domain_counts, using largest-remainder (Hamilton) rounding —
    same method generate_new_scenarios.py already uses for per-domain
    example allocation, reused here for consistency rather than
    introducing a second rounding approach. Guarantees the allocation
    sums to exactly target_total UNLESS a domain's fair share would
    exceed its actual available count, in which case that domain is
    capped at its own size and a warning is logged (falls a little short
    of target_total rather than over-allocating a small domain).
    """
    grand_total = sum(domain_counts.values())
    if grand_total == 0 or target_total <= 0:
        return {d: 0 for d in domain_counts}
    target_total = min(target_total, grand_total)

    raw = {d: target_total * c / grand_total for d, c in domain_counts.items()}
    alloc = {d: min(int(raw[d]), domain_counts[d]) for d in domain_counts}
    remainder = target_total - sum(alloc.values())

    # Distribute the remainder to domains with the largest fractional
    # part first, skipping any domain already capped at its full count.
    order = sorted(domain_counts, key=lambda d: raw[d] - int(raw[d]), reverse=True)
    for d in order:
        if remainder <= 0:
            break
        if alloc[d] < domain_counts[d]:
            alloc[d] += 1
            remainder -= 1

    if remainder > 0:
        log.warning(
            "Could not allocate %d of the requested %d judge examples — one "
            "or more domains ran out of available records. Final judge "
            "sample will be %d, not %d.",
            remainder, target_total, target_total - remainder, target_total,
        )
    return alloc


# Table 3a, DHExP_Generation_Procedure_w_batch_plan.md — the dataset's
# DESIGNED ~10,000-example domain balance, not a snapshot of current counts.
# Domain 1 is currently paused and overrepresented (27.95% actual vs. this
# 17.06% target) specifically because early generation concentrated there;
# stratifying the judge set by CURRENT corpus counts would bake that
# temporary imbalance in permanently. Target share is what "some domains
# matter more than others, and the proportions approximately reflect that"
# actually refers to — the intended final composition, not wherever
# generation happens to be at the moment this split is built.
TABLE_3A_TARGET_SHARE = {
    "domain1":  0.1706,  # Periodontal Assessment and Diagnosis
    "domain2":  0.1220,  # Medical History and Risk Assessment
    "domain3":  0.0488,  # Vital Signs
    "domain4":  0.1220,  # Instrumentation
    "domain5":  0.0854,  # Oral Cancer Screening
    "domain6":  0.0854,  # Patient Communication
    "domain7":  0.0610,  # Infection Control
    "domain8":  0.1098,  # Radiographic Interpretation
    "domain9":  0.0854,  # Scope of Practice
    "domain10": 0.1098,  # Treatment Planning
}


def _target_share_allocation(domain_counts: dict, target_shares: dict,
                              target_total: int) -> dict:
    """
    Allocate target_total judge examples across domains using FIXED target
    shares (TABLE_3A_TARGET_SHARE), not the corpus's current per-domain
    counts — see that constant's comment for why this distinction matters.
    Same largest-remainder rounding as _largest_remainder_allocation();
    this variant just takes the weights as an explicit parameter instead
    of deriving them from domain_counts.

    Domains present in the data but with no Table 3a entry (e.g. an
    "unknown" bucket from an unparseable filename) get 0 allocation and a
    warning — check for mis-named/misplaced files if that's unexpected.
    Domains in Table 3a without enough currently-available records to
    satisfy their target share are capped at what's actually available,
    with a warning, rather than over-allocating a domain that doesn't
    have enough real examples yet.
    """
    unexpected = set(domain_counts) - set(target_shares)
    if unexpected:
        log.warning(
            "Domain(s) found in data with no Table 3a target share: %s — "
            "these get 0 judge allocation. Check for mis-named/misplaced "
            "files if this is unexpected.", sorted(unexpected),
        )

    raw = {d: target_total * s for d, s in target_shares.items()}
    alloc = {d: min(int(raw.get(d, 0)), domain_counts.get(d, 0)) for d in target_shares}
    remainder = target_total - sum(alloc.values())

    order = sorted(target_shares, key=lambda d: raw[d] - int(raw[d]), reverse=True)
    for d in order:
        if remainder <= 0:
            break
        if alloc[d] < domain_counts.get(d, 0):
            alloc[d] += 1
            remainder -= 1

    if remainder > 0:
        log.warning(
            "Could not allocate %d of the requested %d judge examples against "
            "Table 3a target shares — one or more domains don't have enough "
            "records yet to satisfy their target share (check "
            "eval_holdout_domain_summary.json after this run). Final judge "
            "sample will be %d, not %d.",
            remainder, target_total, target_total - remainder, target_total,
        )
    return alloc


def _interleave_stratified(domain_id_lists: dict) -> list:
    """
    Interleave multiple per-domain ID lists (each already shuffled) so
    domains are spread evenly across the WHOLE resulting order, not
    grouped in blocks. This matters because any PREFIX of the frozen eval
    list (e.g. a smaller --max_examples for a quick smoke test) should
    stay approximately proportionally stratified too — a block-concatenated
    order would make a short prefix accidentally all-one-domain.
    """
    items = []
    for domain, ids in domain_id_lists.items():
        n = len(ids)
        for i, rid in enumerate(ids):
            position = (i + 0.5) / n if n else 0.0  # evenly spaced in [0, 1)
            items.append((position, domain, rid))
    items.sort(key=lambda t: (t[0], t[1]))
    return [rid for _, _, rid in items]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_records(data_dir: str) -> list[tuple[str, dict, str]]:
    """
    Load every .jsonl record under data_dir, same file-discovery logic as
    load_sharegpt_jsonl()/load_eval_examples() elsewhere in the pipeline.
    Returns (filepath, record, stable_id) tuples. Sorted by stable_id (not
    by file/glob order) so the return order itself is deterministic and
    independent of filesystem/glob quirks.
    """
    files = sorted(glob.glob(f"{data_dir}/**/*.jsonl", recursive=True))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {data_dir}")

    records = []
    seen_ids = set()
    dupes = 0
    for filepath in files:
        with open(filepath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "conversations" not in obj or not isinstance(obj["conversations"], list):
                    continue
                rid = stable_id(obj)
                if rid in seen_ids:
                    dupes += 1
                    continue
                seen_ids.add(rid)
                records.append((filepath, obj, rid))

    if dupes:
        log.warning("%d duplicate record(s) (identical conversation content) skipped", dupes)
    log.info("Loaded %d unique records from %d files", len(records), len(files))
    return sorted(records, key=lambda r: r[2])


# ---------------------------------------------------------------------------
# Frozen split: build (one-time / append-only) and load
# ---------------------------------------------------------------------------

def _eval_holdout_path(data_dir: str) -> Path:
    return Path(data_dir) / "eval_holdout_ids.json"


def _train_pool_order_path(data_dir: str) -> Path:
    return Path(data_dir) / "train_pool_order.json"


def build_or_extend_frozen_split(data_dir: str, n_judge: int = 200,
                                  seed: int = 42) -> None:
    """
    Create the frozen split files if they don't exist. If they DO exist,
    EXTEND train_pool_order.json with newly-added records only.

    The judge reservation (eval_holdout_ids.json) is a FIXED, ONE-TIME
    selection of n_judge examples, chosen proportionally by domain (see
    extract_domain / _largest_remainder_allocation) so the reservation's
    domain mix matches the corpus's existing domain mix. It is set once,
    at first build, and never grows or changes on later runs — this is
    deliberate: these examples are permanently excluded from every
    training condition at every size (1000/2000/~3800/~9800/...), and
    that guarantee would break if the reservation could change out from
    under an already-completed training run. Growth in the corpus only
    ever adds new records to the TRAINING pool, never to the judge
    reservation.

    A companion file, eval_holdout_domain_summary.json, records the
    per-domain corpus counts and allocated judge counts at build time —
    useful for a methods-section table, and for auditing that the
    stratification actually landed where intended.
    """
    import random

    records = load_all_records(data_dir)  # [(filepath, record, id), ...]
    all_ids = [r[2] for r in records]
    total = len(all_ids)

    eval_path = _eval_holdout_path(data_dir)
    pool_path = _train_pool_order_path(data_dir)
    summary_path = Path(data_dir) / "eval_holdout_domain_summary.json"

    if eval_path.exists() != pool_path.exists():
        raise RuntimeError(
            f"Inconsistent state: exactly one of {eval_path} / {pool_path} "
            f"exists. Resolve manually (restore the missing file or delete "
            f"the other) before re-running — refusing to guess."
        )

    if not eval_path.exists():
        # First build: allocate n_judge across domains proportionally
        # (largest-remainder rounding), select that many per domain via a
        # fixed shuffle, then interleave across domains so any prefix of
        # the final list stays roughly stratified too.
        by_domain = defaultdict(list)
        for filepath, _record, rid in records:
            by_domain[extract_domain(filepath)].append(rid)

        domain_counts = {d: len(ids) for d, ids in by_domain.items()}
        # Stratify by TABLE_3A_TARGET_SHARE (the dataset's designed final
        # balance), not by domain_counts (the corpus's current, possibly
        # skewed, snapshot) — see TABLE_3A_TARGET_SHARE's comment.
        allocation = _target_share_allocation(domain_counts, TABLE_3A_TARGET_SHARE, n_judge)

        rng = random.Random(seed)
        selected_by_domain = {}
        for d in TABLE_3A_TARGET_SHARE:
            ids = by_domain.get(d, [])
            shuffled = ids.copy()
            rng.shuffle(shuffled)
            selected_by_domain[d] = shuffled[:allocation.get(d, 0)]

        eval_ids = _interleave_stratified(selected_by_domain)
        eval_id_set = set(eval_ids)

        pool_ids = [i for i in all_ids if i not in eval_id_set]
        rng2 = random.Random(seed)
        rng2.shuffle(pool_ids)

        eval_path.write_text(json.dumps(eval_ids, indent=2), encoding="utf-8")
        pool_path.write_text(json.dumps(pool_ids, indent=2), encoding="utf-8")

        all_summary_domains = sorted(set(domain_counts) | set(TABLE_3A_TARGET_SHARE))
        summary = {
            d: {
                "corpus_count": domain_counts.get(d, 0),
                "corpus_share": round(domain_counts.get(d, 0) / total, 4) if total else 0,
                "target_share": TABLE_3A_TARGET_SHARE.get(d, 0.0),
                "judge_allocated": len(selected_by_domain.get(d, [])),
                "judge_share": round(len(selected_by_domain.get(d, [])) / len(eval_ids), 4) if eval_ids else 0,
            }
            for d in all_summary_domains
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        log.info("Created NEW frozen split: %d judge-reserved (domain-stratified, "
                 "fixed forever), %d train pool, out of %d total",
                 len(eval_ids), len(pool_ids), total)
        log.info("Domain allocation written to %s", summary_path)
        return

    existing_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    existing_pool = json.loads(pool_path.read_text(encoding="utf-8"))
    existing_eval_set = set(existing_eval)
    existing_pool_set = set(existing_pool)

    new_ids = [i for i in all_ids if i not in existing_eval_set and i not in existing_pool_set]
    if not new_ids:
        log.info("No new records — frozen split unchanged (%d judge-reserved, %d pool)",
                 len(existing_eval), len(existing_pool))
        return

    # Judge reservation is fixed — ALL new records go to the training pool.
    rng = random.Random(seed)
    shuffled_new = new_ids.copy()
    rng.shuffle(shuffled_new)
    updated_pool = existing_pool + shuffled_new  # append-only: old prefix untouched

    pool_path.write_text(json.dumps(updated_pool, indent=2), encoding="utf-8")

    log.info(
        "Extended train pool: %d -> %d (+%d new). Judge reservation unchanged "
        "at %d (fixed since first build).",
        len(existing_pool), len(updated_pool), len(shuffled_new), len(existing_eval),
    )


def load_frozen_split(data_dir: str, records: list[tuple[str, dict, str]] = None
                       ) -> tuple[list[dict], list[dict]]:
    """
    Read the frozen split files and return (eval_records, train_pool_records).
    train_pool_records is ordered per train_pool_order.json — take a prefix
    (e.g. train_pool_records[:1000]) to get a reproducible, nested training
    subset. Raises FileNotFoundError with a clear message if the frozen
    split hasn't been built yet (run this module directly to build it).
    """
    eval_path = _eval_holdout_path(data_dir)
    pool_path = _train_pool_order_path(data_dir)
    if not eval_path.exists() or not pool_path.exists():
        raise FileNotFoundError(
            f"Frozen split not found under {data_dir}. Run once first:\n"
            f"    python split_utils.py --data_dir {data_dir}"
        )

    if records is None:
        records = load_all_records(data_dir)
    by_id = {rid: rec for _, rec, rid in records}

    eval_ids = json.loads(eval_path.read_text(encoding="utf-8"))
    pool_ids = json.loads(pool_path.read_text(encoding="utf-8"))

    missing_eval = [i for i in eval_ids if i not in by_id]
    missing_pool = [i for i in pool_ids if i not in by_id]
    if missing_eval or missing_pool:
        log.warning(
            "%d eval ID(s) and %d train-pool ID(s) from the frozen split are "
            "no longer present in the corpus (records removed/renamed?) — "
            "skipping them rather than erroring.",
            len(missing_eval), len(missing_pool),
        )

    eval_records = [by_id[i] for i in eval_ids if i in by_id]
    train_pool_records = [by_id[i] for i in pool_ids if i in by_id]
    return eval_records, train_pool_records


# ---------------------------------------------------------------------------
# CLI — build/extend the frozen split
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build (once) or extend (as the corpus grows) the frozen "
                     "train/judge split."
    )
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n_judge", type=int, default=200,
                    help="Total judge-reserved examples, allocated proportionally "
                         "across domains (largest-remainder rounding) to match "
                         "the corpus's existing domain mix. FIXED at first build "
                         "— never grows on later runs, regardless of corpus "
                         "growth or this flag's value on a later invocation "
                         "(ignored once the split already exists, since "
                         "changing it later would break comparability with "
                         "training conditions already run against the "
                         "original 200).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_or_extend_frozen_split(args.data_dir, n_judge=args.n_judge, seed=args.seed)

"""
score_judge_examples.py - LLM-as-judge scoring pass for the LoRA-vs-ground-truth
comparison, mirroring the structure and prompt design of the reference RAG
scorer.py.

============================================================================
COMBINED-DB / RUN-ID ARCHITECTURE (2026-09-27)
============================================================================
All judge runs (different judge models, different prompt sets, re-runs)
now write into ONE shared database instead of a separate .db file per
judge. Every row is stamped with:

  judge_model     : the JUDGE_MODEL value this run used (e.g.
                    "qwen3-27b-judge") - lets you filter/compare judges
                    within a single DB instead of opening several files.
  run_id          : identifies "one job" - every example/model scored by
                    one invocation of this script shares the same run_id,
                    even across multiple --models in that same invocation.
                    Defaults to $SLURM_JOB_ID when running under sbatch
                    (already a natural unique-per-job identifier), falls
                    back to a generated "local-<8 hex chars>" id otherwise,
                    or set explicitly with --run_id.
  run_created_at  : ISO8601 UTC timestamp captured ONCE at the start of
                    main() - this is when the JOB started, not when any
                    individual judge response was generated. Every row from
                    the same invocation shares this exact same timestamp.

PRIMARY KEY is now (run_id, example_id, model_key) instead of (example_id,
model_key) - this is what lets the same example/model be scored by
multiple judges/runs and coexist as separate rows in one table, rather
than each new judge's scores overwriting the previous judge's row.

IMPORTANT RESUMABILITY CAVEAT: because the default run_id is
$SLURM_JOB_ID, a job that times out or crashes and gets RESUBMITTED gets
a brand-new SLURM_JOB_ID -> a brand-new run_id by default. Since score_all()
now filters by run_id (so one run's scoring never leaks into another run's
rows), a resubmit with a fresh run_id will NOT resume the crashed run - it
will start a new run from scratch (re-ingesting and re-scoring everything,
since nothing matches the new run_id yet). To actually RESUME a
crashed/timed-out job into the same run, resubmit with
`--run_id <the original job's run_id>` explicitly.

BERTScore is judge-independent (only depends on generated_response/
ground_truth), so compute_bertscore() now auto-copies a cached value from
ANY other row in the DB with the same (example_id, model_key) - regardless
of which run/judge produced it - before computing anything fresh. In a
combined DB this means BERTScore is effectively computed once ever per
example/model, no matter how many judges/runs come through afterward;
--skip_bertscore is no longer necessary for that purpose (it still exists
if you want to skip BERTScore entirely for a given invocation).

============================================================================
PROMPT-FILE ARCHITECTURE (2026-09-27)
============================================================================
Judge prompts are no longer hardcoded Python functions in this file. Each
metric is now a *.txt file in a directory (--prompts_dir, default
"judge_prompts/"): the filename stem becomes both the scored metric name
and its DB column. Adding, removing, or editing a metric means editing
that directory, not this script - point --prompts_dir at a different
folder entirely to try an alternate rubric set without touching code.

Prompt file format (small YAML-lite front matter + a string.Template body):

    ---
    requires_ground_truth: true
    score_type: continuous
    ---
    You are an expert evaluator...
    SCENARIO:
    $scenario
    GROUND TRUTH:
    $ground_truth
    ANSWER:
    $answer
    ...

  requires_ground_truth: true|false (default false). If true, this metric
      is skipped (left NULL) for any row whose ground_truth is empty -
      same gate this file used to hardcode for correctness/completeness -
      and $ground_truth is available in the template. If false, only
      $scenario and $answer are meaningful (there is no reference to
      compare against for that metric).
  score_type: continuous|binary (default continuous). continuous scores
      are stored as REAL (0.0-1.0); binary scores are int(round(...))'d
      and stored as INTEGER (0/1), matching the old hardcoded hallucination
      handling.

The trailing "Respond with only a JSON object..." instruction is NOT part
of any template file - it's appended by response_instruction() based on
score_type and whether reasoning capture is on for this run, so it can
never drift out of sync with the actual enforced schema (SCORE_SCHEMA_
WITH_REASONING / SCORE_SCHEMA_NO_REASONING below), the way a prompt that
hardcodes its own JSON shape could.

ensure_schema() ALTER TABLEs one REAL/INTEGER column (+ a TEXT
*_reasoning column) per metric found in the prompt directory, rather than
a fixed column list - this is fully backward compatible with existing DBs
(judge_scores_cluster*.db etc.): it only adds columns that don't already
exist, so old databases keep working unmodified, and switching to a
prompt set with new/renamed metrics just adds/leaves-alone columns as
appropriate.

Everything else (ollama_judge, vllm_judge, parse_score, ingest_json,
compute_bertscore, ensure_judge_model[_vllm], the vLLM PORT NOTES below)
is unchanged from the previous version of this file.
============================================================================

Input: one or more JSON files produced by generate_judge_examples.py.
Output: a SQLite DB (resumable - only unscored rows are scored on a re-run,
unless RESCORE=1).

Configuration (environment variables, same pattern as scorer.py):
  JUDGE_MODEL    : judge model name/served-name (no default - must be set to
                   whatever you actually register/serve, e.g. Qwen3.8-27B or
                   its fallback)
  OLLAMA_BASE_URL: Ollama API endpoint (default: http://localhost:11434) -
                   only used when --backend ollama
  VLLM_URL       : vLLM OpenAI-compatible API endpoint
                   (default: http://localhost:8000) - only used when
                   --backend vllm
  DB_FILE        : Path to SQLite database (default: outputs/judge_scores.db)
  PROMPTS_DIR    : Directory of *.txt judge-prompt templates (default:
                   "judge_prompts")
  RESCORE        : Set to 1 to re-score already-scored rows (default: 0)

Usage:
    python score_judge_examples.py --input outputs/judge_examples/judge_examples_*.json
    python score_judge_examples.py --input ... --backend vllm --vllm_url http://localhost:8123 --limit 5
    python score_judge_examples.py --input ... --prompts_dir judge_prompts_v2

============================================================================
PORT NOTES (vLLM backend addition) - unchanged from the previous version
============================================================================
This file was ORIGINALLY Ollama-only. It has been extended to also support
a vLLM backend, following the same --backend ollama/vllm dispatch pattern
already established in generate_judge_examples.py. The Ollama-path code
below (ollama_judge, parse_score, ensure_judge_model, ingest_json,
compute_bertscore) is UNCHANGED from the original file. The vLLM-specific
pieces:

  1. get_registered_models_vllm() / ensure_judge_model_vllm() - vLLM
     equivalent of ensure_judge_model(), checks /v1/models instead of
     Ollama's /api/tags.

  2. vllm_judge() - same {"response": ..., "thinking": ...} return
     shape as ollama_judge(), but calls vLLM's OpenAI-compatible
     /v1/chat/completions endpoint instead of Ollama's /api/generate.
     Uses response_format/json_schema (SCORE_SCHEMA_*) to force valid JSON
     at decode time (a plain "respond with only JSON" instruction was not
     reliably followed under vLLM), and repetition_penalty=1.15 to avoid a
     genuine greedy-decoding repetition loop found during smoke testing at
     temperature=0.0.

  3. score_all() takes a `judge_fn` parameter instead of calling
     ollama_judge() directly, so either backend can be passed in
     interchangeably.

  4. --backend {ollama,vllm} (default: whatever cfg.BACKEND says, or
     "ollama") and --vllm_url CLI args. main() picks ensure_judge_model()
     + ollama_judge, or ensure_judge_model_vllm() + vllm_judge,
     accordingly, and passes the chosen judge_fn into score_all().

Recommended before a full run: smoke-test with
--backend vllm --vllm_url <url> --limit 5 against a running server first.
============================================================================
"""

import argparse
import datetime
import glob
import json
import os
import re
import functools
import sqlite3
import sys
import uuid
from pathlib import Path
from string import Template

import requests

import study_config as cfg

try:
    from bert_score import score as bertscore_compute
except ImportError:
    bertscore_compute = None


# --- Configuration (via study_config.py: env var > config file > default) ---

OLLAMA_BASE_URL = cfg.OLLAMA_BASE_URL
JUDGE_MODEL     = cfg.JUDGE_MODEL
DB_FILE         = cfg.DB_FILE
RESCORE         = cfg.RESCORE

# vLLM OpenAI-compatible API endpoint. Only used when --backend vllm.
VLLM_URL = getattr(cfg, "VLLM_URL", "http://localhost:8000")

# Default backend, so an unmodified `python score_judge_examples.py ...`
# invocation keeps behaving exactly as it did before this port (Ollama),
# unless study_config.py sets BACKEND or --backend is passed explicitly.
BACKEND = getattr(cfg, "BACKEND", "ollama")

# Directory of *.txt judge-prompt templates. See module docstring.
DEFAULT_PROMPTS_DIR = getattr(cfg, "PROMPTS_DIR", "judge_prompts")


# --- Ollama call (same pattern as scorer.py) ---

def ollama_judge(prompt: str, retries: int = 3, think: bool = True, num_ctx: int = 4096) -> dict:
    """
    Returns {"response": <final text>, "thinking": <reasoning trace or None>}.
    think=True costs real latency per call (measured ~7.2s vs ~2.8s per call
    on qwen3.8:27b) in exchange for a stored reasoning trace - see module
    docstring for why that trade is worth it here specifically.

    num_ctx is capped explicitly (default 4096) rather than left to Ollama's
    default. Qwen3.8-27B natively supports up to 262,144 tokens, and without
    an explicit cap Ollama reserves KV cache sized against that - actual
    judge prompts here (rubric + scenario + ground truth + answer) are only
    a few hundred to low-thousands of tokens, so the uncapped default was
    reserving far more VRAM than needed and forcing extra CPU/RAM offload
    on top of the model weights alone. 4096 comfortably covers judge
    prompts with headroom; raise it if a specific scenario/answer pair
    turns out to be unusually long.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {"temperature": 0.0, "num_ctx": num_ctx},
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return {
                "response": data.get("response", "").strip(),
                "thinking": data.get("thinking"),
            }
        except requests.RequestException as e:
            print(f"    [Attempt {attempt}/{retries}] Judge error: {e}")
            if attempt == retries:
                return {"response": "", "thinking": None}
    return {"response": "", "thinking": None}


# --- vLLM call (PORT ADDITION - see PORT NOTES above) ---

# Bare schema: fast path, used when capture_reasoning=False. Under strict
# json_schema decoding this produces content like '{"score": 0.75}' and
# nothing else.
SCORE_SCHEMA_NO_REASONING = {
    "type": "object",
    "properties": {"score": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
    "required": ["score"],
    "additionalProperties": False,
}

# Reasoning schema: used when capture_reasoning=True.
# ROOT CAUSE (found via smoke test, 2026-09-25): strict json_schema decoding
# constrains EVERY generated token to the schema from the very first one -
# there's no room in that grammar for a <think> preamble unless the schema
# itself declares a field for it. Putting "reasoning" in the schema captures
# it as a real JSON field instead. Declared before "score" so the model
# writes its justification before committing to a number. maxLength=600
# balances against finish_reason=length truncation (300 was too tight for
# real multi-part clinical reasoning; still watch for truncation at scale).
SCORE_SCHEMA_WITH_REASONING = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 600},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}


def vllm_judge(prompt: str, retries: int = 3, timeout: int = 300,
               capture_reasoning: bool = True) -> dict:
    """
    vLLM equivalent of ollama_judge(): same {"response": ..., "thinking": ...}
    return shape, so score_all() can call either one interchangeably via the
    judge_fn parameter. See PORT NOTES at the top of this file for why
    response_format/json_schema and repetition_penalty are here.

    capture_reasoning: toggles reasoning-trace capture (default True, to
    match ollama_judge's think=True default). Generating reasoning
    multiplies tokens-per-call (and wall-clock time) noticeably - pass
    False (--no_reasoning at the CLI) for a faster run when the reasoning
    trace isn't needed.
    """
    schema = SCORE_SCHEMA_WITH_REASONING if capture_reasoning else SCORE_SCHEMA_NO_REASONING
    max_tokens = 1200 if capture_reasoning else 200
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.15,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judge_score", "schema": schema, "strict": True},
        },
    }
    url = f"{VLLM_URL}/v1/chat/completions"
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            if choice.get("finish_reason") == "length":
                print("    [vllm_judge] WARNING: finish_reason=length - response may be "
                      "truncated before reaching the actual score (see PORT NOTES on "
                      "max_tokens budget)")
            thinking = None
            if capture_reasoning:
                try:
                    thinking = json.loads(content).get("reasoning")
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
            return {"response": content.strip(), "thinking": thinking}
        except requests.RequestException as e:
            print(f"    [Attempt {attempt}/{retries}] Judge error: {e}")
            if attempt == retries:
                return {"response": "", "thinking": None}
    return {"response": "", "thinking": None}


def get_registered_models_vllm(base_url: str) -> set:
    resp = requests.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    return {m["id"] for m in resp.json().get("data", [])}


def ensure_judge_model_vllm():
    if not JUDGE_MODEL:
        print("ERROR: JUDGE_MODEL is not set. Set it to the --served-model-name the "
              "vLLM server was launched with (e.g. 'qwen3-27b-judge').")
        sys.exit(1)
    try:
        registered = get_registered_models_vllm(VLLM_URL)
        if JUDGE_MODEL not in registered:
            print(f"ERROR: Judge model '{JUDGE_MODEL}' is not being served at {VLLM_URL}. "
                  f"Currently served: {sorted(registered)}")
            sys.exit(1)
        print(f"Judge model '{JUDGE_MODEL}' is being served at {VLLM_URL}.")
    except requests.RequestException as e:
        print(f"Warning: could not verify judge model at {VLLM_URL}: {e}")


def parse_score(response: str) -> float | None:
    for match in re.finditer(r"\{[^{}]*\}", response):
        try:
            obj = json.loads(match.group())
            if "score" in obj:
                return max(0.0, min(1.0, float(obj["score"])))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    for n in re.findall(r"\b([01](?:\.[0-9]+)?)\b", response):
        try:
            return max(0.0, min(1.0, float(n)))
        except ValueError:
            pass
    return None


# --- Prompt-file loading (PROMPT-FILE ARCHITECTURE - see module docstring) ---

def parse_prompt_file(path):
    """
    Parses one prompt template file: optional YAML-lite front matter
    (--- ... ---) followed by a string.Template body using $scenario /
    $answer / $ground_truth placeholders (plain $name, not {name} - so the
    rubric text is free to contain literal braces, e.g. a JSON example,
    without needing to escape them).

    Recognized front-matter keys:
      requires_ground_truth: true|false (default false)
      score_type: continuous|binary (default continuous)

    Returns (meta_dict, template_body_str).
    """
    text = Path(path).read_text(encoding="utf-8")
    meta = {"requires_ground_truth": False, "score_type": "continuous"}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            front_matter_raw, body = parts[1], parts[2]
            for line in front_matter_raw.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                meta[key] = (val.lower() == "true") if val.lower() in ("true", "false") else val
            text = body
    return meta, text.strip()


def load_prompt_registry(prompts_dir: str) -> dict:
    """
    Loads every *.txt file in prompts_dir as one metric. The filename stem
    (e.g. correctness.txt -> "correctness") becomes both the DB column name
    and the label used in CLI/log output, so keep filenames as valid,
    lowercase SQL-identifier-safe names (no spaces or hyphens).

    Returns {metric_name: {"template": Template, "requires_ground_truth": bool,
                            "score_type": "continuous"|"binary"}}
    """
    paths = sorted(glob.glob(str(Path(prompts_dir) / "*.txt")))
    if not paths:
        print(f"ERROR: no *.txt prompt templates found in {prompts_dir!r}")
        sys.exit(1)
    registry = {}
    for path in paths:
        name = Path(path).stem
        meta, body = parse_prompt_file(path)
        registry[name] = {
            "template": Template(body),
            "requires_ground_truth": bool(meta.get("requires_ground_truth", False)),
            "score_type": meta.get("score_type", "continuous"),
        }
    return registry


def response_instruction(score_type: str, capture_reasoning: bool) -> str:
    """
    Centralizes the "respond with only a JSON object" instruction so prompt
    template files stay free of it - this is what keeps every metric's
    instruction text in sync with whichever schema (SCORE_SCHEMA_WITH_
    REASONING / NO_REASONING) is actually being enforced for this run,
    instead of each template needing its own copy (and risking drifting out
    of sync with the --no_reasoning flag, the way a hand-written "reasoning"
    key baked into a prompt's own JSON instruction could).
    """
    if score_type == "binary":
        if capture_reasoning:
            return ('\n\nRespond with only a JSON object:\n'
                     '{"reasoning": "<brief explanation>", "score": 0} if no issue is present, or\n'
                     '{"reasoning": "<brief explanation>", "score": 1} if a meaningful issue is present.')
        return '\n\nRespond with only a JSON object:\n{"score": 0} or {"score": 1}.'
    if capture_reasoning:
        return ('\n\nRespond with only a JSON object:\n'
                 '{"reasoning": "<brief explanation>", "score": <value between 0.0 and 1.0>}')
    return '\n\nRespond with only a JSON object:\n{"score": <value between 0.0 and 1.0>}'


def build_prompt(entry: dict, scenario: str, answer: str, ground_truth: str,
                  capture_reasoning: bool) -> str:
    rendered = entry["template"].safe_substitute(
        scenario=scenario or "",
        answer=answer or "",
        ground_truth=ground_truth or "",
    )
    return rendered.strip() + response_instruction(entry["score_type"], capture_reasoning)


# --- Ingest generate_judge_examples.py output into the DB ---

def ensure_schema(conn: sqlite3.Connection, metrics: dict):
    """
    metrics: {metric_name: {"score_type": "continuous"|"binary", ...}} as
    returned by load_prompt_registry(). PRIMARY KEY is (run_id, example_id,
    model_key) - see COMBINED-DB / RUN-ID ARCHITECTURE note at the top of
    this file for why. Metric columns are added dynamically via ALTER
    TABLE ADD COLUMN, one per entry in `metrics`, guarded so this is a
    no-op for columns that already exist - a new/renamed metric just adds
    its own column the first time it's scored, and switching --prompts_dir
    never breaks a DB that already has other metrics' columns in it.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            run_id               TEXT NOT NULL,
            run_created_at       TEXT NOT NULL,
            judge_model          TEXT NOT NULL,
            example_id           TEXT NOT NULL,
            model_key            TEXT NOT NULL,
            ollama_model         TEXT,
            difficulty           TEXT,
            scenario             TEXT,
            generated_response   TEXT,
            ground_truth         TEXT,
            bertscore_precision  REAL,
            bertscore_recall     REAL,
            bertscore_f1         REAL,
            PRIMARY KEY (run_id, example_id, model_key)
        )
    """)
    conn.commit()

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    for name, entry in metrics.items():
        col_type = "INTEGER" if entry["score_type"] == "binary" else "REAL"
        if name not in existing_cols:
            conn.execute(f"ALTER TABLE results ADD COLUMN {name} {col_type}")
        reasoning_col = f"{name}_reasoning"
        if reasoning_col not in existing_cols:
            conn.execute(f"ALTER TABLE results ADD COLUMN {reasoning_col} TEXT")
    conn.commit()


def ingest_json(conn: sqlite3.Connection, paths: list, run_id: str,
                 run_created_at: str, judge_model: str):
    """
    Every row ingested by one invocation of this script gets the same
    run_id/run_created_at/judge_model - including rows for multiple
    different --models values ingested from the same --input set, since
    that's still one job (see module docstring). Re-running ingestion with
    the SAME run_id is a no-op for rows already present (INSERT OR IGNORE
    against the (run_id, example_id, model_key) primary key); a NEW run_id
    always inserts a fresh full set of rows for that run, independent of
    whatever other runs already exist in this DB.
    """
    inserted = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            cur = conn.execute("""
                INSERT OR IGNORE INTO results
                    (run_id, run_created_at, judge_model, example_id, model_key,
                     ollama_model, difficulty, scenario, generated_response, ground_truth)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, run_created_at, judge_model,
                  r["example_id"], r["model_key"], r.get("ollama_model"),
                  r.get("difficulty"), r.get("scenario"),
                  r.get("generated_response"), r.get("ground_truth")))
            if cur.rowcount:
                inserted += 1
        conn.commit()
    print(f"Ingested from {len(paths)} file(s) for run_id={run_id!r}. New rows: {inserted}")


# --- BERTScore (batched, not per-row through Ollama) ---

def copy_cached_bertscore(conn: sqlite3.Connection, run_id: str) -> int:
    """
    BERTScore only depends on (generated_response, ground_truth), which are
    identical for a given (example_id, model_key) no matter which judge/run
    scored it. In a combined DB with multiple runs, that means it only
    ever needs to be computed once per example/model - every later run can
    just copy it from whichever earlier row already has it. Scoped to
    THIS run_id's rows only (the ones that still need a value); the source
    of a copy can be any run_id in the DB.
    """
    cache_rows = conn.execute("""
        SELECT example_id, model_key, bertscore_precision, bertscore_recall, bertscore_f1
        FROM results WHERE bertscore_f1 IS NOT NULL
    """).fetchall()
    cache = {}
    for example_id, model_key, p, r, f1 in cache_rows:
        cache.setdefault((example_id, model_key), (p, r, f1))
    if not cache:
        return 0

    missing = conn.execute("""
        SELECT example_id, model_key FROM results
        WHERE run_id = ? AND bertscore_f1 IS NULL
    """, (run_id,)).fetchall()

    copied = 0
    for example_id, model_key in missing:
        key = (example_id, model_key)
        if key in cache:
            p, r, f1 = cache[key]
            conn.execute("""
                UPDATE results SET bertscore_precision = ?, bertscore_recall = ?, bertscore_f1 = ?
                WHERE run_id = ? AND example_id = ? AND model_key = ?
            """, (p, r, f1, run_id, example_id, model_key))
            copied += 1
    conn.commit()
    return copied


def compute_bertscore(conn: sqlite3.Connection, run_id: str, model_filter: list = None,
                       limit: int = None):
    if bertscore_compute is None:
        print("bert-score not installed (pip install bert-score) - skipping BERTScore.")
        return

    if not RESCORE:
        copied = copy_cached_bertscore(conn, run_id)
        if copied:
            print(f"Copied cached BERTScore (from other runs/judges of the same "
                  f"example+model) into {copied} row(s) - skipped recomputation for those.")

    where_clauses = ["run_id = ?"]
    params = [run_id]
    if not RESCORE:
        where_clauses.append("bertscore_f1 IS NULL")
    if model_filter:
        placeholders = ",".join("?" for _ in model_filter)
        where_clauses.append(f"model_key IN ({placeholders})")
        params.extend(model_filter)
    filter_clause = "WHERE " + " AND ".join(where_clauses)
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    rows = conn.execute(f"""
        SELECT example_id, model_key, generated_response, ground_truth
        FROM results {filter_clause}{limit_clause}
    """, params).fetchall()

    if not rows:
        print("No rows need BERTScore.")
        return

    print(f"Computing BERTScore for {len(rows)} row(s)...")
    candidates = [r[2] or "" for r in rows]
    references = [r[3] or "" for r in rows]

    P, R, F1 = bertscore_compute(candidates, references, lang="en", verbose=True)

    for (example_id, model_key, _, _), p, r, f1 in zip(rows, P.tolist(), R.tolist(), F1.tolist()):
        conn.execute("""
            UPDATE results SET bertscore_precision = ?, bertscore_recall = ?, bertscore_f1 = ?
            WHERE run_id = ? AND example_id = ? AND model_key = ?
        """, (p, r, f1, run_id, example_id, model_key))
    conn.commit()
    print("BERTScore complete.")


# --- LLM-judge scoring loop ---

def score_all(conn: sqlite3.Connection, judge_fn, prompt_registry: dict,
              capture_reasoning: bool, run_id: str, model_filter: list = None,
              limit: int = None, example_id: str = None):
    """
    judge_fn: callable(prompt: str) -> {"response": str, "thinking": str | None}

    prompt_registry: {metric_name: {"template", "requires_ground_truth",
    "score_type"}} from load_prompt_registry(). Iterates generically over
    whatever metrics are present - adding/removing a prompt file changes
    which columns get scored, with no change needed here.

    run_id: scopes scoring to THIS run's rows only. Required now that one
    DB can hold multiple runs/judges - without this, an unscored-row query
    could pick up leftover NULL rows from a DIFFERENT run_id sharing the
    same DB file (e.g. an old crashed job that was never resumed), and
    silently score them under the wrong judge_model. See the
    RESUMABILITY CAVEAT in the module docstring for how this interacts
    with resubmitting a timed-out/crashed job.

    example_id: restricts scoring to one specific example_id, across
    whichever model_keys are present for it (or the --models subset, if
    also given) - useful for a direct same-scenario cross-model comparison.
    """
    metric_names = list(prompt_registry.keys())
    where_clauses = ["run_id = ?"]
    params = [run_id]
    if not RESCORE:
        unscored = " OR ".join(f"{m} IS NULL" for m in metric_names)
        where_clauses.append(f"({unscored})")
    if model_filter:
        placeholders = ",".join("?" for _ in model_filter)
        where_clauses.append(f"model_key IN ({placeholders})")
        params.extend(model_filter)
    if example_id:
        where_clauses.append("example_id = ?")
        params.append(example_id)
    filter_clause = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    rows = conn.execute(f"""
        SELECT example_id, model_key, scenario, generated_response, ground_truth
        FROM results {filter_clause}{limit_clause}
    """, params).fetchall()

    if not rows:
        print("No unscored results found. Use RESCORE=1 to re-score all results.")
        return

    print(f"Scoring {len(rows)} result(s) on {len(metric_names)} metric(s) "
          f"({', '.join(metric_names)}) using judge model: {JUDGE_MODEL}\n")

    for i, (example_id, model_key, scenario, answer, ground_truth) in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {example_id} | {model_key}")

        set_clauses = []
        values = []
        printable = []

        for name, entry in prompt_registry.items():
            if entry["requires_ground_truth"] and not ground_truth:
                # Same gate the old code hardcoded for correctness/
                # completeness: no reference to grade against, so leave
                # this metric NULL rather than call the judge on an empty
                # comparison.
                continue

            prompt = build_prompt(entry, scenario, answer, ground_truth, capture_reasoning)
            result = judge_fn(prompt)
            score = parse_score(result["response"])
            if score is None:
                print(f"  Warning: could not parse {name} from: {result['response'][:80]}")
            elif entry["score_type"] == "binary":
                score = int(round(score))

            set_clauses.append(f"{name} = ?")
            values.append(score)
            set_clauses.append(f"{name}_reasoning = ?")
            values.append(result["thinking"])
            printable.append(f"{name}={score}")

        if set_clauses:
            values.extend([run_id, example_id, model_key])
            conn.execute(
                f"UPDATE results SET {', '.join(set_clauses)} "
                f"WHERE run_id = ? AND example_id = ? AND model_key = ?",
                values,
            )
            conn.commit()

        print("  " + "  ".join(printable))

    print(f"\nLLM-judge scoring complete. {len(rows)} result(s) scored.")


def ensure_judge_model():
    if not JUDGE_MODEL:
        print("ERROR: JUDGE_MODEL is not set. Set it to the Ollama name of your "
              "judge model (e.g. 'set JUDGE_MODEL=dentsim-qwen38-judge' or "
              "whatever name it's registered under once you confirm Qwen3.8-27B "
              "or its fallback is Ollama-servable).")
        sys.exit(1)
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        available = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        judge_base = JUDGE_MODEL.split(":")[0]
        if judge_base not in available:
            print(f"ERROR: Judge model '{JUDGE_MODEL}' is not registered with Ollama. "
                  f"Registered: {sorted(available)}")
            sys.exit(1)
        print(f"Judge model '{JUDGE_MODEL}' is available.")
    except requests.RequestException as e:
        print(f"Warning: could not verify judge model: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Score generate_judge_examples.py output: prompt-file-driven LLM-judge metrics + BERTScore")
    p.add_argument("--input", nargs="+", required=True,
                    help="One or more JSON files (or glob patterns) from generate_judge_examples.py")
    p.add_argument("--models", nargs="+", default=None,
                    help="Restrict scoring to these model_key values (e.g. --models phi-4-mini). "
                         "Ingestion is unaffected - all models in --input still get loaded into "
                         "the DB, just not scored yet. Default: score every model present.")
    p.add_argument("--example_id", default=None,
                    help="Restrict scoring to one specific example_id, across whichever "
                         "model_keys have it (combine with --models to narrow further). "
                         "For a direct same-scenario cross-model comparison instead of an "
                         "arbitrary --limit N slice.")
    p.add_argument("--db_file", default=DB_FILE)
    p.add_argument("--run_id", default=None,
                    help="Identifies this job - every example/model scored by this "
                         "invocation shares this run_id (and one run_created_at "
                         "timestamp captured at startup). Default: $SLURM_JOB_ID if set "
                         "(natural unique-per-job id under sbatch), else a generated "
                         "'local-<8 hex>' id. Pass this explicitly to RESUME a "
                         "crashed/timed-out job into the same run rather than starting a "
                         "new one (see RESUMABILITY CAVEAT in the module docstring) - use "
                         "the run_id that job printed at startup.")
    p.add_argument("--prompts_dir", default=DEFAULT_PROMPTS_DIR,
                    help=f"Directory of *.txt judge-prompt templates (default: "
                         f"{DEFAULT_PROMPTS_DIR!r}, from study_config.PROMPTS_DIR if set). "
                         "Each file's stem becomes a scored metric/DB column - point this at a "
                         "different directory to try alternate rubrics without touching code.")
    p.add_argument("--skip_bertscore", action="store_true")
    p.add_argument("--skip_llm_judge", action="store_true")
    p.add_argument("--backend", choices=["ollama", "vllm"], default=BACKEND,
                    help="Which backend to query for LLM-judge scoring (default: "
                         f"'{BACKEND}', from study_config.BACKEND if set, else 'ollama').")
    p.add_argument("--vllm_url", default=VLLM_URL,
                    help=f"vLLM OpenAI-compatible API base URL (default: {VLLM_URL}). "
                         "Only used when --backend vllm.")
    p.add_argument("--limit", type=int, default=None,
                    help="Score only the first N unscored rows (after --models filtering). "
                         "Useful as a quick smoke test before committing to a full run.")
    p.add_argument("--no_reasoning", action="store_true",
                    help="Disable reasoning-trace capture (*_reasoning columns stay NULL). "
                         "Reasoning is ON by default, but capturing it means the judge schema "
                         "requires a 'reasoning' field before 'score', which measurably "
                         "increases tokens generated (and thus wall-clock time) per call. Pass "
                         "--no_reasoning for a faster run when the reasoning trace isn't needed.")
    return p.parse_args()


def main():
    args = parse_args()
    global DB_FILE, VLLM_URL
    DB_FILE = args.db_file
    VLLM_URL = args.vllm_url
    capture_reasoning = not args.no_reasoning

    prompt_registry = load_prompt_registry(args.prompts_dir)
    print(f"Loaded {len(prompt_registry)} metric(s) from {args.prompts_dir!r}: "
          f"{', '.join(prompt_registry)}")

    # run_id/run_created_at: see COMBINED-DB / RUN-ID ARCHITECTURE note at
    # the top of this file. Captured once, here, before any ingestion or
    # scoring - every row this invocation touches shares these two values.
    run_id = args.run_id or os.environ.get("SLURM_JOB_ID") or f"local-{uuid.uuid4().hex[:8]}"
    run_created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"run_id={run_id!r}  run_created_at={run_created_at}  judge_model={JUDGE_MODEL!r}")
    print("(save this run_id if you need to resume this exact run after a crash/timeout)")

    paths = []
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"Warning: no files matched pattern: {pattern}")
        paths.extend(matched)
    if not paths:
        print("No input files found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(DB_FILE) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    try:
        ensure_schema(conn, prompt_registry)
        ingest_json(conn, paths, run_id, run_created_at, JUDGE_MODEL)

        if not args.skip_bertscore:
            compute_bertscore(conn, run_id, model_filter=args.models, limit=args.limit)

        if not args.skip_llm_judge:
            if args.backend == "ollama":
                ensure_judge_model()
                judge_fn = functools.partial(ollama_judge, think=capture_reasoning)
            else:
                ensure_judge_model_vllm()
                judge_fn = functools.partial(vllm_judge, capture_reasoning=capture_reasoning)
            score_all(conn, judge_fn, prompt_registry, capture_reasoning, run_id,
                      model_filter=args.models, limit=args.limit,
                      example_id=args.example_id)
    finally:
        conn.close()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
"""
generate_judge_examples.py
===========================
Component 3 of the LLM-judge validity pipeline: generates new coaching
responses from each tier-1 model on the held-out eval scenarios (same
seed=42, 90/10 split as lora_finetune.py), and consolidates them into one
file for Component 4 (LLM judge) and Component 5 (human expert review).

Two backends, selected with --backend (default: ollama):
  - ollama: Ollama's REST API, unchanged from the original pipeline. Kept
    for local/personal-machine runs (per faculty guidance: Ollama is fine
    on a personal computer, just not on the cluster). Requires each model
    already registered with Ollama via ollama_pipeline.py first; models
    not yet registered are skipped with a warning rather than failing the
    whole run.
  - vllm: vLLM's OpenAI-compatible REST API, for cluster runs (Turing/
    Wahab). Requires a vllm serve process already running and reachable at
    --vllm_url, with the model(s) loaded as either --served-model-name or
    --lora-modules. See config/vllm_models.yaml for the key -> served-name
    mapping (parallel to config/ollama_models.yaml's ollama_name field).
    On Turing V100s, the server MUST be started with
    --pipeline-parallel-size (not --tensor-parallel-size) - tensor-parallel
    silently produces fluent-looking garbage on this vLLM+Volta combination
    without erroring; see learnings.md.

Works against any key present in the relevant backend's config file,
including "-base" entries (untrained comparison points) alongside
fine-tuned ones - --models accepts a mix of both in the same run, e.g.
--models llama-3.1-8B llama-3.1-8B-base. The two questions this can answer
("did LoRA help" vs. "is the judge valid") aren't mutually exclusive; which
one a given run is for depends entirely on which keys you pass in.

Usage:
    python generate_judge_examples.py                                  # ollama, local
    python generate_judge_examples.py --backend vllm                   # vllm, cluster
    python generate_judge_examples.py --backend vllm --vllm_url http://localhost:8000
    python generate_judge_examples.py --models llama-3.2-3B llama-3.1-8B --max_examples 10
    python generate_judge_examples.py --max_examples 2   # smoke test first

Output:
    ./outputs/judge_examples/judge_examples_<timestamp>.json
"""

import argparse
import asyncio
import glob
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import requests
import yaml

# aiohttp powers the concurrent vLLM generation path only — guarded import
# so the Ollama backend (which stays fully synchronous, using `requests`
# as it always has) never requires it. A clear, actionable error is raised
# in __main__ if --backend vllm is actually selected and this is missing,
# rather than failing at a confusing point deep in an async call.
try:
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
except ImportError:
    ClientSession = ClientTimeout = TCPConnector = None

import study_config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# generation/generate_judge_examples.py -> parent.parent is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OLLAMA_CONFIG = PROJECT_ROOT / "config" / "ollama_models.yaml"
DEFAULT_VLLM_CONFIG = PROJECT_ROOT / "config" / "vllm_models.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "judge_examples"

# split_utils.py lives at the project root (shared with lora_finetune.py) —
# add it to sys.path so this script can import it regardless of cwd.
sys.path.insert(0, str(PROJECT_ROOT))
from split_utils import load_all_records, load_frozen_split, stable_id  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading — mirrors lora_finetune.py's split exactly (backend-agnostic,
# unchanged from the direct-transformers version of this script)
# ---------------------------------------------------------------------------

def load_eval_examples(data_dir: str, max_examples: int = None) -> list[dict]:
    """
    Reads the FROZEN eval holdout (split_utils.py), not a live-recomputed
    split. This is a deliberate change from the original implementation,
    which used its own random.shuffle-based split that — despite sharing
    a seed value with lora_finetune.py's dataset.train_test_split — used a
    genuinely different shuffle algorithm and very likely produced a
    DIFFERENT held-out set than what training actually excluded. See
    split_utils.py's module docstring for the full explanation.

    Run `python split_utils.py --data_dir <data_dir>` once (or again, safely,
    whenever the corpus grows) before calling this, or it raises
    FileNotFoundError with instructions.

    max_examples now takes a PREFIX of the frozen eval set, which is fixed
    order — so "the same 200" really is the same 200 across every run,
    every model, every training-size condition, rather than "whatever the
    first N happened to be under a live-recomputed shuffle."
    """
    records = load_all_records(data_dir)
    eval_records, _train_pool = load_frozen_split(data_dir, records)

    log.info("Frozen eval holdout: %d examples", len(eval_records))

    if max_examples:
        eval_records = eval_records[:max_examples]
        log.info("Capped to first %d examples (--max_examples, fixed prefix "
                 "of the frozen eval order)", max_examples)

    examples = []
    for record in eval_records:
        convs = record["conversations"]
        rid = stable_id(record)

        system_turn = next((c for c in convs if c.get("from", "").lower() == "system"), None)
        human_turn = next((c for c in convs if c.get("from", "").lower() in ("human", "user")), None)
        gpt_turn = next((c for c in convs if c.get("from", "").lower() in ("gpt", "assistant")), None)

        if not human_turn or not gpt_turn:
            continue

        user_text = human_turn.get("value", "").strip()
        difficulty = "UNKNOWN"
        for tag in ("ADVANCED", "INTERMEDIATE", "BEGINNER"):
            if f"[{tag}]" in user_text.upper():
                difficulty = tag
                break

        examples.append({
            "id":           f"eval-{rid}",
            "global_index": rid,
            "source":       "frozen_holdout",
            "difficulty":   difficulty,
            "system":       system_turn["value"].strip() if system_turn else "",
            "user":         user_text,
            "ground_truth": gpt_turn.get("value", "").strip(),
        })

    log.info("Prepared %d eval examples with valid structure", len(examples))
    return examples


# ---------------------------------------------------------------------------
# Ollama client (local / personal-machine runs)
# ---------------------------------------------------------------------------

def get_registered_models_ollama(base_url: str) -> set:
    """
    Query Ollama for currently registered model names, stripped of their
    :tag suffix. Ollama always returns names with an implicit tag (e.g.
    "name:latest") even when none was specified at creation — the config's
    ollama_name never includes one, so tags are normalized away here to
    avoid an exact-string mismatch.
    """
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    resp.raise_for_status()
    return {m["name"].split(":")[0] for m in resp.json().get("models", [])}


def generate_via_ollama(base_url: str, ollama_name: str, system: str, prompt: str,
                         temperature: float = 0.3, num_predict: int = 1536,
                         timeout: int = 300) -> dict:
    """
    Query Ollama's /api/generate. temperature/num_predict are passed
    explicitly here as a deliberate match to what's baked into each
    model's Modelfile (ollama_pipeline.py) — keep the two in sync if
    either changes; this does not silently trust the Modelfile defaults.

    think=False: without this, Qwen-architecture models can trigger
    llama.cpp's reasoning-format output parser (common_chat_peg_parse),
    which expects native <think> block structure. A fine-tuned model that
    no longer produces that native format can hit a parser failure -
    anywhere from a stray leading garbage token to a fully empty response
    with no done_reason at all. Costs nothing for non-reasoning models.
    """
    payload = {
        "model":  ollama_name,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think":  False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return {
        "response":    data.get("response", "").strip(),
        "done_reason": data.get("done_reason", "unknown"),
        "eval_count":  data.get("eval_count"),
    }


# ---------------------------------------------------------------------------
# vLLM client (cluster runs — Turing/Wahab)
# ---------------------------------------------------------------------------

def get_registered_models_vllm(base_url: str) -> set:
    """
    Query vLLM's OpenAI-compatible /v1/models for whatever model or LoRA
    adapter name(s) the running server currently has loaded. Unlike Ollama,
    there's no registry beyond "whatever this one server process was
    started with" — this just confirms the server is up and tells you
    what name(s) to request generation against.
    """
    resp = requests.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    return {m["id"] for m in resp.json().get("data", [])}


def generate_via_vllm(base_url: str, served_name: str, system: str, prompt: str,
                       temperature: float = 0.3, num_predict: int = 1536,
                       timeout: int = 300, stop: list = None) -> dict:
    """
    Query vLLM's OpenAI-compatible /v1/chat/completions endpoint.
    served_name is the vLLM --served-model-name or --lora-modules name
    (e.g. 'granite-dentsim'), analogous to ollama_name in the Ollama
    version. Using chat completions (not raw /v1/completions) so vLLM
    applies the model's actual chat template via the tokenizer it was
    started with — confirmed working end-to-end against a real coaching
    prompt on 2026-09-22.

    stop: optional list of stop strings, sent through to the request
    itself. Added after a real incident (Qwen3-8B-Base, 2026-09-23):
    editing the adapter's tokenizer_config.json to correct eos_token did
    NOT change vLLM's actual stopping behavior — --model and --tokenizer
    are separate sources in this setup (base model weights vs. local
    adapter/tokenizer files), and vLLM's stop-token resolution turned out
    to be tied to the --model source, untouched by any local tokenizer
    edit. An explicit `stop` in the request is honored by the
    OpenAI-compatible API regardless of that resolution, so it's the
    reliable fix — see config/vllm_models.yaml's per-model `stop` field,
    which is where this value actually comes from in real use.
    """
    payload = {
        "model": served_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    if stop:
        payload["stop"] = stop
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return {
        "response":    choice["message"]["content"].strip(),
        "done_reason": choice.get("finish_reason", "unknown"),
        "eval_count":  data.get("usage", {}).get("completion_tokens"),
    }


# ---------------------------------------------------------------------------
# vLLM concurrent generation — the actual throughput lever. vLLM's
# continuous batching lets one server process many requests at once, but
# ONLY if the client sends them concurrently; a sequential loop (one
# request, wait, next request — what every generate_via_ollama()-era run
# and the original generate_via_vllm() loop both did) leaves that
# capacity entirely unused. This does NOT touch the Ollama path at all —
# that stays exactly as it always was, sequential, single-server,
# unaffected by anything below.
# ---------------------------------------------------------------------------

async def generate_via_vllm_async(session, base_url: str, served_name: str,
                                   system: str, prompt: str,
                                   temperature: float = 0.3, num_predict: int = 1536,
                                   timeout: int = 300, stop: list = None) -> dict:
    """
    Async twin of generate_via_vllm() — same request shape, same return
    shape, same `stop` parameter and rationale — used under a bounded
    semaphore so many of these can be in flight against one server at
    once. aiohttp rather than requests, since requests has no async
    story; aiohttp is already a transitive dependency of vLLM's own
    server, so no new install should be needed in the vllm-test env this
    runs in.
    """
    payload = {
        "model": served_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    if stop:
        payload["stop"] = stop
    async with session.post(
        f"{base_url}/v1/chat/completions", json=payload,
        timeout=ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    choice = data["choices"][0]
    return {
        "response":    choice["message"]["content"].strip(),
        "done_reason": choice.get("finish_reason", "unknown"),
        "eval_count":  data.get("usage", {}).get("completion_tokens"),
    }


async def _generate_one_vllm(session, semaphore, server_url, served_name, key, ex,
                              temperature, num_predict, timeout, index, total, stop=None):
    """One example's worth of work under the concurrency limit. Returns the
    same result-dict shape __main__ builds for every backend, or None on
    failure (caller filters Nones out) — mirrors the sequential loop's
    "log and skip" behavior on a per-example request failure, just async."""
    async with semaphore:
        log.info("[%d/%d] %s: %s (%s)", index, total, key, ex["id"], ex["difficulty"])
        try:
            result = await generate_via_vllm_async(
                session, server_url, served_name, ex["system"], ex["user"],
                temperature=temperature, num_predict=num_predict, timeout=timeout, stop=stop,
            )
        except Exception as exc:  # noqa: BLE001 - mirrors the sequential loop's broad catch+log+skip
            log.error("  FAILED: %s: %s", ex["id"], exc)
            return None

        if result["done_reason"] == "length":
            log.warning("  TRUNCATED %s at num_predict=%d (eval_count=%s)",
                        ex["id"], num_predict, result["eval_count"])

        return {
            "model_key":          key,
            "ollama_model":       served_name,  # see NOTE at the sequential loop's own record-building
            "backend":            "vllm",
            "example_id":         ex["id"],
            "global_index":       ex["global_index"],
            "difficulty":         ex["difficulty"],
            "source":             ex["source"],
            "scenario":           ex["user"],
            "generated_response": result["response"],
            "done_reason":        result["done_reason"],
            "ground_truth":       ex["ground_truth"],
        }


async def run_vllm_generation_for_model(server_url, served_name, key, examples,
                                         temperature, num_predict, timeout, max_concurrent,
                                         stop=None):
    """Fires generation for every example against one served model, up to
    max_concurrent in flight at once. Order of results is NOT guaranteed
    to match `examples`' order (whichever finishes first returns first) —
    fine here since every record carries its own example_id/global_index,
    and nothing downstream (score_judge_examples.py) assumes ordering.

    stop: optional list of stop strings for this model specifically —
    see generate_via_vllm()'s docstring for why this exists. Comes from
    the per-model `stop` field in config/vllm_models.yaml, resolved by
    __main__ before this is called."""
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = TCPConnector(limit=max_concurrent + 4)  # a little slack above the semaphore
    async with ClientSession(connector=connector) as session:
        tasks = [
            _generate_one_vllm(
                session, semaphore, server_url, served_name, key, ex,
                temperature, num_predict, timeout, i + 1, len(examples), stop=stop,
            )
            for i, ex in enumerate(examples)
        ]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate Component-3 examples via Ollama or vLLM for each tier-1 model")
    p.add_argument("--backend", choices=["ollama", "vllm"], default=getattr(cfg, "BACKEND", "ollama"),
                    help="Which serving backend to generate against. 'ollama' for local/"
                         "personal-machine runs, 'vllm' for cluster runs (Turing/Wahab). "
                         "Default is 'ollama' so existing local workflows are unaffected; "
                         "pass --backend vllm explicitly for cluster runs.")
    p.add_argument("--ollama_config", default=str(DEFAULT_OLLAMA_CONFIG),
                    help="[ollama backend] Config mapping model keys to registered Ollama "
                         "names (from ollama_pipeline.py)")
    p.add_argument("--vllm_config", default=str(DEFAULT_VLLM_CONFIG),
                    help="[vllm backend] Config mapping model keys to vLLM served/LoRA "
                         "names (see config/vllm_models.yaml)")
    p.add_argument("--models", nargs="+", default=cfg.STUDY_MODELS or None,
                    help="Subset of config keys to run. Defaults to study_models "
                         "in config/study_config.yaml; falls back to all models "
                         "in the selected backend's config if study_models is empty. "
                         "Accepts '-base' keys alongside fine-tuned ones in the same run.")
    p.add_argument("--data_dir", default=cfg.DATA_DIR)
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--max_examples", type=int, default=cfg.MAX_EXAMPLES,
                    help="Held-out examples per model. Config default (see "
                         "config/study_config.yaml) is currently the full "
                         "~188-example held-out set (10%% of the corpus). "
                         "Pass 0 to force the full set regardless of the "
                         "config value.")
    p.add_argument("--ollama_url", default=cfg.OLLAMA_BASE_URL,
                    help="[ollama backend] Ollama API endpoint.")
    p.add_argument("--vllm_url", default=getattr(cfg, "VLLM_BASE_URL", "http://localhost:8000"),
                    help="[vllm backend] vLLM OpenAI-compatible API endpoint.")
    p.add_argument("--temperature", type=float, default=cfg.TEMPERATURE)
    p.add_argument("--num_predict", type=int, default=cfg.NUM_PREDICT,
                    help="512 clipped ~50%% of training responses (median was exactly "
                         "512 tokens). Config default raises this to 1536.")
    p.add_argument("--timeout", type=int, default=cfg.TIMEOUT,
                    help="Per-request timeout in seconds. 8B models generating up to "
                         "num_predict tokens can take a while.")
    p.add_argument("--max_concurrent", type=int, default=getattr(cfg, "MAX_CONCURRENT", 12),
                    help="[vllm backend only] Max requests in flight at once against a "
                         "single server. Ignored for --backend ollama (stays sequential). "
                         "Default 12 leaves headroom under the ~13.97x concurrent-request "
                         "capacity vLLM itself reported for a PP2 Granite server at "
                         "max-model-len=4096 (2026-09-22 server log) — raise cautiously if "
                         "you've confirmed a higher number for your own server config, "
                         "since going over the server's real KV-cache capacity causes "
                         "requests to queue rather than truly run in parallel, not an error.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backend-specific setup: which config file to read, which field holds
    # the name to request generation against, which client/registry
    # functions to call. Everything below this block (the eval-example
    # loop) is backend-agnostic.
    if args.backend == "ollama":
        config_path, name_field = args.ollama_config, "ollama_name"
        server_url = args.ollama_url
        get_registered = get_registered_models_ollama
        generate = generate_via_ollama
        log.info("Backend: ollama (%s)", server_url)
    else:
        config_path, name_field = args.vllm_config, "served_name"
        server_url = args.vllm_url
        get_registered = get_registered_models_vllm
        generate = generate_via_vllm
        log.info("Backend: vllm (%s)", server_url)

    with open(config_path, encoding="utf-8") as f:
        backend_cfg = yaml.safe_load(f)
    all_models = backend_cfg["models"]
    selected = args.models or list(all_models.keys())

    log.info("Checking which models are actually being served...")
    registered = get_registered(server_url)
    log.info("Registered: %s", sorted(registered))

    examples = load_eval_examples(args.data_dir, max_examples=args.max_examples)
    if not examples:
        log.error("No eval examples found. Check --data_dir path.")
        raise SystemExit(1)

    if args.backend == "vllm" and ClientSession is None:
        log.error("--backend vllm needs aiohttp for concurrent generation, and it's not "
                  "installed in this environment. pip install aiohttp (it's very likely "
                  "already present in ~/envs/vllm-test as a vLLM server dependency — "
                  "check you're running this under `crun -p ~/envs/vllm-test python ...`).")
        raise SystemExit(1)

    all_results = []
    truncated_count = 0

    for key in selected:
        if key not in all_models:
            log.error("Unknown model key: %s (not in %s)", key, config_path)
            continue

        served_name = all_models[key][name_field]
        if served_name not in registered:
            log.warning("Skipping %s - '%s' not currently being served (%s backend)",
                        key, served_name, args.backend)
            continue

        log.info("=" * 70)
        log.info("Generating with: %s (%s)", key, served_name)
        log.info("=" * 70)

        if args.backend == "vllm":
            # Concurrent path — up to --max_concurrent requests in flight
            # against this one server at once, exploiting vLLM's continuous
            # batching instead of the one-request-at-a-time pattern below.
            log.info("Generating with up to %d requests in flight...", args.max_concurrent)
            # Per-model stop sequence, if config/vllm_models.yaml declares
            # one for this key (see generate_via_vllm()'s docstring for
            # why this exists — a real incident, not speculative).
            model_stop = all_models[key].get("stop")
            if model_stop:
                log.info("Using explicit stop sequence for %s: %s", key, model_stop)

            model_results = asyncio.run(run_vllm_generation_for_model(
                server_url, served_name, key, examples,
                args.temperature, args.num_predict, args.timeout, args.max_concurrent,
                stop=model_stop,
            ))
            truncated_count += sum(1 for r in model_results if r["done_reason"] == "length")
            all_results.extend(model_results)
            log.info("  %d/%d generations succeeded for %s", len(model_results), len(examples), key)
            continue

        # Sequential path — unchanged from before concurrency existed at
        # all. Ollama backend only; stays exactly as it always has.
        for i, ex in enumerate(examples):
            log.info("[%d/%d] %s: %s (%s)", i + 1, len(examples), key, ex["id"], ex["difficulty"])
            try:
                result = generate(
                    server_url, served_name, ex["system"], ex["user"],
                    temperature=args.temperature, num_predict=args.num_predict,
                    timeout=args.timeout,
                )
            except requests.RequestException as exc:
                log.error("  FAILED: %s", exc)
                continue

            if result["done_reason"] == "length":
                truncated_count += 1
                log.warning("  TRUNCATED at num_predict=%d (eval_count=%s)",
                            args.num_predict, result["eval_count"])

            all_results.append({
                "model_key":          key,
                # NOTE: keeping the key name "ollama_model" (not "served_model")
                # even for the vllm backend — score_judge_examples.py's
                # ingest_json() reads r.get("ollama_model") specifically when
                # loading rows into the DB. Renaming this key would silently
                # insert NULL for every vllm-generated row rather than erroring.
                # "backend" is a new, additive field score_judge_examples.py
                # simply ignores (its INSERT only pulls named fields).
                "ollama_model":       served_name,
                "backend":            args.backend,
                "example_id":         ex["id"],
                "global_index":       ex["global_index"],
                "difficulty":         ex["difficulty"],
                "source":             ex["source"],
                "scenario":           ex["user"],
                "generated_response": result["response"],
                "done_reason":        result["done_reason"],
                "ground_truth":       ex["ground_truth"],
            })

    json_path = output_dir / f"judge_examples_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log.info("=" * 70)
    log.info("Done. %d total generations saved to %s", len(all_results), json_path)
    if truncated_count:
        log.warning("%d/%d generations hit the num_predict cap and were truncated - "
                     "consider raising --num_predict if this is more than a handful.",
                     truncated_count, len(all_results))
    log.info("=" * 70)
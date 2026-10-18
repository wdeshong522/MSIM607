"""
LoRA Fine-Tuning Script for Ollama Models
==========================================
Reads ShareGPT-formatted .jsonl exports, prepares data, and runs LoRA training
via Hugging Face PEFT + TRL. The resulting adapter can then be merged and
re-imported into Ollama.

Dependencies:
    pip install torch transformers peft trl datasets bitsandbytes accelerate pyyaml

Usage:
    python lora_finetune.py \
        --model_name <hf-model-id or local path> \
        --data_dir   ./data \
        --output_dir ./outputs/lora-adapter \
        [--epochs 3] [--batch_size 2] [--lr 2e-4]

    # Or via a config file (see FALLBACK_DEFAULTS / KNOWN_CONFIG_KEYS below):
    python lora_finetune.py --config config/training_sweep/granite-4.1-8B_r32.yaml

    Precedence (highest to lowest): CLI flag > --config file > FALLBACK_DEFAULTS.
    Any CLI flag passed alongside --config still overrides that file's value —
    useful for a quick one-off override without editing the config. If you
    override a tunable flag (e.g. --lora_r 32) without also passing an
    explicit --output_dir, output_dir is auto-suffixed with the override
    (e.g. ..._lora_r32) so sweep candidates never silently overwrite each
    other's adapter or eval files. Pass --output_dir yourself to opt out.

After training, merge & export:
    python lora_finetune.py --merge_only \
        --model_name <base-model> \
        --adapter_dir ./outputs/lora-adapter \
        --output_dir  ./outputs/merged-model
"""

import os
import sys
import json
import glob
import math
import time
import statistics
import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# training/lora_finetune.py -> parent.parent is the project root, where
# split_utils.py lives (shared with generation/generate_judge_examples.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from split_utils import load_all_records, load_frozen_split  # noqa: E402


# ---------------------------------------------------------------------------
# 1. JSONL Loading
# ---------------------------------------------------------------------------

def load_sharegpt_jsonl(data_dir: str) -> list[dict]:
    """
    Recursively find and load every .jsonl file in `data_dir`.

    Expected ShareGPT record shape:
        {
          "conversations": [
            {"from": "human",  "value": "..."},
            {"from": "gpt",    "value": "..."},
            ...
          ]
        }

    Returns a flat list of conversation records (one record = one training
    example with potentially many turns).
    """
    pattern = os.path.join(data_dir, "**", "*.jsonl")
    files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {data_dir}")

    records = []
    skipped = 0

    for filepath in sorted(files):
        log.info("Loading %s", filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed JSON at %s line %d: %s", filepath, lineno, exc)
                    skipped += 1
                    continue

                # Validate minimal ShareGPT structure
                if "conversations" not in obj or not isinstance(obj["conversations"], list):
                    log.warning("Skipping record without 'conversations' at %s line %d", filepath, lineno)
                    skipped += 1
                    continue

                records.append(obj)

    log.info("Loaded %d records (%d skipped) from %d file(s)", len(records), skipped, len(files))
    return records


# ---------------------------------------------------------------------------
# 2. Data Preparation — ShareGPT → chat-template text
# ---------------------------------------------------------------------------

# Jinja2 ChatML template — a safe, widely-used default for models that ship
# without a built-in chat_template (e.g. base Mistral, LLaMA-2 base, etc.).
# Fine-tuned / instruct variants usually have their own template already set.
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)

def ensure_chat_template(tokenizer) -> None:
    """
    If the tokenizer has no chat_template, assign the ChatML default and warn
    the user. Raises if the tokenizer still can't render after assignment.
    """
    if getattr(tokenizer, "chat_template", None):
        return  # already set — nothing to do

    log.warning(
        "Tokenizer has no chat_template. Falling back to ChatML "
        "(<|im_start|> / <|im_end|>). Pass --chat_template <jinja2-string> "
        "on the CLI to override."
    )
    tokenizer.chat_template = _CHATML_TEMPLATE


ROLE_MAP = {
    "human":  "user",
    "user":   "user",
    "gpt":    "assistant",
    "assistant": "assistant",
    "system": "system",
}

def sharegpt_to_messages(record: dict) -> list[dict]:
    """Convert a single ShareGPT record into an OpenAI-style messages list."""
    messages = []
    for turn in record["conversations"]:
        role = ROLE_MAP.get(turn.get("from", "").lower())
        text = turn.get("value", "").strip()
        if role is None or not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def build_dataset(records: list[dict], tokenizer, model_name: str = "") -> Dataset:
    """
    Apply the tokenizer's chat template to every conversation and return a
    Hugging Face Dataset with a single 'text' column ready for SFTTrainer.
    """
    texts = []
    bad = 0
    is_gemma = "gemma" in model_name.lower()
    # Qwen3's chat template unconditionally wraps the FINAL assistant turn
    # in a <think>\n...\n</think>\n\n block (see the template's own logic:
    # `if loop.last or (not loop.last and reasoning_content)` -- loop.last
    # is True for every single-turn training example's assistant message,
    # short-circuiting BEFORE reasoning_content is even checked). Neither
    # enable_thinking=False nor an explicit reasoning_content="" changes
    # this -- confirmed by direct test, not assumption (2026-09-24): both
    # produced byte-identical output to the unmodified call. Every Qwen
    # adapter trained via apply_chat_template() was therefore trained with
    # this invisible wrapper prepended to every single response, which
    # tokenizer_config.json /raw data never shows (it's inserted by the
    # template at render time). This is the confirmed root cause of every
    # Qwen-specific serving symptom from the 2026-09-22/23 incidents:
    # leading garbage tokens, 100% truncation, and --reasoning-parser
    # qwen3 making things worse (an unreliable close of the </think> the
    # model was trained to always open). Bypass the template entirely for
    # Qwen and hand-build plain ChatML instead -- verified to produce
    # token 151667 (<think>) nowhere in the sequence, with assistant
    # content starting immediately on the real text.
    is_qwen = "qwen" in model_name.lower()

    for rec in records:
        messages = sharegpt_to_messages(rec)
        if len(messages) < 2:
            bad += 1
            continue

        # Gemma's chat template does not support a system role.
        # Merge the system message into the first user turn instead.
        if is_gemma:
            system_msgs = [m for m in messages if m["role"] == "system"]
            other_msgs  = [m for m in messages if m["role"] != "system"]
            if system_msgs and other_msgs and other_msgs[0]["role"] == "user":
                other_msgs[0]["content"] = (
                    system_msgs[0]["content"] + "\n\n" + other_msgs[0]["content"]
                )
            messages = other_msgs

        try:
            if is_qwen:
                # Hand-built ChatML -- bypasses apply_chat_template()'s
                # reasoning-aware branching entirely, since no combination
                # of its kwargs suppresses the <think> wrap for a final
                # assistant turn (verified directly, see comment above).
                formatted = "".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                    for m in messages
                )
            else:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append({"text": formatted})
        except Exception as exc:
            log.warning("Skipping record due to template error: %s", exc)
            bad += 1

    log.info("Dataset: %d training examples prepared, %d dropped", len(texts), bad)
    if not texts:
        raise ValueError("No valid training examples. Check your .jsonl files.")

    return Dataset.from_list(texts)


# ---------------------------------------------------------------------------
# 3. Model + LoRA Configuration
# ---------------------------------------------------------------------------

def _gpu_supports_fast_bf16() -> bool:
    """
    torch.cuda.is_bf16_supported() defaults to including_emulation=True,
    which returns True on any CUDA 11+ device — including pre-Ampere GPUs
    like V100 — because PyTorch can emulate bf16 by upcasting to fp32
    under the hood. That's real support in the sense that it won't error,
    but it's far slower than genuine bf16 tensor-core math. Ampere
    (compute capability 8.0) is where bf16 actually gets hardware
    acceleration, so check compute capability directly rather than
    trusting is_bf16_supported()'s permissive default.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _distributed_device_map():
    """
    device_map="auto" is for single-process, multi-GPU model SHARDING
    (splitting one model instance across several GPUs — the inference-time
    / model-parallel pattern). Under DDP (multiple processes, each with its
    own full model copy), it's a documented hard error in accelerate:
    "You can't train a model that has been loaded with device_map='auto'
    in any distributed mode." WORLD_SIZE > 1 is set by Ray Train,
    `accelerate launch`, and `torchrun` alike, so it's a reliable signal
    across all three launch methods — when present, pin this process to
    its own single GPU (LOCAL_RANK) explicitly instead.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return {"": local_rank}
    return "auto" if torch.cuda.is_available() else None


def load_base_model(model_name: str, use_4bit: bool = True):
    """Load the base causal-LM, optionally in 4-bit (QLoRA) mode."""
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        # BF16 tensor cores only exist from Ampere (A100) onward. On
        # pre-Ampere GPUs (e.g. V100/Volta), torch.cuda.is_bf16_supported()
        # still returns True (it allows a slow fp32-emulated fallback by
        # default) — check compute capability directly instead, via
        # _gpu_supports_fast_bf16(), or this silently runs several times
        # slower without ever raising an error.
        compute_dtype = torch.bfloat16 if _gpu_supports_fast_bf16() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        log.info("Using 4-bit (QLoRA) quantization, compute dtype: %s", compute_dtype)
    else:
        log.info("Loading model in full precision (no GPU or 4-bit disabled)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=_distributed_device_map(),
        trust_remote_code=True,
        attn_implementation = "eager" if any(substring in model_name.lower() for substring in ["phi", "gemma"]) else None,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.config.pretraining_tp = 1
    return model


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> LoraConfig:
    """
    Build a LoRA config.

    `target_modules` is auto-detected if None; common overrides:
        LLaMA/Mistral family : ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
        Falcon               : ["query_key_value"]
        GPT-NeoX / Phi       : ["query_key_value"]
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,   # None → PEFT auto-selects linear layers
    )


# ---------------------------------------------------------------------------
# 3b. Effective config — self-documentation per run
# ---------------------------------------------------------------------------

def save_effective_config(args, output_dir: str) -> dict:
    """
    Write the fully-resolved hyperparameters (already merged in `args` from
    --config, any CLI overrides, and FALLBACK_DEFAULTS) to output_dir, so
    each run directory is self-documenting — you can look at
    effective_config.json months later and know exactly what produced the
    eval_loss_per_example.json sitting next to it, without having to
    remember which CLI flags were passed on top of which config file.
    """
    effective = {
        "model_name":    args.model_name,
        "data_dir":      args.data_dir,
        "output_dir":    args.output_dir,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "grad_accum":    args.grad_accum,
        "lr":            args.lr,
        "max_seq_len":   args.max_seq_len,
        "lora_r":        args.lora_r,
        "lora_alpha":    args.lora_alpha,
        "lora_dropout":  args.lora_dropout,
        "no_4bit":       args.no_4bit,
        "attn_only":     args.attn_only,
        "chat_template": args.chat_template,
        "config_file":   args.config,   # None if no --config was passed
        "train_subset_size": args.train_subset_size,  # None = full available pool
        "num_gpus":      args.num_gpus,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "effective_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(effective, f, indent=2)
    log.info("Effective config written to %s", path)
    return effective


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def _is_main_process(args) -> bool:
    """
    True in single-process mode, or for rank 0 when running distributed
    (Ray Train or otherwise, detected via WORLD_SIZE — same signal
    _distributed_device_map() uses). Gates one-time filesystem writes
    (effective_config.json, the saved adapter, eval_loss_per_example.json)
    so N workers training the same model don't race on the same output
    files — trainer.train() itself still runs on every worker, since DDP
    is a synchronized collective operation; only the POST-training
    bookkeeping is rank-0-only.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return True
    import ray.train
    return ray.train.get_context().get_world_rank() == 0


def train(args):
    _train_start_time = time.time()

    # Diagnostic, not a fix: confirm which torch.distributed backend is
    # actually in use before assuming NCCL. Ray Train's process group is
    # already set up by the time this function runs (its "Setting up
    # process group" log line fires before any of this file's own logging
    # does), so this reflects the real, final backend choice. If this
    # prints "gloo" instead of "nccl", that alone would explain the total
    # absence of NCCL_DEBUG output AND the ~5.5x step-time slowdown —
    # gloo routes GPU tensor sync through CPU memory rather than using
    # GPU-direct communication.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        log.info("torch.distributed backend in use: %s", torch.distributed.get_backend())

    # Record the fully-resolved hyperparameters first, before any GPU work
    # starts — so even a run that crashes or OOMs partway through still
    # leaves behind a record of what it was trying to do. Rank-0-only under
    # DDP so N workers don't race writing the same file.
    if _is_main_process(args):
        save_effective_config(args, args.output_dir)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Fast tokenizer failed (%s), retrying with use_fast=False", exc)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, use_fast=False
        )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply a user-supplied template or fall back to ChatML if none is set
    if args.chat_template:
        tokenizer.chat_template = args.chat_template
        log.info("Using chat template from --chat_template argument")
    else:
        ensure_chat_template(tokenizer)

    # Guard against a real bug found the hard way (Qwen3-8B-Base,
    # 2026-09-23): the template above closes every turn with '<|im_end|>',
    # but Qwen3-8B-Base's tokenizer ships with eos_token='<|endoftext|>' by
    # default (correct for raw-text pretraining, wrong for this chat
    # template) — nothing in this pipeline updated it to match. Effect:
    # the fine-tuned model correctly learns to emit '<|im_end|>' to signal
    # "done", but vLLM's stopping logic watches for whatever token
    # tokenizer_config.json calls eos_token — a token the model was never
    # trained to produce in this context — so it never recognizes the
    # model's real stop signal and generation runs to the hard max_tokens
    # cap every single time (confirmed: 100% truncation rate in production
    # use, and under concurrent serving this caused sustained maximal
    # KV-cache pressure that OOM-killed the vLLM server mid-run). This
    # check is model-agnostic — it looks at what the ACTIVE template
    # actually closes a turn with, not a hardcoded model name, so it also
    # protects any future base model fine-tuned with a manually-applied
    # chat template the same way.
    if "<|im_end|>" in (tokenizer.chat_template or "") and tokenizer.eos_token != "<|im_end|>":
        log.warning(
            "Chat template closes turns with '<|im_end|>' but tokenizer.eos_token "
            "is '%s' — the model would learn to emit '<|im_end|>' as its stop "
            "signal, but nothing downstream (including vLLM at inference time) "
            "would recognize it as EOS. Overriding eos_token to '<|im_end|>' to "
            "match what's actually being trained. (See lora_finetune_ray.py "
            "learnings for the 2026-09-23 Qwen3-8B-Base incident this guards "
            "against.)", tokenizer.eos_token,
        )
        tokenizer.eos_token = "<|im_end|>"
        if tokenizer.pad_token in (None, "<|endoftext|>"):
            tokenizer.pad_token = tokenizer.eos_token

    # ── Data ───────────────────────────────────────────────────────────────
    # Reads the FROZEN train pool (split_utils.py), not a live-recomputed
    # split — the judge-reserved examples (permanently excluded from every
    # training condition, at every size) are never in this pool at all.
    # This replaced dataset.train_test_split(seed=42) directly on
    # load_sharegpt_jsonl() output, which used a different shuffle
    # algorithm than generate_judge_examples.py's own split and very
    # likely drew from a different — and non-frozen — set of examples.
    # Run `python split_utils.py --data_dir <data_dir>` once first, or this
    # raises FileNotFoundError with instructions.
    all_records = load_all_records(args.data_dir)
    _judge_records, train_pool = load_frozen_split(args.data_dir, all_records)

    if args.train_subset_size:
        if args.train_subset_size > len(train_pool):
            raise ValueError(
                f"--train_subset_size {args.train_subset_size} exceeds the "
                f"available training pool ({len(train_pool)} examples, after "
                f"excluding the judge reservation). Lower it or grow the corpus."
            )
        # A PREFIX of the frozen pool order — this is what makes subset
        # sizes nest (1000 subset ⊆ 2000 subset ⊆ 4000 subset ⊆ ...) rather
        # than being independent random draws that could overlap partially.
        records = train_pool[:args.train_subset_size]
        log.info("Training subset: %d of %d available pool examples "
                 "(judge-reserved examples excluded; fixed prefix of the "
                 "frozen pool order)", len(records), len(train_pool))
    else:
        records = train_pool
        log.info("Using full available training pool: %d examples "
                 "(judge-reserved examples excluded)", len(records))

    dataset = build_dataset(records, tokenizer, model_name=args.model_name)

    # This 90/10 split is CONDITION-LOCAL — a fresh internal train/val split
    # of just this condition's own N-example pool, purely so this run can
    # watch its own eval loss for overfitting. It is NOT the same thing as,
    # and has no relationship to, the fixed global judge reservation above
    # (which this condition never sees at all, in either half of this split).
    split = dataset.train_test_split(test_size=0.10, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    log.info("Train: %d | Internal val (overfitting check): %d", len(train_ds), len(eval_ds))

    # ── Model + LoRA ───────────────────────────────────────────────────────
    model = load_base_model(args.model_name, use_4bit=not args.no_4bit)
    # Target modules vary by model architecture.
    # NOTE: keep this dict manually in sync with smoke_test.py — there's no
    # shared import between the two scripts, so a change to one won't be
    # reflected in the other automatically.
    TARGET_MODULES = {
        "phi":       ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gemma":     ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "ministral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama":     ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen":      ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "granite":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    def get_target_modules(model_name: str, attn_only: bool = False) -> list[str]:
        if attn_only:
            # Attention-only candidate for the Stage 3 target-modules sweep.
            # q/k/v/o naming is consistent across all seven families in
            # TARGET_MODULES above, so no per-architecture lookup is needed
            # here the way the full attn+MLP list requires.
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        model_name_lower = model_name.lower()
        for key in TARGET_MODULES:
            if key in model_name_lower:
                return TARGET_MODULES[key]
        log.warning(
            "No target_modules match for '%s'; falling back to q_proj/v_proj "
            "only. This LoRA adapter will have far less capacity than the "
            "rest of the roster. Add an entry to TARGET_MODULES instead of "
            "relying on this fallback.",
            model_name,
        )
        return ["q_proj", "v_proj"]  # safe fallback

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=get_target_modules(args.model_name, attn_only=args.attn_only),
        task_type=TaskType.CAUSAL_LM,
    )
    # Required when using 4-bit (QLoRA) — prepares the quantized model for
    # gradient computation and resolves the PreTrainedModel import error that
    # arises from PEFT/transformers version mismatches with quantized models.
    if not args.no_4bit and torch.cuda.is_available():
        use_gc = "gemma" not in args.model_name.lower()
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gc
        )
    elif "gemma" not in args.model_name.lower():
        # Full-precision / --no_4bit path skips prepare_model_for_kbit_training
        # above, but gradient_checkpointing is still turned on below
        # (gradient_checkpointing="gemma" not in ... in sft_config) regardless
        # of quantization. With the base model frozen (only the LoRA adapter
        # trains), checkpointed layers see no input requiring grad unless this
        # is called explicitly — prepare_model_for_kbit_training does this
        # internally on the quantized path, so this is its equivalent here.
        # Without it: "RuntimeError: element 0 of tensors does not require
        # grad and does not have a grad_fn" on the first backward pass.
        model.enable_input_require_grads()

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Training arguments ─────────────────────────────────────────────────
    # SFTConfig is the modern replacement for passing TrainingArguments to
    # SFTTrainer. SFT-specific options (dataset_text_field, max_seq_length,
    # packing) now live here instead of as SFTTrainer kwargs.
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing="gemma" not in args.model_name.lower(),
        # use_reentrant=False: the default ("reentrant") checkpointing mode
        # re-executes part of the forward pass DURING backward, which fires
        # DDP's per-parameter "mark ready" hook twice for the same parameter
        # within one iteration — DDP raises "Expected to mark a variable
        # ready only once" for this (confirmed on Turing, --num_gpus 4,
        # 2026-09-22; harmless on --num_gpus 1, since DDP's hooks aren't in
        # play there at all). Non-reentrant checkpointing doesn't have this
        # interaction and is the generally-recommended mode regardless of
        # DDP. Only matters when args.num_gpus > 1, but harmless to set
        # unconditionally.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # DDP-specific: without this explicit False, HF's default heuristic
        # for ddp_find_unused_parameters often resolves to True for PEFT/LoRA
        # models — DDP then re-traverses the ENTIRE autograd graph every
        # step to detect which parameters didn't get a gradient. With a
        # ~49M-trainable / 8B-frozen split, that traversal touches almost
        # the whole model for zero benefit each iteration, since which
        # parameters are frozen never changes between steps. Confirmed on
        # Turing, --num_gpus 4, 2026-09-22: this was very likely the cause
        # of steps taking ~37s (vs. ~6.7s on --num_gpus 1) at only 13-24%
        # GPU utilization — low utilization + high step time together point
        # at graph-traversal/sync overhead dominating, not real compute.
        # Only takes effect when args.num_gpus > 1 (DDP); harmless no-op
        # for single-GPU runs.
        ddp_find_unused_parameters=False,
        learning_rate=args.lr,
        weight_decay=0.001,
        fp16=not _gpu_supports_fast_bf16() and torch.cuda.is_available(),
        bf16=_gpu_supports_fast_bf16(),
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        # load_best_model_at_end=True (as it was) makes HF Trainer RELOAD
        # the best checkpoint's adapter weights onto the GPU at the very
        # end of training, via a separate safetensors load — an extra
        # allocation on top of whatever's already resident (model,
        # optimizer state, activations), at the exact point in the run
        # where CUDA memory fragmentation is highest. Confirmed as the
        # exact, sole cause of every failure in the first overnight batch
        # (Granite nfull, Qwen n=1000/2000/nfull — all four failed with
        # identical stack traces: _load_best_model() -> load_adapter() ->
        # safe_load_file() -> "CUDA error: out of memory", 2026-09-23).
        # This project already has its own, deliberate post-training
        # quality check — compute_per_example_eval_loss(), called right
        # after trainer.train() returns — so HF's built-in "reload best"
        # behavior was redundant on top of being the actual crash cause.
        # Disabled: the LAST checkpoint (already resident, no reload
        # needed) is what gets saved instead, via the existing
        # trainer.save_model() call.
        load_best_model_at_end=False,
        report_to="tensorboard",
        packing=False,
        dataset_text_field="text",        # ← add this
        max_seq_length=args.max_seq_len,  # ← add this
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info("Starting LoRA training …")
    # If output_dir already has checkpoint-N subdirectories (e.g. a prior
    # attempt hit the partition's wall-time limit before finishing),
    # resume from the latest one instead of starting over — this makes
    # resubmitting the exact same command after a timeout pick up where
    # it left off automatically.
    last_checkpoint = None
    if Path(args.output_dir).is_dir():
        ckpts = sorted(
            Path(args.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if ckpts:
            last_checkpoint = str(ckpts[-1])
            log.info("Found existing checkpoint — resuming from: %s", last_checkpoint)
            # Resuming loads rng_state.pth via torch.load(weights_only=True)
            # (a security default) internally in Trainer._load_rng_state().
            # Allowlisting individual globals one at a time
            # (numpy.core.multiarray._reconstruct, then _codecs.encode, ...)
            # turned out to be whack-a-mole — legacy numpy RandomState
            # pickles reference several. Since this file is always our own
            # checkpoint from a training run we just ran ourselves, not a
            # file from an untrusted source (the scenario weights_only
            # actually protects against), it's safe to just make torch.load
            # default to weights_only=False for the remainder of this
            # process instead of continuing to chase individual globals.
            _original_torch_load = torch.load
            def _torch_load_trusted(*args, **kwargs):
                # NOTE: setdefault() here would NOT work — HF Trainer's
                # own _load_rng_state() explicitly passes weights_only=True
                # itself (confirmed via traceback, 2026-09-22), so the key
                # is already present and setdefault is a no-op. Force the
                # override instead, since we've already established this
                # file is our own trusted checkpoint regardless of what
                # the internal caller requests.
                kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)
            torch.load = _torch_load_trusted

    # Ray Train wiring — ONLY when running distributed (args.num_gpus > 1;
    # see launch_training() below, which is what actually starts N worker
    # processes with WORLD_SIZE set). In single-GPU mode this block is
    # skipped entirely and trainer.train() runs exactly as it always did —
    # ray is not even imported, so single-GPU/local runs have no new
    # dependency. prepare_trainer() does NOT reimplement DDP: HF
    # Trainer/accelerate already knows how to do standard
    # DistributedDataParallel whenever it sees WORLD_SIZE/RANK/MASTER_ADDR
    # set (the same env vars `torchrun`/`accelerate launch` would set).
    # Ray Train's job is launching the N worker processes with those env
    # vars set correctly and wiring up checkpoint/metric reporting —
    # glue around the training math, not a replacement for it.
    if args.num_gpus > 1:
        from ray.train.huggingface.transformers import (
            prepare_trainer,
            RayTrainReportCallback,
        )
        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Rank-0-only from here down — every worker just ran trainer.train()
    # together (that part is a synchronized DDP collective and MUST run on
    # all of them), but only one process should compute/write the
    # per-example eval loss and save the adapter, or N workers race on the
    # same output files.
    if _is_main_process(args):
        # Per-example held-out loss — lets sweep candidates be ranked with a
        # standard error (SEM = std / sqrt(n)) instead of eyeballing the
        # single aggregate eval_loss TRL logs to TensorBoard. Note: since
        # this script's SFTTrainer has no completion-only data collator,
        # both this number and TRL's own eval_loss are full-sequence
        # (prompt+response) loss — the two stay directly comparable.
        compute_per_example_eval_loss(
            trainer.model, tokenizer, eval_ds, args.max_seq_len,
            output_path=str(Path(args.output_dir) / "eval_loss_per_example.json"),
        )

        log.info("Saving LoRA adapter to %s", args.output_dir)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Timing capture for later num_gpus=1 vs num_gpus=N comparisons —
        # wall-clock only (this process's view from before data loading to
        # after save), not a breakdown of where time went within that.
        # Deliberately its own small file rather than folded into
        # effective_config.json, since that file is written BEFORE training
        # starts on purpose (so a crashed run still leaves a config record)
        # and shouldn't be reopened/rewritten after the fact.
        duration_seconds = time.time() - _train_start_time
        timing = {
            "num_gpus": args.num_gpus,
            "train_subset_size": args.train_subset_size,
            "n_train_examples": len(train_ds),
            "duration_seconds": round(duration_seconds, 1),
            "duration_minutes": round(duration_seconds / 60, 2),
        }
        timing_path = Path(args.output_dir) / "training_time.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        log.info("Training took %.1f min (num_gpus=%d) -> %s",
                 timing["duration_minutes"], args.num_gpus, timing_path)

        log.info("Done. Adapter saved.")
    else:
        log.info("Worker rank>0: training complete, skipping save/eval-loss "
                 "(handled by rank 0).")


# ---------------------------------------------------------------------------
# 4b. Per-example held-out eval loss
# ---------------------------------------------------------------------------

def compute_per_example_eval_loss(model, tokenizer, eval_ds, max_seq_len: int, output_path: str) -> dict:
    """
    Run each held-out example through the model individually (batch=1, no
    grad) and record its loss, so hyperparameter-sweep candidates can be
    compared via mean +/- SEM rather than a single aggregate number.

    Writes a JSON file with n_examples, mean_loss, std_loss, sem, and the
    full per-example loss list (so you can re-derive any other statistic
    later without re-running the model).
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for row in eval_ds:
            enc = tokenizer(
                row["text"],
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(model.device)
            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())

    # Plain float arithmetic instead of the statistics module: if training
    # diverged, some/all per-example losses come back as NaN. Ordinary
    # float math propagates NaN harmlessly (mean/std just come out NaN
    # too), but statistics.stdev()'s exact-fraction internals raise
    # AttributeError on NaN input — which would otherwise crash this
    # function AFTER training already finished, destroying the run's eval
    # file instead of recording "this candidate diverged," which is
    # itself a real, useful result.
    n = len(losses)
    num_nan = sum(1 for x in losses if math.isnan(x))
    if num_nan:
        log.warning(
            "%d of %d per-example eval losses are NaN — likely training "
            "divergence (check learning rate / grad_norm in the training log).",
            num_nan, n,
        )

    mean_loss = sum(losses) / n if n else float("nan")
    if n > 1:
        variance = sum((x - mean_loss) ** 2 for x in losses) / (n - 1)
        std_loss = math.sqrt(variance) if variance == variance and variance >= 0 else float("nan")
    else:
        std_loss = 0.0
    sem = std_loss / math.sqrt(n) if n else float("nan")

    result = {
        "n_examples": n,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "sem": sem,
        "num_nan": num_nan,
        "per_example_loss": losses,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(
        "Per-example eval loss -> %s (mean=%.4f, sem=%.4f, n=%d)",
        output_path, mean_loss, sem, len(losses),
    )

    model.train()
    return result


# ---------------------------------------------------------------------------
# 5. Merge adapter → full model (optional, for Ollama re-import)
# ---------------------------------------------------------------------------

def merge_and_save(args):
    """
    Merge the LoRA adapter back into the base weights and save a standard
    Hugging Face model that can be converted to GGUF for Ollama.

    After merging, convert with:
        python llama.cpp/convert_hf_to_gguf.py ./outputs/merged-model \
              --outtype q4_k_m --outfile my-model.gguf
    Then create an Ollama Modelfile:
        FROM ./my-model.gguf
        ollama create my-custom-model -f Modelfile
    """
    log.info("Loading tokenizer from adapter dir (preserves chat template used in training): %s", args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)

    # offload_folder is required once accelerate's device_map needs disk
    # offload, not just CPU offload - it only shows up as a hard failure at
    # PeftModel.from_pretrained() time (attaching the adapter forces a
    # device_map recompute for base+adapter combined), even when the plain
    # base-only load succeeded fine with pure CPU offloading. Passing it to
    # both calls is harmless when it isn't needed - the folder just sits
    # empty - so there's no downside to always providing it.
    offload_dir = Path(args.output_dir).parent / f"{Path(args.output_dir).name}-offload-tmp"
    offload_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        offload_folder=str(offload_dir),
        trust_remote_code=True,
    )
    log.info("Loading adapter from: %s", args.adapter_dir)
    model = PeftModel.from_pretrained(base, args.adapter_dir, offload_folder=str(offload_dir))

    log.info("Merging weights …")
    model = model.merge_and_unload()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("Saving merged model to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Merge complete. Convert to GGUF and import into Ollama.")


# ---------------------------------------------------------------------------
# 5b. Save base weights only, no adapter (optional, for Ollama re-import)
# ---------------------------------------------------------------------------

def save_base_only(args):
    """
    Save the raw base model + tokenizer to output_dir, with no LoRA adapter
    involved at all. This exists so a truly untrained model can be pushed
    through the same GGUF-convert -> Ollama-register pipeline as a
    fine-tuned one (see ollama_pipeline.py), letting judge scoring compare
    base vs. fine-tuned using identical infrastructure on both sides.

    Deliberately mirrors merge_and_save()'s save step, minus the PeftModel
    load/merge — this is NOT a stand-in for --merge_only, it's for models
    that were never fine-tuned in the first place.
    """
    log.info("Loading tokenizer directly from base model (no adapter dir): %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("Saving base model (untrained) to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Base save complete. Convert to GGUF and import into Ollama.")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

# Fallback defaults — used only for values not supplied via --config or CLI.
# Precedence (highest to lowest): CLI flag > --config file > these fallbacks.
# Mirrors the precedence convention already used in config/study_config.yaml.
FALLBACK_DEFAULTS = {
    "data_dir":     "./data",
    "output_dir":   "./outputs/lora-adapter",
    "epochs":       3,
    "batch_size":   2,
    "grad_accum":   4,
    "lr":           2e-4,
    "max_seq_len":  2048,
    "lora_r":       16,
    "lora_alpha":   32,
    "lora_dropout": 0.05,
    "no_4bit":      False,
    "attn_only":    False,
}

# Keys a --config YAML file is allowed to set. Anything else is a likely
# typo — set_defaults() would otherwise accept and silently ignore it
# (argparse doesn't validate set_defaults() keys against registered flags),
# so unrecognized keys are logged and dropped rather than failing silently.
KNOWN_CONFIG_KEYS = {
    "model_name", "data_dir", "output_dir",
    "epochs", "batch_size", "grad_accum", "lr", "max_seq_len",
    "lora_r", "lora_alpha", "lora_dropout",
    "no_4bit", "chat_template", "attn_only",
}


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune an Ollama-compatible model on ShareGPT JSONL data")

    p.add_argument("--config", default=None,
                   help="Path to a YAML file of default hyperparameters (e.g. "
                        "config/training_sweep/granite-4.1-8B_r32.yaml). CLI "
                        "flags override values from this file.")

    p.add_argument("--model_name",   default=None,
                   help="HuggingFace model ID or local path (e.g. mistralai/Mistral-7B-v0.1). "
                        "Required, via --config or this flag.")
    p.add_argument("--data_dir",     default=None,
                   help="Directory containing one or more .jsonl files (searched recursively)")
    p.add_argument("--output_dir",   default=None,
                   help="Where to write the trained LoRA adapter (or merged model)")

    # Training hyper-parameters
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--grad_accum",   type=int,   default=None,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--max_seq_len",  type=int,   default=None)

    # LoRA hyper-parameters
    p.add_argument("--lora_r",       type=int,   default=None,
                   help="LoRA rank (higher = more capacity, more VRAM)")
    p.add_argument("--lora_alpha",   type=int,   default=None,
                   help="LoRA scaling factor (commonly 2× rank)")
    p.add_argument("--lora_dropout", type=float, default=None)

    # Quantization
    p.add_argument("--no_4bit",      action="store_true", default=None,
                   help="Disable 4-bit QLoRA (use full / fp16 precision)")

    # Target modules
    p.add_argument("--attn_only",    action="store_true", default=None,
                   help="Attach LoRA to attention projections only (q/k/v/o), "
                        "instead of the default attention+MLP set.")

    # Chat template
    p.add_argument("--chat_template", default=None,
                   help="Jinja2 chat template string to assign to the tokenizer. "
                        "Use this when your base model has no built-in template and "
                        "the ChatML fallback does not match your desired format.")

    # Merge-only mode
    p.add_argument("--merge_only",   action="store_true",
                   help="Skip training; merge an existing adapter into the base model")
    p.add_argument("--adapter_dir",  default=None,
                   help="Path to an existing adapter (required with --merge_only)")

    # Base-only mode — no adapter at all, just save the raw base weights
    # (for pushing an untrained model through the same Ollama pipeline)
    p.add_argument("--base_only",    action="store_true",
                   help="Skip training and adapter merging; save the raw base "
                        "model + tokenizer to --output_dir as-is")

    # Training-scale study support: which slice of the frozen train pool
    # (split_utils.py) to train on, and how many GPUs to spread that
    # training across. Deliberately kept OUTSIDE KNOWN_CONFIG_KEYS/
    # FALLBACK_DEFAULTS — these are "which data / how much hardware"
    # concerns, not tunable hyperparameters a --config sweep file should
    # set, same reasoning as --adapter_dir being kept separate.
    p.add_argument("--train_subset_size", type=int, default=None,
                   help="Train on a fixed-size PREFIX of the frozen train pool "
                        "(split_utils.py) instead of the full available pool — "
                        "e.g. --train_subset_size 1000. Prefixes of the same "
                        "pool nest (1000 subset-of 2000 subset-of 4000 ...), "
                        "which is what makes a training-data-scaling study comparable across "
                        "conditions. Omit to train on the full available pool.")
    p.add_argument("--num_gpus", type=int, default=1,
                   help="Number of GPUs to train on. 1 (default) runs exactly "
                        "as before, no Ray dependency. >1 launches distributed "
                        "data-parallel training via Ray Train across that many "
                        "GPUs on this node (requires `pip install ray[train]`).")

    # First pass: peek at --config, and separately record which tunable
    # flags were typed directly on the CLI (as opposed to inherited from
    # --config or FALLBACK_DEFAULTS). At this point every tunable arg's
    # default is still None, so "not None" here means the user typed it —
    # regardless of whether --config also sets that same key. Used below to
    # auto-suffix output_dir.
    known, _ = p.parse_known_args()
    cli_overrides = {
        key for key in (KNOWN_CONFIG_KEYS - {"model_name", "data_dir", "chat_template"})
        if getattr(known, key, None) is not None
    }
    user_set_output_dir = known.output_dir is not None

    if known.config:
        with open(known.config, encoding="utf-8") as f:
            config_values = yaml.safe_load(f) or {}
        unknown = set(config_values) - KNOWN_CONFIG_KEYS
        if unknown:
            log.warning("Ignoring unrecognized --config key(s): %s", sorted(unknown))
            for key in unknown:
                config_values.pop(key)
        p.set_defaults(**config_values)

    args = p.parse_args()

    # Fill anything still unset (not in --config, not on CLI) from FALLBACK_DEFAULTS
    for key, val in FALLBACK_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    if args.model_name is None:
        p.error("--model_name is required (via --config or --model_name)")

    # Auto-suffix output_dir when tunable flags were overridden on the CLI,
    # unless --output_dir was itself set explicitly (that choice wins
    # as-is). Without this, a fixed baseline-config output_dir means every
    # sweep candidate silently overwrites the previous candidate's adapter,
    # effective_config.json, and eval_loss_per_example.json.
    sweep_overrides = cli_overrides - {"output_dir"}
    if sweep_overrides and not user_set_output_dir:
        suffix = "__" + "_".join(f"{k}{getattr(args, k)}" for k in sorted(sweep_overrides))
        args.output_dir = f"{args.output_dir.rstrip('/')}{suffix}"
        log.info(
            "CLI override(s) %s detected — output_dir auto-suffixed to %s",
            sorted(sweep_overrides), args.output_dir,
        )

    # Separately suffix for --train_subset_size (not part of cli_overrides/
    # KNOWN_CONFIG_KEYS — see its help text). Without this, running the
    # 1000/2000/4000 conditions of a training-scale study back to back with
    # the same base --output_dir would have each one silently overwrite the
    # previous condition's adapter.
    if args.train_subset_size and not user_set_output_dir:
        args.output_dir = f"{args.output_dir.rstrip('/')}__n{args.train_subset_size}"
        log.info("--train_subset_size %d detected — output_dir auto-suffixed to %s",
                 args.train_subset_size, args.output_dir)

    return args


def launch_training(args):
    """
    Dispatches to single-process training (args.num_gpus == 1, the default —
    identical to every run before this flag existed, no Ray dependency) or
    Ray Train distributed data-parallel training across args.num_gpus GPUs
    on this node (> 1).

    Multi-node note: this ScalingConfig doesn't specify placement, so Ray
    schedules all num_gpus workers on whatever GPUs are available in
    whatever Ray cluster this process can see — which is just "this node's
    local GPUs" unless a multi-node Ray cluster was started first (`ray
    start --head` / `ray start --address=...` on each node, before this
    script runs). Going multi-node later is additive: this function and
    train() don't change at all, only how/where the Ray cluster this
    connects to was started.
    """
    if args.num_gpus <= 1:
        train(args)
        return

    import ray
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    log.info("Launching distributed training across %d GPUs via Ray Train", args.num_gpus)

    # Ray Train workers are SEPARATE OS PROCESSES, not forked copies of this
    # one — they don't inherit the sys.path.insert(PROJECT_ROOT) this file
    # did for itself at import time. When Ray serializes the training
    # function via cloudpickle, anything it calls that came FROM split_utils
    # (load_all_records, load_frozen_split) gets pickled BY REFERENCE — the
    # worker tries to `import split_utils` on its own to resolve it, fails
    # with ModuleNotFoundError, since that module only exists on the
    # driver's in-memory sys.path, not the worker's real PYTHONPATH.
    # Fix: explicitly set PYTHONPATH via Ray's runtime_env so every worker
    # process gets it at OS level, not just the driver. Must call ray.init()
    # ourselves with this BEFORE constructing TorchTrainer — otherwise
    # TorchTrainer auto-inits Ray with no runtime_env and this never takes
    # effect. (Confirmed necessary and sufficient on Turing, 2026-09-22 —
    # see learnings.md.)
    if not ray.is_initialized():
        # include_dashboard=False: the dashboard's periodic state-polling
        # of the Train Controller actor fails repeatedly on this node
        # (compute node, likely restricted ports/no dashboard UI needed)
        # and floods the log with "Failed to query Ray Train Controller
        # actor state" every polling interval, drowning out real training
        # output. Pure log noise, unrelated to training correctness or
        # speed — disabling the dashboard component entirely removes the
        # thing doing the failing polling, rather than living with the
        # spam. Confirmed needed on Turing, 2026-09-22.
        ray.init(
            runtime_env={
            "env_vars": {
                "PYTHONPATH": str(PROJECT_ROOT),
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,NET",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "TOKENIZERS_PARALLELISM": "false",
            }
        },
            include_dashboard=False,
        )

    def _ray_worker(config):
        train(config["args"])

    # CPU-per-worker: must divide evenly into whatever --cpus-per-task the
    # SLURM allocation actually grants, NOT a fixed number — Turing's GPU
    # node has CPUTot=28 (7/worker at num_gpus=4), Wahab's has CPUTot=32
    # (8/worker). A hardcoded value that's right for one cluster causes an
    # immediate "Requested node configuration is not available" on the
    # other (confirmed on Turing, 2026-09-22, after being validated only
    # on Wahab). SLURM sets SLURM_CPUS_PER_TASK to whatever --cpus-per-task
    # was actually granted — read that directly instead of hardcoding
    # either cluster's number, so this works on both (and anywhere else)
    # without editing this file per-cluster. Falls back to 4 if not running
    # under SLURM at all (e.g. a bare interactive test).
    _total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    _cpus_per_worker = max(1, _total_cpus // args.num_gpus) if args.num_gpus > 0 else 1
    scaling_config = ScalingConfig(
        num_workers=args.num_gpus,
        use_gpu=True,
        resources_per_worker={
            "CPU": _cpus_per_worker,
            "GPU": 1,
        },
        placement_strategy="STRICT_PACK",
    )

    trainer = TorchTrainer(
        train_loop_per_worker=_ray_worker,
        train_loop_config={"args": args},
        scaling_config=scaling_config,
    )
    result = trainer.fit()
    log.info("Ray Train finished: %s", result)


if __name__ == "__main__":
    args = parse_args()

    if args.base_only:
        save_base_only(args)
    elif args.merge_only:
        if not args.adapter_dir:
            raise ValueError("--adapter_dir is required when using --merge_only")
        merge_and_save(args)
    else:
        launch_training(args)
"""
study_config.py
=================
Shared configuration loader for the LLM-as-judge validity study scripts
(generate_judge_examples.py, generate_new_scenarios.py, score_judge_examples.py).

Mirrors scorer.py's config precedence exactly - environment variable >
config file > hardcoded default - factored into one module so all three
scripts read the same values instead of each copy-pasting the loading
block and risking drift between them.

Config file location: config/study_config.yaml by default. Override the
path itself with the CONFIG_FILE environment variable (same variable name
scorer.py uses, for consistency).

Usage in a script:
    import study_config as cfg
    parser.add_argument("--ollama_url", default=cfg.OLLAMA_BASE_URL)
"""

import os
from pathlib import Path

import yaml

# generation/study_config.py -> parent.parent is the project root
_project_root = Path(__file__).resolve().parent.parent

_cfg = {}
_config_path = os.getenv("CONFIG_FILE", str(_project_root / "config" / "study_config.yaml"))
if os.path.exists(_config_path):
    with open(_config_path, encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f) or {}
else:
    print(f"[study_config] No config file found at '{_config_path}' - "
          f"using environment variables and hardcoded defaults only.")


def _get(key: str, env_var: str, default):
    """env var > config file > hardcoded default."""
    env_val = os.getenv(env_var)
    if env_val is not None:
        return env_val
    return _cfg.get(key, default)


def _get_bool(key: str, env_var: str, default: bool) -> bool:
    val = _get(key, env_var, default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes")


def _get_int(key: str, env_var: str, default: int) -> int:
    return int(_get(key, env_var, default))


def _get_float(key: str, env_var: str, default: float) -> float:
    return float(_get(key, env_var, default))


def _get_list(key: str, env_var: str, default: list) -> list:
    """env var (comma-separated) > config file list > hardcoded default."""
    env_val = os.getenv(env_var)
    if env_val is not None:
        return [v.strip() for v in env_val.split(",") if v.strip()]
    val = _cfg.get(key, default)
    return val if isinstance(val, list) else default


def _resolve_path(value: str) -> str:
    """Anchor a relative path at the project root, not at whatever the
    current working directory happens to be when a script is launched."""
    p = Path(value)
    return str(p) if p.is_absolute() else str(_project_root / p)


OLLAMA_BASE_URL = _get("ollama_base_url", "OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL     = _get("judge_model",     "JUDGE_MODEL",     "")
STUDY_MODELS    = _get_list("study_models", "STUDY_MODELS",  [])
DB_FILE         = _resolve_path(_get("db_file",  "DB_FILE",  "generation/outputs/judge_scores.db"))
DATA_DIR        = _resolve_path(_get("data_dir", "DATA_DIR", "data"))
MAX_EXAMPLES    = _get_int("max_examples", "MAX_EXAMPLES",   20)
TEMPERATURE     = _get_float("temperature","TEMPERATURE",    0.3)
NUM_PREDICT     = _get_int("num_predict",  "NUM_PREDICT",    1536)
TIMEOUT         = _get_int("timeout",      "TIMEOUT",        300)
RESCORE         = _get_bool("rescore",     "RESCORE",        False)
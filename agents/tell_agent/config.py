from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("tell_agent.yaml")
CONFIG_PATH_ENV = "TELL_AGENT_CONFIG_PATH"
_RUNTIME_OVERRIDES: Dict[str, Any] = {}


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _to_int(v: Any, default: int) -> int:
    try:
        return int(float(_to_str(v).strip()))
    except Exception:
        return default


def _to_float(v: Any, default: float) -> float:
    try:
        return float(_to_str(v).strip())
    except Exception:
        return default


def _to_bool(v: Any, default: bool) -> bool:
    s = _to_str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def load_tell_yaml_config() -> Dict[str, Any]:
    override = os.getenv(CONFIG_PATH_ENV, "").strip()
    path = Path(override).expanduser().resolve() if override else DEFAULT_CONFIG_PATH.resolve()
    cfg = _load_yaml(path)
    cfg["_config_path"] = str(path)
    return cfg


def reload_tell_yaml_config() -> Dict[str, Any]:
    load_tell_yaml_config.cache_clear()
    return load_tell_yaml_config()


def get_tell_config_path() -> str:
    return str(load_tell_yaml_config().get("_config_path") or DEFAULT_CONFIG_PATH)


def get_yaml_env(name: str, default: Any = None) -> Any:
    cfg = load_tell_yaml_config()
    env_cfg = cfg.get("env")
    if not isinstance(env_cfg, dict):
        return default
    return env_cfg.get(name, default)


def get_yaml_env_map() -> Dict[str, Any]:
    cfg = load_tell_yaml_config()
    env_cfg = cfg.get("env")
    return env_cfg if isinstance(env_cfg, dict) else {}


def set_runtime_override(name: str, value: Any) -> None:
    if not isinstance(name, str) or not name:
        return
    _RUNTIME_OVERRIDES[name] = value


def clear_runtime_overrides() -> None:
    _RUNTIME_OVERRIDES.clear()


def _get_env_value(name: str, default: Any) -> Any:
    if name in _RUNTIME_OVERRIDES:
        return _RUNTIME_OVERRIDES[name]
    return get_yaml_env(name, default)


def get_env_str(name: str, default: str) -> str:
    return _to_str(_get_env_value(name, default)) or default


def get_env_int(name: str, default: int) -> int:
    return _to_int(_get_env_value(name, default), default)


def get_env_float(name: str, default: float) -> float:
    return _to_float(_get_env_value(name, default), default)


def get_prompt_templates() -> Dict[str, Any]:
    cfg = load_tell_yaml_config()
    prompts = cfg.get("prompts")
    return prompts if isinstance(prompts, dict) else {}


def get_reminder_templates() -> Dict[str, Any]:
    cfg = load_tell_yaml_config()
    prompts = cfg.get("prompts")
    if not isinstance(prompts, dict):
        return {}
    reminders = prompts.get("reminders")
    return reminders if isinstance(reminders, dict) else {}


def get_tools_config() -> Any:
    cfg = load_tell_yaml_config()
    return cfg.get("tools")

def get_workspace_config() -> Dict[str, Any]:
    cfg = load_tell_yaml_config()
    workspace = cfg.get("workspace")
    return workspace if isinstance(workspace, dict) else {}

def get_workspace_str(name: str, default: str = "") -> str:
    return _to_str(get_workspace_config().get(name, default)) or default


def get_workspace_int(name: str, default: int = 0) -> int:
    return _to_int(get_workspace_config().get(name, default), default)


def get_workspace_bool(name: str, default: bool = False) -> bool:
    return _to_bool(get_workspace_config().get(name, default), default)


def get_workspace_size_limit_bytes(default: int = 0) -> int:
    workspace = get_workspace_config()
    raw = workspace.get("size_limit")
    if raw is None or raw == "":
        raw = workspace.get("size_limit_bytes", default)
    if isinstance(raw, (int, float)):
        return max(0, int(raw))
    text = _to_str(raw).strip().lower()
    if not text:
        return max(0, int(default))
    m = None
    try:
        import re

        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?", text)
    except Exception:
        m = None
    if not m:
        return max(0, int(default))
    value = float(m.group(1))
    suffix = (m.group(2) or "").strip()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "ki": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mi": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gi": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    multiplier = multipliers.get(suffix)
    if multiplier is None:
        return max(0, int(default))
    return max(0, int(value * multiplier))


def _get_mapping_section(name: str) -> Dict[str, Any]:
    cfg = load_tell_yaml_config()
    section = cfg.get(name)
    return section if isinstance(section, dict) else {}


def get_memory_checkpoint_clear_config() -> Dict[str, Any]:
    return _get_mapping_section("memory_checkpoint_clear")


def get_history_log_config() -> Dict[str, Any]:
    return _get_mapping_section("history_log")


def get_history_log_str(name: str, default: str = "") -> str:
    return _to_str(get_history_log_config().get(name, default)) or default


def get_history_log_bool(name: str, default: bool = False) -> bool:
    return _to_bool(get_history_log_config().get(name, default), default)


def get_llm_config() -> Dict[str, Any]:
    cfg = load_tell_yaml_config()
    llm = cfg.get("llm")
    return llm if isinstance(llm, dict) else {}


def get_llm_max_tokens_retry_config() -> Dict[str, Any]:
    llm = get_llm_config()
    cfg = llm.get("max_tokens_retry")
    return cfg if isinstance(cfg, dict) else {}


def get_llm_max_tokens_retry_bool(name: str, default: bool) -> bool:
    return _to_bool(get_llm_max_tokens_retry_config().get(name, default), default)


def get_llm_max_tokens_retry_int(name: str, default: int) -> int:
    return _to_int(get_llm_max_tokens_retry_config().get(name, default), default)


def get_llm_max_tokens_retry_str(name: str, default: str) -> str:
    return _to_str(get_llm_max_tokens_retry_config().get(name, default)) or default


def get_llm_empty_response_recovery_config() -> Dict[str, Any]:
    llm = get_llm_config()
    cfg = llm.get("empty_response_recovery")
    return cfg if isinstance(cfg, dict) else {}


def get_llm_empty_response_recovery_bool(name: str, default: bool) -> bool:
    return _to_bool(get_llm_empty_response_recovery_config().get(name, default), default)


def get_llm_empty_response_recovery_str(name: str, default: str) -> str:
    return _to_str(get_llm_empty_response_recovery_config().get(name, default)) or default


def get_llm_truncation_recovery_config() -> Dict[str, Any]:
    llm = get_llm_config()
    cfg = llm.get("truncation_recovery")
    return cfg if isinstance(cfg, dict) else {}


def get_llm_truncation_recovery_bool(name: str, default: bool) -> bool:
    return _to_bool(get_llm_truncation_recovery_config().get(name, default), default)


def get_llm_truncation_recovery_int(name: str, default: int) -> int:
    return _to_int(get_llm_truncation_recovery_config().get(name, default), default)


def get_llm_truncation_recovery_str(name: str, default: str) -> str:
    return _to_str(get_llm_truncation_recovery_config().get(name, default)) or default


def get_llm_request_budget_config() -> Dict[str, Any]:
    llm = get_llm_config()
    cfg = llm.get("request_budget")
    return cfg if isinstance(cfg, dict) else {}


def get_llm_request_budget_int(name: str, default: int) -> int:
    return _to_int(get_llm_request_budget_config().get(name, default), default)


def get_llm_request_budget_str(name: str, default: str) -> str:
    return _to_str(get_llm_request_budget_config().get(name, default)) or default


def get_compaction_config() -> Dict[str, Any]:
    cfg = load_tell_yaml_config()
    compaction = cfg.get("compaction")
    return compaction if isinstance(compaction, dict) else {}


def get_compaction_bool(name: str, default: bool) -> bool:
    return _to_bool(get_compaction_config().get(name, default), default)


def get_compaction_int(name: str, default: int) -> int:
    return _to_int(get_compaction_config().get(name, default), default)


def get_compaction_float(name: str, default: float) -> float:
    return _to_float(get_compaction_config().get(name, default), default)


def get_compaction_str(name: str, default: str) -> str:
    return _to_str(get_compaction_config().get(name, default)) or default

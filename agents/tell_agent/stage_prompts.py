from __future__ import annotations

import re
from typing import Any, Dict

from .config import get_prompt_templates


_TOKEN_RE = re.compile(r"\{([A-Z0-9_]+)\}")


def render_template(template: str, values: Dict[str, Any]) -> str:
    if not template:
        return ""

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        val = values.get(key)
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        return str(val)

    return _TOKEN_RE.sub(repl, template)


def _resolve_stage_map() -> Dict[str, Dict[str, str]]:
    prompts = get_prompt_templates()

    # Flat format: prompts.system + prompts.user directly (no prompts.stages)
    if "system" in prompts and "user" in prompts:
        system = prompts.get("system")
        user = prompts.get("user")
        if isinstance(system, str) and system.strip() and isinstance(user, str) and user.strip():
            return {"main": {"system": system, "user": user}}

    # Legacy format: prompts.stages.<name>.system / .user
    stages_cfg = prompts.get("stages")
    if not isinstance(stages_cfg, dict):
        raise ValueError("prompts.stages is required (or use flat format with prompts.system + prompts.user)")
    out: Dict[str, Dict[str, str]] = {}
    for stage_name, raw in stages_cfg.items():
        if not isinstance(stage_name, str) or not isinstance(raw, dict):
            continue
        system = raw.get("system")
        user = raw.get("user")
        if not isinstance(system, str) or not system.strip():
            raise ValueError(f"prompts.stages.{stage_name}.system is required")
        if not isinstance(user, str) or not user.strip():
            raise ValueError(f"prompts.stages.{stage_name}.user is required")
        out[stage_name] = {"system": system, "user": user}
    if not out:
        raise ValueError("prompts.stages has no valid stage definitions")
    return out


def get_stage_system_prompt(stage: str, runtime_api_base: str, values: Dict[str, Any] | None = None) -> str:
    stage_map = _resolve_stage_map()
    if stage not in stage_map:
        raise ValueError(f"Unknown stage for system prompt: {stage}")
    base_values = {"RUNTIME_API_BASE": runtime_api_base.rstrip("/")}
    if values:
        base_values.update(values)
    return render_template(stage_map[stage]["system"], base_values)


def get_stage_user_prompt_template(stage: str) -> str:
    stage_map = _resolve_stage_map()
    if stage not in stage_map:
        raise ValueError(f"Unknown stage for user prompt template: {stage}")
    return stage_map[stage]["user"]


STAGE_SYSTEM_PROMPT_TEMPLATES: Dict[str, str] = {
    name: cfg["system"] for name, cfg in _resolve_stage_map().items()
}

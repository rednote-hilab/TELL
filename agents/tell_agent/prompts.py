from __future__ import annotations

from .config import get_prompt_templates


def load_system_prompt() -> str:
    prompts = get_prompt_templates()
    stages = prompts.get("stages")
    if isinstance(stages, dict):
        for _, cfg in stages.items():
            if isinstance(cfg, dict):
                system = cfg.get("system")
                if isinstance(system, str) and system.strip():
                    return system
    return "You are TELL Agent."

SYSTEM_PROMPT = load_system_prompt()

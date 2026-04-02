import json
from pathlib import Path
from typing import Any, Dict, List

from .config import get_tools_config


def load_tools() -> List[Dict[str, Any]]:
    path = Path(__file__).resolve().parent / "tools.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    default_tools = payload.get("tools", [])
    defaults = [t for t in default_tools if isinstance(t, dict)] if isinstance(default_tools, list) else []

    from_yaml = get_tools_config()
    if not isinstance(from_yaml, list):
        return defaults

    merged: List[Dict[str, Any]] = []
    seen_names = set()

    for tool in from_yaml:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if name:
            seen_names.add(name)
        merged.append(tool)

    for tool in defaults:
        name = str(tool.get("name") or "").strip()
        if name and name in seen_names:
            continue
        merged.append(tool)
    return merged


TOOLS = load_tools()

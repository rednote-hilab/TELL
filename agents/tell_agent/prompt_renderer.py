from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_TOKEN_RE = re.compile(r"\{([^{}\n]+)\}")
_MAX_RENDER_DEPTH = 20


class PromptRenderer:
    def __init__(self, workspace: Path, fragments: Optional[Dict[str, Any]] = None) -> None:
        self.workspace = workspace
        self.fragments = _flatten_fragments(fragments or {})

    def render(self, template: str, values: Dict[str, Any]) -> str:
        return self._render_with_context(template, values, include_stack=(), depth=0)

    def _render_with_context(
        self,
        template: str,
        values: Dict[str, Any],
        include_stack: Tuple[str, ...],
        depth: int,
    ) -> str:
        if not template:
            return ""
        if depth > _MAX_RENDER_DEPTH:
            return template

        def repl(match: re.Match[str]) -> str:
            token = match.group(1).strip()
            if not token:
                return match.group(0)
            if token in values:
                return _to_text(values.get(token))
            if token.startswith("frag:") or token.startswith("fragment:") or token.startswith("prompt:"):
                _, _, name = token.partition(":")
                key = name.strip()
                if not key:
                    return match.group(0)
                if key in include_stack:
                    return ""
                frag_template = self.fragments.get(key)
                if not isinstance(frag_template, str):
                    return match.group(0)
                return self._render_with_context(
                    frag_template,
                    values,
                    include_stack=include_stack + (key,),
                    depth=depth + 1,
                )
            if token.startswith("file:"):
                return self._read_rel_file(token[5:])
            return match.group(0)

        return _TOKEN_RE.sub(repl, template)

    def _read_rel_file(self, raw_path: str) -> str:
        rel = (raw_path or "").strip()
        if not rel:
            return ""
        # Keep reads scoped to workspace.
        abs_path = (self.workspace / rel).resolve()
        workspace_root = self.workspace.resolve()
        try:
            abs_path.relative_to(workspace_root)
        except Exception:
            return ""
        if not abs_path.exists() or not abs_path.is_file():
            return ""
        try:
            return abs_path.read_text(encoding="utf-8")
        except Exception:
            return ""


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _flatten_fragments(raw: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    _flatten_into(out, raw, prefix="")
    return out


def _flatten_into(out: Dict[str, str], raw: Dict[str, Any], prefix: str) -> None:
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(v, str):
            out[full_key] = v
            continue
        if isinstance(v, dict):
            _flatten_into(out, v, full_key)

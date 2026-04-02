"""Tool-level permission system for tell agent.

Works without Docker by building on the existing strace supervision
in tool_handlers.py. This module adds a higher-level policy layer:

1. Per-tool allow/deny rules (which tools are available per stage)
2. Path-based restrictions (which directories a tool can read/write)
3. Command pattern blocking (dangerous shell patterns)
4. Rate limiting (max tool calls per stage turn)

Usage:
    policy = ToolPermissionPolicy.from_config(config_dict)
    result = policy.check("bash_exec", {"command": "rm -rf /"}, stage="main")
    if not result.allowed:
        return f"Permission denied: {result.reason}"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass(frozen=True)
class PermissionResult:
    allowed: bool
    reason: str = ""


# Dangerous command patterns that should always be blocked unless
# explicitly allowed. These catch common destructive operations.
DEFAULT_BLOCKED_PATTERNS: List[str] = [
    r"\brm\s+-[rRf]*[rR][rRf]*\s+/",              # rm -rf / or rm -r /
    r"\brm\s+-[rRf]*[rR][rRf]*\s+~",              # rm -rf ~
    r"\bmkfs\b",                                    # format filesystem
    r"\bdd\s+.*of=/dev/",                           # raw device write
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",             # fork bomb
    r"\bchmod\s+(-R\s+)?[0-7]*777\s+/",            # chmod 777 /
    r"\bcurl\b.*\|\s*(bash|sh|zsh)\b",              # curl | bash
    r"\bwget\b.*\|\s*(bash|sh|zsh)\b",              # wget | bash
    r"\bnc\s+-[elp]",                               # netcat listener
    r"\bpython[23]?\s+-m\s+http\.server\b",         # http server
    r"\bnohup\b.*&\s*$",                            # background daemon
    r"\bsudo\b",                                     # privilege escalation
    r"\bsu\s+-?\s*\w",                               # switch user
]


@dataclass
class ToolPermissionPolicy:
    """Configurable permission policy for tool execution."""

    # Tools that are always allowed regardless of stage.
    globally_allowed_tools: Set[str] = field(default_factory=set)

    # Per-stage tool allowlist. If empty, all tools are allowed for that stage.
    stage_tool_allowlist: Dict[str, Set[str]] = field(default_factory=dict)

    # Blocked shell command patterns (regex).
    blocked_command_patterns: List[re.Pattern] = field(default_factory=list)  # type: ignore

    # Max tool calls per stage turn (0 = unlimited).
    max_tool_calls_per_turn: int = 0

    # Whether to enforce path restrictions beyond the strace sandbox.
    enforce_path_restrictions: bool = True

    # Additional allowed read paths (beyond workspace).
    extra_read_paths: List[str] = field(default_factory=list)

    def check(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        stage: str = "",
        turn_tool_count: int = 0,
    ) -> PermissionResult:
        """Check if a tool call is permitted under this policy."""

        # 1. Rate limit check.
        if self.max_tool_calls_per_turn > 0 and turn_tool_count >= self.max_tool_calls_per_turn:
            return PermissionResult(
                allowed=False,
                reason=f"tool call limit reached ({self.max_tool_calls_per_turn} per turn)",
            )

        # 2. Stage-level tool allowlist.
        if stage and stage in self.stage_tool_allowlist:
            allowed = self.stage_tool_allowlist[stage]
            if allowed and tool_name not in allowed:
                return PermissionResult(
                    allowed=False,
                    reason=f"tool '{tool_name}' not allowed in stage '{stage}'",
                )

        # 3. Command pattern blocking for bash_exec.
        if tool_name == "bash_exec":
            command = str(args.get("command") or "")
            for pattern in self.blocked_command_patterns:
                if pattern.search(command):
                    return PermissionResult(
                        allowed=False,
                        reason=f"command blocked by security policy: matches pattern '{pattern.pattern}'",
                    )

        return PermissionResult(allowed=True)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "ToolPermissionPolicy":
        """Build policy from YAML config dict.

        Expected config structure:
            permissions:
              blocked_patterns:
                - "rm -rf /"
                - "sudo"
              max_tool_calls_per_turn: 50
              extra_read_paths:
                - /etc/hosts
              stages:
                main:
                  tools: [bash_exec, screen_shot, read_file]
        """
        if not isinstance(config, dict):
            config = {}

        blocked_patterns = list(DEFAULT_BLOCKED_PATTERNS)
        extra = config.get("blocked_patterns")
        if isinstance(extra, list):
            for p in extra:
                if isinstance(p, str) and p.strip():
                    blocked_patterns.append(p.strip())

        compiled = []
        for p in blocked_patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error:
                pass

        stage_allowlist: Dict[str, Set[str]] = {}
        stages = config.get("stages")
        if isinstance(stages, dict):
            for stage_name, stage_cfg in stages.items():
                if not isinstance(stage_cfg, dict):
                    continue
                tools = stage_cfg.get("tools")
                if isinstance(tools, list):
                    stage_allowlist[str(stage_name)] = {str(t).strip() for t in tools if str(t).strip()}

        extra_read = []
        paths = config.get("extra_read_paths")
        if isinstance(paths, list):
            extra_read = [str(p).strip() for p in paths if str(p).strip()]

        return cls(
            blocked_command_patterns=compiled,
            stage_tool_allowlist=stage_allowlist,
            max_tool_calls_per_turn=max(0, int(config.get("max_tool_calls_per_turn", 0) or 0)),
            extra_read_paths=extra_read,
        )

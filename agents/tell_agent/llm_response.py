from __future__ import annotations

from typing import Any, Dict, List, Optional


class LLMResponse:
    def __init__(
        self,
        text: str,
        tool_calls: List[Dict[str, Any]],
        usage: Dict[str, int],
        raw: Dict[str, Any],
        stop_reason: str = "",
        request_generation_config: Optional[Dict[str, Any]] = None,
        raw_request: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.tool_calls = tool_calls
        self.usage = usage
        self.raw = raw
        self.stop_reason = stop_reason
        self.request_generation_config = request_generation_config or {}
        self.raw_request = raw_request or {}

from __future__ import annotations

from .claude_client import ClaudeClient


def create_llm_client(model: str) -> ClaudeClient:
    return ClaudeClient(model=model)

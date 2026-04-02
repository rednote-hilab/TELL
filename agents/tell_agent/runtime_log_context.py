from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .config import get_env_str


class RuntimeLogContextMixin:
    def __init__(self) -> None:
        self._runtime_log_run_id = ""
        self._runtime_log_replay_dir = ""

    def set_runtime_log_context(self, *, run_id: str = "", replay_dir: str = "") -> None:
        self._runtime_log_run_id = str(run_id or "").strip()
        self._runtime_log_replay_dir = str(replay_dir or "").strip()

    def get_runtime_log_context(self) -> Tuple[str, str]:
        run_id = str(self._runtime_log_run_id or "").strip()
        replay_dir = str(self._runtime_log_replay_dir or "").strip()
        if replay_dir or run_id:
            return run_id, replay_dir
        return (
            str(get_env_str("LOG_RUN_ID", "") or "").strip(),
            str(get_env_str("LOG_REPLAY_DIR", "") or "").strip(),
        )

    def get_runtime_log_path(self, filename: str = "llm_error_requests.jsonl") -> Optional[Path]:
        _, replay_dir = self.get_runtime_log_context()
        if not replay_dir:
            return None
        return Path(replay_dir) / filename

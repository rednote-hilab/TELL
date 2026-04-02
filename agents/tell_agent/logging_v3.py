from __future__ import annotations

import base64
import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import get_history_log_bool, get_history_log_str

logger = logging.getLogger(__name__)


def _resolve_history_log_path(raw_path: str, *, workspace: Path, run_id: str, game_id: str) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    try:
        rendered = text.format(
            RUN_ID=run_id,
            GAME_ID=game_id,
            WORKSPACE_NAME=workspace.name,
        )
    except Exception:
        rendered = text
    candidate = Path(rendered)
    return candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()


@dataclass
class LogContext:
    run_id: str
    game_id: str
    agent_name: str
    workspace: Path
    log_dir: Path


class LogWriterV3:
    """Async three-file logger for messages/action_frames/fs_versions."""

    def __init__(self, ctx: LogContext, resume_from: Optional[Path] = None) -> None:
        self._ctx = ctx
        self._seq_lock = threading.Lock()
        self._seq = 0
        self._created_at = self._now()

        if resume_from is not None and resume_from.is_dir():
            # --- Resume mode: reuse existing replay directory ---
            self._out_dir = resume_from
            self._init_resume(resume_from)
        else:
            # --- Fresh mode ---
            self._out_dir = ctx.log_dir.resolve() / "replays" / ctx.run_id
            self._out_dir.mkdir(parents=True, exist_ok=True)

        self._messages_path = self._out_dir / "messages.jsonl"
        self._raw_requests_path = self._out_dir / "raw_requests.jsonl"
        self._action_frames_path = self._out_dir / "action_frames.jsonl"
        self._runtime_observations_path = self._out_dir / "runtime_observations.jsonl"
        self._fs_versions_path = self._out_dir / "fs_versions.jsonl"
        self._manifest_path = self._out_dir / "manifest.json"
        self._stats_path = self._out_dir / "run_stats.json"
        self._history_path: Optional[Path] = None
        if get_history_log_bool("enabled", False):
            raw_history_path = get_history_log_str("path", "").strip()
            if raw_history_path:
                self._history_path = _resolve_history_log_path(
                    raw_history_path,
                    workspace=ctx.workspace,
                    run_id=ctx.run_id,
                    game_id=ctx.game_id,
                )
            if self._history_path is not None:
                self._history_path.parent.mkdir(parents=True, exist_ok=True)

        if resume_from is None or not resume_from.is_dir():
            self._message_count = 0
            self._raw_request_count = 0
            self._action_frame_count = 0
            self._runtime_observation_count = 0
            self._fs_version_count = 0
            self._max_levels_completed_observed = 0
            self._level_completion_steps: list[Dict[str, Any]] = []
            self._session_ids: set[str] = set()
            self._state_ids: set[str] = set()
            self._loop_ids: set[str] = set()
            self._fs_version_idx = 0
            self._fs_last: Dict[str, Dict[str, Any]] = {}
            self._stats: Dict[str, Any] = {
                "schema_version": "log_v3_stats",
                "run_id": ctx.run_id,
                "game_id": ctx.game_id,
                "agent": ctx.agent_name,
                "created_at": self._created_at,
                "updated_at": self._created_at,
                "messages_total": 0,
                "messages_by_stage": {},
                "tool_calls_total": 0,
                "tool_calls_by_stage": {},
                "tool_results_total": 0,
                "tool_results_by_stage": {},
                "reminder_messages_total": 0,
                "reminder_types": {},
                "compaction_count": 0,
                "tokens": {
                    "prompt_total": 0,
                    "completion_total": 0,
                    "total_total": 0,
                    "prompt_max": 0,
                    "completion_max": 0,
                    "total_max": 0,
                    "last_prompt": 0,
                    "last_completion": 0,
                    "last_total": 0,
                },
                "last_compaction": {},
            }

        self._queue: queue.Queue[Optional[tuple[str, Dict[str, Any]]]] = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        if resume_from is None or not resume_from.is_dir():
            self._write_manifest(closed=False)

    def _init_resume(self, resume_dir: Path) -> None:
        """Restore internal counters from an existing replay directory."""
        manifest_path = resume_dir / "manifest.json"
        stats_path = resume_dir / "run_stats.json"
        fs_versions_path = resume_dir / "fs_versions.jsonl"

        # Load manifest
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("resume: failed to read manifest.json: %s", exc)
            manifest = {}

        # Restore counters from manifest
        self._seq = int(manifest.get("seq_max", 0) or 0)
        self._created_at = str(manifest.get("created_at", "") or "") or self._created_at
        self._message_count = int(manifest.get("message_count", 0) or 0)
        self._raw_request_count = int(manifest.get("raw_request_count", 0) or 0)
        self._action_frame_count = int(manifest.get("action_frame_count", 0) or 0)
        self._runtime_observation_count = int(manifest.get("runtime_observation_count", 0) or 0)
        self._fs_version_count = int(manifest.get("fs_version_count", 0) or 0)
        self._max_levels_completed_observed = int(manifest.get("max_levels_completed_observed", 0) or 0)
        self._level_completion_steps = list(manifest.get("level_completion_steps", []) or [])
        self._fs_version_idx = self._fs_version_count

        # Scope tracking sets - approximate from manifest counts
        self._session_ids = set(f"sess_{i:04d}" for i in range(1, int(manifest.get("session_count", 1) or 1) + 1))
        self._state_ids = set()  # will be populated from new records
        self._loop_ids = set()

        # Load stats
        try:
            self._stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("resume: failed to read run_stats.json, using defaults: %s", exc)
            self._stats = {
                "schema_version": "log_v3_stats",
                "run_id": self._ctx.run_id,
                "game_id": self._ctx.game_id,
                "agent": self._ctx.agent_name,
                "created_at": self._created_at,
                "updated_at": self._now(),
                "messages_total": self._message_count,
                "messages_by_stage": {},
                "tool_calls_total": 0,
                "tool_calls_by_stage": {},
                "tool_results_total": 0,
                "tool_results_by_stage": {},
                "reminder_messages_total": 0,
                "reminder_types": {},
                "compaction_count": 0,
                "tokens": {
                    "prompt_total": 0, "completion_total": 0, "total_total": 0,
                    "prompt_max": 0, "completion_max": 0, "total_max": 0,
                    "last_prompt": 0, "last_completion": 0, "last_total": 0,
                },
                "last_compaction": {},
            }

        # Rebuild _fs_last from fs_versions.jsonl for correct patch diffs
        self._fs_last = {}
        if fs_versions_path.exists():
            try:
                from .session_resume import _load_workspace_state
                ws_files = _load_workspace_state(fs_versions_path)
                for fpath, fbytes in ws_files.items():
                    self._fs_last[fpath] = {
                        "path": fpath,
                        "sha256": hashlib.sha256(fbytes).hexdigest(),
                        "size": len(fbytes),
                        "content_b64": base64.b64encode(fbytes).decode("ascii"),
                    }
            except Exception as exc:
                logger.warning("resume: failed to rebuild fs_last: %s", exc)

        logger.info(
            "resume: restored log state (seq=%d, messages=%d, requests=%d, actions=%d, fs=%d)",
            self._seq, self._message_count, self._raw_request_count,
            self._action_frame_count, self._fs_version_count,
        )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _base_record(self, session_id: str, state_id: str, loop_id: str, source: str) -> Dict[str, Any]:
        return {
            "schema_version": "log_v3",
            "run_id": self._ctx.run_id,
            "seq": self._next_seq(),
            "ts": self._now(),
            "session_id": session_id,
            "state_id": state_id,
            "loop_id": loop_id,
            "source": source,
        }

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _append_history_record(self, record: Dict[str, Any]) -> None:
        if self._history_path is None:
            return
        simplified: Dict[str, Any] = {
            "ts": record.get("ts", ""),
            "seq": int(record.get("seq", 0) or 0),
            "run_id": self._ctx.run_id,
            "game_id": self._ctx.game_id,
            "source": record.get("source", ""),
            "state_id": record.get("state_id", ""),
            "loop_id": record.get("loop_id", ""),
            "message_id": record.get("message_id", ""),
            "role": record.get("role", ""),
            "message_type": record.get("message_type", ""),
            "parts": self._simplify_message_parts(record.get("parts")),
        }
        self._append_jsonl(self._history_path, simplified)

    @staticmethod
    def _simplify_message_parts(parts: Any) -> list[dict[str, Any]]:
        if not isinstance(parts, list):
            return []
        out: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                out.append({"type": "text", "text": text})
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                out.append(
                    {
                        "type": "functionCall",
                        "name": str(function_call.get("name") or ""),
                        "args": function_call.get("args", {}),
                    }
                )
                continue
            function_response = part.get("functionResponse")
            if isinstance(function_response, dict):
                out.append(
                    {
                        "type": "functionResponse",
                        "name": str(function_response.get("name") or ""),
                        "response": function_response.get("response", {}),
                    }
                )
        return out

    def _extract_stage(self, source: str) -> str:
        s = (source or "").strip()
        if s.startswith("tell_"):
            return s[len("tell_") :]
        return s or "unknown"

    def _inc(self, bucket: Dict[str, int], key: str, delta: int = 1) -> None:
        bucket[key] = int(bucket.get(key, 0) or 0) + int(delta)

    def _extract_tokens_from_raw_response(self, raw_response: Dict[str, Any]) -> Dict[str, int]:
        prompt = completion = total = 0
        usage_meta = raw_response.get("usageMetadata")
        if isinstance(usage_meta, dict):
            prompt = int(usage_meta.get("promptTokenCount", 0) or 0)
            completion = int(usage_meta.get("candidatesTokenCount", 0) or 0)
            total = int(usage_meta.get("totalTokenCount", 0) or 0)
        usage = raw_response.get("usage")
        if isinstance(usage, dict):
            prompt = max(prompt, int(usage.get("prompt_tokens", 0) or 0))
            completion = max(completion, int(usage.get("completion_tokens", 0) or 0))
            total = max(total, int(usage.get("total_tokens", 0) or 0))
            # Anthropic-style usage compatibility.
            prompt = max(prompt, int(usage.get("input_tokens", 0) or 0))
            completion = max(completion, int(usage.get("output_tokens", 0) or 0))
        if total <= 0:
            total = prompt + completion
        return {"prompt": prompt, "completion": completion, "total": total}

    def _update_stats_from_message(self, record: Dict[str, Any]) -> None:
        stage = self._extract_stage(str(record.get("source") or ""))
        self._stats["messages_total"] = int(self._stats.get("messages_total", 0) or 0) + 1
        self._inc(self._stats.setdefault("messages_by_stage", {}), stage, 1)
        parts = record.get("parts")
        if isinstance(parts, list):
            fc = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict))
            fr = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("functionResponse"), dict))
            if fc > 0:
                self._stats["tool_calls_total"] = int(self._stats.get("tool_calls_total", 0) or 0) + fc
                self._inc(self._stats.setdefault("tool_calls_by_stage", {}), stage, fc)
            if fr > 0:
                self._stats["tool_results_total"] = int(self._stats.get("tool_results_total", 0) or 0) + fr
                self._inc(self._stats.setdefault("tool_results_by_stage", {}), stage, fr)
        meta = record.get("message_meta")
        if isinstance(meta, dict):
            special = meta.get("special_types")
            if isinstance(special, list):
                nonempty = [str(x).strip() for x in special if str(x).strip()]
                if nonempty:
                    self._stats["reminder_messages_total"] = int(
                        self._stats.get("reminder_messages_total", 0) or 0
                    ) + 1
                    rem_types = self._stats.setdefault("reminder_types", {})
                    for t in nonempty:
                        self._inc(rem_types, t, 1)
            ev = str(meta.get("event") or "")
            if ev in {"compaction", "forced_compaction", "turn_limit_compaction"}:
                self._stats["compaction_count"] = int(self._stats.get("compaction_count", 0) or 0) + 1
                self._stats["last_compaction"] = {
                    "ts": record.get("ts", ""),
                    "stage": stage,
                    "event": ev,
                    "estimated_tokens": int(meta.get("estimated_tokens", 0) or 0),
                    "messages_before": int(meta.get("messages_before", 0) or 0),
                    "messages_after": int(meta.get("messages_after", 0) or 0),
                }

    def _update_stats_from_raw_request(self, record: Dict[str, Any]) -> None:
        raw_response = record.get("raw_response")
        if not isinstance(raw_response, dict):
            return
        tk = self._extract_tokens_from_raw_response(raw_response)
        tokens = self._stats.setdefault("tokens", {})
        tokens["prompt_total"] = int(tokens.get("prompt_total", 0) or 0) + tk["prompt"]
        tokens["completion_total"] = int(tokens.get("completion_total", 0) or 0) + tk["completion"]
        tokens["total_total"] = int(tokens.get("total_total", 0) or 0) + tk["total"]
        tokens["prompt_max"] = max(int(tokens.get("prompt_max", 0) or 0), tk["prompt"])
        tokens["completion_max"] = max(int(tokens.get("completion_max", 0) or 0), tk["completion"])
        tokens["total_max"] = max(int(tokens.get("total_max", 0) or 0), tk["total"])
        tokens["last_prompt"] = tk["prompt"]
        tokens["last_completion"] = tk["completion"]
        tokens["last_total"] = tk["total"]

    def _to_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _extract_level_completion_fields(self, observation: Any) -> tuple[Optional[int], Optional[int]]:
        if not isinstance(observation, dict):
            return None, None
        levels_completed = self._to_int(observation.get("levels_completed"))
        step = self._to_int(observation.get("step"))
        return levels_completed, step

    def _record_level_completion(
        self,
        *,
        levels_completed: Optional[int],
        step: Optional[int],
        af_id: str = "",
        action_frame_count: Optional[int] = None,
    ) -> None:
        if levels_completed is None or levels_completed <= 0:
            return
        if levels_completed > self._max_levels_completed_observed:
            for level in range(self._max_levels_completed_observed + 1, levels_completed + 1):
                row: Dict[str, Any] = {"level": level}
                if step is not None:
                    row["step"] = int(step)
                if action_frame_count is not None:
                    row["action_frame_count"] = int(action_frame_count)
                if af_id:
                    row["af_id"] = af_id
                self._level_completion_steps.append(row)
            self._max_levels_completed_observed = levels_completed
            return
        # If runtime observation arrived before action_frame, enrich the latest row.
        if levels_completed == self._max_levels_completed_observed and self._level_completion_steps:
            last = self._level_completion_steps[-1]
            if int(last.get("level", 0) or 0) != levels_completed:
                return
            if step is not None and "step" not in last:
                last["step"] = int(step)
            if action_frame_count is not None and "action_frame_count" not in last:
                last["action_frame_count"] = int(action_frame_count)
            if af_id and not str(last.get("af_id") or "").strip():
                last["af_id"] = af_id

    def _write_stats(self, closed: bool) -> None:
        self._stats["updated_at"] = self._now()
        if closed:
            self._stats["closed_at"] = self._stats["updated_at"]
        self._stats_path.write_text(json.dumps(self._stats, ensure_ascii=True, indent=2), encoding="utf-8")

    def _write_manifest(self, closed: bool) -> None:
        payload = {
            "schema_version": "log_v3",
            "run_id": self._ctx.run_id,
            "game_id": self._ctx.game_id,
            "agent": self._ctx.agent_name,
            "created_at": self._created_at,
            "closed_at": self._now() if closed else "",
            "message_count": self._message_count,
            "raw_request_count": self._raw_request_count,
            "action_frame_count": self._action_frame_count,
            "runtime_observation_count": self._runtime_observation_count,
            "fs_version_count": self._fs_version_count,
            "max_levels_completed_observed": self._max_levels_completed_observed,
            "level_completion_steps": list(self._level_completion_steps),
            "session_count": len(self._session_ids),
            "state_count": len(self._state_ids),
            "loop_count": len(self._loop_ids),
            "seq_max": self._seq,
        }
        self._manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _iter_workspace_files(self) -> Iterable[Path]:
        for p in sorted(self._ctx.workspace.resolve().rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts:
                continue
            yield p

    def _file_entry(self, path: Path) -> Optional[Dict[str, Any]]:
        data: Optional[bytes] = None
        for attempt in range(3):
            try:
                data = path.read_bytes()
                break
            except OSError as exc:
                # ESTALE(116) is common on distributed filesystems under churn.
                if getattr(exc, "errno", None) == 116 and attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning("skip fs snapshot file due to read error: path=%s err=%s", path, exc)
                return None
            except Exception as exc:
                logger.warning("skip fs snapshot file due to unexpected error: path=%s err=%s", path, exc)
                return None
        if data is None:
            return None
        return {
            "path": path.relative_to(self._ctx.workspace).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "content_b64": base64.b64encode(data).decode("ascii"),
        }

    def _build_fs_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        current: Dict[str, Dict[str, Any]] = {}
        for fp in self._iter_workspace_files():
            entry = self._file_entry(fp)
            if entry is None:
                continue
            current[str(entry["path"])] = entry

        self._fs_version_idx += 1
        record["fs_version_id"] = f"fs_{self._fs_version_idx:06d}"

        if not self._fs_last:
            record.update({"kind": "snapshot", "root": "workspace", "files": list(current.values())})
        else:
            ops: list[dict[str, Any]] = []
            old_paths = set(self._fs_last.keys())
            new_paths = set(current.keys())
            for p in sorted(old_paths - new_paths):
                ops.append({"op": "delete", "path": p})
            for p in sorted(new_paths):
                cur = current[p]
                prev = self._fs_last.get(p)
                if prev is None or prev.get("sha256") != cur.get("sha256"):
                    ops.append(
                        {
                            "op": "write",
                            "path": p,
                            "sha256": cur["sha256"],
                            "size": cur["size"],
                            "content_b64": cur["content_b64"],
                        }
                    )
            record.update(
                {
                    "kind": "patch",
                    "base_fs_version_id": f"fs_{(self._fs_version_idx - 1):06d}",
                    "ops": ops,
                }
            )
        self._fs_last = current
        return record

    def _touch_scope(self, rec: Dict[str, Any]) -> None:
        self._session_ids.add(str(rec.get("session_id") or ""))
        self._state_ids.add(str(rec.get("state_id") or ""))
        loop_id = str(rec.get("loop_id") or "")
        if loop_id:
            self._loop_ids.add(loop_id)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            kind, record = item
            try:
                self._touch_scope(record)
                if kind == "message":
                    self._append_jsonl(self._messages_path, record)
                    self._append_history_record(record)
                    self._message_count += 1
                    self._update_stats_from_message(record)
                elif kind == "raw_request":
                    self._append_jsonl(self._raw_requests_path, record)
                    self._raw_request_count += 1
                    self._update_stats_from_raw_request(record)
                elif kind == "action_frame":
                    self._append_jsonl(self._action_frames_path, record)
                    self._action_frame_count += 1
                    result = record.get("result")
                    observation = result.get("observation") if isinstance(result, dict) else None
                    levels_completed, step = self._extract_level_completion_fields(observation)
                    self._record_level_completion(
                        levels_completed=levels_completed,
                        step=step,
                        af_id=str(record.get("af_id") or ""),
                        action_frame_count=self._action_frame_count,
                    )
                elif kind == "runtime_observation":
                    self._append_jsonl(self._runtime_observations_path, record)
                    self._runtime_observation_count += 1
                    levels_completed, step = self._extract_level_completion_fields(record.get("observation"))
                    if step is None:
                        step = self._to_int(record.get("step"))
                    self._record_level_completion(levels_completed=levels_completed, step=step)
                elif kind == "fs_version":
                    full = self._build_fs_record(record)
                    self._append_jsonl(self._fs_versions_path, full)
                    self._fs_version_count += 1
                self._write_manifest(closed=False)
                self._write_stats(closed=False)
            except Exception:
                logger.exception("log worker failed: kind=%s", kind)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._worker.join(timeout=2.0)
        self._write_manifest(closed=True)
        self._write_stats(closed=True)

    def log_message(
        self,
        *,
        session_id: str,
        state_id: str,
        loop_id: str,
        source: str,
        message_id: str,
        role: str,
        message_type: str,
        parts: list[dict[str, Any]],
        parent_message_id: str = "",
        token_usage: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tools_digest: str = "",
        message_meta: Optional[dict[str, Any]] = None,
    ) -> int:
        record = self._base_record(session_id, state_id, loop_id, source)
        record.update(
            {
                "message_id": message_id,
                "role": role,
                "message_type": message_type,
                "parts": parts,
                "parent_message_id": parent_message_id,
                "token_usage": token_usage or {},
                "tools": tools or [],
                "tools_digest": tools_digest or "",
                "message_meta": message_meta or {},
            }
        )
        self._queue.put(("message", record))
        return int(record["seq"])

    def log_raw_request(
        self,
        *,
        session_id: str,
        state_id: str,
        loop_id: str,
        source: str,
        message_id: str,
        raw_request: Optional[dict[str, Any]] = None,
        raw_response: Optional[dict[str, Any]] = None,
        request_generation_config: Optional[dict[str, Any]] = None,
    ) -> int:
        record = self._base_record(session_id, state_id, loop_id, source)
        record.update(
            {
                "message_id": message_id,
                "raw_request": raw_request or {},
                "raw_response": raw_response or {},
                "request_generation_config": request_generation_config or {},
            }
        )
        self._queue.put(("raw_request", record))
        return int(record["seq"])

    def log_action_frame(
        self,
        *,
        session_id: str,
        state_id: str,
        loop_id: str,
        source: str,
        af_id: str,
        message_id: str,
        action_name: str,
        action_args: dict[str, Any],
        status: str,
        observation: dict[str, Any],
        error: str = "",
    ) -> int:
        record = self._base_record(session_id, state_id, loop_id, source)
        record.update(
            {
                "af_id": af_id,
                "message_id": message_id,
                "action": {"name": action_name, "args": action_args},
                "result": {"status": status, "error": error, "observation": observation},
            }
        )
        self._queue.put(("action_frame", record))
        return int(record["seq"])

    def log_fs_version(
        self,
        *,
        session_id: str,
        state_id: str,
        loop_id: str,
        source: str,
        message_id: str,
    ) -> int:
        record = self._base_record(session_id, state_id, loop_id, source)
        record.update({"message_id": message_id})
        self._queue.put(("fs_version", record))
        return int(record["seq"])

    def log_runtime_observation(
        self,
        *,
        session_id: str,
        state_id: str,
        loop_id: str,
        source: str,
        step: int,
        observation: dict[str, Any],
        message_id: str = "",
    ) -> int:
        record = self._base_record(session_id, state_id, loop_id, source)
        record.update(
            {
                "message_id": message_id,
                "step": int(step),
                "observation": dict(observation),
            }
        )
        self._queue.put(("runtime_observation", record))
        return int(record["seq"])

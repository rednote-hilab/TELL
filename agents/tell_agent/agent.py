from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..game_bridge import BoundGameBridge
from .compaction import compact_messages, estimate_message_tokens
from .config import (
    get_tell_config_path,
    get_env_float,
    get_env_int,
    get_env_str,
    get_history_log_bool,
    get_history_log_str,
    get_prompt_templates,
    get_reminder_templates,
    get_workspace_bool,
    get_workspace_size_limit_bytes,
    get_workspace_str,
)
from .llm_client import create_llm_client
from .logging_v3 import LogContext, LogWriterV3, _resolve_history_log_path
from .prompt_renderer import PromptRenderer
from .request_formatter import build_request_data, build_tool_result_parts
from .runtime_service import RuntimeGameService, pick_available_port
from .session_resume import (
    ResumeState,
    build_resume_context_messages,
    load_resume_state,
    restore_workspace,
)
from .stage_prompts import render_template
from .state_machine import TELLStateMachine
from .tool_handlers import TELLToolHandlers
from .tools import TOOLS
from .workspace_volume import WorkspaceVolume, create_workspace_volume

logger = logging.getLogger()


class TELLAgent(Agent):
    MAX_ACTIONS = 2048
    SUBAGENT_TODO_REMINDER_INTERVAL = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.MAX_ACTIONS = get_env_int("MAX_ACTIONS", self.MAX_ACTIONS)
        self.model = get_env_str("TELL_AGENT_MODEL", "claude-opus-4-6")
        self.max_output_tokens = get_env_int("TELL_MAX_OUTPUT_TOKENS", 8192)
        self.shell_timeout = get_env_float("TELL_SHELL_TIMEOUT", 30.0)
        self.shell_output_max_chars = get_env_int("TELL_BASH_OUTPUT_MAX_CHARS", 8000)
        self.max_stage_turns = get_env_int("TELL_STAGE_MAX_TURNS", 4)
        self.action_wait_timeout = get_env_float("TELL_ACTION_WAIT_TIMEOUT", 40.0)
        self.worker_poll_interval = get_env_float("TELL_WORKER_POLL_INTERVAL", 0.5)
        self.recent_frames_limit = get_env_int("TELL_RECENT_FRAMES_LIMIT", 12)
        self.render_scale = get_env_int("TELL_RENDER_SCALE", 2)
        self.workspace_size_limit_bytes = get_workspace_size_limit_bytes(0)

        safe_game = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.game_id)
        run_id = uuid.uuid4().hex[:8]
        explicit_run_id = str(os.environ.get("LOG_RUN_ID", "") or "").strip()
        resume_path: Optional[Path] = None
        _resume_log_dir = os.environ.get("RESUME_LOG_DIR", "").strip()
        if _resume_log_dir:
            candidate = Path(_resume_log_dir)
            if candidate.is_dir() and (candidate / "manifest.json").exists():
                resume_path = candidate
                try:
                    _manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
                    original_run_id = str(_manifest.get("run_id", "") or "").strip()
                    if original_run_id:
                        explicit_run_id = original_run_id
                except Exception:
                    pass
        self._run_id = explicit_run_id or f"run_{safe_game}_{run_id}"
        resolved_log_dir = str(os.environ.get("LOG_DIR", "logs") or "logs").strip() or "logs"
        self._log_dir = Path(resolved_log_dir).resolve()
        self._replay_dir = (self._log_dir / "replays" / self._run_id).resolve()

        self.llm = create_llm_client(model=self.model)
        if hasattr(self.llm, "set_runtime_log_context"):
            self.llm.set_runtime_log_context(
                run_id=self._run_id,
                replay_dir=str(self._replay_dir),
            )
        self.bridge = BoundGameBridge(game_id=self.game_id)
        project_root = Path(__file__).resolve().parents[2]
        workspace_root = project_root / "logs" / "tmp" / "tell_agent"
        workspace_name = f"{safe_game}_{run_id}"
        self._workspace_volume: WorkspaceVolume = create_workspace_volume(
            base_root=workspace_root,
            workspace_name=workspace_name,
            size_limit_bytes=self.workspace_size_limit_bytes,
            backend=get_workspace_str("backend", "auto"),
            preserve_image=get_workspace_bool("preserve_image", False),
            cleanup_stale=get_workspace_bool("cleanup_stale", True),
        )
        self.workspace = self._workspace_volume.workspace
        self.tmp_dir = self.workspace / "tmp"
        self.memory_root = self.workspace / "memory"
        self._prepare_workspace()
        self._maybe_load_prior_memory()
        self._maybe_load_prior_workspace()

        self.runtime_host = get_env_str("TELL_RUNTIME_HOST", "127.0.0.1")
        preferred_port = get_env_int("TELL_RUNTIME_PORT", 8000)
        max_tries = get_env_int("TELL_RUNTIME_PORT_TRIES", 32)
        self.runtime_port = pick_available_port(self.runtime_host, preferred_port, max_tries=max_tries)
        self._runtime_service = RuntimeGameService(
            self.game_id,
            self.runtime_host,
            self.runtime_port,
            render_scale=self.render_scale,
            workspace=self.workspace,
        )
        self._runtime_service.start()

        self.turn_count = 0
        self._last_action_taken: Dict[str, Any] = {}
        self._last_action_message_id: str = ""
        self._last_action_state_id: str = ""
        self._last_action_loop_id: str = ""
        self._last_observed_state_id: str = ""
        self._session_id = "sess_0001"

        self._log_v3 = LogWriterV3(
            LogContext(
                run_id=self._run_id,
                game_id=self.game_id,
                agent_name=self.agent_name,
                workspace=self.workspace,
                log_dir=self._log_dir,
            ),
            resume_from=resume_path,
        )
        atexit.register(self._log_v3.close)
        atexit.register(self._cleanup_workspace_volume)
        self._snapshot_replay_config()

        extra_read_paths: List[Path] = []
        if get_history_log_bool("enabled", False):
            raw_history_path = get_history_log_str("path", "").strip()
            if raw_history_path:
                history_resolved = _resolve_history_log_path(
                    raw_history_path,
                    workspace=self.workspace,
                    run_id=self._run_id,
                    game_id=self.game_id,
                )
                if history_resolved is not None:
                    extra_read_paths.append(history_resolved)

        self.tool_handlers = TELLToolHandlers(
            workspace=self.workspace,
            memory_root=self.memory_root,
            runtime_port=self.runtime_port,
            shell_timeout=self.shell_timeout,
            output_limit=self.shell_output_max_chars,
            workspace_size_limit_bytes=self.workspace_size_limit_bytes,
            workspace_hard_limited=self._workspace_volume.hard_limited,
            extra_read_paths=extra_read_paths,
            subagent_runner=self._run_subagent_task,
        )

        self.state_machine = TELLStateMachine(
            llm=self.llm,
            tools=TOOLS,
            tool_dispatch=self.tool_handlers.dispatch,
            log_event=self._log_event,
            workspace=self.workspace,
            memory_root=self.memory_root,
            max_stage_turns=self.max_stage_turns,
            max_output_tokens=self.max_output_tokens,
            recent_frames_limit=self.recent_frames_limit,
            runtime_api_base=f"http://{self.runtime_host}:{self.runtime_port}",
            run_id=self._run_id,
            step_getter=lambda: int(self.action_counter),
            turn_getter=lambda: int(self.turn_count),
            runtime_step_getter=lambda: int(self._runtime_service.current_step()),
            runtime_observation_getter=lambda: dict(self._runtime_service.get_observation_snapshot()),
            runtime_action_history_getter=lambda: list(self._runtime_service.get_action_history_snapshot()),
            stop_requested_getter=lambda: bool(self._worker_stop.is_set()),
            action_plan_submitter=self._submit_action_plan,
            workspace_budget_getter=self.tool_handlers.workspace_budget_status,
            max_actions=self.MAX_ACTIONS,
        )
        self._runtime_service.set_action_context_provider(
            lambda: self.state_machine.current_action_context()
        )
        self._runtime_service.set_runtime_observation_logger(self._log_runtime_observation_event)
        self._runtime_service.set_runtime_action_frame_logger(self._log_runtime_action_frame_event)

        self._worker_stop = threading.Event()
        self._worker_started = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_error: Optional[str] = None
        self._win_stop_requested = False

        # --- Session resume from interrupted run ---
        self._is_resumed = False
        if resume_path is not None:
            if self.try_resume_from_log(str(resume_path)):
                self._is_resumed = True
                logger.info(
                    "session resume: ready (actions=%d, requests=%d, tokens=%d)",
                    self.action_counter,
                    self.state_machine._llm_requests_used,
                    self.state_machine._total_prompt_tokens_used,
                )
            else:
                logger.warning("session resume: failed to load state from %s", str(resume_path))

    def _is_win_epilogue_complete(self) -> bool:
        try:
            loop_end_reason = str(self.state_machine.last_loop_end_reason() or "")
        except Exception:
            loop_end_reason = ""
        return (
            bool(self._win_stop_requested)
            and bool(self._worker_stop.is_set())
            and loop_end_reason in {
            "assistant_no_tool_calls",
            "win_after_epilogue",
            "win_before_llm_request",
        }
        )

    def _wait_for_win_epilogue(self) -> None:
        self._ensure_worker_started()
        while True:
            if self._worker_error:
                raise RuntimeError(f"worker_failed: {self._worker_error}")
            if self._is_win_epilogue_complete():
                return
            worker_thread = self._worker_thread
            if worker_thread is not None and not worker_thread.is_alive():
                raise RuntimeError("worker_stopped_before_win_epilogue")
            self._worker_stop.wait(timeout=max(0.01, float(self.worker_poll_interval)))

    def _should_stop_worker_after_stage(self) -> tuple[bool, str]:
        try:
            loop_end_reason = str(self.state_machine.last_loop_end_reason() or "")
        except Exception:
            loop_end_reason = ""

        if loop_end_reason == "assistant_no_tool_calls":
            return True, loop_end_reason

        if loop_end_reason in {
            "llm_request_limit_reached",
            "context_clear_limit_reached",
        }:
            return True, loop_end_reason

        if self._runtime_is_win() and loop_end_reason in {
            "win_after_epilogue",
            "win_before_llm_request",
        }:
            return True, loop_end_reason

        return False, loop_end_reason

    def _sync_win_observation(self, latest_frame: FrameData) -> None:
        obs = self._build_observation(latest_frame)
        try:
            self.state_machine.ingest_observation(obs)
        except Exception:
            pass
        try:
            self._runtime_service.update_observation(obs)
        except Exception:
            pass

    def _get_subagent_config(self, kind: str) -> Dict[str, Any]:
        prompts = get_prompt_templates()
        subagents = prompts.get("subagents")
        if not isinstance(subagents, dict):
            return {}
        cfg = subagents.get(kind)
        return cfg if isinstance(cfg, dict) else {}

    @staticmethod
    def _json_compact(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _runtime_is_win(self) -> bool:
        try:
            obs = self._runtime_service.get_observation_snapshot()
        except Exception:
            obs = {}
        if not isinstance(obs, dict):
            return False
        return str(obs.get("state") or "").strip().upper() == "WIN"

    def _build_subagent_todo_reminder(
        self,
        *,
        include_empty: bool = True,
        tool_owner: Optional[TELLToolHandlers] = None,
    ) -> str:
        todos: List[Dict[str, Any]] = []
        try:
            owner = tool_owner or self.tool_handlers
            raw = getattr(owner, "todos", [])
            if isinstance(raw, list):
                todos = [t for t in raw if isinstance(t, dict)]
        except Exception:
            todos = []

        cfg = get_reminder_templates()
        todo_empty_tpl = str(cfg.get("todo_empty") or "").strip()
        todo_tpl = str(cfg.get("todo") or "").strip()
        if not todo_empty_tpl:
            todo_empty_tpl = (
                "[todo-reminder]\n"
                "Checklist is currently empty. For multi-step tasks, keep a short internal checklist and update it after each checkpoint."
            )
        if not todo_tpl:
            todo_tpl = "[todo-reminder]\nCurrent todo list:\n{TODO_ITEMS}"

        if not todos:
            if not include_empty:
                return ""
            return todo_empty_tpl

        lines: List[str] = []
        for t in todos:
            status = str(t.get("status", "pending"))
            marker = " "
            if status == "completed":
                marker = "x"
            elif status == "in_progress":
                marker = ">"
            elif status == "cancelled":
                marker = "-"
            tid = str(t.get("id", ""))
            content = str(t.get("content", ""))
            lines.append(f"- [{marker}] {tid} {content}".strip())
        return render_template(todo_tpl, {"TODO_ITEMS": "\n".join(lines)}).strip()

    def _default_subagent_system_prompt(self) -> str:
        try:
            base_values = self.state_machine._base_prompt_values()
            rendered = self.state_machine._prompt_renderer.render(self.state_machine._system_template, base_values).strip()
            if rendered:
                return rendered
        except Exception:
            pass
        return "You are a focused analysis subagent. Solve the provided objective using the given guidance and tools."

    @staticmethod
    def _default_subagent_user_template(kind: str) -> str:
        if kind == "deep_analysis":
            return (
                "Deep analysis mode.\n\n"
                "Subagent id: {SUBAGENT_ID}\n"
                "Objective:\n"
                "{ANALYSIS_OBJECTIVE}\n\n"
                "Guidance:\n"
                "{ANALYSIS_GUIDANCE}\n\n"
                "Task spec JSON:\n"
                "{TASK_SPEC_JSON}\n\n"
                "Requirements:\n"
                "- Stay focused on this objective only.\n"
                "- Use the provided guidance as your starting point.\n"
                "- Follow the provided guidance while deciding what to inspect.\n"
                "- Ground claims in evidence from runtime, files, or tool outputs.\n"
                "- If unresolved, say what remains uncertain and what check would resolve it.\n"
                "- Output the final conclusion directly."
            )
        return (
            "Task id: {TASK_ID}\n"
            "Task: {TASK_TEXT}\n"
            "Task spec JSON:\n"
            "{TASK_SPEC_JSON}\n"
        )

    def _maybe_compact_subagent(
        self,
        *,
        log_source: str,
        state_id: str,
        loop_id: str,
        task_id: str,
        kind: str,
        turn: int,
        messages: List[Dict[str, Any]],
        tool_defs: List[Dict[str, Any]],
        tool_dispatch: Any,
        max_output_tokens: int,
        observed_total_tokens: int,
    ) -> List[Dict[str, Any]]:
        cfg = self.state_machine._compaction_cfg
        if not cfg.enabled:
            return messages
        threshold = int(max(1, cfg.max_context_tokens) * max(0.0, min(1.0, cfg.trigger_ratio)))
        estimated_request_tokens = int(estimate_message_tokens(messages))
        trigger_tokens = max(int(observed_total_tokens), estimated_request_tokens)
        if trigger_tokens < threshold:
            return messages
        try:
            compacted, info = compact_messages(
                llm=self.llm,
                messages=messages,
                cfg=cfg,
                max_output_tokens=max_output_tokens,
                tools=tool_defs,
                tool_dispatch=tool_dispatch,
                request_hook=lambda _kind: self.state_machine._consume_llm_request_budget(
                    sub_turn=int(turn),
                    request_kind=f"subagent:{kind}:compaction",
                ),
            )
            summary = self.state_machine._extract_compaction_summary(compacted)
            parts: List[Dict[str, Any]] = [
                {
                    "text": (
                        "[subagent_compaction] "
                        f"kind={kind} "
                        f"turn={turn} "
                        f"observed_total_tokens={int(observed_total_tokens)} "
                        f"estimated_request_tokens={estimated_request_tokens} "
                        f"threshold={threshold} "
                        f"messages_before={int(info.get('messages_before', len(messages)))} "
                        f"messages_after={int(info.get('messages_after', len(compacted)))}"
                    )
                }
            ]
            if summary:
                parts.append({"text": summary})
            self._log_event(
                "message",
                {
                    "source": log_source,
                    "state_id": state_id,
                    "loop_id": loop_id,
                    "message_id": f"{task_id}_compaction_{turn}",
                    "message": {"role": "assistant", "parts": parts},
                    "message_meta": {
                        "subagent_kind": kind,
                        "subagent_task_id": task_id,
                        "subagent_turn": int(turn),
                        "subagent_phase": "compaction",
                    },
                },
            )
            return compacted
        except Exception as exc:
            if isinstance(exc, RuntimeError) and (
                "LLM_REQUEST_LIMIT_REACHED" in str(exc or "")
                or "CONTEXT_CLEAR_LIMIT_REACHED" in str(exc or "")
            ):
                raise
            self._log_event(
                "message",
                {
                    "source": log_source,
                    "state_id": state_id,
                    "loop_id": loop_id,
                    "message_id": f"{task_id}_compaction_error_{turn}",
                    "message": {"role": "assistant", "parts": [{"text": f"[subagent_compaction_error] {exc}"}]},
                    "message_meta": {
                        "subagent_kind": kind,
                        "subagent_task_id": task_id,
                        "subagent_turn": int(turn),
                        "subagent_phase": "compaction_error",
                    },
                },
            )
            return messages

    def _complete_subagent_with_retry(
        self,
        *,
        kind: str,
        task_id: str,
        log_source: str,
        state_id: str,
        loop_id: str,
        turn: int,
        messages: List[Dict[str, Any]],
        tool_defs: List[Dict[str, Any]],
        tool_dispatch: Any,
        max_output_tokens: int,
    ) -> Any:
        attempt = 0
        empty_recovery_used = False
        input_overflow_recovery_used = False
        invalid_argument_recovery_used = False
        temperature = float(self.state_machine._llm_temperature)
        top_p = self.state_machine._llm_top_p
        while True:
            if self._runtime_is_win():
                raise RuntimeError("WIN_GUARD_BEFORE_LLM_REQUEST")
            request_data = build_request_data(
                messages=messages,
                tools=tool_defs,
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=top_p,
            )
            try:
                self.state_machine._consume_llm_request_budget(
                    sub_turn=int(turn),
                    request_kind=f"subagent:{kind}:stage",
                )
                response = self.llm.complete(
                    messages=messages,
                    tools=tool_defs,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    request_data=request_data,
                )
            except RuntimeError as exc:
                msg = str(exc or "")
                is_empty = "Empty response: no candidates" in msg
                is_input_overflow = (
                    "INPUT_TOKENS_EXCEEDED" in msg
                    or "input token count exceeds" in msg.lower()
                    or "maximum number of tokens allowed" in msg.lower()
                )
                is_invalid_argument = (
                    "INVALID_ARGUMENT" in msg
                    or "invalid argument" in msg.lower()
                    or "request contains an invalid argument" in msg.lower()
                )
                if is_input_overflow and self.state_machine._compaction_cfg.enabled and not input_overflow_recovery_used:
                    input_overflow_recovery_used = True
                    compacted, info = compact_messages(
                        llm=self.llm,
                        messages=messages,
                        cfg=self.state_machine._compaction_cfg,
                        max_output_tokens=max_output_tokens,
                        tools=tool_defs,
                        tool_dispatch=tool_dispatch,
                        overflow_mode=True,
                        request_hook=lambda _kind: self.state_machine._consume_llm_request_budget(
                            sub_turn=int(turn),
                            request_kind=f"subagent:{kind}:forced_compaction_input_tokens",
                        ),
                    )
                    messages[:] = compacted
                    summary = self.state_machine._extract_compaction_summary(compacted)
                    parts: List[Dict[str, Any]] = [
                        {
                            "text": (
                                "[subagent_forced_compaction] "
                                f"reason=input_tokens_exceeded "
                                f"kind={kind} "
                                f"turn={turn} "
                                f"messages_before={int(info.get('messages_before', 0))} "
                                f"messages_after={int(info.get('messages_after', 0))}"
                            )
                        }
                    ]
                    if summary:
                        parts.append({"text": summary})
                    self._log_event(
                        "message",
                        {
                            "source": log_source,
                            "state_id": state_id,
                            "loop_id": loop_id,
                            "message_id": f"{task_id}_forced_compaction_{turn}",
                            "message": {"role": "assistant", "parts": parts},
                            "message_meta": {
                                "subagent_kind": kind,
                                "subagent_task_id": task_id,
                                "subagent_turn": int(turn),
                                "subagent_phase": "forced_compaction",
                            },
                        },
                    )
                    continue
                if is_invalid_argument and self.state_machine._compaction_cfg.enabled and not invalid_argument_recovery_used:
                    request_size_chars = 0
                    try:
                        request_size_chars = len(
                            json.dumps(request_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        )
                    except Exception:
                        request_size_chars = 0
                    if request_size_chars >= 1_500_000:
                        invalid_argument_recovery_used = True
                        compacted, info = compact_messages(
                            llm=self.llm,
                            messages=messages,
                            cfg=self.state_machine._compaction_cfg,
                            max_output_tokens=max_output_tokens,
                            tools=tool_defs,
                            tool_dispatch=tool_dispatch,
                            overflow_mode=True,
                            request_hook=lambda _kind: self.state_machine._consume_llm_request_budget(
                                sub_turn=int(turn),
                                request_kind=f"subagent:{kind}:forced_compaction_invalid_argument",
                            ),
                        )
                        messages[:] = compacted
                        summary = self.state_machine._extract_compaction_summary(compacted)
                        parts: List[Dict[str, Any]] = [
                            {
                                "text": (
                                    "[subagent_forced_compaction] "
                                    f"reason=invalid_argument_oversized_request "
                                    f"kind={kind} "
                                    f"turn={turn} "
                                    f"request_size_chars={int(request_size_chars)} "
                                    f"messages_before={int(info.get('messages_before', 0))} "
                                    f"messages_after={int(info.get('messages_after', 0))}"
                                )
                            }
                        ]
                        if summary:
                            parts.append({"text": summary})
                        self._log_event(
                            "message",
                            {
                                "source": log_source,
                                "state_id": state_id,
                                "loop_id": loop_id,
                                "message_id": f"{task_id}_forced_compaction_{turn}",
                                "message": {"role": "assistant", "parts": parts},
                                "message_meta": {
                                    "subagent_kind": kind,
                                    "subagent_task_id": task_id,
                                    "subagent_turn": int(turn),
                                    "subagent_phase": "forced_compaction",
                                },
                            },
                        )
                        continue
                if is_empty and self.state_machine._empty_response_recovery_enabled and not empty_recovery_used:
                    empty_recovery_used = True
                    reminder = str(self.state_machine._empty_response_recovery_reminder or "").strip()
                    if reminder:
                        reminder_message = {"role": "user", "parts": [{"text": reminder}]}
                        messages.append(reminder_message)
                        self._log_event(
                            "message",
                            {
                                "source": log_source,
                                "state_id": state_id,
                                "loop_id": loop_id,
                                "message_id": f"{task_id}_empty_response_reminder_{turn}",
                                "message": reminder_message,
                                "message_meta": {
                                    "subagent_kind": kind,
                                    "subagent_task_id": task_id,
                                    "subagent_turn": int(turn),
                                    "subagent_phase": "empty_response_recovery",
                                    "special_types": ["system_reminder"],
                                },
                            },
                        )
                    continue
                raise
            stop_reason = str(getattr(response, "stop_reason", "") or "").upper()
            if not self.state_machine._max_tokens_retry_enabled or stop_reason != "MAX_TOKENS":
                return response
            if attempt < self.state_machine._max_tokens_retry_count:
                attempt += 1
                continue
            reminder = str(self.state_machine._max_tokens_retry_reminder or "").strip()
            if reminder:
                reminder_message = {"role": "user", "parts": [{"text": reminder}]}
                messages.append(reminder_message)
                self._log_event(
                    "message",
                    {
                        "source": log_source,
                        "state_id": state_id,
                        "loop_id": loop_id,
                        "message_id": f"{task_id}_max_tokens_reminder_{turn}_{attempt}",
                        "message": reminder_message,
                        "message_meta": {
                            "subagent_kind": kind,
                            "subagent_task_id": task_id,
                            "subagent_turn": int(turn),
                            "subagent_phase": "max_tokens_retry",
                            "special_types": ["system_reminder"],
                        },
                    },
                )
            if self._runtime_is_win():
                raise RuntimeError("WIN_GUARD_BEFORE_LLM_REQUEST")
            self.state_machine._consume_llm_request_budget(
                sub_turn=int(turn),
                request_kind=f"subagent:{kind}:stage:max_tokens_retry",
            )
            return self.llm.complete(
                messages=messages,
                tools=tool_defs,
                max_tokens=max_output_tokens,
                temperature=temperature,
                request_data=build_request_data(
                    messages=messages,
                    tools=tool_defs,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    top_p=top_p,
                ),
            )

    def _run_subagent_task(self, kind: str, args: Dict[str, Any]) -> str:
        cfg = self._get_subagent_config(kind)
        task_spec = args.get("task_spec")
        objective = str(args.get("objective") or args.get("task") or "").strip()
        guidance = str(args.get("guidance") or "").strip()
        if not objective:
            return f"Error: run_{kind}_task requires non-empty 'objective'"
        subagent_id = str(args.get("subagent_id") or "").strip()
        if kind == "deep_analysis":
            if not guidance:
                return "Error: deep_analysis requires non-empty 'guidance'"

        prompt_cfg = get_prompt_templates()
        fragments = prompt_cfg.get("fragments")
        renderer = PromptRenderer(
            self.workspace,
            fragments=fragments if isinstance(fragments, dict) else None,
        )

        state_id = f"level_{int(self.levels_completed):04d}"
        loop_id = str(getattr(self.state_machine, "current_loop_id", "") or "")
        task_id = f"{kind}_{uuid.uuid4().hex[:10]}"
        log_source = f"tell_subagent_{kind}"

        values = {
            "RUNTIME_API_BASE": f"http://{self.runtime_host}:{self.runtime_port}",
            "LEVEL": str(int(self.levels_completed)),
            "STATE": str(getattr(self, "current_state", "")),
            "AVAILABLE_ACTIONS_JSON": self._json_compact(getattr(self, "available_actions", [])),
            "TASK_SPEC_JSON": self._json_compact(task_spec if task_spec is not None else {}),
            "TASK_TEXT": objective,
            "TASK_ID": task_id,
            "SUBAGENT_ID": subagent_id,
            "ANALYSIS_OBJECTIVE": objective,
            "ANALYSIS_GUIDANCE": guidance,
        }

        system_template = str(cfg.get("system") or "").strip()
        user_template = str(cfg.get("user") or "").strip()
        if not system_template:
            system_template = self._default_subagent_system_prompt()
        if not user_template:
            user_template = self._default_subagent_user_template(kind)

        system_text = renderer.render(system_template, values).strip()
        user_text = renderer.render(user_template, values).strip()

        subagent_tools = TELLToolHandlers(
            workspace=self.workspace,
            memory_root=self.memory_root,
            runtime_port=self.runtime_port,
            shell_timeout=self.shell_timeout,
            output_limit=self.shell_output_max_chars,
            subagent_runner=None,
        )
        if isinstance(getattr(self.tool_handlers, "todos", None), list):
            subagent_tools.todos = [
                dict(todo) for todo in getattr(self.tool_handlers, "todos", []) if isinstance(todo, dict)
            ]

        todo_reminder = self._build_subagent_todo_reminder(include_empty=True, tool_owner=subagent_tools)
        if todo_reminder:
            user_text = f"{user_text.rstrip()}\n\n{todo_reminder}"

        allowed_tool_names: List[str] = []
        for raw in cfg.get("tools", []):
            if isinstance(raw, str):
                name = raw.strip()
                if name:
                    allowed_tool_names.append(name)
        if not allowed_tool_names:
            allowed_tool_names = ["bash_exec"]
        allowed_tool_names = [n for n in allowed_tool_names if n not in {"run_grid_survey_task", "run_deep_analysis"}]
        tool_defs = [t for t in TOOLS if str(t.get("name") or "") in set(allowed_tool_names)]
        try:
            tools_digest = hashlib.sha256(
                json.dumps(tool_defs, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        except Exception:
            tools_digest = ""

        configured_turns = int(cfg.get("max_turns") or 6)
        subagent_turn_cap = max(1, int(getattr(self.state_machine, "_stage_hard_turn_limit", 16)))
        max_turns = max(1, min(subagent_turn_cap, configured_turns))
        cfg_max_tokens = int(cfg.get("max_output_tokens") or min(self.max_output_tokens, 4096))
        max_output_tokens = max(256, min(self.max_output_tokens, cfg_max_tokens))

        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "parts": [{"text": system_text}]})
        messages.append({"role": "user", "parts": [{"text": user_text}]})
        self._log_event(
            "message",
            {
                "source": log_source,
                "state_id": state_id,
                "loop_id": loop_id,
                "message_id": f"{task_id}_system",
                "message": {"role": "system", "parts": [{"text": system_text}]},
                "message_meta": {"subagent_kind": kind, "subagent_task_id": task_id, "subagent_phase": "bootstrap"},
            },
        )
        self._log_event(
            "message",
            {
                "source": log_source,
                "state_id": state_id,
                "loop_id": loop_id,
                "message_id": f"{task_id}_user",
                "message": {"role": "user", "parts": [{"text": user_text}]},
                "tools": tool_defs,
                "tools_digest": tools_digest,
                    "message_meta": {
                        "subagent_kind": kind,
                        "subagent_task_id": task_id,
                        "subagent_phase": "bootstrap",
                        "task_text": objective,
                        "special_types": ["system_reminder"],
                    },
                },
        )
        final_text = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_tool_calls = 0
        end_reason = "assistant_no_tool_calls"
        observed_total_tokens = 0

        for turn in range(max_turns):
            if self._runtime_is_win():
                end_reason = "win_before_llm_request"
                final_text = "[win_guard] runtime state is WIN; skip subagent LLM request and end task"
                self._log_event(
                    "message",
                    {
                        "source": log_source,
                        "state_id": state_id,
                        "loop_id": loop_id,
                        "message_id": f"{task_id}_win_guard_{turn}",
                        "message": {"role": "assistant", "parts": [{"text": final_text}]},
                        "message_meta": {
                            "subagent_kind": kind,
                            "subagent_task_id": task_id,
                            "subagent_turn": int(turn),
                            "subagent_phase": "win_guard",
                        },
                    },
                )
                break
            messages = self._maybe_compact_subagent(
                log_source=log_source,
                state_id=state_id,
                loop_id=loop_id,
                task_id=task_id,
                kind=kind,
                turn=turn,
                messages=messages,
                tool_defs=tool_defs,
                tool_dispatch=subagent_tools.dispatch,
                max_output_tokens=max_output_tokens,
                observed_total_tokens=observed_total_tokens,
            )
            try:
                rsp = self._complete_subagent_with_retry(
                    kind=kind,
                    task_id=task_id,
                    log_source=log_source,
                    state_id=state_id,
                    loop_id=loop_id,
                    turn=turn,
                    messages=messages,
                    tool_defs=tool_defs,
                    tool_dispatch=subagent_tools.dispatch,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                err_msg = str(exc or "")
                if "LLM_REQUEST_LIMIT_REACHED" in err_msg:
                    end_reason = "llm_request_limit_reached"
                elif "CONTEXT_CLEAR_LIMIT_REACHED" in err_msg:
                    end_reason = "context_clear_limit_reached"
                else:
                    end_reason = "llm_error"
                err_text = f"[subagent_error] {type(exc).__name__}: {exc}"
                self._log_event(
                    "message",
                    {
                        "source": log_source,
                        "state_id": state_id,
                        "loop_id": loop_id,
                        "message_id": f"{task_id}_error_{turn}",
                        "message": {"role": "assistant", "parts": [{"text": err_text}]},
                        "raw_request": {},
                        "message_meta": {
                            "subagent_kind": kind,
                            "subagent_task_id": task_id,
                            "subagent_turn": int(turn),
                            "subagent_phase": "error",
                        },
                    },
                )
                final_text = err_text
                break
            usage = rsp.usage or {}
            total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            observed_total_tokens = int(usage.get("total_tokens", 0) or 0)
            final_text = str(rsp.text or "").strip()
            tool_calls = list(rsp.tool_calls or [])
            total_tool_calls += len(tool_calls)

            # Use raw parts for logging fidelity (including thought parts),
            # but exclude thought parts from context to avoid leaking model thoughts
            # into subsequent requests.
            assistant_parts_raw: List[Dict[str, Any]] = []
            try:
                candidates = (rsp.raw or {}).get("candidates", []) if isinstance(rsp.raw, dict) else []
                if isinstance(candidates, list) and candidates:
                    content = (candidates[0] or {}).get("content", {})
                    parts = content.get("parts", []) if isinstance(content, dict) else []
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict):
                                assistant_parts_raw.append(dict(p))
            except Exception:
                assistant_parts_raw = []

            def _is_blank_text_part(part: Dict[str, Any]) -> bool:
                return set(part.keys()) == {"text"} and not str(part.get("text") or "").strip()

            assistant_parts_ctx: List[Dict[str, Any]] = []
            filtered_raw: List[Dict[str, Any]] = []
            for p in assistant_parts_raw:
                if p.get("thought") is True:
                    continue
                if _is_blank_text_part(p):
                    continue
                part_copy = dict(p)
                filtered_raw.append(part_copy)
                assistant_parts_ctx.append(dict(part_copy))
            assistant_parts_raw = filtered_raw

            # Fallback path when provider doesn't return candidate parts.
            if not assistant_parts_raw:
                if str(final_text or "").strip():
                    cleaned_text = str(final_text).strip()
                    assistant_parts_raw.append({"text": cleaned_text})
                    assistant_parts_ctx.append({"text": cleaned_text})
                for tc in tool_calls:
                    part: Dict[str, Any] = {
                        "functionCall": {
                            "name": str(tc.get("name", "")),
                            "args": tc.get("args", {}),
                        }
                    }
                    if tc.get("thoughtSignature"):
                        part["thoughtSignature"] = tc.get("thoughtSignature")
                    assistant_parts_raw.append(part)
                    assistant_parts_ctx.append(dict(part))

            if not assistant_parts_raw:
                assistant_parts_raw = [{"text": ""}]
            if not assistant_parts_ctx:
                # Keep context syntactically valid; if this turn is thought-only and no tool call,
                # an empty text part is safer than re-injecting thought text.
                assistant_parts_ctx = [{"text": ""}]

            assistant_msg_ctx: Dict[str, Any] = {
                "role": "assistant",
                "parts": assistant_parts_ctx,
            }
            messages.append(assistant_msg_ctx)
            self._log_event(
                "message",
                {
                    "source": log_source,
                    "state_id": state_id,
                    "loop_id": loop_id,
                    "message_id": f"{task_id}_assistant_{turn}",
                    "message": {"role": "assistant", "parts": assistant_parts_raw},
                    "raw_request": rsp.raw_request if isinstance(getattr(rsp, "raw_request", None), dict) else {},
                    "raw_response": rsp.raw if isinstance(getattr(rsp, "raw", None), dict) else {},
                    "request_generation_config": (
                        rsp.request_generation_config
                        if isinstance(getattr(rsp, "request_generation_config", None), dict)
                        else {}
                    ),
                    "message_meta": {
                        "subagent_kind": kind,
                        "subagent_task_id": task_id,
                        "subagent_turn": int(turn),
                        "subagent_phase": "assistant",
                    },
                },
            )

            if not tool_calls:
                end_reason = "assistant_no_tool_calls"
                break

            tool_results: List[Dict[str, Any]] = []
            for call in tool_calls:
                name = str(call.get("name") or "")
                call_args = call.get("args")
                call_args = call_args if isinstance(call_args, dict) else {}
                if name in {"run_grid_survey_task", "run_deep_analysis"}:
                    result = "Error: nested subagent calls are not allowed"
                elif name not in allowed_tool_names:
                    result = f"Error: tool {name} is not allowed for subagent {kind}"
                else:
                    result = subagent_tools.dispatch(name, call_args)
                tool_results.append({"name": name, "result": result})
            messages.append({"role": "user", "tool_results": tool_results})
            tool_parts = build_tool_result_parts(tool_results)
            self._log_event(
                "message",
                {
                    "source": log_source,
                    "state_id": state_id,
                    "loop_id": loop_id,
                    "message_id": f"{task_id}_tool_result_{turn}",
                    "message": {"role": "user", "parts": tool_parts},
                    "message_meta": {
                        "subagent_kind": kind,
                        "subagent_task_id": task_id,
                        "subagent_turn": int(turn),
                        "subagent_phase": "tool_result",
                    },
                },
            )
            if (turn + 1) % int(self.SUBAGENT_TODO_REMINDER_INTERVAL) == 0:
                periodic_reminder = self._build_subagent_todo_reminder(
                    include_empty=False,
                    tool_owner=subagent_tools,
                )
                if periodic_reminder:
                    reminder_parts = [{"text": periodic_reminder}]
                    messages.append({"role": "user", "parts": reminder_parts})
                    self._log_event(
                        "message",
                        {
                            "source": log_source,
                            "state_id": state_id,
                            "loop_id": loop_id,
                            "message_id": f"{task_id}_todo_reminder_{turn}",
                            "message": {"role": "user", "parts": reminder_parts},
                            "message_meta": {
                                "subagent_kind": kind,
                                "subagent_task_id": task_id,
                                "subagent_turn": int(turn),
                                "subagent_phase": "todo_reminder",
                                "special_types": ["system_reminder"],
                            },
                        },
                    )

            if turn == max_turns - 1:
                end_reason = "max_turns_reached"
                final_text = (final_text + "\n[truncated: subagent max_turns reached]").strip()
            else:
                end_reason = "tool_calls_continue"

        max_chars = max(1024, min(32000, int(self.shell_output_max_chars) * 4))
        if len(final_text) > max_chars:
            final_text = f"{final_text[:max_chars]}...[truncated]"

        turns_count = len([m for m in messages if m.get("role") == "assistant"])
        self._log_event(
            "message",
            {
                "source": log_source,
                "state_id": state_id,
                "loop_id": loop_id,
                "message_id": f"{task_id}_end",
                "message": {
                    "role": "assistant",
                    "parts": [
                        {
                            "text": (
                                "[subagent_end] "
                                f"kind={kind} "
                                f"task_id={task_id} "
                                f"reason={end_reason} "
                                f"turns={turns_count} "
                                f"tool_calls={total_tool_calls}"
                            )
                        }
                    ],
                },
                "message_meta": {
                    "subagent_kind": kind,
                    "subagent_task_id": task_id,
                    "subagent_phase": "end",
                    "end_reason": end_reason,
                    "turns": int(turns_count),
                    "tool_calls": int(total_tool_calls),
                },
            },
        )
        return final_text

    def _log_runtime_observation_event(self, payload: Dict[str, Any]) -> None:
        observation = payload.get("observation")
        if not isinstance(observation, dict):
            return
        step_raw = payload.get("step")
        try:
            step = int(step_raw)
        except Exception:
            return
        self._log_v3.log_runtime_observation(
            session_id=str(payload.get("session_id") or self._session_id),
            state_id=str(payload.get("state_id") or f"level_{int(observation.get('levels_completed') or 0):04d}"),
            loop_id=str(payload.get("loop_id") or ""),
            source=str(payload.get("source") or "runtime"),
            step=step,
            observation=observation,
            message_id=str(payload.get("message_id") or ""),
        )

    def _log_runtime_action_frame_event(self, payload: Dict[str, Any]) -> None:
        action = payload.get("action")
        result = payload.get("result")
        if not isinstance(action, dict) or not isinstance(result, dict):
            return
        self._log_v3.log_action_frame(
            session_id=str(payload.get("session_id") or self._session_id),
            state_id=str(payload.get("state_id") or f"level_{int(self.levels_completed):04d}"),
            loop_id=str(payload.get("loop_id") or ""),
            source=str(payload.get("source") or "runtime"),
            af_id=str(payload.get("af_id") or ""),
            message_id=str(payload.get("message_id") or ""),
            action_name=str(action.get("name") or ""),
            action_args=dict(action.get("args") or {}),
            status=str(result.get("status") or "ok"),
            observation=dict(result.get("observation") or {}),
            error=str(result.get("error") or ""),
        )

    def _snapshot_replay_config(self) -> None:
        try:
            self._replay_dir.mkdir(parents=True, exist_ok=True)
            src = Path(get_tell_config_path()).resolve()
            if not src.is_file():
                return
            shutil.copy2(src, self._replay_dir / "agent_config.yaml")
            (self._replay_dir / "agent_config.path.txt").write_text(str(src), encoding="utf-8")
        except Exception:
            logger.warning("failed to snapshot agent yaml into replay dir", exc_info=True)

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        self._worker_stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join()
        try:
            self._runtime_service.stop()
        except Exception:
            pass
        try:
            self._log_v3.close()
        except Exception:
            pass
        self._cleanup_workspace_volume()
        super().cleanup(scorecard)

    def _cleanup_workspace_volume(self) -> None:
        volume = getattr(self, "_workspace_volume", None)
        if not isinstance(volume, WorkspaceVolume):
            return
        try:
            volume.cleanup()
        except Exception:
            logger.warning("failed to cleanup workspace volume", exc_info=True)

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        # Do not stop on GAME_OVER; the agent can reset and continue this run.
        if latest_frame.state == GameState.WIN:
            if self._is_win_epilogue_complete():
                return True
            # On WIN, allow the worker to use one final epilogue turn to update MEMORY.md
            # and terminate naturally with a no-tool-call reply.
            if not self._win_stop_requested:
                self._win_stop_requested = True
                self._log_event(
                    "message",
                    {
                        "source": "main",
                        "state_id": f"level_{int(latest_frame.levels_completed):04d}",
                        "loop_id": self._last_action_loop_id,
                        "message_id": f"win_graceful_stop_{int(self.action_counter)}",
                        "message": {
                            "role": "assistant",
                            "parts": [
                                {
                                    "text": (
                                        "[win_detected] WIN detected; "
                                        "allowing final epilogue turn for MEMORY.md update before worker stop"
                                    )
                                }
                            ],
                        },
                    },
                )
                self._sync_win_observation(latest_frame)
            self._wait_for_win_epilogue()
            return True
        return False

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        obs = self._build_observation(latest_frame)
        self.state_machine.ingest_observation(obs)
        self._runtime_service.update_observation(obs)

        self._ensure_worker_started()

        # Do not auto-fallback to a random action: keep waiting until the model submits an action,
        # or until the worker dies/stops (to avoid silent policy degradation).
        pending = None
        while pending is None:
            step = self._runtime_service.current_step()
            pending = self._runtime_service.wait_for_action(step=step, timeout=self.action_wait_timeout)
            if pending is not None:
                action_label = str(pending.get("action", "")).strip().lower()
                x = pending.get("x")
                y = pending.get("y")
                try:
                    action = self.bridge.build_action(latest_frame, action_label, x=x, y=y)
                    self._last_action_taken = self._action_trace(action)
                    self._last_action_message_id = str(pending.get("message_id") or "")
                    self._last_action_state_id = str(pending.get("state_id") or "")
                    self._last_action_loop_id = str(pending.get("loop_id") or "")
                    return action
                except Exception as exc:
                    available = self.bridge.available_action_labels(latest_frame)
                    detail = (
                        f"invalid_pending_action: action={action_label!r}, x={x!r}, y={y!r}, "
                        f"available={available!r}, err={exc}"
                    )
                    self._log_event(
                        "message",
                        {
                            "source": "main",
                            "state_id": str(pending.get("state_id") or f"level_{int(latest_frame.levels_completed):04d}"),
                            "loop_id": str(pending.get("loop_id") or self._last_action_loop_id),
                            "message_id": f"invalid_pending_action_{int(self.action_counter)}",
                            "message": {
                                "role": "assistant",
                                "parts": [{"text": f"[action_error] {detail}"}],
                            },
                        },
                    )
                    self._runtime_service.record_action_error(
                        action=action_label,
                        x=x,
                        y=y,
                        error=detail,
                        source=str(pending.get("source") or "plan"),
                        message_id=str(pending.get("message_id") or ""),
                        state_id=str(pending.get("state_id") or ""),
                        loop_id=str(pending.get("loop_id") or ""),
                        session_id=str(pending.get("session_id") or "sess_0001"),
                    )
                    self._inject_action_error_for_model(detail)
                    self._restart_worker_after_action_error()
                    pending = None
                    continue
            if self._worker_error:
                self._log_event(
                    "message",
                    {
                        "source": "main",
                        "state_id": f"level_{int(latest_frame.levels_completed):04d}",
                        "loop_id": self._last_action_loop_id,
                        "message_id": f"worker_failed_{int(self.action_counter)}",
                        "message": {
                            "role": "assistant",
                            "parts": [{"text": f"[worker_failed] {self._worker_error}"}],
                        },
                    },
                )
                raise RuntimeError(f"worker_failed: {self._worker_error}")
            if self._worker_stop.is_set():
                self._log_event(
                    "message",
                    {
                        "source": "main",
                        "state_id": f"level_{int(latest_frame.levels_completed):04d}",
                        "loop_id": self._last_action_loop_id,
                        "message_id": f"worker_stopped_{int(self.action_counter)}",
                        "message": {
                            "role": "assistant",
                            "parts": [{"text": "[worker_stopped] worker stop flag set while waiting for action"}],
                        },
                    },
                )
                raise RuntimeError("worker_stopped")
        raise RuntimeError("worker_stopped")

    def _ensure_worker_started(self) -> None:
        if self._worker_started:
            return
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._worker_started = True

    def _inject_action_error_for_model(self, detail: str) -> None:
        reminder = (
            "<system-reminder>\n"
            "The runtime rejected your submitted action/action plan.\n"
            f"{detail}\n"
            "Do not assume the action executed. Inspect available_actions and observation history, "
            "then submit a corrected action or action plan.\n"
            "</system-reminder>"
        )
        message = {"role": "user", "parts": [{"text": reminder}]}
        try:
            self.state_machine._messages.append(message)
        except Exception:
            return
        self._log_event(
            "message",
            {
                "source": "main",
                "state_id": self._last_action_state_id or f"level_{int(self.levels_completed):04d}",
                "loop_id": self._last_action_loop_id,
                "message_id": f"invalid_pending_action_reminder_{int(self.action_counter)}",
                "message": message,
                "message_meta": {
                    "special_types": ["system_reminder"],
                    "reminder_kind": "invalid_pending_action",
                },
            },
        )

    def _restart_worker_after_action_error(self) -> None:
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            self._worker_stop.set()
            thread.join(timeout=1.0)
        self._worker_stop = threading.Event()
        self._worker_error = None
        self._worker_started = False
        self._worker_thread = None
        self._ensure_worker_started()

    def _prepare_workspace(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        template_root = self._resolve_template_workspace_root()
        _skip_dirs = {"__pycache__", ".git", ".mypy_cache", ".ruff_cache", ".fseventsd", ".DS_Store"}
        if template_root.exists() and template_root.is_dir():
            for item in sorted(template_root.iterdir()):
                if item.name in _skip_dirs:
                    continue
                dst = self.workspace / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns(*_skip_dirs))
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dst)

    def _maybe_load_prior_memory(self) -> None:
        """If PRIOR_MEMORY_DIR is set, copy a pre-built MEMORY.md into workspace.

        Looks for ``{game_prefix}_MEMORY.md`` inside the directory. This is
        simpler than RESUME_WORKSPACE_FROM and intended for cleaned/curated
        memory files that aren't tied to a replay directory.
        """
        mem_dir = str(get_env_str("PRIOR_MEMORY_DIR", "") or "").strip()
        if not mem_dir:
            return
        mem_path = Path(mem_dir).resolve()
        if not mem_path.is_dir():
            logger.warning("PRIOR_MEMORY_DIR: not a directory: %s", mem_path)
            return
        game_prefix = self.game_id.split("-")[0]
        candidates = list(mem_path.glob(f"{game_prefix}_MEMORY.md")) + list(
            mem_path.glob(f"{game_prefix}_*.md")
        )
        if not candidates:
            logger.info("PRIOR_MEMORY_DIR: no file for game %s in %s", game_prefix, mem_path)
            return
        src = candidates[0]
        dst = self.workspace / "MEMORY.md"
        shutil.copy2(src, dst)
        logger.info("PRIOR_MEMORY_DIR: copied %s -> workspace MEMORY.md", src.name)

    def _maybe_load_prior_workspace(self) -> None:
        """If RESUME_WORKSPACE_FROM is set, overlay workspace files from a prior run.

        This copies workspace files (especially MEMORY.md) from a previous run's
        log directory into the current workspace, giving the agent prior knowledge
        while starting a fresh game.
        """
        prior_log_dir = str(get_env_str("RESUME_WORKSPACE_FROM", "") or "").strip()
        if not prior_log_dir:
            return
        prior_path = Path(prior_log_dir).resolve()
        if not prior_path.exists():
            logger.warning("RESUME_WORKSPACE_FROM path does not exist: %s", prior_path)
            return
        try:
            state = load_resume_state(str(prior_path))
            if not state.workspace_files:
                logger.warning("RESUME_WORKSPACE_FROM: no workspace files found in %s", prior_path)
                return
            count = restore_workspace(self.workspace, state.workspace_files)
            logger.info(
                "RESUME_WORKSPACE_FROM: restored %d workspace files from %s",
                count,
                prior_path,
            )
        except Exception as exc:
            logger.warning("RESUME_WORKSPACE_FROM: failed to load: %s", exc)

    def _resolve_template_workspace_root(self) -> Path:
        configured = str(get_workspace_str("template_workspace", "") or "").strip()
        default_root = Path(__file__).resolve().with_name("template_workspace")
        if not configured:
            return default_root
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            # Project-relative path: resolve from repository root.
            repo_root = Path(__file__).resolve().parents[2]
            candidate = (repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
        logger.warning(
            "configured workspace.template_workspace not found or not a directory: %s; fallback=%s",
            str(candidate),
            str(default_root),
        )
        return default_root

    def _worker_loop(self) -> None:
        try:
            # Preflight: observation interface must be healthy before stage loop starts.
            if not self._wait_for_runtime_observation_ready(timeout=self.action_wait_timeout):
                raise RuntimeError("runtime_observation_unhealthy")
            while not self._worker_stop.is_set():
                if self._runtime_service.has_action_backlog():
                    self._worker_stop.wait(timeout=max(0.01, float(self.worker_poll_interval)))
                    continue
                self.state_machine.run_stage_loop_once()
                _should_stop, loop_end_reason = self._should_stop_worker_after_stage()
                if not loop_end_reason:
                    loop_end_reason = "stage_loop_complete"
                self._worker_stop.set()
                is_win = self._runtime_is_win()
                event_name = (
                    "win_epilogue_complete"
                    if is_win
                    else "worker_stage_complete"
                )
                event_text = (
                    "[win_epilogue_complete] "
                    f"loop ended with reason={loop_end_reason}; stopping worker"
                    if is_win
                    else "[worker_stage_complete] "
                    f"loop ended with reason={loop_end_reason}; stopping worker"
                )
                self._log_event(
                    "message",
                    {
                        "source": "worker",
                        "state_id": self._last_action_state_id or f"level_{int(self.levels_completed):04d}",
                        "loop_id": self._last_action_loop_id,
                        "message_id": f"{event_name}_{int(self.action_counter)}",
                        "message": {
                            "role": "assistant",
                            "parts": [{"text": event_text}],
                        },
                    },
                )
                break
        except Exception as exc:
            self._worker_error = str(exc)
            tb = traceback.format_exc(limit=20)
            self._log_event(
                "message",
                {
                    "source": "worker",
                    "state_id": self._last_action_state_id or f"level_{int(self.levels_completed):04d}",
                    "loop_id": self._last_action_loop_id,
                    "message_id": f"worker_exception_{int(self.action_counter)}",
                    "message": {
                        "role": "assistant",
                        "parts": [{"text": f"[worker_exception] {self._worker_error}\n{tb}"}],
                    },
                },
            )
            logger.exception("tell worker loop failed")

    def try_resume_from_log(self, resume_log_dir: str) -> bool:
        """Attempt to resume agent from a previous run's log directory.

        This restores:
        1. Workspace files from fs_versions.jsonl
        2. LLM conversation context into state_machine
        3. Game actions for environment replay (caller must replay via env.step())

        Returns True if resume state was loaded successfully.
        The caller is responsible for replaying game actions through the environment.
        """
        state = load_resume_state(resume_log_dir)
        if not state.valid:
            logger.warning("resume failed: %s", state.error)
            return False

        # 1. Restore workspace files.
        if state.workspace_files:
            count = restore_workspace(self.workspace, state.workspace_files)
            logger.info("resume: restored %d workspace files", count)

        # 2. Restore state machine context.
        if state.stage_name:
            self.state_machine.current_stage = state.stage_name
        if state.loop_counter > 0:
            self.state_machine.loop_counter = state.loop_counter
        if state.last_observation:
            self.state_machine.ingest_observation(state.last_observation)

        # 3. Restore LLM messages directly into the state machine.
        if state.messages and state.stage_name:
            max_tokens = self.state_machine._compaction_cfg.max_context_tokens
            system_content = str(getattr(self.state_machine, "_system_content", "") or "").strip()
            if not system_content:
                try:
                    system_values = self.state_machine._base_prompt_values()
                    system_content = self.state_machine._prompt_renderer.render(
                        self.state_machine._system_template,
                        system_values,
                    )
                except Exception:
                    system_content = ""
            resumed_messages = build_resume_context_messages(
                state, system_content or "", max_context_tokens=max_tokens
            )
            self.state_machine._messages = resumed_messages
            self.state_machine._system_content = system_content
            if state.last_compaction_summary:
                self.state_machine._last_compaction_summary = state.last_compaction_summary

        # 4. Store the action replay list for the caller.
        self._resume_actions = state.actions
        self._resume_state = state

        # 5. Restore budget counters from manifest and run_stats.
        manifest = state.manifest
        stats = state.stats

        # LLM request budget
        self.state_machine._llm_requests_used = int(manifest.get("raw_request_count", 0) or 0)

        # Context clear budget
        self.state_machine._context_clears_used = int(stats.get("compaction_count", 0) or 0)

        # Token budget
        tokens = stats.get("tokens", {})
        self.state_machine._total_prompt_tokens_used = int(tokens.get("prompt_total", 0) or 0)
        self.state_machine._total_completion_tokens_used = int(tokens.get("completion_total", 0) or 0)

        # Action counter (base Agent field)
        self.action_counter = int(manifest.get("action_frame_count", 0) or 0)

        # Prevent spurious level-up / action-submitted events
        obs = state.last_observation
        levels_completed = int(obs.get("levels_completed", 0) or 0) if obs else 0
        self.state_machine._last_level = levels_completed
        self.state_machine._last_action_count = self.action_counter

        self._log_event(
            "message",
            {
                "source": "main",
                "state_id": self.state_machine._state_id(),
                "loop_id": "",
                "message_id": "session_resume",
                "message": {
                    "role": "assistant",
                    "parts": [
                        {
                            "text": (
                                f"[session_resume] "
                                f"messages={len(state.messages)} "
                                f"actions={len(state.actions)} "
                                f"workspace_files={len(state.workspace_files)} "
                                f"stage={state.stage_name} "
                                f"loop={state.loop_counter} "
                                f"llm_requests={self.state_machine._llm_requests_used} "
                                f"context_clears={self.state_machine._context_clears_used} "
                                f"prompt_tokens={self.state_machine._total_prompt_tokens_used} "
                                f"action_counter={self.action_counter}"
                            )
                        }
                    ],
                },
                "message_meta": {
                    "event": "session_resume",
                    "resume_log_dir": str(resume_log_dir),
                },
            },
        )
        logger.info(
            "resume: loaded %d messages, %d actions, stage=%s, loop=%d, "
            "llm_requests=%d, context_clears=%d, tokens=%d+%d, action_counter=%d",
            len(state.messages),
            len(state.actions),
            state.stage_name,
            state.loop_counter,
            self.state_machine._llm_requests_used,
            self.state_machine._context_clears_used,
            self.state_machine._total_prompt_tokens_used,
            self.state_machine._total_completion_tokens_used,
            self.action_counter,
        )
        return True

    def get_resume_actions(self) -> List[Dict[str, Any]]:
        """Get the list of game actions to replay for environment recovery.

        Each entry has {name: str, args: dict}. The caller should map
        these to GameAction and call env.step() for each.
        """
        return getattr(self, "_resume_actions", [])

    def replay_resume_actions(self) -> int:
        """Replay logged game actions to restore environment state.

        Must be called after try_resume_from_log() and before the main loop.
        Games are deterministic so replaying actions rebuilds the exact state.
        Returns the number of actions replayed.
        """
        actions = self.get_resume_actions()
        if not actions:
            return 0

        from agents.game_bridge import ACTION_BY_LABEL

        replayed = 0
        for action_record in actions:
            name = str(action_record.get("name", "")).strip().lower()
            args = action_record.get("args") or {}

            if name not in ACTION_BY_LABEL:
                logger.warning("resume replay: unknown action %r, skipping", name)
                continue

            game_action = ACTION_BY_LABEL[name]
            data: Dict[str, Any] = {}
            if name == "click":
                x, y = args.get("x"), args.get("y")
                if x is not None and y is not None:
                    game_action.set_data({"x": int(x), "y": int(y)})
                    data = {"x": int(x), "y": int(y)}

            try:
                raw = self.arc_env.step(game_action, data=data)
                frame = self._convert_raw_frame_data(raw)
                self.frames.append(frame)
                if frame.guid:
                    self.guid = frame.guid
                replayed += 1
            except Exception as exc:
                logger.error("resume replay failed at action %d (%s): %s", replayed, name, exc)
                break

        # Sync latest observation to state machine and runtime service
        if replayed > 0 and len(self.frames) > 1:
            latest = self.frames[-1]
            obs = self._build_observation(latest)
            self.state_machine.ingest_observation(obs)
            self._runtime_service.update_observation(obs)
            # Verify state consistency
            expected_levels = int(getattr(self, "_resume_state", None) and
                                  self._resume_state.manifest.get("max_levels_completed_observed", 0) or 0)
            actual_levels = latest.levels_completed
            if actual_levels != expected_levels:
                logger.warning(
                    "resume replay: levels_completed mismatch (expected=%d, actual=%d)",
                    expected_levels, actual_levels,
                )

        logger.info("resume: replayed %d/%d actions through environment", replayed, len(actions))
        return replayed

    def _on_pre_loop(self) -> None:
        """Replay actions to restore game state before the main loop starts."""
        if self._is_resumed:
            replayed = self.replay_resume_actions()
            logger.info("resume: replayed %d actions before main loop", replayed)

    def _submit_action_plan(self, actions: List[Dict[str, Any]], meta: Dict[str, str]) -> Dict[str, Any]:
        result = self._runtime_service.enqueue_action_plan(
            actions,
            source="plan",
            message_id=str(meta.get("message_id") or ""),
            state_id=str(meta.get("state_id") or ""),
            loop_id=str(meta.get("loop_id") or ""),
            replace_existing=True,
        )
        try:
            self._log_event(
                "message",
                {
                    "source": "main",
                    "state_id": str(meta.get("state_id") or ""),
                    "loop_id": str(meta.get("loop_id") or ""),
                    "message_id": f"action_plan_{int(self.action_counter)}",
                    "message": {
                        "role": "assistant",
                        "parts": [
                            {
                                "text": (
                                    "[action_plan_submit] "
                                    f"queued={int(result.get('queued', 0))} "
                                    f"skipped={int(result.get('skipped', 0))}"
                                )
                            }
                        ],
                    },
                    "message_meta": {
                        "event": "action_plan_submit",
                        "queued": int(result.get("queued", 0)),
                        "skipped": int(result.get("skipped", 0)),
                    },
                },
            )
        except Exception:
            pass
        return result

    def _wait_for_runtime_observation_ready(self, timeout: float) -> bool:
        snap = self._runtime_service.get_observation_snapshot()
        if isinstance(snap, dict) and snap:
            return True
        step = self._runtime_service.wait_for_observation_after(step=-1, timeout=timeout)
        return step is not None

    def _build_observation(self, latest_frame: FrameData) -> Dict[str, Any]:
        grids = self._normalize_frames(latest_frame.frame)
        grid_last = grids[-1] if grids else []
        width = len(grid_last[0]) if grid_last and isinstance(grid_last[0], list) else 0
        height = len(grid_last)
        frame_count = len(grids) if grids else 1
        summary = f"{frame_count} frames, {width}x{height} grid" if frame_count > 1 else f"{width}x{height} grid"
        state = latest_frame.state.value if hasattr(latest_frame.state, "value") else str(latest_frame.state)
        levels_completed = int(latest_frame.levels_completed)
        frames = [{"frame_index": idx, "grid": frame} for idx, frame in enumerate(grids)] if grids else []
        default_state_id = f"level_{levels_completed:04d}"
        state_upper = str(state).strip().upper()
        if state_upper == "GAME_OVER":
            state_id = (
                self._last_observed_state_id
                or self._last_action_state_id
                or default_state_id
            )
        else:
            state_id = default_state_id
            self._last_observed_state_id = state_id
        return {
            "game_id": self.game_id,
            "state_id": state_id,
            "state": state,
            "levels_completed": levels_completed,
            "available_actions": self.bridge.available_action_labels(latest_frame),
            "frame_count": frame_count,
            "summary": summary,
            "frames": frames,
        }

    def _normalize_frames(self, frame: List[Any]) -> List[List[List[int]]]:
        if not frame:
            return [[]]
        if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], int):
            return [frame]
        return frame

    def _action_trace(self, action: GameAction) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if action == GameAction.ACTION6:
            action_data = getattr(action, "data", None) or {}
            if isinstance(action_data, dict):
                if "x" in action_data:
                    data["x"] = action_data["x"]
                if "y" in action_data:
                    data["y"] = action_data["y"]
        return {"name": action.name, "data": data}

    def _fallback_game_action(self, latest_frame: FrameData) -> GameAction:
        available = self.bridge.available_action_labels(latest_frame)
        if not available:
            raise RuntimeError("No available actions")
        action = self.bridge.build_action(latest_frame, available[0])
        self._last_action_taken = self._action_trace(action)
        try:
            ctx = self.state_machine.current_action_context()
            self._last_action_message_id = str(ctx.get("message_id") or "")
            self._last_action_state_id = str(ctx.get("state_id") or "")
            self._last_action_loop_id = str(ctx.get("loop_id") or "")
        except Exception:
            pass
        # Emit an explicit log message so fallback actions are always visible in message logs.
        # (action_frames.jsonl is written on the next observation, but this is immediate.)
        try:
            self._log_event(
                "message",
                {
                    "source": "main",
                    "state_id": self._last_action_state_id or f"level_{int(latest_frame.levels_completed):04d}",
                    "loop_id": self._last_action_loop_id,
                    "message_id": f"fallback_action_{int(self.action_counter)}",
                    "message": {
                        "role": "assistant",
                        "parts": [{"text": f"[fallback_action] using available_actions[0]={available[0]}"}],
                    },
                },
            )
        except Exception:
            logger.exception("failed to log fallback action")
        return action

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "raw_request":
            raw_request = payload.get("raw_request")
            raw_response = payload.get("raw_response")
            request_generation_config = payload.get("request_generation_config")
            if not (
                isinstance(raw_request, dict)
                or isinstance(raw_response, dict)
                or isinstance(request_generation_config, dict)
            ):
                return
            source = str(payload.get("source") or "main")
            state_id = str(payload.get("state_id") or f"level_{int(self.levels_completed):04d}")
            loop_id = str(payload.get("loop_id") or "")
            msg_id = str(payload.get("message_id") or "")
            try:
                self._log_v3.log_raw_request(
                    session_id=self._session_id,
                    state_id=state_id,
                    loop_id=loop_id,
                    source=source,
                    message_id=msg_id,
                    raw_request=raw_request if isinstance(raw_request, dict) else None,
                    raw_response=raw_response if isinstance(raw_response, dict) else None,
                    request_generation_config=(
                        request_generation_config if isinstance(request_generation_config, dict) else None
                    ),
                )
            except Exception:
                logger.exception("tell log_v3 raw_request write failed")
            return
        if event_type != "message":
            return
        message = payload.get("message")
        if not isinstance(message, dict):
            return
        raw_response = payload.get("raw_response")
        raw_request = payload.get("raw_request")
        request_generation_config = payload.get("request_generation_config")
        tools = payload.get("tools")
        tools_digest = str(payload.get("tools_digest") or "")
        message_meta = payload.get("message_meta")
        msg_id = str(payload.get("message_id") or "")
        source = str(payload.get("source") or "main")
        state_id = str(payload.get("state_id") or f"level_{int(self.levels_completed):04d}")
        loop_id = str(payload.get("loop_id") or "")
        role = str(message.get("role") or "")
        parts = message.get("parts")
        if not isinstance(parts, list):
            parts = [{"text": str(message.get("content") or "")}]
        message_type = "text"
        if any(isinstance(p, dict) and "functionCall" in p for p in parts):
            message_type = "function_call"
        elif any(isinstance(p, dict) and "functionResponse" in p for p in parts):
            message_type = "tool_result"
        try:
            self._log_v3.log_message(
                session_id=self._session_id,
                state_id=state_id,
                loop_id=loop_id,
                source=source,
                message_id=msg_id,
                role=role,
                message_type=message_type,
                parts=parts,
                tools=tools if isinstance(tools, list) else None,
                tools_digest=tools_digest,
                message_meta=message_meta if isinstance(message_meta, dict) else None,
            )
            if (
                isinstance(raw_request, dict)
                or isinstance(raw_response, dict)
                or isinstance(request_generation_config, dict)
            ):
                self._log_v3.log_raw_request(
                    session_id=self._session_id,
                    state_id=state_id,
                    loop_id=loop_id,
                    source=source,
                    message_id=msg_id,
                    raw_request=raw_request if isinstance(raw_request, dict) else None,
                    raw_response=raw_response if isinstance(raw_response, dict) else None,
                    request_generation_config=(
                        request_generation_config if isinstance(request_generation_config, dict) else None
                    ),
                )
            self._log_v3.log_fs_version(
                session_id=self._session_id,
                state_id=state_id,
                loop_id=loop_id,
                source=source,
                message_id=msg_id,
            )
        except Exception:
            logger.exception("tell log_v3 write failed")

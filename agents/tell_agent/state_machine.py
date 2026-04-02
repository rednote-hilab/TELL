from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .compaction import (
    DEFAULT_COMPACTION_PROMPT,
    DEFAULT_COMPACTION_USER_TEMPLATE,
    CompactionConfig,
    compact_messages,
    estimate_message_tokens,
)
from .config import (
    get_compaction_bool,
    get_compaction_float,
    get_compaction_int,
    get_compaction_str,
    get_env_int,
    get_env_float,
    get_env_str,
    get_history_log_bool,
    get_history_log_str,
    get_llm_max_tokens_retry_bool,
    get_llm_max_tokens_retry_int,
    get_llm_max_tokens_retry_str,
    get_llm_request_budget_int,
    get_llm_request_budget_str,
    get_llm_empty_response_recovery_bool,
    get_llm_empty_response_recovery_str,
    get_llm_truncation_recovery_bool,
    get_llm_truncation_recovery_int,
    get_llm_truncation_recovery_str,
    get_memory_checkpoint_clear_config,
    get_reminder_templates,
    get_prompt_templates,
    get_workspace_size_limit_bytes,
)
from .claude_client import ClaudeClient
from .logging_v3 import _resolve_history_log_path
from .request_formatter import (
    build_request_data,
    build_text_parts_with_inline_media,
    build_tool_result_parts,
)
from .prompt_renderer import PromptRenderer
from .stage_prompts import render_template
from .permissions import ToolPermissionPolicy

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Resume policy: controls what happens on game events and context overflow.
# ---------------------------------------------------------------------------

@dataclass
class ResumePolicy:
    on_level_up: str = "compact"
    on_action_submitted: str = "keep"
    on_context_limit: str = "compact"


@dataclass
class MemoryCheckpointClearConfig:
    enabled: bool = False
    memory_path: str = "MEMORY.md"
    max_grace_turns: int = 1


def _normalize_resume_action(raw: Any, default: str, *, allow_keep: bool = True) -> str:
    val = str(raw or default).strip().lower()
    if val in ("compact", "clear"):
        return val
    if val == "keep" and allow_keep:
        return val
    return default


def _parse_resume_policy(raw: Any) -> ResumePolicy:
    if not isinstance(raw, dict):
        return ResumePolicy()
    return ResumePolicy(
        on_level_up=_normalize_resume_action(raw.get("on_level_up"), "compact"),
        on_action_submitted=_normalize_resume_action(raw.get("on_action_submitted"), "keep"),
        on_context_limit=_normalize_resume_action(raw.get("on_context_limit"), "compact", allow_keep=False),
    )


DISK_BACKED_PROMPT_KEYS = {
    "KNOWLEDGE_MEMORY_MD",
    "HYPOTHESES_MEMORY_MD",
    "SKILLS_MEMORY_MD",
    "POLICY_MEMORY_MD",
}


# ---------------------------------------------------------------------------
# TELLStateMachine — single-loop LLM agent with inline event handling.
# ---------------------------------------------------------------------------

class TELLStateMachine:
    def __init__(
        self,
        llm: ClaudeClient,
        tools: List[Dict[str, Any]],
        tool_dispatch: Callable[[str, Dict[str, Any]], str],
        log_event: Callable[[str, Dict[str, Any]], None],
        workspace: Path,
        memory_root: Path,
        max_stage_turns: int,
        max_output_tokens: int,
        recent_frames_limit: int,
        runtime_api_base: str,
        run_id: str,
        step_getter: Callable[[], int],
        turn_getter: Callable[[], int],
        runtime_step_getter: Optional[Callable[[], int]] = None,
        runtime_observation_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        runtime_action_history_getter: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        stop_requested_getter: Optional[Callable[[], bool]] = None,
        action_plan_submitter: Optional[Callable[[List[Dict[str, Any]], Dict[str, str]], Dict[str, Any]]] = None,
        workspace_budget_getter: Optional[Callable[[], str]] = None,
        max_actions: int = 2048,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_dispatch = tool_dispatch
        self.log_event = log_event
        self.workspace = workspace
        self.memory_root = memory_root
        self.max_stage_turns = max_stage_turns
        self._stage_hard_turn_limit = max(1, int(max_stage_turns) if int(max_stage_turns) > 0 else 32)
        self.max_output_tokens = max_output_tokens
        self.recent_frames_limit = recent_frames_limit
        self.runtime_api_base = runtime_api_base.rstrip("/")
        self.run_id = str(run_id or "").strip()
        self.step_getter = step_getter
        self.turn_getter = turn_getter
        self.runtime_step_getter = runtime_step_getter or step_getter
        self.runtime_observation_getter = runtime_observation_getter
        self.runtime_action_history_getter = runtime_action_history_getter
        self.stop_requested_getter = stop_requested_getter
        self.action_plan_submitter = action_plan_submitter
        self._workspace_budget_getter = workspace_budget_getter
        self._max_actions = max_actions

        # Tools that modify workspace files (trigger budget reminder)
        self._file_modifying_tools: set[str] = {"write_file"}

        # Level-up reminder flag (consumed by reminder injection)
        self._pending_level_up_reminder: Optional[str] = None
        # Deferred compact/clear — set by inline event handlers, executed after
        # tool_results are appended to preserve functionCall/functionResponse pairing.
        self._deferred_context_action: Optional[tuple[str, str]] = None
        self._memory_checkpoint_pending = False
        self._memory_checkpoint_turns_waited = 0
        self._memory_checkpoint_reason = ""

        # Game state
        self.current_level = 0
        self.policy: Dict[str, Any] = {"rules": []}
        self.recent_frames: List[Dict[str, Any]] = []
        self.recent_observations: List[Dict[str, Any]] = []
        self.last_observation: Dict[str, Any] = {}

        # Progress tracking (for <game-status> reminder)
        self._action_count_at_level_up = 0
        self._resets_on_current_level = 0      # game_over recovery resets
        self._voluntary_resets_on_current_level = 0  # agent-initiated resets
        self._last_observed_state = ""

        # Loop state
        self.loop_counter = 0
        self.current_loop_id = ""
        self._last_stage_message_id = ""
        self._last_loop_end_reason = ""
        self._recent_tool_signatures: List[str] = []
        self._doom_loop_threshold = 16
        self._win_epilogue_done = False

        # Global token budget
        self._total_prompt_tokens_used = 0
        self._total_completion_tokens_used = 0
        self._global_token_budget = max(0, get_env_int("TELL_GLOBAL_TOKEN_BUDGET", 0))
        self._llm_request_limit = max(0, get_llm_request_budget_int("max_requests_per_run", 0))
        self._llm_request_limit_message = get_llm_request_budget_str(
            "reminder_message",
            "LLM request budget exhausted for this run. Stop here instead of sending more model requests.",
        )
        self._llm_requests_used = 0
        self._max_context_clears_per_run = max(0, get_compaction_int("max_context_clears_per_run", 0))
        self._context_clears_used = 0

        # --- Load config from YAML ---
        prompts_cfg = get_prompt_templates()
        fragments = prompts_cfg.get("fragments")
        self._prompt_renderer = PromptRenderer(
            self.workspace,
            fragments=fragments if isinstance(fragments, dict) else None,
        )

        # Support two YAML layouts:
        #   Flat format (preferred):  prompts.system, prompts.user, prompts.tools, prompts.resume_policy
        #   Legacy format:            prompts.workflow.stages.<entry>.* + prompts.stages.<ref>.*
        if "system" in prompts_cfg and "user" in prompts_cfg:
            # --- Flat format ---
            self._system_template = str(prompts_cfg.get("system", ""))
            self._user_template = str(prompts_cfg.get("user", ""))
            self._tool_names = [str(t) for t in (prompts_cfg.get("tools") or [])]
            self._resume_policy = _parse_resume_policy(prompts_cfg.get("resume_policy"))
            entry_name = "main"
        else:
            # --- Legacy workflow format ---
            workflow_cfg = prompts_cfg.get("workflow", {})
            stages_cfg = workflow_cfg.get("stages", {})
            entry_name = str(workflow_cfg.get("entry_stage", "main")).strip().lower()
            stage_raw = stages_cfg.get(entry_name, {})
            prompt_ref = str(stage_raw.get("prompt_ref", entry_name))
            stage_prompts = prompts_cfg.get("stages", {}).get(prompt_ref, {})
            self._system_template = str(stage_raw.get("system") or stage_prompts.get("system", ""))
            self._user_template = str(stage_raw.get("user") or stage_prompts.get("user", ""))
            self._tool_names = [str(t) for t in (stage_raw.get("tools") or [])]
            self._resume_policy = _parse_resume_policy(stage_raw.get("resume_policy"))

        self._log_source = f"tell_{entry_name}"
        # Keep entry_name accessible for agent.py compatibility.
        self.current_stage = entry_name

        # Persistent conversation state (survives across run_loop_once calls)
        self._messages: List[Dict[str, Any]] = []
        self._system_content: str = ""
        self._last_level: int = 0
        self._last_action_count: int = 0
        self._last_compaction_summary: str = ""
        self._memory_checkpoint_deadline_missed = False

        # Todo
        self._todo_cache: List[Dict[str, str]] = []

        # Permissions
        permissions_cfg = prompts_cfg.get("permissions")
        self._permission_policy = ToolPermissionPolicy.from_config(
            permissions_cfg if isinstance(permissions_cfg, dict) else None
        )
        self._turn_tool_call_count = 0

        # Compaction
        self._compaction_raw_seq = 0
        self._compaction_cfg = self._build_compaction_config()
        self._memory_checkpoint_cfg = self._build_memory_checkpoint_clear_config()
        self._last_prompt_tokens_observed = 0
        self._last_total_tokens_observed = 0

        # Reminders
        self._reminders = self._load_reminder_templates()
        self._todo_reminder_interval = max(1, int(self._reminders.get("todo_reminder_interval") or 32))

        # Rendering
        self._render_scale = max(1, get_env_int("TELL_RENDER_SCALE", 2))

        # LLM parameters
        self._llm_temperature = float(get_env_float("LLM_TEMPERATURE", 0.0))
        top_p_raw = get_env_float("LLM_TOP_P", -1.0)
        self._llm_top_p: Optional[float] = None
        try:
            tp = float(top_p_raw)
            if 0.0 < tp <= 1.0:
                self._llm_top_p = tp
        except Exception:
            self._llm_top_p = None

        # Max-tokens retry
        self._max_tokens_retry_enabled = get_llm_max_tokens_retry_bool("enabled", True)
        self._max_tokens_retry_count = max(0, get_llm_max_tokens_retry_int("max_retries", 2))
        self._max_tokens_retry_reminder = render_template(
            get_llm_max_tokens_retry_str(
                "reminder_message",
                (
                    "<system-reminder>\n"
                    "Your output may be too long. Please keep the response within "
                    "{TELL_MAX_OUTPUT_TOKENS} tokens.\n"
                    "</system-reminder>"
                ),
            ),
            {"TELL_MAX_OUTPUT_TOKENS": str(self.max_output_tokens)},
        )

        # Empty-response recovery
        self._empty_response_recovery_enabled = get_llm_empty_response_recovery_bool("enabled", True)
        self._empty_response_recovery_reminder = get_llm_empty_response_recovery_str(
            "reminder_message",
            (
                "<system-reminder>\n"
                "The previous response was empty. Please continue and complete the task.\n"
                "</system-reminder>"
            ),
        )

        # Truncation recovery: when stop_reason is MAX_TOKENS and no tool calls,
        # keep the truncated text and inject a continuation reminder instead of stopping.
        self._truncation_recovery_enabled = get_llm_truncation_recovery_bool("enabled", True)
        self._truncation_recovery_max_retries = get_llm_truncation_recovery_int("max_retries", 999)
        self._truncation_recovery_reminder = get_llm_truncation_recovery_str(
            "reminder_message",
            (
                "<system-reminder>\n"
                "Your previous response was truncated due to length. Do NOT repeat or re-analyze "
                "what you already wrote. Resume from where you left off and immediately call a tool "
                "(bash_exec, write_file, or screen_shot). Pick the single most important next action "
                "and execute it now.\n"
                "</system-reminder>"
            ),
        )

        # Workspace budget reminder (injected after file-modifying tool calls)
        self._workspace_budget_reminder_enabled = str(
            self._reminders.get("enable_workspace_budget_reminder") or "auto"
        ).strip().lower() not in {"0", "false", "no", "n", "off", "disable", "disabled"}

        # Context token budget reminder (injected every turn after tool results)
        self._context_budget_reminder_enabled = str(
            self._reminders.get("enable_context_budget_reminder") or "auto"
        ).strip().lower() not in {"0", "false", "no", "n", "off", "disable", "disabled"}
        # Threshold (fraction of compaction trigger) at which context budget reminder starts appearing
        _ctx_thr_raw = self._reminders.get("context_budget_reminder_threshold")
        self._context_budget_reminder_threshold: float = (
            float(_ctx_thr_raw) if _ctx_thr_raw is not None else 0.6
        )

        # Max-steps reminder (injected on the last allowed turn)
        self._max_steps_reminder = str(self._reminders.get("max_steps") or "").strip()

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def ingest_observation(self, obs: Dict[str, Any]) -> None:
        self.last_observation = obs
        self.recent_frames.append(
            {
                "step": int(self.step_getter()),
                "state": obs.get("state"),
                "levels_completed": obs.get("levels_completed"),
                "available_actions": obs.get("available_actions", []),
                "summary": obs.get("summary"),
            }
        )
        self.recent_frames = self.recent_frames[-self.recent_frames_limit:]
        self.recent_observations.append(dict(obs))
        self.recent_observations = self.recent_observations[-self.recent_frames_limit:]

        # Track game overs and voluntary resets
        current_state = str(obs.get("state") or "").strip().lower()
        last_action = str(obs.get("last_action") or "").strip().lower()
        if self._last_observed_state == "game_over" and current_state != "game_over":
            self._resets_on_current_level += 1
        elif last_action == "reset" and self._last_observed_state not in ("game_over", ""):
            self._voluntary_resets_on_current_level += 1
        self._last_observed_state = current_state

        state_level = self._level_from_state_id(obs.get("state_id"))
        if state_level is not None:
            self.current_level = state_level
        else:
            self.current_level = int(obs.get("levels_completed") or 0)

    # ------------------------------------------------------------------
    # Main loop — replaces run_stage_loop_once + run_stage
    # ------------------------------------------------------------------

    def run_stage_loop_once(self) -> None:
        """Single iteration of the agent loop.

        Called repeatedly by the worker thread.  Maintains ``self._messages``
        across calls so the conversation is persistent.
        """
        self.loop_counter += 1
        self.current_loop_id = f"level_{self.current_level:04d}_loop_{self.loop_counter:04d}"

        # --- Global token budget check ---
        if self._global_token_budget > 0:
            total_used = self._total_prompt_tokens_used + self._total_completion_tokens_used
            if total_used >= self._global_token_budget:
                self._last_loop_end_reason = "global_budget_exhausted"
                self.log_event(
                    "message",
                    {
                        "source": self._log_source,
                        "state_id": self._state_id(),
                        "loop_id": self.current_loop_id,
                        "message_id": f"tell_global_budget_exhausted_{self.loop_counter}",
                        "message": {
                            "role": "assistant",
                            "parts": [
                                {"text": f"[global_budget_exhausted] total_used={total_used} budget={self._global_token_budget}"}
                            ],
                        },
                        "message_meta": {"event": "global_budget_exhausted"},
                    },
                )
                return

        # --- Initialize messages on first call (or after a full clear) ---
        self._last_loop_end_reason = ""

        if not self._messages:
            self._init_messages()

        stage_tools = [t for t in self.tools if str(t.get("name")) in set(self._tool_names)]
        try:
            tools_digest = hashlib.sha256(
                json.dumps(stage_tools, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        except Exception:
            tools_digest = ""

        todo_mode = str(self._reminders.get("enable_todo_reminder") or "auto").strip().lower()
        todo_enabled = (
            todo_mode in {"1", "true", "yes", "y", "on"}
            or (todo_mode not in {"0", "false", "no", "n", "off", "disable", "disabled"} and "todo_write" in self._tool_names)
        )

        last_runtime_step = int(self.runtime_step_getter())
        self._win_epilogue_done = False
        end_reason = "assistant_no_tool_calls"
        next_periodic_compaction = int(self.max_stage_turns) if self.max_stage_turns > 0 else 0
        final_text = ""
        sub_turn = 0
        consecutive_empty_no_tool_calls = 0
        consecutive_truncation_no_tool_calls = 0

        while True:
            if self._stop_requested():
                end_reason = "worker_stop_before_turn"
                break
            if sub_turn >= self._stage_hard_turn_limit:
                end_reason = "hard_turn_limit_reached"
                self._log_simple_event("hard_turn_limit", f"hard_turn_limit={self._stage_hard_turn_limit} reached", sub_turn)
                break

            self._turn_tool_call_count = 0

            # --- Periodic compaction at turn limit ---
            while next_periodic_compaction > 0 and sub_turn >= next_periodic_compaction:
                try:
                    self._do_periodic_compaction(stage_tools, sub_turn)
                except RuntimeError as exc:
                    msg = str(exc or "")
                    if "LLM_REQUEST_LIMIT_REACHED" in msg:
                        end_reason = "llm_request_limit_reached"
                        self._log_simple_event("llm_request_limit", msg, sub_turn)
                        break
                    if "CONTEXT_CLEAR_LIMIT_REACHED" in msg:
                        end_reason = "context_clear_limit_reached"
                        self._log_simple_event("context_clear_limit", msg, sub_turn)
                        break
                    raise
                next_periodic_compaction += int(self.max_stage_turns)
            if end_reason in {"llm_request_limit_reached", "context_clear_limit_reached"}:
                break

            # --- Win guard ---
            if self._runtime_is_win():
                if self._win_epilogue_done:
                    end_reason = "win_after_epilogue"
                    self._log_simple_event("win_guard", "epilogue complete; ending stage", sub_turn)
                    break
                # First detection: inject reminder and allow one more turn
                self._win_epilogue_done = True
                used = self._runtime_action_count()
                win_epilogue_tpl = str(self._reminders.get("win_epilogue") or "").strip()
                if not win_epilogue_tpl:
                    win_epilogue_tpl = (
                        "<game-complete>{TOTAL_ACTIONS_USED} of {MAX_ACTIONS} actions used. "
                        "The game is WON at level {CURRENT_LEVEL}. "
                        "Update MEMORY.md with your final findings: confirmed rules, "
                        "successful strategies, level solutions, and transferable insights. "
                        "Then respond with exactly `I win` and do not call any more tools. "
                        "After that response, the run will end.</game-complete>"
                    )
                win_reminder = render_template(
                    win_epilogue_tpl,
                    {
                        "TOTAL_ACTIONS_USED": str(used),
                        "MAX_ACTIONS": str(self._max_actions),
                        "CURRENT_LEVEL": str(self.current_level),
                    },
                )
                self._messages.append({"role": "user", "parts": [{"text": win_reminder}]})
                self._log_simple_event("win_epilogue", "game won — injected epilogue reminder for memory update", sub_turn)

            # --- Context limit check before LLM call ---
            try:
                self._messages = self._maybe_compact(self._messages, stage_tools)
            except RuntimeError as exc:
                msg = str(exc or "")
                if "LLM_REQUEST_LIMIT_REACHED" in msg:
                    end_reason = "llm_request_limit_reached"
                    self._log_simple_event("llm_request_limit", msg, sub_turn)
                    break
                if "CONTEXT_CLEAR_LIMIT_REACHED" in msg:
                    end_reason = "context_clear_limit_reached"
                    self._log_simple_event("context_clear_limit", msg, sub_turn)
                    break
                raise
            if self._memory_checkpoint_deadline_missed:
                self._apply_missed_memory_checkpoint_fallback(sub_turn=sub_turn)
                sub_turn += 1
                continue

            # --- Max-steps reminder on the last allowed turn ---
            is_last_turn = (sub_turn + 1) >= self._stage_hard_turn_limit
            if is_last_turn and self._max_steps_reminder:
                max_steps_msg = {"role": "user", "parts": [{"text": self._max_steps_reminder}]}
                self._messages.append(max_steps_msg)
                self._log_simple_event(
                    "max_steps_reminder",
                    f"injected max_steps reminder at sub_turn={sub_turn} (limit={self._stage_hard_turn_limit})",
                    sub_turn,
                )

            # --- LLM call ---
            try:
                response = self._complete_with_max_tokens_retry(
                    self._messages, stage_tools, sub_turn,
                )
            except RuntimeError as exc:
                msg = str(exc or "")
                if "WIN_GUARD_BEFORE_LLM_REQUEST" in msg:
                    end_reason = "win_before_llm_request"
                    self._log_simple_event("win_guard", "runtime state is WIN; skip LLM request", sub_turn)
                    break
                if "LLM_REQUEST_LIMIT_REACHED" in msg:
                    end_reason = "llm_request_limit_reached"
                    self._log_simple_event("llm_request_limit", msg, sub_turn)
                    break
                if "CONTEXT_CLEAR_LIMIT_REACHED" in msg:
                    end_reason = "context_clear_limit_reached"
                    self._log_simple_event("context_clear_limit", msg, sub_turn)
                    break
                raise

            # --- Track token usage ---
            try:
                usage = response.usage or {}
                self._last_prompt_tokens_observed = int(usage.get("prompt_tokens", 0) or 0)
                self._last_total_tokens_observed = int(usage.get("total_tokens", 0) or 0)
            except Exception:
                self._last_prompt_tokens_observed = 0
                self._last_total_tokens_observed = 0

            self._total_prompt_tokens_used += int(self._last_prompt_tokens_observed)
            completion_tokens = (
                int(self._last_total_tokens_observed - self._last_prompt_tokens_observed)
                if self._last_total_tokens_observed > self._last_prompt_tokens_observed
                else 0
            )
            self._total_completion_tokens_used += completion_tokens

            final_text = response.text or self._extract_text_from_raw(response.raw)
            mid = f"{self._log_source}_{self.step_getter()}_{self.turn_getter()}_{sub_turn}"
            self._last_stage_message_id = mid

            # --- Build assistant message (exclude thought parts from context) ---
            assistant_parts_raw, assistant_parts_ctx = self._build_assistant_parts(response, final_text)
            assistant_message: Dict[str, Any] = {"role": "assistant", "parts": assistant_parts_ctx}
            self._messages.append(assistant_message)
            self.log_event(
                "message",
                {
                    "source": self._log_source,
                    "state_id": self._state_id(),
                    "loop_id": self.current_loop_id,
                    "message_id": mid,
                    "message": {"role": "assistant", "parts": assistant_parts_raw},
                    "raw_response": response.raw if isinstance(response.raw, dict) else {},
                    "raw_request": (
                        response.raw_request
                        if isinstance(getattr(response, "raw_request", None), dict)
                        else {}
                    ),
                    "request_generation_config": (
                        response.request_generation_config
                        if isinstance(getattr(response, "request_generation_config", None), dict)
                        else {}
                    ),
                },
            )

            # --- No tool calls → extract action plan, done ---
            if not response.tool_calls:
                if self._memory_checkpoint_pending:
                    self._log_simple_event(
                        "memory_checkpoint_wait",
                        "assistant replied without writing MEMORY.md; awaiting checkpoint write or fallback clear",
                        sub_turn,
                    )
                    sub_turn += 1
                    continue
                if not str(final_text or "").strip():
                    consecutive_empty_no_tool_calls += 1
                    if (
                        self._empty_response_recovery_enabled
                        and consecutive_empty_no_tool_calls < 2
                    ):
                        reminder = (self._empty_response_recovery_reminder or "").strip()
                        extra_instruction = (
                            "Continue from the current state. If the game is not won, "
                            "your next reply must call a tool or provide a valid action plan."
                        )
                        if reminder.endswith("</system-reminder>"):
                            reminder = reminder.replace(
                                "</system-reminder>",
                                f"{extra_instruction}\n</system-reminder>",
                            )
                        elif reminder:
                            reminder = f"{reminder}\n{extra_instruction}"
                        else:
                            reminder = (
                                "<system-reminder>\n"
                                "The previous response was empty. Continue from the current state. "
                                "If the game is not won, your next reply must call a tool or "
                                "provide a valid action plan.\n"
                                "</system-reminder>"
                            )
                        rem_message = {
                            "role": "user",
                            "parts": build_text_parts_with_inline_media(reminder),
                        }
                        self._messages.append(rem_message)
                        self.log_event(
                            "message",
                            {
                                "source": self._log_source,
                                "state_id": self._state_id(),
                                "loop_id": self.current_loop_id,
                                "message_id": f"{mid}_empty_response_reminder",
                                "message": rem_message,
                                "message_meta": {
                                    "special_types": ["system_reminder"],
                                    "reminder_kind": "empty_response_recovery",
                                    "consecutive_empty_no_tool_calls": consecutive_empty_no_tool_calls,
                                },
                            },
                        )
                        continue
                else:
                    consecutive_empty_no_tool_calls = 0
                if self.action_plan_submitter is not None and final_text:
                    payload = self._extract_actions_payload(final_text)
                    if payload is not None:
                        plan = self._normalize_action_plan(payload)
                        if plan:
                            submit_meta = self.current_action_context()
                            submit_result = self.action_plan_submitter(plan, submit_meta)
                            errors = submit_result.get("errors")
                            errors_text = ""
                            if isinstance(errors, list) and errors:
                                rendered: List[str] = []
                                for item in errors[:3]:
                                    if not isinstance(item, dict):
                                        continue
                                    rendered.append(
                                        json.dumps(item, ensure_ascii=False, sort_keys=True)
                                    )
                                if rendered:
                                    errors_text = "\nerrors=" + "\n".join(rendered)
                            self.log_event(
                                "message",
                                {
                                    "source": self._log_source,
                                    "state_id": self._state_id(),
                                    "loop_id": self.current_loop_id,
                                    "message_id": f"{mid}_action_plan",
                                    "message": {
                                        "role": "assistant",
                                        "parts": [
                                            {
                                                "text": (
                                                    f"[action_plan_enqueued] queued={int(submit_result.get('queued', 0))} "
                                                    f"skipped={int(submit_result.get('skipped', 0))}"
                                                    f"{errors_text}"
                                                )
                                            }
                                        ],
                                    },
                                    "message_meta": {
                                        "event": "action_plan_enqueued",
                                        "queued": int(submit_result.get("queued", 0)),
                                        "skipped": int(submit_result.get("skipped", 0)),
                                    },
                                },
                            )
                            if errors_text and int(submit_result.get("queued", 0)) <= 0:
                                rem_message = {
                                    "role": "user",
                                    "parts": [
                                        {
                                            "text": (
                                                "<system-reminder>\n"
                                                "The runtime rejected your submitted action plan.\n"
                                                f"{errors_text.lstrip()}\n"
                                                "Do not assume any action executed. Inspect available_actions and submit a corrected plan.\n"
                                                "</system-reminder>"
                                            )
                                        }
                                    ],
                                }
                                self._messages.append(rem_message)
                                self.log_event(
                                    "message",
                                    {
                                        "source": self._log_source,
                                        "state_id": self._state_id(),
                                        "loop_id": self.current_loop_id,
                                        "message_id": f"{mid}_action_plan_error_reminder",
                                        "message": rem_message,
                                        "message_meta": {
                                            "special_types": ["system_reminder"],
                                            "reminder_kind": "action_plan_validation_error",
                                        },
                                    },
                                )
                                continue

                # --- Truncation recovery: stop_reason is MAX_TOKENS with text but no tool calls ---
                stop_reason_upper = str(getattr(response, "stop_reason", "") or "").upper()
                is_truncated = stop_reason_upper in ("MAX_TOKENS", "LENGTH")
                if (
                    is_truncated
                    and self._truncation_recovery_enabled
                    and consecutive_truncation_no_tool_calls < self._truncation_recovery_max_retries
                    and str(final_text or "").strip()
                    and not self._win_epilogue_done
                ):
                    consecutive_truncation_no_tool_calls += 1
                    reminder = (self._truncation_recovery_reminder or "").strip()
                    if not reminder:
                        reminder = (
                            "<system-reminder>\n"
                            "Your previous response was truncated due to length. Do NOT repeat or "
                            "re-analyze what you already wrote. Resume from where you left off and "
                            "immediately call a tool (bash_exec, write_file, or screen_shot). "
                            "Pick the single most important next action and execute it now.\n"
                            "</system-reminder>"
                        )
                    rem_message = {
                        "role": "user",
                        "parts": build_text_parts_with_inline_media(reminder),
                    }
                    self._messages.append(rem_message)
                    self.log_event(
                        "message",
                        {
                            "source": self._log_source,
                            "state_id": self._state_id(),
                            "loop_id": self.current_loop_id,
                            "message_id": f"{mid}_truncation_recovery_reminder",
                            "message": rem_message,
                            "message_meta": {
                                "special_types": ["system_reminder"],
                                "reminder_kind": "truncation_recovery",
                                "consecutive_truncation_no_tool_calls": consecutive_truncation_no_tool_calls,
                                "stop_reason": stop_reason_upper,
                            },
                        },
                    )
                    sub_turn += 1
                    continue

                end_reason = "assistant_no_tool_calls"
                break

            # --- Tool dispatch loop ---
            tool_results: List[Dict[str, Any]] = []
            todo_written = False
            stop_after_current_tool = False
            clear_after_memory_checkpoint = False
            doom_loop_detected = False
            self._deferred_context_action = None

            for i, tc in enumerate(response.tool_calls):
                name = str(tc.get("name", ""))
                args = tc.get("args", {})

                # Doom loop detection
                try:
                    sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=True, separators=(',', ':'))}"
                except Exception:
                    sig = f"{name}:{str(args)}"
                self._recent_tool_signatures.append(sig)
                if len(self._recent_tool_signatures) > self._doom_loop_threshold * 2:
                    self._recent_tool_signatures = self._recent_tool_signatures[-(self._doom_loop_threshold * 2):]
                if (
                    len(self._recent_tool_signatures) >= self._doom_loop_threshold
                    and len(set(self._recent_tool_signatures[-self._doom_loop_threshold:])) == 1
                ):
                    doom_loop_detected = True
                    tool_results.append({"name": name, "result": f"Error: doom loop detected — tool '{name}' called {self._doom_loop_threshold} times consecutively with identical arguments. Try a different approach."})
                    self._log_simple_event("doom_loop", f"tool={name} repeated={self._doom_loop_threshold}", sub_turn)
                    for rest in response.tool_calls[i + 1:]:
                        tool_results.append({"name": str((rest or {}).get("name", "")), "result": "Error: tool skipped due to doom loop"})
                    break

                # Permission check
                perm = self._permission_policy.check(
                    name, args if isinstance(args, dict) else {},
                    stage=self.current_stage,
                    turn_tool_count=self._turn_tool_call_count,
                )
                if not perm.allowed:
                    tool_results.append({"name": name, "result": f"Permission denied: {perm.reason}"})
                    continue
                self._turn_tool_call_count += 1

                # Execute tool
                try:
                    result = self.tool_dispatch(name, args)
                except Exception as tool_exc:
                    result = f"Error: tool '{name}' raised {type(tool_exc).__name__}: {tool_exc}"
                tool_results.append({"name": name, "result": result})

                if self._is_memory_checkpoint_write_success(name, result):
                    clear_after_memory_checkpoint = True
                    self._deferred_context_action = ("clear", "memory_checkpoint")
                    self._reset_memory_checkpoint_state()
                    self._log_simple_event(
                        "memory_checkpoint_write",
                        f"memory checkpoint written to {self._memory_checkpoint_cfg.memory_path}; clearing context",
                        sub_turn,
                    )
                    for rest in response.tool_calls[i + 1:]:
                        tool_results.append(
                            {
                                "name": str((rest or {}).get("name", "")),
                                "result": "Error: tool skipped because context reset was triggered after MEMORY.md update",
                            }
                        )
                    break

                # Sync todo state
                if name == "todo_write":
                    todo_written = True
                    self._sync_todo_cache()

                # ============================================================
                # INLINE EVENT: level change (fixes the old stage-boundary bug)
                # ============================================================
                if int(self.current_level) > int(self._last_level):
                    self._handle_level_up(stage_tools, sub_turn)

                # ============================================================
                # INLINE EVENT: action submitted
                # ============================================================
                current_actions = self._runtime_action_count()
                if current_actions > self._last_action_count:
                    self._handle_action_submitted(stage_tools, sub_turn)

                # Stop check
                if self._stop_requested():
                    stop_after_current_tool = True
                    end_reason = "worker_stop_after_tool_call"
                    for rest in response.tool_calls[i + 1:]:
                        tool_results.append({"name": str((rest or {}).get("name", "")), "result": "Error: tool skipped because worker stop was requested"})
                    break

            # --- Append tool results + reminders ---
            todo_trigger = todo_enabled and (
                todo_written or ((sub_turn + 1) % self._todo_reminder_interval == 0)
            )
            current_runtime_step = int(self.runtime_step_getter())
            game_trigger = current_runtime_step > last_runtime_step
            if game_trigger:
                last_runtime_step = current_runtime_step
            reminder = ""
            rem_types: List[str] = []
            if todo_trigger or game_trigger:
                reminder = self._build_merged_reminder(
                    include_todo=todo_trigger,
                    game_mode=("update" if game_trigger else None),
                )
                if reminder:
                    if game_trigger:
                        rem_types.append("game_reminder")
                    if todo_trigger:
                        rem_types.append("system_reminder")

            # --- Workspace budget reminder (after file-modifying tools) ---
            if (
                self._workspace_budget_reminder_enabled
                and self._workspace_budget_getter is not None
                and any(
                    str(tc.get("name", "")) in self._file_modifying_tools
                    for tc in response.tool_calls
                )
            ):
                try:
                    budget_line = self._workspace_budget_getter()
                except Exception:
                    budget_line = ""
                if budget_line:
                    budget_text = f"<workspace-budget>{budget_line}</workspace-budget>"
                    if reminder:
                        reminder = f"{reminder}\n{budget_text}"
                    else:
                        reminder = budget_text
                    if "workspace_budget" not in rem_types:
                        rem_types.append("workspace_budget")

            # --- Context token budget reminder ---
            if self._context_budget_reminder_enabled and self._compaction_cfg.enabled:
                ctx_line = self._context_budget_status()
                if ctx_line:
                    ctx_text = f"<context-budget>{ctx_line}</context-budget>"
                    if reminder:
                        reminder = f"{reminder}\n{ctx_text}"
                    else:
                        reminder = ctx_text
                    if "context_budget" not in rem_types:
                        rem_types.append("context_budget")

            # --- Game status reminder (after actions are submitted) ---
            if game_trigger and self._max_actions > 0:
                used = self._runtime_action_count()
                on_level = used - self._action_count_at_level_up
                resets = self._resets_on_current_level
                parts = [f"{used} of {self._max_actions} actions used."]
                if self._pending_level_up_reminder:
                    # Level-up event: no extra hint needed
                    parts.append(self._pending_level_up_reminder)
                    self._pending_level_up_reminder = None
                else:
                    # Normal: show current level progress
                    is_game_over = self._last_observed_state == "game_over"
                    if is_game_over:
                        parts.append(f"Level {self.current_level}, now game over (reset to continue).")
                    else:
                        parts.append(f"Level {self.current_level}.")
                    # resets counts completed recoveries; if currently game_over, +1 for this one
                    game_overs = resets + (1 if is_game_over else 0)
                    vol_resets = self._voluntary_resets_on_current_level
                    level_detail = f"{on_level} actions"
                    extras = []
                    if game_overs > 0:
                        extras.append(f"{game_overs} game over{'s' if game_overs != 1 else ''}")
                    if vol_resets > 0:
                        extras.append(f"{vol_resets} reset{'s' if vol_resets != 1 else ''}")
                    if extras:
                        level_detail += ", " + ", ".join(extras) + ","
                    parts.append(f"{level_detail} on this level.")
                    # Situational hint
                    if is_game_over and game_overs >= 3:
                        parts.append("Repeated game overs — consider a different approach before resetting.")
                    elif is_game_over:
                        parts.append("Review what caused the game over before resetting.")
                    else:
                        parts.append("Fewer actions = better score.")
                status_text = f"<game-status>{' '.join(parts)}</game-status>"
                if reminder:
                    reminder = f"{reminder}\n{status_text}"
                else:
                    reminder = status_text
                if "game_status" not in rem_types:
                    rem_types.append("game_status")

            tool_mid = f"{self._log_source}_tool_{self.step_getter()}_{self.turn_getter()}_{sub_turn}"
            tool_parts = build_tool_result_parts(tool_results)

            # --- Append reminder inline to tool result (opencode style) ---
            if reminder:
                reminder_wrapped = f"<system-reminder>\n{reminder}\n</system-reminder>"
                tool_parts.append({"text": reminder_wrapped})

            tool_message = {"role": "user", "parts": tool_parts}
            self._messages.append(tool_message)
            self.log_event(
                "message",
                {
                    "source": self._log_source,
                    "state_id": self._state_id(),
                    "loop_id": self.current_loop_id,
                    "message_id": tool_mid,
                    "message": tool_message,
                    "message_meta": {"special_types": rem_types} if rem_types else {},
                },
            )

            # --- Execute deferred compact/clear (after tool_results are paired) ---
            if self._deferred_context_action is not None:
                ctx_action, ctx_reason = self._deferred_context_action
                self._deferred_context_action = None
                try:
                    if ctx_action == "clear":
                        if ctx_reason == "memory_checkpoint":
                            self._apply_context_clear_to(
                                self._messages,
                                self._build_memory_checkpoint_resume_messages(
                                    "the latest durable findings were written to MEMORY.md"
                                ),
                                trigger="memory_checkpoint",
                                sub_turn=sub_turn,
                            )
                            self.log_event("message", {
                                "source": self._log_source,
                                "state_id": self._state_id(),
                                "loop_id": self.current_loop_id,
                                "message_id": f"{self._log_source}_context_clear_memory_checkpoint_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
                                "message": {
                                    "role": "assistant",
                                    "parts": [{"text": "[context_clear] trigger=memory_checkpoint"}],
                                },
                                "message_meta": {"event": "context_clear", "trigger": "memory_checkpoint"},
                            })
                        else:
                            self._apply_context_clear_to(
                                self._messages,
                                self._build_fresh_messages(f"context cleared after {ctx_reason}"),
                                trigger=str(ctx_reason or "deferred_clear"),
                                sub_turn=sub_turn,
                            )
                    elif ctx_action == "compact":
                        self._do_compact(self._messages, stage_tools, ctx_reason, sub_turn)
                        self._reset_context_token_observation()
                except RuntimeError as exc:
                    msg = str(exc or "")
                    if "LLM_REQUEST_LIMIT_REACHED" in msg:
                        end_reason = "llm_request_limit_reached"
                        self._log_simple_event("llm_request_limit", msg, sub_turn)
                        break
                    if "CONTEXT_CLEAR_LIMIT_REACHED" in msg:
                        end_reason = "context_clear_limit_reached"
                        self._log_simple_event("context_clear_limit", msg, sub_turn)
                        break
                    raise

            if doom_loop_detected:
                end_reason = "doom_loop_detected"
                break
            if clear_after_memory_checkpoint:
                sub_turn += 1
                continue
            if stop_after_current_tool or self._stop_requested():
                end_reason = "worker_stop_after_tool_result"
                break
            consecutive_truncation_no_tool_calls = 0  # reset on successful tool call
            sub_turn += 1

        # --- Log loop end ---
        self.log_event(
            "message",
            {
                "source": self._log_source,
                "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_end_{self.step_getter()}_{self.turn_getter()}",
                "message": {
                    "role": "assistant",
                    "parts": [{"text": f"[loop_end] reason={end_reason} turns_used={sub_turn}/{self.max_stage_turns}"}],
                },
                "message_meta": {
                    "event": "loop_end",
                    "reason": end_reason,
                    "turns_used": sub_turn,
                    "max_stage_turns": int(self.max_stage_turns),
                },
            },
        )
        self._last_loop_end_reason = end_reason

    def last_loop_end_reason(self) -> str:
        return str(self._last_loop_end_reason or "")

    # ------------------------------------------------------------------
    # Message initialization
    # ------------------------------------------------------------------

    def _init_messages(self) -> None:
        """Build fresh system + user messages."""
        self._reset_memory_checkpoint_state()
        self._todo_cache = []
        self._sync_todo_cache_clear()
        system_values = self._base_prompt_values()
        self._system_content = self._prompt_renderer.render(self._system_template, system_values)
        user_prompt = self._render_user_prompt()
        initial_parts: List[str] = []
        game_text = self._build_game_reminder(game_mode="initial")
        if game_text:
            initial_parts.append(game_text)
        user_content = user_prompt.rstrip()
        if initial_parts:
            user_content = f"{user_content}\n\n" + "\n".join(initial_parts)
        user_parts = build_text_parts_with_inline_media(user_content)
        self._messages = [
            {"role": "system", "parts": [{"text": self._system_content}]},
            {"role": "user", "parts": user_parts},
        ]
        self._last_level = int(self.current_level)
        self._last_action_count = self._runtime_action_count()
        # Log
        base_mid = f"{self._log_source}_base_{self.step_getter()}_{self.turn_getter()}"
        self.log_event("message", {
            "source": self._log_source, "state_id": self._state_id(),
            "loop_id": self.current_loop_id, "message_id": f"{base_mid}_system",
            "message": {"role": "system", "parts": [{"text": self._system_content}]},
        })
        self.log_event("message", {
            "source": self._log_source, "state_id": self._state_id(),
            "loop_id": self.current_loop_id, "message_id": f"{base_mid}_user",
            "message": {"role": "user", "parts": user_parts},
        })

    def _build_fresh_messages(self, clear_reason: str = "") -> List[Dict[str, Any]]:
        """Build cleared messages — same structure as compaction output."""
        self._reset_memory_checkpoint_state()
        system_values = self._base_prompt_values()
        self._system_content = self._prompt_renderer.render(self._system_template, system_values)
        user_prompt = self._render_user_prompt()
        reason = clear_reason or "the context was reset"
        summary = f"Context was cleared: {reason}. No prior summary available."
        self._last_compaction_summary = ""
        context_cleared_tpl = str(self._reminders.get("context_cleared") or "").strip()
        if context_cleared_tpl:
            cleared_prompt = render_template(
                context_cleared_tpl,
                self._reminder_template_values(
                    user_prompt=user_prompt,
                    trigger="context_clear",
                    clear_reason=reason,
                ),
            ).strip()
        else:
            cleared_prompt = user_prompt
        return [
            {"role": "system", "parts": [{"text": self._system_content}]},
            {"role": "user", "parts": build_text_parts_with_inline_media(cleared_prompt)},
            {"role": "assistant", "parts": [{"text": f"[Compaction — {reason}]\n\n{summary}"}]},
            {"role": "user", "parts": [{"text": "Continue."}]},
        ]

    # ------------------------------------------------------------------
    # Inline event handlers (called inside tool dispatch loop)
    # ------------------------------------------------------------------

    def _handle_level_up(self, stage_tools: List[Dict[str, Any]], sub_turn: int) -> None:
        action = self._resume_policy.on_level_up
        prev_level = int(self._last_level)
        new_level = int(self.current_level)
        self._log_simple_event(
            "level_up",
            f"level {prev_level} -> {new_level} action={action}",
            sub_turn,
        )
        # Defer compact/clear until after tool_results are appended, so the
        # functionCall/functionResponse pairing stays intact.
        if action in ("clear", "compact"):
            self._deferred_context_action = (action, "level_up")
        # In keep mode, set a reminder so the agent knows about the level change
        level_up_tpl = str(self._reminders.get("level_up") or "").strip() or (
            "Level up: {PREV_LEVEL} -> {NEW_LEVEL}. "
            "Save level {PREV_LEVEL} discoveries to MEMORY.md."
        )
        self._pending_level_up_reminder = render_template(
            level_up_tpl,
            {
                "PREV_LEVEL": str(prev_level),
                "NEW_LEVEL": str(new_level),
                "CURRENT_LEVEL": str(new_level),
            },
        )
        self._last_level = new_level
        self._action_count_at_level_up = self._runtime_action_count()
        self._resets_on_current_level = 0
        self._voluntary_resets_on_current_level = 0

    def _handle_action_submitted(self, stage_tools: List[Dict[str, Any]], sub_turn: int) -> None:
        action = self._resume_policy.on_action_submitted
        if action == "keep":
            self._last_action_count = self._runtime_action_count()
            return
        self._log_simple_event(
            "action_submitted",
            f"action_count {self._last_action_count} -> {self._runtime_action_count()} action={action}",
            sub_turn,
        )
        if action in ("clear", "compact"):
            self._deferred_context_action = (action, "action_submitted")
        self._last_action_count = self._runtime_action_count()

    # ------------------------------------------------------------------
    # Compaction helpers
    # ------------------------------------------------------------------

    def _do_compact(self, messages: List[Dict[str, Any]], stage_tools: List[Dict[str, Any]], reason: str, sub_turn: int) -> None:
        """In-place compaction of self._messages."""
        try:
            compacted, info = compact_messages(
                llm=self.llm,
                messages=messages,
                cfg=self._compaction_cfg,
                max_output_tokens=self.max_output_tokens,
                tools=stage_tools,
                tool_dispatch=self.tool_dispatch,
                round_logger=self._build_compaction_round_logger(sub_turn=sub_turn, event_kind=f"event_{reason}"),
                request_hook=lambda _kind: self._consume_llm_request_budget(
                    sub_turn=sub_turn,
                    request_kind=f"compaction:{reason}",
                ),
            )
            self._messages[:] = compacted
            self._last_compaction_summary = self._extract_compaction_summary(compacted)
            self.log_event("message", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_compact_{reason}_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
                "message": {
                    "role": "assistant",
                    "parts": [{"text": f"[compaction] reason={reason} messages_before={int(info.get('messages_before', 0))} messages_after={int(info.get('messages_after', 0))}"}],
                },
                "message_meta": {"event": "compaction", "reason": reason},
            })
        except Exception as exc:
            if isinstance(exc, RuntimeError) and (
                "LLM_REQUEST_LIMIT_REACHED" in str(exc or "")
                or "CONTEXT_CLEAR_LIMIT_REACHED" in str(exc or "")
            ):
                raise
            self.log_event("message", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_compact_error_{reason}_{sub_turn}",
                "message": {"role": "assistant", "parts": [{"text": f"[compaction_error] reason={reason} {exc}"}]},
            })
            # Fallback: hard context clear to prevent unbounded growth.
            self._apply_context_clear_to(
                self._messages,
                self._build_fresh_messages(f"compaction failed (reason={reason}), context was force-cleared"),
                trigger="compaction_fallback_clear",
                sub_turn=sub_turn,
            )
            self.log_event("message", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_compact_fallback_clear_{reason}_{sub_turn}",
                "message": {"role": "assistant", "parts": [{"text": f"[compaction_fallback_clear] reason={reason} error={exc}"}]},
                "message_meta": {"event": "compaction_fallback_clear", "reason": reason},
            })

    def _do_periodic_compaction(self, stage_tools: List[Dict[str, Any]], sub_turn: int) -> None:
        self._do_compact(self._messages, stage_tools, "turn_limit", sub_turn)

    def _maybe_compact(self, messages: List[Dict[str, Any]], stage_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._compaction_cfg.enabled:
            return messages
        threshold = int(max(1, self._compaction_cfg.max_context_tokens) * max(0.0, min(1.0, self._compaction_cfg.trigger_ratio)))
        observed_prompt_tokens = int(self._last_prompt_tokens_observed)
        estimated_request_tokens = int(estimate_message_tokens(messages))
        trigger_tokens = observed_prompt_tokens if observed_prompt_tokens > 0 else estimated_request_tokens
        if trigger_tokens < threshold:
            return messages
        if self._memory_checkpoint_cfg.enabled:
            if self._memory_checkpoint_pending:
                self._memory_checkpoint_turns_waited += 1
                if self._memory_checkpoint_turns_waited > self._memory_checkpoint_cfg.max_grace_turns:
                    self._memory_checkpoint_deadline_missed = True
                    self.log_event("message", {
                        "source": self._log_source, "state_id": self._state_id(),
                        "loop_id": self.current_loop_id,
                        "message_id": f"{self._log_source}_memory_checkpoint_deadline_missed_{self.step_getter()}_{self.turn_getter()}_{len(messages)}",
                        "message": {
                            "role": "assistant",
                            "parts": [{"text": f"[memory_checkpoint_deadline_missed] trigger_tokens={trigger_tokens} threshold={threshold}"}],
                        },
                        "message_meta": {"event": "memory_checkpoint_deadline_missed", "trigger_tokens": int(trigger_tokens), "threshold": int(threshold)},
                    })
                    return messages
            else:
                self._memory_checkpoint_pending = True
                self._memory_checkpoint_turns_waited = 0
                self._memory_checkpoint_reason = "context limit reached"
                self.log_event("message", {
                    "source": self._log_source, "state_id": self._state_id(),
                    "loop_id": self.current_loop_id,
                    "message_id": f"{self._log_source}_memory_checkpoint_{self.step_getter()}_{self.turn_getter()}_{len(messages)}",
                    "message": {
                        "role": "assistant",
                        "parts": [{"text": f"[memory_checkpoint] trigger_tokens={trigger_tokens} threshold={threshold}"}],
                    },
                    "message_meta": {"event": "memory_checkpoint", "trigger_tokens": int(trigger_tokens), "threshold": int(threshold)},
                })
            reminder_text = self._memory_checkpoint_reminder_text(trigger_tokens=trigger_tokens, threshold=threshold)
            if self._memory_checkpoint_turns_waited == self._memory_checkpoint_cfg.max_grace_turns:
                reminder_text = self._memory_checkpoint_final_reminder_text(trigger_tokens=trigger_tokens, threshold=threshold)
            reminded = self._append_memory_checkpoint_reminder(messages, reminder_text)
            return reminded
        on_limit = self._resume_policy.on_context_limit
        if on_limit == "clear":
            cleared = self._build_fresh_messages("the preserved context exceeded its limit")
            self._consume_context_clear_budget(trigger="context_limit", sub_turn=-1)
            self.log_event("message", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_context_clear_{self.step_getter()}_{self.turn_getter()}_{len(messages)}",
                "message": {"role": "assistant", "parts": [{"text": f"[context_clear] trigger_tokens={trigger_tokens} threshold={threshold}"}]},
                "message_meta": {"event": "context_clear", "trigger": "context_limit"},
            })
            return cleared
        try:
            compacted, info = compact_messages(
                llm=self.llm, messages=messages, cfg=self._compaction_cfg,
                max_output_tokens=self.max_output_tokens, tools=stage_tools,
                tool_dispatch=self.tool_dispatch,
                round_logger=self._build_compaction_round_logger(sub_turn=-1, event_kind="threshold_compaction"),
                request_hook=lambda _kind: self._consume_llm_request_budget(
                    sub_turn=-1,
                    request_kind="compaction:threshold",
                ),
            )
            self._last_compaction_summary = self._extract_compaction_summary(compacted)
            self.log_event("message", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_compaction_{self.step_getter()}_{self.turn_getter()}_{len(messages)}",
                "message": {
                    "role": "assistant",
                    "parts": [{"text": f"[compaction] trigger_tokens={trigger_tokens} threshold={threshold} messages_before={int(info.get('messages_before', len(messages)))} messages_after={int(info.get('messages_after', len(compacted)))}"}],
                },
                "message_meta": {"event": "compaction", "trigger_tokens": int(trigger_tokens), "threshold": int(threshold)},
            })
            return compacted
        except Exception as exc:
            if isinstance(exc, RuntimeError) and (
                "LLM_REQUEST_LIMIT_REACHED" in str(exc or "")
                or "CONTEXT_CLEAR_LIMIT_REACHED" in str(exc or "")
            ):
                raise
            self.log_event("message", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_compaction_error_{self.step_getter()}_{self.turn_getter()}_{len(messages)}",
                "message": {"role": "assistant", "parts": [{"text": f"[compaction_error] {exc}"}]},
            })
            # Fallback: if compaction fails, do a hard context clear to prevent
            # unbounded growth.  Without this, every subsequent turn re-triggers
            # compaction, which fails again, creating an infinite error loop.
            cleared = self._build_fresh_messages("compaction failed, context was force-cleared")
            self._consume_context_clear_budget(trigger="compaction_fallback_clear", sub_turn=-1)
            self.log_event("message", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_compaction_fallback_clear_{self.step_getter()}_{self.turn_getter()}_{len(messages)}",
                "message": {"role": "assistant", "parts": [{"text": f"[compaction_fallback_clear] trigger_tokens={trigger_tokens} threshold={threshold} error={exc}"}]},
                "message_meta": {"event": "compaction_fallback_clear"},
            })
            return cleared

    # ------------------------------------------------------------------
    # LLM call with error recovery
    # ------------------------------------------------------------------

    def _complete_with_max_tokens_retry(
        self,
        messages: List[Dict[str, Any]],
        stage_tools: List[Dict[str, Any]],
        sub_turn: int,
    ) -> Any:
        attempt = 0
        empty_recovery_used = False
        input_overflow_recovery_used = False
        invalid_argument_recovery_used = False
        while True:
            if self._runtime_is_win() and not self._win_epilogue_done:
                raise RuntimeError("WIN_GUARD_BEFORE_LLM_REQUEST")
            request_data = build_request_data(
                messages=messages,
                tools=stage_tools,
                temperature=self._llm_temperature,
                max_tokens=self.max_output_tokens,
                top_p=self._llm_top_p,
            )
            try:
                self._consume_llm_request_budget(sub_turn=sub_turn, request_kind="stage")
                response = self.llm.complete(
                    messages=messages,
                    tools=stage_tools,
                    max_tokens=self.max_output_tokens,
                    temperature=self._llm_temperature,
                    request_data=request_data,
                )
            except RuntimeError as exc:
                msg = str(exc or "")
                is_empty = (
                    "Empty response: no candidates" in msg
                    or "Incomplete Claude response: empty content" in msg
                    or "Incomplete Claude stream response: empty content" in msg
                    or "Claude non-stream fallback failed after stream empty content" in msg
                    or "Incomplete Claude response: thinking-only content" in msg
                    or "Incomplete Claude stream response: thinking-only content" in msg
                    or "Incomplete Claude response: no text/tool_use" in msg
                    or "Incomplete Claude stream response: no text/tool_use" in msg
                )
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
                # --- Input overflow recovery ---
                if is_input_overflow and self._compaction_cfg.enabled and not input_overflow_recovery_used:
                    input_overflow_recovery_used = True
                    if self._resume_policy.on_context_limit == "clear":
                        self._apply_context_clear_to(
                            messages,
                            self._build_fresh_messages("the preserved context exceeded its token limit"),
                            trigger="input_tokens_exceeded_clear",
                            sub_turn=sub_turn,
                        )
                        continue
                    prompt_tokens_before = int(self._last_prompt_tokens_observed)
                    try:
                        compacted, info = compact_messages(
                            llm=self.llm, messages=messages, cfg=self._compaction_cfg,
                            max_output_tokens=self.max_output_tokens, tools=stage_tools,
                            tool_dispatch=self.tool_dispatch, overflow_mode=True,
                            round_logger=self._build_compaction_round_logger(sub_turn=sub_turn, event_kind="forced_compaction_input_tokens"),
                            request_hook=lambda _kind: self._consume_llm_request_budget(
                                sub_turn=sub_turn,
                                request_kind="compaction:forced_input_tokens",
                            ),
                        )
                        messages[:] = compacted
                        self._last_compaction_summary = self._extract_compaction_summary(compacted)
                        self.log_event("message", {
                            "source": self._log_source, "state_id": self._state_id(),
                            "loop_id": self.current_loop_id,
                            "message_id": f"{self._log_source}_forced_compaction_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
                            "message": {"role": "assistant", "parts": [{"text": f"[forced_compaction] reason=input_tokens_exceeded prompt_tokens_before={prompt_tokens_before} messages_before={int(info.get('messages_before', 0))} messages_after={int(info.get('messages_after', 0))}"}]},
                            "message_meta": {"event": "forced_compaction", "reason": "input_tokens_exceeded"},
                        })
                    except Exception as compact_exc:
                        if isinstance(compact_exc, RuntimeError) and (
                            "LLM_REQUEST_LIMIT_REACHED" in str(compact_exc or "")
                            or "CONTEXT_CLEAR_LIMIT_REACHED" in str(compact_exc or "")
                        ):
                            raise
                        self._apply_context_clear_to(
                            messages,
                            self._build_fresh_messages("compaction failed during overflow recovery, context was force-cleared"),
                            trigger="compaction_fallback_clear",
                            sub_turn=sub_turn,
                        )
                        self.log_event("message", {
                            "source": self._log_source, "state_id": self._state_id(),
                            "loop_id": self.current_loop_id,
                            "message_id": f"{self._log_source}_forced_compaction_fallback_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
                            "message": {"role": "assistant", "parts": [{"text": f"[forced_compaction_fallback_clear] reason=input_tokens_exceeded error={compact_exc}"}]},
                            "message_meta": {"event": "compaction_fallback_clear", "reason": "input_tokens_exceeded"},
                        })
                    continue
                # --- Invalid argument (oversized) recovery ---
                if is_invalid_argument and self._compaction_cfg.enabled and not invalid_argument_recovery_used:
                    request_size_chars = 0
                    try:
                        request_size_chars = len(json.dumps(request_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    except Exception:
                        pass
                    if request_size_chars >= 1_500_000:
                        invalid_argument_recovery_used = True
                        if self._resume_policy.on_context_limit == "clear":
                            self._apply_context_clear_to(
                                messages,
                                self._build_fresh_messages("the preserved context exceeded the oversized request limit"),
                                trigger="invalid_argument_oversized_request_clear",
                                sub_turn=sub_turn,
                            )
                            continue
                        try:
                            compacted, info = compact_messages(
                                llm=self.llm, messages=messages, cfg=self._compaction_cfg,
                                max_output_tokens=self.max_output_tokens, tools=stage_tools,
                                tool_dispatch=self.tool_dispatch, overflow_mode=True,
                                round_logger=self._build_compaction_round_logger(sub_turn=sub_turn, event_kind="forced_compaction_invalid_argument"),
                                request_hook=lambda _kind: self._consume_llm_request_budget(
                                    sub_turn=sub_turn,
                                    request_kind="compaction:forced_invalid_argument",
                                ),
                            )
                            messages[:] = compacted
                            self._last_compaction_summary = self._extract_compaction_summary(compacted)
                            self.log_event("message", {
                                "source": self._log_source, "state_id": self._state_id(),
                                "loop_id": self.current_loop_id,
                                "message_id": f"{self._log_source}_forced_compaction_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
                                "message": {"role": "assistant", "parts": [{"text": f"[forced_compaction] reason=invalid_argument request_size_chars={request_size_chars}"}]},
                                "message_meta": {"event": "forced_compaction", "reason": "invalid_argument_oversized_request"},
                            })
                        except Exception as compact_exc:
                            if isinstance(compact_exc, RuntimeError) and (
                                "LLM_REQUEST_LIMIT_REACHED" in str(compact_exc or "")
                                or "CONTEXT_CLEAR_LIMIT_REACHED" in str(compact_exc or "")
                            ):
                                raise
                            self._apply_context_clear_to(
                                messages,
                                self._build_fresh_messages("compaction failed during oversized recovery, context was force-cleared"),
                                trigger="compaction_fallback_clear",
                                sub_turn=sub_turn,
                            )
                            self.log_event("message", {
                                "source": self._log_source, "state_id": self._state_id(),
                                "loop_id": self.current_loop_id,
                                "message_id": f"{self._log_source}_forced_compaction_fallback_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
                                "message": {"role": "assistant", "parts": [{"text": f"[forced_compaction_fallback_clear] reason=invalid_argument error={compact_exc}"}]},
                                "message_meta": {"event": "compaction_fallback_clear", "reason": "invalid_argument_oversized_request"},
                            })
                        continue
                # --- Empty response recovery ---
                if is_empty and self._empty_response_recovery_enabled and not empty_recovery_used:
                    empty_recovery_used = True
                    reminder = (self._empty_response_recovery_reminder or "").strip()
                    if reminder:
                        rem_message = {"role": "user", "parts": build_text_parts_with_inline_media(reminder)}
                        messages.append(rem_message)
                        self.log_event("message", {
                            "source": self._log_source, "state_id": self._state_id(),
                            "loop_id": self.current_loop_id,
                            "message_id": f"{self._log_source}_empty_response_reminder_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
                            "message": rem_message,
                            "message_meta": {"special_types": ["system_reminder"], "reminder_kind": "empty_response_recovery"},
                        })
                    continue
                raise
            # --- Max tokens retry ---
            stop_reason = str(getattr(response, "stop_reason", "") or "").upper()
            if not self._max_tokens_retry_enabled or stop_reason != "MAX_TOKENS":
                return response
            if attempt < self._max_tokens_retry_count:
                attempt += 1
                continue
            reminder = self._max_tokens_retry_reminder.strip()
            if reminder:
                rem_message = {"role": "user", "parts": build_text_parts_with_inline_media(reminder)}
                messages.append(rem_message)
                self.log_event("message", {
                    "source": self._log_source, "state_id": self._state_id(),
                    "loop_id": self.current_loop_id,
                    "message_id": f"{self._log_source}_max_tokens_reminder_{self.step_getter()}_{self.turn_getter()}_{sub_turn}_{attempt}",
                    "message": rem_message,
                    "message_meta": {"special_types": ["system_reminder"], "reminder_kind": "max_tokens_retry"},
                })
            if self._runtime_is_win() and not self._win_epilogue_done:
                raise RuntimeError("WIN_GUARD_BEFORE_LLM_REQUEST")
            self._consume_llm_request_budget(sub_turn=sub_turn, request_kind="stage:max_tokens_retry")
            return self.llm.complete(
                messages=messages, tools=stage_tools,
                max_tokens=self.max_output_tokens, temperature=self._llm_temperature,
                request_data=build_request_data(
                    messages=messages, tools=stage_tools,
                    temperature=self._llm_temperature, max_tokens=self.max_output_tokens,
                    top_p=self._llm_top_p,
                ),
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _stop_requested(self) -> bool:
        try:
            if self.stop_requested_getter is None:
                return False
            return bool(self.stop_requested_getter())
        except Exception:
            return False

    def _runtime_is_win(self) -> bool:
        obs: Dict[str, Any] = {}
        if self.runtime_observation_getter is not None:
            try:
                got = self.runtime_observation_getter()
                if isinstance(got, dict):
                    obs = dict(got)
            except Exception:
                obs = {}
        if not obs:
            obs = dict(self.last_observation or {})
        state = str(obs.get("state") or "").strip().upper()
        return state == "WIN"

    def _state_id(self) -> str:
        return f"level_{self.current_level:04d}"

    @staticmethod
    def _level_from_state_id(state_id: Any) -> Optional[int]:
        text = str(state_id or "").strip().lower()
        if not text.startswith("level_"):
            return None
        try:
            return int(text.split("_", 1)[1])
        except Exception:
            return None

    def _runtime_action_count(self) -> int:
        try:
            return int(self.step_getter())
        except Exception:
            return 0

    def current_action_context(self) -> Dict[str, str]:
        return {
            "message_id": str(self._last_stage_message_id or ""),
            "state_id": self._state_id(),
            "loop_id": str(self.current_loop_id or ""),
        }

    @staticmethod
    def _extract_actions_payload(text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "")
        marker = "[ACTIONS]"
        idx = raw.find(marker)
        if idx < 0:
            return None
        candidate = raw[idx + len(marker):].lstrip()
        if not candidate:
            return None
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalize_action_plan(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        plan_raw = payload.get("plan")
        if not isinstance(plan_raw, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in plan_raw:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or item.get("name") or "").strip().lower()
            if not action:
                continue
            out: Dict[str, Any] = {"action": action}
            if "x" in item:
                out["x"] = item.get("x")
            if "y" in item:
                out["y"] = item.get("y")
            normalized.append(out)
        return normalized

    @staticmethod
    def _clone_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            return json.loads(json.dumps(messages, ensure_ascii=False))
        except Exception:
            return [dict(item) for item in messages]

    def _log_simple_event(self, event: str, text: str, sub_turn: int) -> None:
        self.log_event("message", {
            "source": self._log_source,
            "state_id": self._state_id(),
            "loop_id": self.current_loop_id,
            "message_id": f"{self._log_source}_{event}_{self.step_getter()}_{self.turn_getter()}_{sub_turn}",
            "message": {"role": "assistant", "parts": [{"text": f"[{event}] {text}"}]},
            "message_meta": {"event": event},
        })

    def _sync_todo_cache(self) -> None:
        tool_owner = getattr(self.tool_dispatch, "__self__", None)
        if tool_owner is not None and hasattr(tool_owner, "todos"):
            try:
                todos_obj = getattr(tool_owner, "todos", [])
                normalized: List[Dict[str, str]] = []
                if isinstance(todos_obj, list):
                    for t in todos_obj:
                        if not isinstance(t, dict):
                            continue
                        normalized.append({
                            "id": str(t.get("id", "")),
                            "content": str(t.get("content", "")),
                            "status": str(t.get("status", "pending")),
                            "priority": str(t.get("priority", "medium")),
                        })
                self._todo_cache = normalized
            except Exception:
                pass

    def _sync_todo_cache_clear(self) -> None:
        tool_owner = getattr(self.tool_dispatch, "__self__", None)
        if tool_owner is not None and hasattr(tool_owner, "todos"):
            try:
                setattr(tool_owner, "todos", [])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Assistant message builder
    # ------------------------------------------------------------------

    def _build_assistant_parts(self, response: Any, final_text: str):
        def _is_blank_text_part(part: Dict[str, Any]) -> bool:
            return set(part.keys()) == {"text"} and not str(part.get("text") or "").strip()

        assistant_parts_raw: List[Dict[str, Any]] = []
        try:
            candidates = (response.raw or {}).get("candidates", []) if isinstance(response.raw, dict) else []
            if isinstance(candidates, list) and candidates:
                content = (candidates[0] or {}).get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                if isinstance(parts, list):
                    for p in parts:
                        if not isinstance(p, dict):
                            continue
                        part = dict(p)
                        if _is_blank_text_part(part):
                            continue
                        assistant_parts_raw.append(part)
        except Exception:
            assistant_parts_raw = []

        assistant_parts_ctx: List[Dict[str, Any]] = []
        for p in assistant_parts_raw:
            if _is_blank_text_part(p):
                continue
            assistant_parts_ctx.append(dict(p))

        if not assistant_parts_raw:
            if str(final_text or "").strip():
                cleaned_text = str(final_text).strip()
                assistant_parts_raw.append({"text": cleaned_text})
                assistant_parts_ctx.append({"text": cleaned_text})
            for tc in response.tool_calls:
                part: Dict[str, Any] = {
                    "functionCall": {"name": str(tc.get("name", "")), "args": tc.get("args", {})}
                }
                if tc.get("thoughtSignature"):
                    part["thoughtSignature"] = tc.get("thoughtSignature")
                assistant_parts_raw.append(part)
                assistant_parts_ctx.append(dict(part))

        if not assistant_parts_raw:
            assistant_parts_raw = [{"text": ""}]
        if not assistant_parts_ctx:
            assistant_parts_ctx = [{"text": ""}]

        return assistant_parts_raw, assistant_parts_ctx

    # ------------------------------------------------------------------
    # Compaction config builder
    # ------------------------------------------------------------------

    def _context_budget_status(self) -> str:
        """Return a short context-token budget summary string."""
        if not self._compaction_cfg.enabled or self._memory_checkpoint_cfg.enabled:
            return ""
        max_ctx = int(self._compaction_cfg.max_context_tokens)
        ratio = float(self._compaction_cfg.trigger_ratio)
        threshold = int(max_ctx * ratio)
        observed = int(self._last_prompt_tokens_observed)
        if observed <= 0:
            return ""
        pct_used = min(100.0, (observed / threshold) * 100.0) if threshold > 0 else 0.0
        reminder_pct = self._context_budget_reminder_threshold * 100.0
        if pct_used < reminder_pct:
            return ""
        remaining = max(0, threshold - observed)
        context_budget_tpl = str(self._reminders.get("context_budget") or "").strip()
        if not context_budget_tpl:
            context_budget_tpl = (
                "Context usage: {CONTEXT_OBSERVED_TOKENS}/{CONTEXT_TRIGGER_TOKENS} "
                "tokens ({CONTEXT_USED_PCT}% used). Save important findings to "
                "MEMORY.md now — context will be compacted soon."
            )
        return render_template(
            context_budget_tpl,
            {
                "CONTEXT_OBSERVED_TOKENS": str(observed),
                "CONTEXT_TRIGGER_TOKENS": str(threshold),
                "CONTEXT_USED_PCT": f"{pct_used:.0f}",
                "CONTEXT_REMAINING_TOKENS": str(remaining),
                "CONTEXT_REMINDER_THRESHOLD_PCT": f"{reminder_pct:.0f}",
            },
        ).strip()

    def _build_compaction_config(self) -> CompactionConfig:
        summary_prompt = get_compaction_str("summary_prompt", DEFAULT_COMPACTION_PROMPT).strip()
        if not summary_prompt:
            summary_prompt = DEFAULT_COMPACTION_PROMPT
        summary_user_template = get_compaction_str("summary_user_template", DEFAULT_COMPACTION_USER_TEMPLATE).strip()
        if not summary_user_template:
            summary_user_template = DEFAULT_COMPACTION_USER_TEMPLATE
        return CompactionConfig(
            enabled=get_compaction_bool("enabled", False),
            max_context_tokens=max(1, get_compaction_int("max_context_tokens", 800_000)),
            trigger_ratio=max(0.0, min(1.0, get_compaction_float("trigger_ratio", 0.8))),
            summary_max_tokens=max(1, get_compaction_int("summary_max_tokens", 4096)),
            summary_prompt=summary_prompt,
            summary_user_template=summary_user_template,
            multi_round_enabled=get_compaction_bool("multi_round_enabled", True),
            max_rounds=max(1, get_compaction_int("max_rounds", 4)),
            tool_names=tuple(
                str(x).strip()
                for x in str(get_compaction_str("tool_names", "bash_exec")).split(",")
                if str(x).strip()
            ),
            pin_first_user_message=get_compaction_bool("pin_first_user_message", False),
        )

    def _build_memory_checkpoint_clear_config(self) -> MemoryCheckpointClearConfig:
        cfg = get_memory_checkpoint_clear_config()
        enabled_raw = cfg.get("enabled", False)
        enabled = bool(enabled_raw) if isinstance(enabled_raw, bool) else str(enabled_raw).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        memory_path = str(cfg.get("memory_path") or "MEMORY.md").strip() or "MEMORY.md"
        try:
            max_grace_turns = max(0, int(cfg.get("max_grace_turns", 1) or 1))
        except Exception:
            max_grace_turns = 1
        return MemoryCheckpointClearConfig(
            enabled=enabled,
            memory_path=memory_path,
            max_grace_turns=max_grace_turns,
        )

    def _reset_memory_checkpoint_state(self) -> None:
        self._memory_checkpoint_pending = False
        self._memory_checkpoint_turns_waited = 0
        self._memory_checkpoint_reason = ""
        self._memory_checkpoint_deadline_missed = False

    def _memory_checkpoint_reminder_text(self, *, trigger_tokens: int, threshold: int) -> str:
        tpl = str(self._reminders.get("memory_checkpoint") or "").strip()
        if not tpl:
            tpl = (
                "<system-reminder>\n"
                "Context usage reached {CONTEXT_OBSERVED_TOKENS}/{CONTEXT_TRIGGER_TOKENS} tokens. "
                "Before context is cleared, write {MEMORY_CHECKPOINT_PATH} with the durable facts, "
                "current hypothesis, failures, and next steps needed to continue. "
                "As soon as that write succeeds, context will be cleared and the task will resume.\n"
                "</system-reminder>"
            )
        return render_template(
            tpl,
            {
                "CONTEXT_OBSERVED_TOKENS": str(trigger_tokens),
                "CONTEXT_TRIGGER_TOKENS": str(threshold),
                "MEMORY_CHECKPOINT_PATH": self._memory_checkpoint_cfg.memory_path,
            },
        ).strip()

    def _memory_checkpoint_final_reminder_text(self, *, trigger_tokens: int, threshold: int) -> str:
        tpl = str(self._reminders.get("memory_checkpoint_final") or "").strip()
        if not tpl:
            tpl = (
                "<system-reminder>\n"
                "Final reminder: write {MEMORY_CHECKPOINT_PATH} in this response. "
                "If this response does not successfully write that file, the run will stop.\n"
                "</system-reminder>"
            )
        return render_template(
            tpl,
            {
                "CONTEXT_OBSERVED_TOKENS": str(trigger_tokens),
                "CONTEXT_TRIGGER_TOKENS": str(threshold),
                "MEMORY_CHECKPOINT_PATH": self._memory_checkpoint_cfg.memory_path,
            },
        ).strip()

    def _build_memory_checkpoint_resume_messages(self, clear_reason: str) -> List[Dict[str, Any]]:
        system_values = self._base_prompt_values()
        self._system_content = self._prompt_renderer.render(self._system_template, system_values)
        user_prompt = self._render_user_prompt()
        self._last_compaction_summary = ""
        history_path = str(self._base_prompt_values().get("HISTORY_LOG_PATH", "") or "").strip()
        assistant_note = (
            "[Memory checkpoint clear]\n\n"
            f"Context was cleared: {clear_reason}\n"
            f"Re-read {self._memory_checkpoint_cfg.memory_path}"
        )
        if history_path:
            assistant_note += f", consult {history_path} if needed"
        assistant_note += ", and continue the same task from the latest runtime context."
        return [
            {"role": "system", "parts": [{"text": self._system_content}]},
            {"role": "user", "parts": build_text_parts_with_inline_media(user_prompt)},
            {"role": "assistant", "parts": [{"text": assistant_note}]},
            {"role": "user", "parts": [{"text": "Continue."}]},
        ]

    def _apply_missed_memory_checkpoint_fallback(self, *, sub_turn: int) -> None:
        memory_path = str(self._memory_checkpoint_cfg.memory_path or "MEMORY.md").strip()
        clear_reason = (
            f"required {memory_path} checkpoint was not written after final reminder; "
            f"continuing from the existing {memory_path}"
        )
        new_messages = self._build_memory_checkpoint_resume_messages(clear_reason)
        self._consume_context_clear_budget(trigger="memory_checkpoint_fallback", sub_turn=sub_turn)
        self._messages = new_messages
        self._reset_memory_checkpoint_state()
        self._reset_context_token_observation()
        self._log_simple_event("memory_checkpoint_fallback", clear_reason, sub_turn)
        self.log_event(
            "message",
            {
                "source": self._log_source,
                "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": (
                    f"{self._log_source}_context_clear_memory_checkpoint_fallback_"
                    f"{self.step_getter()}_{self.turn_getter()}_{sub_turn}"
                ),
                "message": {
                    "role": "assistant",
                    "parts": [{"text": "[context_clear] trigger=memory_checkpoint_fallback"}],
                },
                "message_meta": {
                    "event": "context_clear",
                    "trigger": "memory_checkpoint_fallback",
                },
            },
        )

    def _reset_context_token_observation(self) -> None:
        self._last_prompt_tokens_observed = 0
        self._last_total_tokens_observed = 0

    def _append_memory_checkpoint_reminder(
        self, messages: List[Dict[str, Any]], reminder_text: str
    ) -> List[Dict[str, Any]]:
        reminded = list(messages)
        if reminded:
            last = dict(reminded[-1])
            parts = list(last.get("parts") or [])
            if str(last.get("role") or "") == "user" and any(
                isinstance(part, dict) and "functionResponse" in part for part in parts
            ):
                parts.append({"text": reminder_text})
                last["parts"] = parts
                reminded[-1] = last
                return reminded
        reminded.append({"role": "user", "parts": [{"text": reminder_text}]})
        return reminded

    def _is_memory_checkpoint_write_success(self, tool_name: str, result: str) -> bool:
        if tool_name != "write_file":
            return False
        if not self._memory_checkpoint_pending:
            return False
        try:
            payload = json.loads(result)
        except Exception:
            return False
        if not isinstance(payload, dict) or not payload.get("ok"):
            return False
        path_value = str(payload.get("path") or "").strip()
        target = str(self._memory_checkpoint_cfg.memory_path or "MEMORY.md").strip()
        return path_value == target

    def _consume_llm_request_budget(self, *, sub_turn: int, request_kind: str) -> None:
        if self._llm_request_limit > 0 and self._llm_requests_used >= self._llm_request_limit:
            detail = (
                f"LLM_REQUEST_LIMIT_REACHED used={self._llm_requests_used} "
                f"limit={self._llm_request_limit} request_kind={request_kind}"
            )
            parts: List[Dict[str, Any]] = [{"text": f"[llm_request_limit_reached] {detail}"}]
            reminder_text = str(self._llm_request_limit_message or "").strip()
            if reminder_text:
                parts.append({"text": reminder_text})
            self.log_event(
                "message",
                {
                    "source": self._log_source,
                    "state_id": self._state_id(),
                    "loop_id": self.current_loop_id,
                    "message_id": (
                        f"{self._log_source}_llm_request_limit_"
                        f"{self.step_getter()}_{self.turn_getter()}_{sub_turn}_{self._llm_requests_used}"
                    ),
                    "message": {"role": "assistant", "parts": parts},
                    "message_meta": {
                        "event": "llm_request_limit_reached",
                        "requests_used": int(self._llm_requests_used),
                        "request_limit": int(self._llm_request_limit),
                        "request_kind": str(request_kind),
                    },
                },
            )
            raise RuntimeError(detail)
        self._llm_requests_used += 1

    def _consume_context_clear_budget(self, *, trigger: str, sub_turn: int) -> None:
        if self._max_context_clears_per_run > 0 and self._context_clears_used >= self._max_context_clears_per_run:
            detail = (
                f"CONTEXT_CLEAR_LIMIT_REACHED used={self._context_clears_used} "
                f"limit={self._max_context_clears_per_run} trigger={trigger}"
            )
            self.log_event(
                "message",
                {
                    "source": self._log_source,
                    "state_id": self._state_id(),
                    "loop_id": self.current_loop_id,
                    "message_id": (
                        f"{self._log_source}_context_clear_limit_"
                        f"{self.step_getter()}_{self.turn_getter()}_{sub_turn}_{self._context_clears_used}"
                    ),
                    "message": {"role": "assistant", "parts": [{"text": f"[context_clear_limit_reached] {detail}"}]},
                    "message_meta": {
                        "event": "context_clear_limit_reached",
                        "clears_used": int(self._context_clears_used),
                        "clear_limit": int(self._max_context_clears_per_run),
                        "trigger": str(trigger),
                    },
                },
            )
            raise RuntimeError(detail)
        self._context_clears_used += 1

    def _apply_context_clear_to(
        self,
        target_messages: List[Dict[str, Any]],
        replacement_messages: List[Dict[str, Any]],
        *,
        trigger: str,
        sub_turn: int,
    ) -> None:
        self._consume_context_clear_budget(trigger=trigger, sub_turn=sub_turn)
        target_messages[:] = replacement_messages
        self._reset_context_token_observation()

    def _build_compaction_round_logger(self, *, sub_turn: int, event_kind: str) -> Callable[[int, Any], None]:
        def _log_round(round_idx: int, resp: Any) -> None:
            raw_request = getattr(resp, "raw_request", None)
            raw_response = getattr(resp, "raw", None)
            req_cfg = getattr(resp, "request_generation_config", None)
            if not (isinstance(raw_request, dict) or isinstance(raw_response, dict) or isinstance(req_cfg, dict)):
                return
            self._compaction_raw_seq += 1
            self.log_event("raw_request", {
                "source": self._log_source, "state_id": self._state_id(),
                "loop_id": self.current_loop_id,
                "message_id": f"{self._log_source}_compaction_round_raw_{self.step_getter()}_{self.turn_getter()}_{sub_turn}_{event_kind}_{self._compaction_raw_seq}_{int(round_idx)}",
                "raw_request": raw_request if isinstance(raw_request, dict) else {},
                "raw_response": raw_response if isinstance(raw_response, dict) else {},
                "request_generation_config": req_cfg if isinstance(req_cfg, dict) else {},
            })
        return _log_round

    @staticmethod
    def _extract_compaction_summary(compacted_messages: List[Dict[str, Any]]) -> str:
        """Extract the compaction summary text from compacted messages.

        With pin_first_user_message the summary is in the first assistant
        message (after system + pinned user).  Without pinning it is in
        the first user message (the handoff).
        """
        # Try assistant first (pinned mode), then user (legacy mode).
        for target_role in ("assistant", "user"):
            for msg in compacted_messages:
                if str(msg.get("role") or "") != target_role:
                    continue
                parts = msg.get("parts")
                if not isinstance(parts, list):
                    continue
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    text = p.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        return ""

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _base_prompt_values(self) -> Dict[str, str]:
        values: Dict[str, str] = {
            "RUNTIME_API_BASE": self.runtime_api_base,
            "LEVEL": str(self.current_level),
            "STATE_ID": self._state_id(),
            "STATE": str(self.last_observation.get("state") or ""),
            "WORKSPACE_ROOT": str(self.workspace.resolve()),
            "DEFAULT_TOOL_CWD": str(self.workspace.resolve()),
            "ARC_UTILS_PATH": str((self.workspace / "arc_utils.py").resolve()),
            "MEMO_PATH": str((self.workspace / "MEMORY.md").resolve()),
            "MEMORY_PATH": str((self.workspace / "MEMORY.md").resolve()),
            "MEMORY_CHECKPOINT_PATH": str(self._memory_checkpoint_cfg.memory_path or "MEMORY.md"),
            "HISTORY_LOG_PATH": self._history_log_display_path(),
            "WORKSPACE_SIZE_LIMIT": self._format_workspace_size_limit(),
            "RESUME_ON_LEVEL_UP": self._resume_policy.on_level_up,
            "RESUME_ON_ACTION_SUBMITTED": self._resume_policy.on_action_submitted,
            "RESUME_ON_CONTEXT_LIMIT": self._resume_policy.on_context_limit,
            "HARD_ACTION_LIMIT": "10",
            "AVAILABLE_ACTIONS_JSON": json.dumps(self.last_observation.get("available_actions", []), ensure_ascii=False),
            "KNOWLEDGE_MEMORY_MD": self._read_memory_md("knowledge/MEMORY.md"),
            "HYPOTHESES_MEMORY_MD": self._read_memory_md("hypotheses/MEMORY.md"),
            "SKILLS_MEMORY_MD": self._read_memory_md("skills/MEMORY.md"),
            "POLICY_JSON": json.dumps(self.policy or {}, ensure_ascii=False, indent=2),
            "POLICY_MEMORY_MD": self._read_memory_md("policy/MEMORY.md"),
        }
        values.update(self._observation_placeholders())
        return values

    def _history_log_display_path(self) -> str:
        """Return the history log path for display in prompts.

        Uses the relative path from the YAML config (e.g. ``../history_{RUN_ID}.jsonl``)
        with placeholders expanded, keeping it relative to the workspace so the
        prompt does not leak long absolute paths.  The absolute path is still
        used for ``extra_read_paths`` / actual file I/O elsewhere.
        """
        if not get_history_log_bool("enabled", False):
            return ""
        raw = get_history_log_str("path", "").strip()
        if not raw:
            return ""
        try:
            rendered = raw.format(
                RUN_ID=self.run_id,
                GAME_ID=str(self.last_observation.get("game_id") or ""),
                WORKSPACE_NAME=self.workspace.name,
            )
        except Exception:
            rendered = raw
        return rendered

    @staticmethod
    def _format_workspace_size_limit() -> str:
        limit = get_workspace_size_limit_bytes(0)
        if limit <= 0:
            return "unlimited"
        for unit, div in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
            if limit >= div:
                val = limit / div
                return f"{val:.0f}{unit}" if val == int(val) else f"{val:.1f}{unit}"
        return f"{limit}B"

    def _render_user_prompt(self) -> str:
        values = self._base_prompt_values()
        return self._prompt_renderer.render(self._user_template, values)

    def _reminder_template_values(self, *, user_prompt: str = "", trigger: str = "", clear_reason: str = "") -> Dict[str, str]:
        values = self._base_prompt_values()
        values.update({
            "STAGE_NAME": self.current_stage,
            "LATEST_USER_PROMPT": user_prompt.rstrip(),
            "RESUME_TRIGGER": str(trigger),
            "RESUME_TRIGGER_NOTE": "",
            "CONTEXT_CLEAR_REASON": str(clear_reason),
            "LATEST_COMPACTION_SUMMARY": str(self._last_compaction_summary or ""),
        })
        return values

    def _read_memory_md(self, relative_path: str) -> str:
        path = self.memory_root / relative_path
        try:
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8")
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    def _build_merged_reminder(self, include_todo: bool, game_mode: Optional[str]) -> str:
        parts: List[str] = []
        if game_mode:
            game_text = self._build_game_reminder(game_mode=game_mode)
            if game_text:
                parts.append(game_text)
        if include_todo:
            todo_text = self._todo_reminder_body()
            if todo_text:
                parts.append(todo_text)
        return "\n".join(parts) if parts else ""

    def _build_game_reminder(self, game_mode: str) -> str:
        obs = None
        if self.runtime_observation_getter is not None:
            try:
                got = self.runtime_observation_getter()
                if isinstance(got, dict):
                    obs = dict(got)
            except Exception:
                obs = None
        if not obs:
            obs = dict(self.last_observation or {})
        if not obs:
            return "Game update detected."
        level = self._level_from_state_id(obs.get("state_id"))
        if level is None:
            level = int(obs.get("levels_completed") or 0)
        available_actions = obs.get("available_actions", [])
        image_data_url = ""
        grid_any = self._observation_frames_input(obs)
        if grid_any is not None:
            image_data_url = self._render_observation_b64(grid_any)
        actions = []
        if self.runtime_action_history_getter is not None:
            try:
                got = self.runtime_action_history_getter()
                if isinstance(got, list):
                    actions = [a for a in got if isinstance(a, dict)]
            except Exception:
                actions = []
        total_actions = len(actions)
        if game_mode == "initial":
            tpl = str(self._reminders.get("game_initial") or "").strip()
        else:
            tpl = str(self._reminders.get("game_update") or "").strip()
        if not tpl:
            tpl = str(self._reminders.get("game") or "").strip()
        values = {
            "GAME_TOTAL_ACTIONS": str(total_actions),
            "GAME_LEVEL": str(level),
            "GAME_STATE_ID": f"level_{level:04d}",
            "GAME_STATE": str(obs.get("state") or ""),
            "GAME_AVAILABLE_ACTIONS": json.dumps(available_actions, ensure_ascii=False),
            "GAME_IMAGE_DATA_URL": image_data_url,
        }
        return render_template(tpl, values).strip()

    def _build_todo_reminder(self) -> str:
        return self._todo_reminder_body()

    def _todo_reminder_body(self) -> str:
        if not self._todo_cache:
            return str(self._reminders.get("todo_empty") or "").strip()
        lines: List[str] = []
        for t in self._todo_cache:
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
        tpl = str(self._reminders.get("todo") or "").strip()
        return render_template(tpl, {"TODO_ITEMS": "\n".join(lines)}).strip()

    def _load_reminder_templates(self) -> Dict[str, str]:
        cfg = get_reminder_templates()
        def _as_bool(v: Any, default: bool = False) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v or "").strip().lower()
            if not s:
                return default
            return s in {"1", "true", "yes", "y", "on"} if s in {"1", "true", "yes", "y", "on", "0", "false", "no", "n", "off"} else default

        todo_empty = str(cfg.get("todo_empty") or "").strip()
        todo = str(cfg.get("todo") or "").strip()
        game_initial = str(cfg.get("game_initial") or "").strip()
        game_update = str(cfg.get("game_update") or "").strip()
        game = str(cfg.get("game") or "").strip()
        level_up = str(cfg.get("level_up") or "").strip()
        context_cleared = str(cfg.get("context_cleared") or "").strip()
        context_budget = str(cfg.get("context_budget") or "").strip()
        memory_checkpoint = str(cfg.get("memory_checkpoint") or "").strip()
        win_epilogue = str(cfg.get("win_epilogue") or "").strip()
        max_steps = str(cfg.get("max_steps") or "").strip()
        enable_game_reminder = _as_bool(cfg.get("enable_game_reminder"), True)
        enable_todo_reminder = str(cfg.get("enable_todo_reminder") or "auto").strip().lower() or "auto"
        todo_reminder_interval_raw = cfg.get("todo_reminder_interval", 32)
        try:
            todo_reminder_interval = max(1, int(todo_reminder_interval_raw))
        except Exception:
            todo_reminder_interval = 32
        todo_enabled_by_config = enable_todo_reminder not in {"0", "false", "no", "n", "off", "disable", "disabled"}
        if todo_enabled_by_config:
            if not todo_empty:
                todo_empty = "[todo-reminder]\nKeep an internal checklist (objective / test / stop condition) and refresh it after each observation batch."
            if not todo:
                todo = "[todo-reminder]\nCurrent checklist:\n{TODO_ITEMS}"
        else:
            todo_empty = ""
            todo = ""
        if enable_game_reminder:
            default_game_tpl = (
                "<game-reminder>\ntotal_actions={GAME_TOTAL_ACTIONS}\n"
                "current_level={GAME_LEVEL}\ncurrent_state_id={GAME_STATE_ID}\n"
                "current_state={GAME_STATE}\navailable_actions={GAME_AVAILABLE_ACTIONS}\n</game-reminder>"
            )
            if not game_initial:
                game_initial = default_game_tpl
            if not game_update:
                game_update = default_game_tpl
            if not game:
                game = game_update
        else:
            game_initial = ""
            game_update = ""
            game = ""
        if not context_cleared:
            context_cleared = (
                "<system-reminder>\n"
                "Context was reset because {CONTEXT_CLEAR_REASON}.\n"
                "Continue from the latest runtime context below.\n\n"
                "{LATEST_USER_PROMPT}\n"
                "</system-reminder>"
            )
        if not level_up:
            level_up = (
                "Level up: {PREV_LEVEL} -> {NEW_LEVEL}. "
                "Save level {PREV_LEVEL} discoveries to MEMORY.md."
            )
        if not context_budget:
            context_budget = (
                "Context usage: {CONTEXT_OBSERVED_TOKENS}/{CONTEXT_TRIGGER_TOKENS} "
                "tokens ({CONTEXT_USED_PCT}% used). Save important findings to "
                "MEMORY.md now — context will be compacted soon."
            )
        if not memory_checkpoint:
            memory_checkpoint = (
                "<system-reminder>\n"
                "Context usage reached {CONTEXT_OBSERVED_TOKENS}/{CONTEXT_TRIGGER_TOKENS} tokens. "
                "Write {MEMORY_CHECKPOINT_PATH} now with the durable findings needed to continue. "
                "When that write succeeds, context will be cleared and the task will resume.\n"
                "</system-reminder>"
            )
        if not max_steps:
            max_steps = (
                "<system-reminder>\n"
                "CRITICAL — FINAL STEP\n\n"
                "You have reached the maximum number of steps allowed for this loop.\n"
                "This is your LAST opportunity to respond.\n\n"
                "REQUIREMENTS:\n"
                "1. Do NOT start new multi-step explorations — there is no next turn.\n"
                "2. Write your updated MEMORY.md with everything discovered so far.\n"
                "3. Output your best action plan based on current knowledge.\n"
                "4. If the task is incomplete, note what remains in MEMORY.md for the next loop.\n"
                "</system-reminder>"
            )
        return {
            "enable_todo_reminder": enable_todo_reminder,
            "enable_workspace_budget_reminder": str(cfg.get("enable_workspace_budget_reminder") or "auto").strip().lower(),
            "enable_context_budget_reminder": str(cfg.get("enable_context_budget_reminder") or "auto").strip().lower(),
            "context_budget_reminder_threshold": cfg.get("context_budget_reminder_threshold"),
            "todo_reminder_interval": str(todo_reminder_interval),
            "todo_empty": todo_empty,
            "todo": todo,
            "game_initial": game_initial,
            "game_update": game_update,
            "game": game,
            "level_up": level_up,
            "context_budget": context_budget,
            "memory_checkpoint": memory_checkpoint,
            "context_cleared": context_cleared,
            "win_epilogue": win_epilogue,
            "max_steps": max_steps,
        }

    # ------------------------------------------------------------------
    # Text extraction from raw LLM response
    # ------------------------------------------------------------------

    def _extract_text_from_raw(self, raw: Any) -> str:
        try:
            if isinstance(raw, dict):
                choices = raw.get("choices")
                if isinstance(choices, list) and choices:
                    msg = (choices[0] or {}).get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        texts: List[str] = []
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                texts.append(str(p.get("text") or ""))
                        return "\n".join([t for t in texts if t.strip()])
            candidates = raw.get("candidates") if isinstance(raw, dict) else None
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            texts: List[str] = []
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    if p.get("thought") is True:
                        continue
                    texts.append(str(p["text"]))
            return "\n".join([t for t in texts if t.strip()])
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Observation rendering
    # ------------------------------------------------------------------

    def _observation_placeholders(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for before in range(3):
            obs = self._get_observation_before(before)
            image_b64 = ""
            ascii_text = ""
            if obs is not None:
                grid_any = self._observation_frames_input(obs)
                if grid_any is not None:
                    image_b64 = self._render_observation_b64(grid_any)
                    frames = self._extract_grid_frames(grid_any)
                    ascii_frames = [self._grid_to_ascii_text(frame) for frame in frames]
                    ascii_text = "\n\n".join(ascii_frames)
            out[f"OBSERVATION_IMAGE_BEFORE_{before}"] = image_b64
            out[f"OBSERVATION_ASCII_BEFORE_{before}"] = ascii_text
        return out

    def _get_observation_before(self, before: int) -> Optional[Dict[str, Any]]:
        if before < 0:
            return None
        idx = len(self.recent_observations) - 1 - before
        if idx < 0 or idx >= len(self.recent_observations):
            return None
        return self.recent_observations[idx]

    def _render_observation_b64(self, grid_any: Any) -> str:
        if Image is None:
            return ""
        frames = self._extract_grid_frames(grid_any)
        if not frames:
            return ""
        try:
            if len(frames) == 1:
                img = self._grid_to_image(frames[0])
                buff = io.BytesIO()
                img.save(buff, format="PNG")
                b64 = base64.b64encode(buff.getvalue()).decode("ascii")
                return f"data:image/png;base64,{b64}"
            if cv2 is None or np is None:
                img = self._grid_to_image(frames[-1])
                buff = io.BytesIO()
                img.save(buff, format="PNG")
                b64 = base64.b64encode(buff.getvalue()).decode("ascii")
                return f"data:image/png;base64,{b64}"
            imgs = [self._grid_to_image(frame).convert("RGB") for frame in frames]
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tmp_path = tf.name
            try:
                w, h = imgs[0].size
                writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("mp4_writer_open_failed")
                for img in imgs:
                    arr = np.array(img)
                    frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
                writer.release()
                data = open(tmp_path, "rb").read()
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:video/mp4;base64,{b64}"
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        except Exception:
            return ""

    def _observation_frames_input(self, obs: Dict[str, Any]) -> Any:
        frames_any = obs.get("frames")
        if isinstance(frames_any, list):
            return frames_any
        return []

    def _extract_grid_frames(self, grid_any: Any) -> List[List[List[int]]]:
        if not isinstance(grid_any, list) or not grid_any:
            return []
        if isinstance(grid_any[0], dict):
            out: List[List[List[int]]] = []
            for frame_item in grid_any:
                if not isinstance(frame_item, dict):
                    continue
                frame_grid = frame_item.get("grid")
                if isinstance(frame_grid, list) and frame_grid and isinstance(frame_grid[0], list) and frame_grid[0] and isinstance(frame_grid[0][0], int):
                    out.append(frame_grid)
            return out
        first = grid_any[0]
        if isinstance(first, list) and first and isinstance(first[0], int):
            return [grid_any]
        if isinstance(first, list) and first and isinstance(first[0], list):
            out = []
            for frame in grid_any:
                if isinstance(frame, list) and frame and isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], int):
                    out.append(frame)
            return out
        return []

    def _grid_to_image(self, grid: List[List[int]]) -> Any:
        assert Image is not None
        h = len(grid)
        w = len(grid[0]) if h else 0
        scale = self._render_scale
        img = Image.new("RGB", (max(1, w * scale), max(1, h * scale)), (0, 0, 0))
        px = img.load()
        palette = [
            (255, 255, 255), (248, 244, 234), (217, 217, 217), (158, 158, 158),
            (97, 97, 97), (0, 0, 0), (255, 0, 255), (255, 154, 213),
            (255, 0, 0), (30, 64, 255), (142, 203, 255), (255, 230, 0),
            (255, 152, 0), (128, 0, 0), (0, 166, 81), (126, 34, 206),
        ]
        for y, row in enumerate(grid):
            for x, raw in enumerate(row):
                try:
                    v = int(raw)
                except Exception:
                    v = 0
                if v < 0 or v >= len(palette):
                    v = 0
                c = palette[v]
                for dy in range(scale):
                    for dx in range(scale):
                        px[x * scale + dx, y * scale + dy] = c
        return img

    def _grid_to_ascii_text(self, grid: List[List[int]]) -> str:
        palette = ["W", "w", "g", "G", "D", "K", "m", "p", "R", "B", "b", "Y", "O", "n", "g", "u"]
        lines: List[str] = []
        for row in grid:
            chars = []
            for raw in row:
                v = int(raw) if isinstance(raw, int) else 0
                if v < 0 or v >= len(palette):
                    v = 0
                chars.append(palette[v])
            lines.append("".join(chars))
        return "\n".join(lines)

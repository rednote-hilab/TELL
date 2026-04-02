from __future__ import annotations

from pathlib import Path

import pytest

from agents.tell_agent.llm_response import LLMResponse
from agents.tell_agent.stage_workflow import ResumePolicy as WorkflowResumePolicy, build_stage_workflow
from agents.tell_agent.state_machine import TELLStateMachine, ResumePolicy
from agents.tell_agent.tool_handlers import TELLToolHandlers


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools or [],
                "kwargs": kwargs,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


def _make_response(text: str = "") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        raw={},
        stop_reason="STOP",
    )


def _build_state_machine(
    monkeypatch,
    tmp_path: Path,
    llm: FakeLLM,
    log_events: list[tuple[str, dict]],
    *,
    reminders: dict | None = None,
    resume_policy: dict | None = None,
):
    workflow_cfg = {
        "workflow": {
            "entry_stage": "main",
            "stages": {
                "main": {
                    "system": "system",
                    "user": "user",
                    "tools": [],
                    "transitions": [{"type": "default", "to": "main"}],
                }
            },
        }
    }
    if resume_policy is not None:
        workflow_cfg["workflow"]["stages"]["main"]["resume_policy"] = resume_policy

    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_prompt_templates",
        lambda: workflow_cfg,
    )
    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_reminder_templates",
        lambda: reminders or {},
    )
    counters = {"step": 0, "turn": 0, "runtime_step": 0}

    def log_event(kind: str, payload: dict) -> None:
        log_events.append((kind, payload))

    machine = TELLStateMachine(
        llm=llm,
        tools=[],
        tool_dispatch=lambda name, args: "",
        log_event=log_event,
        workspace=tmp_path,
        memory_root=tmp_path,
        max_stage_turns=2,
        max_output_tokens=256,
        recent_frames_limit=4,
        runtime_api_base="http://127.0.0.1:8000",
        run_id="test-run",
        step_getter=lambda: counters["step"],
        turn_getter=lambda: counters["turn"],
        runtime_step_getter=lambda: counters["runtime_step"],
        runtime_observation_getter=lambda: {},
        runtime_action_history_getter=lambda: [],
        stop_requested_getter=lambda: False,
    )
    machine.current_level = 0
    return machine, counters


def test_build_stage_workflow_parses_resume_policy():
    workflow = build_stage_workflow(
        {
            "workflow": {
                "entry_stage": "main",
                "stages": {
                    "main": {
                        "system": "system",
                        "user": "user",
                        "tools": ["read_file"],
                        "resume": True,
                        "resume_policy": {
                            "on_level_up": "clear",
                            "on_action_submitted": "keep",
                            "on_context_limit": "compact",
                        },
                        "transitions": [{"type": "default", "to": "main"}],
                    }
                },
            }
        }
    )

    stage = workflow.get_stage("main")
    assert stage.resume is True
    assert isinstance(stage.resume_policy, WorkflowResumePolicy)
    assert stage.resume_policy.on_level_up == "clear"
    assert stage.resume_policy.on_action_submitted == "keep"
    assert stage.resume_policy.on_context_limit == "compact"


def test_read_file_returns_numbered_lines(tmp_path):
    (tmp_path / "notes.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
    )

    result = handler.handle_read_file({"path": "notes.txt", "offset": 2, "limit": 2})

    assert "2\tb" in result
    assert "3\tc" in result


def test_list_dir_lists_entries(tmp_path):
    (tmp_path / "alpha.txt").write_text("x", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
    )

    result = handler.handle_list_dir({"path": ".", "limit": 10})

    assert "alpha.txt" in result
    assert "beta/" in result


def test_read_file_supports_configured_extra_read_paths(tmp_path):
    history_path = tmp_path.parent / "history.log"
    history_path.write_text('{"hello":"world"}\n', encoding="utf-8")
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
        extra_read_paths=[history_path],
    )

    result = handler.handle_read_file({"path": "../history.log"})

    assert "hello" in result
    assert str(history_path) in result


def test_grep_text_finds_matches(tmp_path):
    src = tmp_path / "pkg"
    src.mkdir()
    (src / "demo.py").write_text("value = 3\nother = 5\n", encoding="utf-8")
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
    )

    result = handler.handle_grep_text({"pattern": "value", "path": ".", "include": "*.py"})

    assert "demo.py:1:value = 3" in result


def test_dispatch_supports_new_tools(tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
    )

    result = handler.dispatch("read_file", {"path": "notes.txt"})

    assert "hello" in result


def test_handle_level_up_clears_messages(monkeypatch, tmp_path):
    """When resume_policy.on_level_up == 'clear', _handle_level_up rebuilds messages."""
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _counters = _build_state_machine(
        monkeypatch, tmp_path, llm, log_events,
        resume_policy={"on_level_up": "clear", "on_action_submitted": "keep", "on_context_limit": "compact"},
    )

    # Simulate having existing conversation messages
    machine._messages = [
        {"role": "system", "parts": [{"text": "system prompt"}]},
        {"role": "user", "parts": [{"text": "user prompt"}]},
        {"role": "assistant", "parts": [{"text": "some previous response"}]},
        {"role": "user", "parts": [{"text": "more context"}]},
    ]
    machine._last_level = 0
    machine.current_level = 1

    machine._handle_level_up(stage_tools=[], sub_turn=0)

    # Clear is deferred — check that the deferred action is set
    assert machine._deferred_context_action == ("clear", "level_up")
    assert machine._last_level == 1


def test_handle_action_submitted_keeps_messages(monkeypatch, tmp_path):
    """When resume_policy.on_action_submitted == 'keep', messages are preserved."""
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _counters = _build_state_machine(
        monkeypatch, tmp_path, llm, log_events,
        resume_policy={"on_level_up": "compact", "on_action_submitted": "keep", "on_context_limit": "compact"},
    )

    original_messages = [
        {"role": "system", "parts": [{"text": "system prompt"}]},
        {"role": "user", "parts": [{"text": "user prompt"}]},
        {"role": "assistant", "parts": [{"text": "response"}]},
    ]
    machine._messages = list(original_messages)
    machine._last_action_count = 0

    machine._handle_action_submitted(stage_tools=[], sub_turn=0)

    # Messages should be unchanged
    assert len(machine._messages) == 3
    assert machine._messages[2]["parts"][0]["text"] == "response"


def test_handle_action_submitted_clears_messages(monkeypatch, tmp_path):
    """When resume_policy.on_action_submitted == 'clear', messages are rebuilt."""
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _counters = _build_state_machine(
        monkeypatch, tmp_path, llm, log_events,
        resume_policy={"on_level_up": "compact", "on_action_submitted": "clear", "on_context_limit": "compact"},
    )

    machine._messages = [
        {"role": "system", "parts": [{"text": "system prompt"}]},
        {"role": "user", "parts": [{"text": "user prompt"}]},
        {"role": "assistant", "parts": [{"text": "old response"}]},
        {"role": "user", "parts": [{"text": "old context"}]},
    ]
    machine._last_action_count = 0

    machine._handle_action_submitted(stage_tools=[], sub_turn=0)

    # Clear is deferred — check that the deferred action is set
    assert machine._deferred_context_action == ("clear", "action_submitted")


def test_resume_policy_parsed_from_yaml(monkeypatch, tmp_path):
    """Resume policy from YAML config is parsed into the state machine."""
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(
        monkeypatch, tmp_path, llm, log_events,
        resume_policy={"on_level_up": "clear", "on_action_submitted": "compact", "on_context_limit": "compact"},
    )

    assert machine._resume_policy.on_level_up == "clear"
    assert machine._resume_policy.on_action_submitted == "compact"
    assert machine._resume_policy.on_context_limit == "compact"


def test_build_fresh_messages_uses_context_cleared_template(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(
        monkeypatch,
        tmp_path,
        llm,
        log_events,
        reminders={
            "context_cleared": (
                "<system-reminder>\n"
                "History was cleared because {CONTEXT_CLEAR_REASON}.\n"
                "Latest:\n"
                "{LATEST_USER_PROMPT}\n"
                "</system-reminder>"
            )
        },
    )

    fresh = machine._build_fresh_messages("context limit exceeded")

    assert fresh[1]["role"] == "user"
    text = fresh[1]["parts"][0]["text"]
    assert "History was cleared because context limit exceeded." in text
    assert "Latest:\nuser" in text


def test_win_epilogue_uses_configured_template(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([_make_response("I win")])
    machine, _ = _build_state_machine(
        monkeypatch,
        tmp_path,
        llm,
        log_events,
        reminders={
            "win_epilogue": (
                "<game-complete>\n"
                "Won at level {CURRENT_LEVEL} after {TOTAL_ACTIONS_USED}/{MAX_ACTIONS} actions.\n"
                "Do not use tools. Reply exactly `I win`.\n"
                "</game-complete>"
            )
        },
    )
    machine.current_level = 3
    machine.last_observation = {"state": "win"}
    machine.runtime_observation_getter = lambda: {"state": "win"}
    machine._runtime_action_count = lambda: 17

    machine.run_stage_loop_once()

    assert llm.calls
    sent_messages = llm.calls[0]["messages"]
    all_text = []
    for msg in sent_messages:
        for part in msg.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                all_text.append(str(part["text"]))
    joined = "\n".join(all_text)
    assert "Won at level 3 after 17/2048 actions." in joined
    assert "Do not use tools. Reply exactly `I win`." in joined


def test_empty_text_without_tool_calls_retries_once(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([
        _make_response(""),
        _make_response("done"),
    ])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)

    machine.run_stage_loop_once()

    assert len(llm.calls) == 2
    second_call_messages = llm.calls[1]["messages"]
    all_text = []
    for msg in second_call_messages:
        for part in msg.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                all_text.append(str(part["text"]))
    joined = "\n".join(all_text)
    assert "Continue from the current state." in joined
    assert "must call a tool or provide a valid action plan" in joined
    assert machine.last_loop_end_reason() == "assistant_no_tool_calls"


def test_llm_request_budget_limit_stops_stage(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_llm_request_budget_int",
        lambda name, default=0: 1 if name == "max_requests_per_run" else default,
    )
    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_llm_request_budget_str",
        lambda _name, default="": default,
    )
    llm = FakeLLM([
        _make_response(""),
        _make_response("done"),
    ])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)

    machine.run_stage_loop_once()

    assert len(llm.calls) == 1
    assert machine.last_loop_end_reason() == "llm_request_limit_reached"


def test_context_clear_budget_limit_raises(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)
    machine._max_context_clears_per_run = 1
    machine._context_clears_used = 1

    with pytest.raises(RuntimeError, match="CONTEXT_CLEAR_LIMIT_REACHED"):
        machine._apply_missed_memory_checkpoint_fallback(sub_turn=0)


def test_todo_reminder_interval_is_configurable(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([_make_response("first")])
    machine, _ = _build_state_machine(
        monkeypatch,
        tmp_path,
        llm,
        log_events,
        reminders={"todo_reminder_interval": 7},
    )

    assert machine._todo_reminder_interval == 7


def test_workspace_budget_reminder_enabled_by_default(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)

    assert machine._workspace_budget_reminder_enabled is True


def test_workspace_budget_reminder_disabled(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(
        monkeypatch, tmp_path, llm, log_events,
        reminders={"enable_workspace_budget_reminder": "false"},
    )

    assert machine._workspace_budget_reminder_enabled is False


def test_workspace_budget_status_format(tmp_path):
    """workspace_budget_status returns a formatted string with usage info."""
    (tmp_path / "file.txt").write_text("hello world", encoding="utf-8")
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
        workspace_size_limit_bytes=8192,
    )

    status = handler.workspace_budget_status()

    assert "workspace_usage=" in status
    assert "remaining=" in status
    assert "free" in status


def test_workspace_budget_status_unlimited(tmp_path):
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
        workspace_size_limit_bytes=0,
    )

    status = handler.workspace_budget_status()

    assert "unlimited" in status


def test_context_budget_status_with_observed_tokens(monkeypatch, tmp_path):
    """context budget status returns formatted string when tokens are observed."""
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)

    # Simulate having observed prompt tokens
    machine._last_prompt_tokens_observed = 50000

    status = machine._context_budget_status()

    assert "Context usage: 50000/104857 tokens" in status
    assert "48% used" in status
    assert "MEMORY.md" in status


def test_context_budget_status_uses_configured_template(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(
        monkeypatch,
        tmp_path,
        llm,
        log_events,
        reminders={
            "context_budget": (
                "CTX {CONTEXT_OBSERVED_TOKENS}/{CONTEXT_TRIGGER_TOKENS} "
                "remaining={CONTEXT_REMAINING_TOKENS} "
                "threshold={CONTEXT_REMINDER_THRESHOLD_PCT}%"
            )
        },
    )
    machine._last_prompt_tokens_observed = 50000

    status = machine._context_budget_status()

    assert status == "CTX 50000/104857 remaining=54857 threshold=60%"


def test_context_budget_status_empty_when_no_tokens(monkeypatch, tmp_path):
    """context budget status returns empty string when no tokens observed yet."""
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)

    status = machine._context_budget_status()

    assert status == ""


def test_context_budget_reminder_disabled(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(
        monkeypatch, tmp_path, llm, log_events,
        reminders={"enable_context_budget_reminder": "false"},
    )

    assert machine._context_budget_reminder_enabled is False


def test_memory_checkpoint_trigger_replaces_compaction(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)
    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_memory_checkpoint_clear_config",
        lambda: {"enabled": True, "memory_path": "MEMORY.md", "max_grace_turns": 1},
    )
    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_history_log_bool",
        lambda _name, default=False: default,
    )
    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_history_log_str",
        lambda _name, default="": default,
    )
    machine._memory_checkpoint_cfg = machine._build_memory_checkpoint_clear_config()
    machine._compaction_cfg.enabled = True
    machine._compaction_cfg.max_context_tokens = 100
    machine._compaction_cfg.trigger_ratio = 0.8
    machine._last_prompt_tokens_observed = 90

    messages = machine._maybe_compact(
        [
            {"role": "system", "parts": [{"text": "system"}]},
            {"role": "user", "parts": [{"text": "user"}]},
        ],
        [],
    )

    assert machine._memory_checkpoint_pending is True
    assert messages[-1]["role"] == "user"
    assert "Write MEMORY.md now" in messages[-1]["parts"][0]["text"]


def test_missed_memory_checkpoint_clears_context_and_reuses_existing_memory(monkeypatch, tmp_path):
    log_events: list[tuple[str, dict]] = []
    llm = FakeLLM([])
    machine, _ = _build_state_machine(monkeypatch, tmp_path, llm, log_events)
    monkeypatch.setattr(
        "agents.tell_agent.state_machine.get_memory_checkpoint_clear_config",
        lambda: {"enabled": True, "memory_path": "MEMORY.md", "max_grace_turns": 1},
    )
    machine._memory_checkpoint_cfg = machine._build_memory_checkpoint_clear_config()
    machine._messages = [
        {"role": "system", "parts": [{"text": "system"}]},
        {"role": "user", "parts": [{"text": "user"}]},
        {"role": "assistant", "parts": [{"text": "analysis"}]},
    ]
    machine._memory_checkpoint_pending = True
    machine._memory_checkpoint_turns_waited = 2
    machine._memory_checkpoint_deadline_missed = True

    machine._apply_missed_memory_checkpoint_fallback(sub_turn=3)

    assert machine._memory_checkpoint_pending is False
    assert machine._memory_checkpoint_turns_waited == 0
    assert machine._memory_checkpoint_deadline_missed is False
    assert machine._messages[2]["parts"][0]["text"].startswith("[Memory checkpoint clear]")
    assert "continuing from the existing MEMORY.md" in machine._messages[2]["parts"][0]["text"]
    assert any(
        kind == "message"
        and payload.get("message_meta", {}).get("trigger") == "memory_checkpoint_fallback"
        for kind, payload in log_events
    )

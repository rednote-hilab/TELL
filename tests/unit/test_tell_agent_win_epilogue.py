from __future__ import annotations

import threading

from arcengine import FrameData, GameState

from agents.tell_agent.agent import TELLAgent


class _FakeStateMachine:
    def __init__(self) -> None:
        self.loop_end_reason = ""

    def last_loop_end_reason(self) -> str:
        return self.loop_end_reason


def test_is_done_waits_for_win_epilogue_before_exiting():
    agent = TELLAgent.__new__(TELLAgent)
    agent._worker_stop = threading.Event()
    agent._worker_started = False
    agent._worker_thread = None
    agent._worker_error = None
    agent._win_stop_requested = False
    agent._last_action_loop_id = ""
    agent.action_counter = 17
    agent.worker_poll_interval = 0.01
    agent.state_machine = _FakeStateMachine()
    agent._runtime_service = type("RuntimeSvc", (), {"update_observation": lambda self, obs: None})()
    synced = {"obs": None}

    log_events: list[tuple[str, dict]] = []

    def _log_event(kind: str, payload: dict) -> None:
        log_events.append((kind, payload))

    agent._log_event = _log_event
    agent._build_observation = lambda frame: {"state": "WIN", "levels_completed": frame.levels_completed, "state_id": f"level_{int(frame.levels_completed):04d}"}

    def _ingest_observation(obs) -> None:
        synced["obs"] = obs

    agent.state_machine.ingest_observation = _ingest_observation

    def _ensure_worker_started() -> None:
        if agent._worker_started:
            return
        agent._worker_started = True

        def _complete_epilogue() -> None:
            agent.state_machine.loop_end_reason = "win_after_epilogue"
            agent._worker_stop.set()

        worker = threading.Thread(target=_complete_epilogue, daemon=True)
        agent._worker_thread = worker
        worker.start()

    agent._ensure_worker_started = _ensure_worker_started

    latest_frame = FrameData(state=GameState.WIN, levels_completed=5, score=0)

    assert agent.is_done([latest_frame], latest_frame) is True
    assert agent._worker_started is True
    assert agent._win_stop_requested is True
    assert agent._worker_stop.is_set() is True
    assert synced["obs"] == {"state": "WIN", "levels_completed": 5, "state_id": "level_0005"}
    assert any(
        payload.get("message_id") == "win_graceful_stop_17"
        for kind, payload in log_events
        if kind == "message"
    )


def test_is_done_returns_immediately_when_win_epilogue_already_complete():
    agent = TELLAgent.__new__(TELLAgent)
    agent._worker_stop = threading.Event()
    agent._worker_started = False
    agent._worker_thread = None
    agent._worker_error = None
    agent._win_stop_requested = True
    agent._worker_stop.set()
    agent._last_action_loop_id = ""
    agent.action_counter = 3
    agent.worker_poll_interval = 0.01
    agent.state_machine = _FakeStateMachine()
    agent.state_machine.loop_end_reason = "assistant_no_tool_calls"
    agent._runtime_service = type("RuntimeSvc", (), {"update_observation": lambda self, obs: None})()
    agent._log_event = lambda *_args, **_kwargs: None
    agent._build_observation = lambda frame: {"state": "WIN", "levels_completed": frame.levels_completed, "state_id": f"level_{int(frame.levels_completed):04d}"}

    ensure_calls = {"count": 0}

    def _ensure_worker_started() -> None:
        ensure_calls["count"] += 1

    agent._ensure_worker_started = _ensure_worker_started

    latest_frame = FrameData(state=GameState.WIN, levels_completed=2, score=0)

    assert agent.is_done([latest_frame], latest_frame) is True
    assert ensure_calls["count"] == 0


def test_is_done_does_not_treat_stale_no_tool_reply_as_completed_epilogue():
    agent = TELLAgent.__new__(TELLAgent)
    agent._worker_stop = threading.Event()
    agent._worker_started = False
    agent._worker_thread = None
    agent._worker_error = None
    agent._win_stop_requested = False
    agent._last_action_loop_id = ""
    agent.action_counter = 9
    agent.worker_poll_interval = 0.01
    agent.state_machine = _FakeStateMachine()
    agent.state_machine.loop_end_reason = "assistant_no_tool_calls"
    agent._runtime_service = type("RuntimeSvc", (), {"update_observation": lambda self, obs: None})()
    agent._log_event = lambda *_args, **_kwargs: None
    agent._build_observation = lambda frame: {"state": "WIN", "levels_completed": frame.levels_completed, "state_id": f"level_{int(frame.levels_completed):04d}"}

    started = {"count": 0}

    def _ensure_worker_started() -> None:
        if agent._worker_started:
            return
        agent._worker_started = True
        started["count"] += 1

        def _complete_epilogue() -> None:
            agent.state_machine.loop_end_reason = "win_after_epilogue"
            agent._worker_stop.set()

        worker = threading.Thread(target=_complete_epilogue, daemon=True)
        agent._worker_thread = worker
        worker.start()

    agent._ensure_worker_started = _ensure_worker_started

    latest_frame = FrameData(state=GameState.WIN, levels_completed=4, score=0)

    assert agent.is_done([latest_frame], latest_frame) is True
    assert started["count"] == 1


def test_is_done_syncs_latest_win_observation_before_waiting():
    agent = TELLAgent.__new__(TELLAgent)
    agent._worker_stop = threading.Event()
    agent._worker_started = False
    agent._worker_thread = None
    agent._worker_error = None
    agent._win_stop_requested = False
    agent._last_action_loop_id = ""
    agent.action_counter = 21
    agent.worker_poll_interval = 0.01
    agent.state_machine = _FakeStateMachine()

    synced = {"ingest": None, "runtime": None}

    agent._build_observation = lambda frame: {
        "state": "WIN",
        "levels_completed": frame.levels_completed,
        "state_id": f"level_{int(frame.levels_completed):04d}",
    }

    def _ingest_observation(obs) -> None:
        synced["ingest"] = obs

    agent.state_machine.ingest_observation = _ingest_observation

    class _RuntimeSvc:
        def update_observation(self, obs) -> None:
            synced["runtime"] = obs

    agent._runtime_service = _RuntimeSvc()
    agent._log_event = lambda *_args, **_kwargs: None

    def _ensure_worker_started() -> None:
        if agent._worker_started:
            return
        agent._worker_started = True

        def _complete_epilogue() -> None:
            agent.state_machine.loop_end_reason = "win_after_epilogue"
            agent._worker_stop.set()

        worker = threading.Thread(target=_complete_epilogue, daemon=True)
        agent._worker_thread = worker
        worker.start()

    agent._ensure_worker_started = _ensure_worker_started

    latest_frame = FrameData(state=GameState.WIN, levels_completed=7, score=0)

    assert agent.is_done([latest_frame], latest_frame) is True
    expected = {"state": "WIN", "levels_completed": 7, "state_id": "level_0007"}
    assert synced["ingest"] == expected
    assert synced["runtime"] == expected


def test_worker_loop_stops_after_non_win_assistant_no_tool_calls():
    agent = TELLAgent.__new__(TELLAgent)
    agent._worker_stop = threading.Event()
    agent._worker_error = None
    agent._last_action_loop_id = ""
    agent._last_action_state_id = "level_0006"
    agent.action_counter = 1024
    agent.levels_completed = 6
    agent.worker_poll_interval = 0.01
    agent.stage_loop_limit = 128

    class _RuntimeSvc:
        def has_action_backlog(self) -> bool:
            return False

    class _StateMachine:
        def __init__(self) -> None:
            self.calls = 0

        def run_stage_loop_once(self) -> None:
            self.calls += 1

        def last_loop_end_reason(self) -> str:
            return "assistant_no_tool_calls"

    logs: list[tuple[str, dict]] = []

    def _log_event(kind: str, payload: dict) -> None:
        logs.append((kind, payload))

    agent._runtime_service = _RuntimeSvc()
    agent.state_machine = _StateMachine()
    agent._log_event = _log_event
    agent._wait_for_runtime_observation_ready = lambda timeout: True
    agent._runtime_is_win = lambda: False

    agent._worker_loop()

    assert agent._worker_stop.is_set() is True
    assert agent.state_machine.calls == 1
    assert any(
        payload.get("message_id") == "worker_stage_complete_1024"
        for kind, payload in logs
        if kind == "message"
    )


def test_memory_checkpoint_not_written_does_not_stop_worker():
    agent = TELLAgent.__new__(TELLAgent)
    agent.state_machine = _FakeStateMachine()
    agent.state_machine.loop_end_reason = "memory_checkpoint_not_written"
    agent._runtime_is_win = lambda: False

    assert agent._should_stop_worker_after_stage() == (
        False,
        "memory_checkpoint_not_written",
    )

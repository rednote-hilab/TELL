from __future__ import annotations

import json
import socket
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


def pick_available_port(host: str, preferred: int, max_tries: int = 32) -> int:
    for i in range(max_tries):
        cand = preferred + i
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, cand))
            sock.close()
            return cand
        except OSError:
            sock.close()
            continue
    raise RuntimeError(f"Failed to allocate runtime port from {preferred}")


class RuntimeGameService:
    def __init__(
        self,
        game_id: str,
        host: str,
        port: int,
        render_scale: int = 2,
        workspace: Optional[Path] = None,
    ) -> None:
        self.game_id = game_id
        self.host = host
        self.port = port
        self.render_scale = max(1, int(render_scale))
        self.workspace = (workspace.resolve() if isinstance(workspace, Path) else Path.cwd().resolve())
        self.owner = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._observation: Dict[str, Any] = {}
        self._pending_action: Optional[Dict[str, Any]] = None
        self._inflight_action: Optional[Dict[str, Any]] = None
        self._planned_actions: List[Dict[str, Any]] = []
        self._action_frame_seq: int = 0
        self._step: int = -1
        self._observation_history: list[Dict[str, Any]] = []
        self._action_history: list[Dict[str, Any]] = []
        self._observations_api: list[Dict[str, Any]] = []
        self._current_level: Optional[int] = None
        self._current_level_has_action: bool = False
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._action_context_provider: Optional[Callable[[], Dict[str, str]]] = None
        self._runtime_observation_logger: Optional[Callable[[Dict[str, Any]], None]] = None
        self._runtime_action_frame_logger: Optional[Callable[[Dict[str, Any]], None]] = None

    def _available_actions_locked(self) -> List[str]:
        available_actions_raw = self._observation.get("available_actions") or []
        if not isinstance(available_actions_raw, list):
            available_actions_raw = []
        available = [str(a).strip().lower() for a in available_actions_raw if str(a).strip()]
        if "reset" not in available:
            available.append("reset")
        return available

    def _current_state_locked(self) -> str:
        return str(self._observation.get("state") or "").strip().upper()

    def _validate_action_locked(self, action: str, x: Any = None, y: Any = None) -> Dict[str, Any]:
        action_norm = str(action or "").strip().lower()
        available_actions = self._available_actions_locked()
        if not action_norm:
            return {"ok": False, "error": "missing_action", "available_actions": available_actions}
        if action_norm not in available_actions:
            return {
                "ok": False,
                "error": "invalid_action",
                "action": action_norm,
                "available_actions": available_actions,
            }
        if action_norm == "click":
            if x is None or y is None:
                return {
                    "ok": False,
                    "error": "invalid_action_args",
                    "action": action_norm,
                    "detail": "click requires x and y",
                    "available_actions": available_actions,
                }
            try:
                x_int = int(x)
                y_int = int(y)
            except Exception:
                return {
                    "ok": False,
                    "error": "invalid_action_args",
                    "action": action_norm,
                    "detail": "click x/y must be integers",
                    "available_actions": available_actions,
                }
            if not (0 <= x_int <= 63 and 0 <= y_int <= 63):
                return {
                    "ok": False,
                    "error": "invalid_action_args",
                    "action": action_norm,
                    "detail": "click x/y must be within 0..63",
                    "available_actions": available_actions,
                }
        elif x is not None or y is not None:
            return {
                "ok": False,
                "error": "invalid_action_args",
                "action": action_norm,
                "detail": "x/y are only valid for click",
                "available_actions": available_actions,
            }
        return {"ok": True, "action": action_norm, "available_actions": available_actions}

    def _validate_runtime_action_locked(self, action: str, x: Any = None, y: Any = None) -> Dict[str, Any]:
        validation = self._validate_action_locked(action, x, y)
        if not validation.get("ok", False):
            return validation
        current_state = self._current_state_locked()
        if current_state in {"GAME_OVER", "WIN"} and validation.get("action") != "reset":
            error_name = "action_not_allowed_in_game_over"
            if current_state == "WIN":
                error_name = "action_not_allowed_in_win"
            return {
                "ok": False,
                "error": error_name,
                "action": str(validation.get("action") or ""),
                "state": current_state.lower(),
                "allowed_actions": ["reset"],
                "available_actions": list(validation.get("available_actions") or []),
            }
        return validation

    def record_action_error(
        self,
        *,
        action: str,
        x: Any = None,
        y: Any = None,
        error: str,
        source: str = "runtime",
        message_id: str = "",
        state_id: str = "",
        loop_id: str = "",
        session_id: str = "sess_0001",
    ) -> None:
        action_frame_event: Optional[Dict[str, Any]] = None
        action_frame_logger: Optional[Callable[[Dict[str, Any]], None]] = None
        with self._cv:
            step = int(self._step)
            available_actions = self._available_actions_locked()
            self._action_history.append(
                {
                    "kind": "action_error",
                    "step": step,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "action": str(action).strip().lower(),
                    "x": x,
                    "y": y,
                    "source": source,
                    "message_id": message_id,
                    "state_id": state_id,
                    "loop_id": loop_id,
                    "error": error,
                    "available_actions": list(available_actions),
                }
            )
            latest = self._observations_api[-1] if self._observations_api else {}
            if latest:
                row = dict(latest)
                row["observation_index"] = len(self._observations_api)
                row["last_action"] = str(action).strip().lower()
                if x is not None:
                    row["last_action"] += f" x={x}"
                if y is not None:
                    row["last_action"] += f" y={y}"
                row["action_error"] = error
                row["available_actions"] = list(available_actions)
                self._observations_api.append(row)
            self._planned_actions = []
            self._pending_action = None
            self._inflight_action = None
            action_frame_logger = self._runtime_action_frame_logger
            self._action_frame_seq += 1
            action_args: Dict[str, Any] = {}
            if x is not None:
                action_args["x"] = x
            if y is not None:
                action_args["y"] = y
            action_frame_event = {
                "af_id": f"af_{self._action_frame_seq:06d}",
                "session_id": session_id or "sess_0001",
                "state_id": state_id or f"level_{int(self._observation.get('levels_completed') or 0):04d}",
                "loop_id": loop_id,
                "source": source,
                "message_id": message_id,
                "action": {"name": str(action).strip().lower(), "args": action_args},
                "result": {"status": "error", "error": error, "observation": dict(self._observation or {})},
            }
            self._cv.notify_all()
        if action_frame_event is not None and action_frame_logger is not None:
            try:
                action_frame_logger(dict(action_frame_event))
            except Exception:
                pass

    def set_action_context_provider(self, provider: Optional[Callable[[], Dict[str, str]]]) -> None:
        with self._lock:
            self._action_context_provider = provider

    def set_runtime_observation_logger(self, logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        with self._lock:
            self._runtime_observation_logger = logger

    def set_runtime_action_frame_logger(self, logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        with self._lock:
            self._runtime_action_frame_logger = logger

    def start(self) -> None:
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query, keep_blank_values=True)
                if path.startswith("/health"):
                    self._send_json(200, {"ok": True, "mode": "runtime", "game_id": svc.game_id})
                    return
                if path == "/observations":
                    index_expr = (query.get("index", ["-1"])[0] or "-1").strip()
                    with svc._lock:
                        rows = [dict(x) for x in svc._observations_api]
                    selected = svc._select_observations(rows, index_expr)
                    if selected is None:
                        self._send_json(400, {"ok": False, "error": "invalid_index"})
                        return
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "observations": selected,
                        },
                    )
                    return
                self._send_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.startswith("/action"):
                    self._send_json(404, {"ok": False, "error": "not_found"})
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._send_json(400, {"ok": False, "error": "invalid_json"})
                    return
                if not isinstance(payload, dict):
                    self._send_json(400, {"ok": False, "error": "invalid_payload"})
                    return
                allowed_fields = {
                    "action",
                    "x",
                    "y",
                    "step",
                    "timeout_sec",
                    "message_id",
                    "state_id",
                    "loop_id",
                }
                unknown_fields = sorted(str(k) for k in payload.keys() if str(k) not in allowed_fields)
                if unknown_fields:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "invalid_payload_fields",
                            "fields": unknown_fields,
                        },
                    )
                    return
                action = str(payload.get("action", "")).strip().lower()
                if not action:
                    self._send_json(400, {"ok": False, "error": "missing_action"})
                    return
                with svc._lock:
                    level_has_action = bool(svc._current_level_has_action)
                    current_state = svc._current_state_locked()
                    validation = svc._validate_runtime_action_locked(action, payload.get("x"), payload.get("y"))
                    available_actions = list(validation.get("available_actions") or [])
                if not validation.get("ok", False):
                    self._send_json(
                        409 if str(validation.get("error") or "").startswith("action_not_allowed_in_") else 400,
                        dict(validation),
                    )
                    return
                if action == "reset" and not level_has_action and current_state not in {"GAME_OVER", "WIN"}:
                    current_step = svc.current_step()
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "ignored": True,
                            "reason": "reset_ignored_at_level_start",
                            "step": current_step,
                            "next_step": current_step,
                            "available_actions": available_actions,
                            "accepted": {
                                "action": action,
                                "x": payload.get("x"),
                                "y": payload.get("y"),
                            },
                        },
                    )
                    return
                step_raw = payload.get("step")
                step_arg: Optional[int] = None
                if step_raw is not None:
                    try:
                        step_arg = int(step_raw)
                    except Exception:
                        self._send_json(400, {"ok": False, "error": "invalid_step"})
                        return
                timeout_raw = payload.get("timeout_sec", 40)
                try:
                    timeout_sec = max(0.5, min(120.0, float(timeout_raw)))
                except Exception:
                    self._send_json(400, {"ok": False, "error": "invalid_timeout_sec"})
                    return
                accepted_step = svc.current_step()
                accepted = svc.submit_action(
                    action=action,
                    x=payload.get("x"),
                    y=payload.get("y"),
                    step=step_arg,
                    source="api",
                    message_id=str(payload.get("message_id") or ""),
                    state_id=str(payload.get("state_id") or ""),
                    loop_id=str(payload.get("loop_id") or ""),
                    enqueue_timeout=timeout_sec,
                )
                if not accepted:
                    self._send_json(
                        409,
                        {
                            "ok": False,
                            "error": "action_not_enqueued",
                            "expected_step": svc.current_step(),
                        },
                    )
                    return
                next_step = svc.wait_for_observation_after(step=accepted_step, timeout=timeout_sec)
                if next_step is None:
                    self._send_json(
                        504,
                        {
                            "ok": False,
                            "error": "action_wait_timeout",
                            "step": accepted_step,
                            "accepted": {
                                "action": action,
                                "x": payload.get("x"),
                                "y": payload.get("y"),
                            },
                        },
                    )
                    return
                out: Dict[str, Any] = {
                    "ok": True,
                    "step": accepted_step,
                    "next_step": int(next_step),
                    "available_actions": available_actions,
                    "accepted": {
                        "action": action,
                        "x": payload.get("x"),
                        "y": payload.get("y"),
                    },
                }
                self._send_json(200, out)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def update_observation(self, observation: Dict[str, Any]) -> None:
        event: Dict[str, Any]
        action_frame_event: Optional[Dict[str, Any]] = None
        runtime_logger: Optional[Callable[[Dict[str, Any]], None]]
        action_frame_logger: Optional[Callable[[Dict[str, Any]], None]]
        with self._cv:
            obs_state_id = str(observation.get("state_id") or "").strip()
            if not obs_state_id:
                obs_state_id = f"level_{int(observation.get('levels_completed') or 0):04d}"
            obs_level = int(observation.get("levels_completed") or 0)
            obs_state = str(observation.get("state") or "").strip().upper()
            ctx: Dict[str, str] = {}
            if self._action_context_provider is not None:
                try:
                    ctx = self._action_context_provider() or {}
                except Exception:
                    ctx = {}
            self._step += 1
            self._observation = dict(observation)
            self._pending_action = None
            prev_level = self._current_level
            if prev_level is None or obs_level != prev_level:
                self._current_level = obs_level
                self._current_level_has_action = False
                self._planned_actions = []
            if obs_state in {"GAME_OVER", "WIN"}:
                self._planned_actions = []
            level_changed = prev_level is not None and obs_level != prev_level
            state_id = str(ctx.get("state_id") or obs_state_id)
            event = {
                "kind": "observation",
                "step": self._step,
                "at": datetime.now(timezone.utc).isoformat(),
                "session_id": str(ctx.get("session_id") or "sess_0001"),
                "state_id": state_id,
                "loop_id": str(ctx.get("loop_id") or ""),
                "message_id": str(ctx.get("message_id") or ""),
                "source": "runtime",
                "observation": dict(observation),
            }
            self._observation_history.append(dict(event))
            runtime_logger = self._runtime_observation_logger
            action_frame_logger = self._runtime_action_frame_logger
            inflight = self._inflight_action
            if inflight is not None:
                self._action_frame_seq += 1
                action_args: Dict[str, Any] = {}
                if inflight.get("x") is not None:
                    action_args["x"] = inflight.get("x")
                if inflight.get("y") is not None:
                    action_args["y"] = inflight.get("y")
                frame_observation = dict(observation)
                frame_observation["step"] = self._step
                action_frame_event = {
                    "af_id": f"af_{self._action_frame_seq:06d}",
                    "session_id": str(inflight.get("session_id") or event["session_id"]),
                    # action_frame state_id must reflect the resulting observation level,
                    # not the action submission context.
                    "state_id": obs_state_id,
                    "loop_id": str(inflight.get("loop_id") or event["loop_id"]),
                    "source": str(inflight.get("source") or "runtime"),
                    "message_id": str(inflight.get("message_id") or ""),
                    "action": {"name": str(inflight.get("action") or ""), "args": action_args},
                    "result": {"status": "ok", "error": "", "observation": frame_observation},
                }
                if not level_changed:
                    inflight_action_name = str(inflight.get("action") or "").strip().lower()
                    if inflight_action_name == "reset":
                        # A successful reset returns to level start — clear the
                        # flag so the next reset (without intervening actions)
                        # will be blocked instead of causing a full game reset.
                        self._current_level_has_action = False
                    else:
                        self._current_level_has_action = True
                self._inflight_action = None
            self._cv.notify_all()
        self._append_observations_api(observation=dict(observation), inflight=inflight)
        if runtime_logger is not None:
            try:
                runtime_logger(dict(event))
            except Exception:
                pass
        if action_frame_event is not None and action_frame_logger is not None:
            try:
                action_frame_logger(dict(action_frame_event))
            except Exception:
                pass

    def current_step(self) -> int:
        with self._lock:
            return self._step

    def get_observation_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._observation)

    def get_action_history_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(x) for x in self._action_history]

    def has_action_backlog(self) -> bool:
        with self._lock:
            return bool(self._pending_action is not None or self._inflight_action is not None or self._planned_actions)

    def enqueue_action_plan(
        self,
        plan: List[Dict[str, Any]],
        *,
        source: str = "plan",
        message_id: str = "",
        state_id: str = "",
        loop_id: str = "",
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        queued: List[Dict[str, Any]] = []
        skipped = 0
        with self._cv:
            session_id = "sess_0001"
            if (not message_id or not state_id or not loop_id) and self._action_context_provider is not None:
                try:
                    ctx = self._action_context_provider() or {}
                except Exception:
                    ctx = {}
                session_id = str(ctx.get("session_id") or session_id)
                if not message_id:
                    message_id = str(ctx.get("message_id") or "")
                if not state_id:
                    state_id = str(ctx.get("state_id") or "")
                if not loop_id:
                    loop_id = str(ctx.get("loop_id") or "")
            if replace_existing:
                self._planned_actions = []
            available_actions = self._available_actions_locked()
            for item in plan:
                if not isinstance(item, dict):
                    skipped += 1
                    continue
                action = str(item.get("action") or item.get("name") or "").strip().lower()
                if not action:
                    skipped += 1
                    continue
                validation = self._validate_runtime_action_locked(action, item.get("x"), item.get("y"))
                if not validation.get("ok", False):
                    skipped += 1
                    err = dict(validation)
                    err.setdefault("action", action)
                    err.setdefault("available_actions", list(available_actions))
                    queued.append({"error": err})
                    continue
                queued_item = {
                    "action": action,
                    "x": item.get("x"),
                    "y": item.get("y"),
                    "source": source,
                    "message_id": message_id,
                    "state_id": state_id,
                    "loop_id": loop_id,
                    "session_id": session_id,
                }
                self._planned_actions.append(queued_item)
                queued.append(dict(queued_item))
            self._cv.notify_all()
            errors = [dict(x.get("error") or {}) for x in queued if isinstance(x, dict) and "error" in x]
            return {
                "queued": len([x for x in queued if isinstance(x, dict) and "error" not in x]),
                "skipped": skipped,
                "pending_queue_size": len(self._planned_actions),
                "errors": errors,
            }

    def wait_for_observation_after(self, step: int, timeout: float) -> Optional[int]:
        with self._cv:
            if self._step > step:
                return self._step
            ok = self._cv.wait_for(lambda: self._step > step, timeout=timeout)
            if not ok:
                return None
            return self._step

    def submit_action(
        self,
        action: str,
        x: Any = None,
        y: Any = None,
        step: Optional[int] = None,
        source: str = "worker",
        message_id: str = "",
        state_id: str = "",
        loop_id: str = "",
        enqueue_timeout: float = 0.0,
    ) -> bool:
        with self._cv:
            session_id = "sess_0001"
            if (not message_id or not state_id or not loop_id) and self._action_context_provider is not None:
                try:
                    ctx = self._action_context_provider() or {}
                except Exception:
                    ctx = {}
                session_id = str(ctx.get("session_id") or session_id)
                if not message_id:
                    message_id = str(ctx.get("message_id") or "")
                if not state_id:
                    state_id = str(ctx.get("state_id") or "")
                if not loop_id:
                    loop_id = str(ctx.get("loop_id") or "")
            if step is not None and step != self._step:
                return False
            # If another request already occupies this step, wait for an enqueue window
            # (or step transition) instead of reporting false success.
            if self._pending_action is not None and int(self._pending_action.get("step", -1)) == self._step:
                wait_sec = max(0.0, float(enqueue_timeout))
                if wait_sec <= 0.0:
                    return False
                ok = self._cv.wait_for(
                    lambda: self._pending_action is None or int(self._pending_action.get("step", -1)) != self._step,
                    timeout=wait_sec,
                )
                if not ok:
                    return False
            if step is not None and step != self._step:
                return False
            if self._pending_action is not None and int(self._pending_action.get("step", -1)) == self._step:
                return False
            self._pending_action = {
                "action": str(action).strip().lower(),
                "x": x,
                "y": y,
                "step": self._step,
                "source": source,
                "message_id": message_id,
                "state_id": state_id,
                "loop_id": loop_id,
                "session_id": session_id,
            }
            self._action_history.append(
                {
                    "kind": "action",
                    "step": self._step,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "action": str(action).strip().lower(),
                    "x": x,
                    "y": y,
                    "source": source,
                    "message_id": message_id,
                    "state_id": state_id,
                    "loop_id": loop_id,
                }
            )
            self._cv.notify_all()
            return True

    def wait_for_action(self, step: int, timeout: float) -> Optional[Dict[str, Any]]:
        with self._cv:
            if self._pending_action is not None and int(self._pending_action.get("step", -1)) == step:
                out = dict(self._pending_action)
                out["session_id"] = str(out.get("session_id") or "sess_0001")
                self._inflight_action = dict(out)
                self._pending_action = None
                return out
            while self._planned_actions:
                queued = dict(self._planned_actions.pop(0))
                action_name = str(queued.get("action") or "").strip().lower()
                queued["step"] = step
                queued["session_id"] = str(queued.get("session_id") or "sess_0001")
                self._action_history.append(
                    {
                        "kind": "action",
                        "step": step,
                        "at": datetime.now(timezone.utc).isoformat(),
                        "action": action_name,
                        "x": queued.get("x"),
                        "y": queued.get("y"),
                        "source": str(queued.get("source") or "plan"),
                        "message_id": str(queued.get("message_id") or ""),
                        "state_id": str(queued.get("state_id") or ""),
                        "loop_id": str(queued.get("loop_id") or ""),
                    }
                )
                self._inflight_action = dict(queued)
                return queued
            ok = self._cv.wait_for(
                lambda: (
                    self._pending_action is not None and int(self._pending_action.get("step", -1)) == step
                ) or bool(self._planned_actions),
                timeout=timeout,
            )
            if not ok or self._pending_action is None:
                if not ok:
                    return None
                if self._pending_action is None:
                    while self._planned_actions:
                        queued = dict(self._planned_actions.pop(0))
                        action_name = str(queued.get("action") or "").strip().lower()
                        queued["step"] = step
                        queued["session_id"] = str(queued.get("session_id") or "sess_0001")
                        self._action_history.append(
                            {
                                "kind": "action",
                                "step": step,
                                "at": datetime.now(timezone.utc).isoformat(),
                                "action": action_name,
                                "x": queued.get("x"),
                                "y": queued.get("y"),
                                "source": str(queued.get("source") or "plan"),
                                "message_id": str(queued.get("message_id") or ""),
                                "state_id": str(queued.get("state_id") or ""),
                                "loop_id": str(queued.get("loop_id") or ""),
                            }
                        )
                        self._inflight_action = dict(queued)
                        return queued
                    return None
            out = dict(self._pending_action)
            out["session_id"] = str(out.get("session_id") or "sess_0001")
            self._inflight_action = dict(out)
            self._pending_action = None
            return out

    def _format_observation_payload(self, obs: Dict[str, Any], step: int, fmt: str) -> Dict[str, Any]:
        out = dict(obs)
        out.pop("frames", None)
        summary = str(out.get("summary") or "").strip()
        if summary and "grid" in summary:
            out["summary"] = summary.replace("grid", "observation")
        return out

    def _append_observations_api(self, observation: Dict[str, Any], inflight: Optional[Dict[str, Any]]) -> None:
        frames = self._extract_grid_frames(self._observation_frames_input(observation))
        if not frames:
            return
        ascii_frames = [self._grid_to_ascii(frame) for frame in frames]
        levels_completed = int(observation.get("levels_completed") or 0)
        available_actions_raw = observation.get("available_actions") or []
        if not isinstance(available_actions_raw, list):
            available_actions_raw = []
        available_actions = [str(a) for a in available_actions_raw if str(a).strip()]
        step = int(self._step)
        action_value = "INIT" if step <= 0 else "unknown"
        x = None
        y = None
        if isinstance(inflight, dict):
            action_value = str(inflight.get("action") or "unknown")
            x = inflight.get("x")
            y = inflight.get("y")
        if x is not None:
            action_value += f" x={x}"
        if y is not None:
            action_value += f" y={y}"
        state = str(observation.get("state") or "").strip().lower()
        with self._lock:
            obs_index = len(self._observations_api)
            self._observations_api.append(
                {
                    "observation_index": obs_index,
                    "total_actions": obs_index,
                    "last_action": action_value,
                    "state": state,
                    "current_level": levels_completed,
                    "available_actions": available_actions,
                    "frames": ascii_frames,
                    "frame_count": len(ascii_frames),
                }
            )

    def _select_observations(self, rows: List[Dict[str, Any]], index_expr: str) -> Any:
        """Python indexing semantics: single index -> dict, slice -> list."""
        expr = str(index_expr or "").strip()
        if not expr:
            expr = "-1"
        try:
            if ":" in expr:
                parts = expr.split(":")
                if len(parts) != 2:
                    return None
                start_s, stop_s = parts
                start = int(start_s) if start_s.strip() else None
                stop = int(stop_s) if stop_s.strip() else None
                return [dict(x) for x in rows[slice(start, stop)]]
            idx = int(expr)
        except Exception:
            return None
        if not rows:
            return {}
        if -len(rows) <= idx < len(rows):
            return dict(rows[idx])
        return {}

    def _observation_frames_input(self, obs: Dict[str, Any]) -> Any:
        # Internal canonical shape uses observation.frames[].grid.
        frames_any = obs.get("frames")
        if isinstance(frames_any, list):
            return frames_any
        return []

    def _extract_grid_frames(self, grid_any: Any) -> List[List[List[int]]]:
        if not isinstance(grid_any, list) or not grid_any:
            return []
        # Canonical shape: [{"frame_index": 0, "grid": [[...]]}, ...]
        if isinstance(grid_any[0], dict):
            out_from_dicts: List[List[List[int]]] = []
            for frame_item in grid_any:
                if not isinstance(frame_item, dict):
                    continue
                frame_grid = frame_item.get("grid")
                if (
                    isinstance(frame_grid, list)
                    and frame_grid
                    and isinstance(frame_grid[0], list)
                    and frame_grid[0]
                    and isinstance(frame_grid[0][0], int)
                ):
                    out_from_dicts.append(frame_grid)
            return out_from_dicts
        first = grid_any[0]
        if isinstance(first, list) and first and isinstance(first[0], int):
            return [grid_any]
        if isinstance(first, list) and first and isinstance(first[0], list):
            out: List[List[List[int]]] = []
            for frame in grid_any:
                if (
                    isinstance(frame, list)
                    and frame
                    and isinstance(frame[0], list)
                    and frame[0]
                    and isinstance(frame[0][0], int)
                ):
                    out.append(frame)
            return out
        return []

    def _color_to_ascii(self) -> List[str]:
        # 0..15 -> unique single-char tokens.
        # Uniqueness is required so text-only board parsing does not lose color identity.
        return [
            "W",  # 0 white
            "w",  # 1 off-white
            ".",  # 2 light-gray
            ":",  # 3 gray
            "d",  # 4 dark-gray
            "K",  # 5 black
            "M",  # 6 magenta
            "p",  # 7 pink
            "R",  # 8 red
            "B",  # 9 blue
            "b",  # 10 light-blue
            "Y",  # 11 yellow
            "O",  # 12 orange
            "n",  # 13 maroon
            "G",  # 14 green
            "u",  # 15 purple
        ]

    def _grid_to_ascii(self, grid: List[List[int]]) -> str:
        chars = self._color_to_ascii()
        lines: List[str] = []
        for row in grid:
            row_chars: List[str] = []
            for raw in row:
                v = int(raw) if isinstance(raw, int) else 0
                if v < 0 or v >= len(chars):
                    v = 0
                row_chars.append(chars[v])
            lines.append("".join(row_chars))
        return "\n".join(lines)

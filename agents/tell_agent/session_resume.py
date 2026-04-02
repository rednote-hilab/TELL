"""Session resume: rebuild agent context and game state from log files.

Usage:
    from .session_resume import load_resume_state, ResumeState

    state = load_resume_state(log_dir)
    # state.messages  -> List[Dict] for LLM context restoration
    # state.actions   -> List[Dict] for game environment replay
    # state.workspace_files -> Dict[path, bytes] for workspace restoration
    # state.manifest  -> Dict with run metadata
    # state.last_observation -> Dict with latest game state
    # state.stage_name -> str, which stage to resume from
    # state.loop_counter -> int, loop counter to resume from
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResumeState:
    """Everything needed to resume an interrupted agent session."""

    # LLM conversation messages to restore into state_machine context.
    messages: List[Dict[str, Any]] = field(default_factory=list)
    # Game actions to replay to bring environment to the same state.
    actions: List[Dict[str, Any]] = field(default_factory=list)
    # Workspace files to restore: {relative_path: bytes_content}.
    workspace_files: Dict[str, bytes] = field(default_factory=dict)
    # Run manifest metadata.
    manifest: Dict[str, Any] = field(default_factory=dict)
    # Latest runtime observation for state_machine.last_observation.
    last_observation: Dict[str, Any] = field(default_factory=dict)
    # Stage and loop to resume from.
    stage_name: str = ""
    loop_counter: int = 0
    # Last compaction summary if available.
    last_compaction_summary: str = ""
    # Stats from previous run.
    stats: Dict[str, Any] = field(default_factory=dict)
    # Whether resume data was successfully loaded.
    valid: bool = False
    # Error message if loading failed.
    error: str = ""


def load_resume_state(log_dir: str | Path) -> ResumeState:
    """Load resume state from a log directory (replays/<run_id>/).

    Returns a ResumeState with valid=True if enough data exists to resume,
    or valid=False with an error message otherwise.
    """
    log_path = Path(log_dir).resolve()
    state = ResumeState()

    # 1. Load manifest
    manifest_path = log_path / "manifest.json"
    if not manifest_path.exists():
        state.error = f"manifest.json not found in {log_path}"
        return state
    try:
        state.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        state.error = f"failed to read manifest.json: {exc}"
        return state

    # 2. Load run stats
    stats_path = log_path / "run_stats.json"
    if stats_path.exists():
        try:
            state.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 3. Rebuild LLM messages from messages.jsonl
    messages_path = log_path / "messages.jsonl"
    if messages_path.exists():
        state.messages = _load_messages(messages_path)

    # 4. Extract stage and loop info from messages
    _extract_stage_info(state)

    # 5. Load game actions for replay from action_frames.jsonl
    af_path = log_path / "action_frames.jsonl"
    if af_path.exists():
        state.actions = _load_action_frames(af_path)

    # 6. Load latest runtime observation
    obs_path = log_path / "runtime_observations.jsonl"
    if obs_path.exists():
        state.last_observation = _load_last_observation(obs_path)

    # 7. Restore workspace from fs_versions.jsonl
    fs_path = log_path / "fs_versions.jsonl"
    if fs_path.exists():
        state.workspace_files = _load_workspace_state(fs_path)

    # 8. Extract last compaction summary if present
    state.last_compaction_summary = _extract_last_compaction_summary(state.messages)

    # Validate: we need at least messages to resume
    if state.messages:
        state.valid = True
    else:
        state.error = "no messages found in log"

    return state


def _load_messages(path: Path) -> List[Dict[str, Any]]:
    """Load and reconstruct LLM conversation messages from messages.jsonl.

    The log format stores individual message records; we rebuild the
    conversation as a list of {role, parts} dicts suitable for feeding
    back into the state machine.
    """
    records: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        logger.warning("failed to read messages.jsonl: %s", exc)
        return []

    # Sort by seq to ensure correct order.
    records.sort(key=lambda r: int(r.get("seq", 0) or 0))

    # Rebuild conversation: extract the 'message' field from each record.
    messages: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for rec in records:
        msg = rec.get("message")
        if not isinstance(msg, dict):
            # Try parts-based record (LogWriterV3.log_message format).
            parts = rec.get("parts")
            role = rec.get("role")
            if isinstance(parts, list) and role:
                msg = {"role": str(role), "parts": parts}
            else:
                continue

        msg_id = rec.get("message_id", "")
        # Skip duplicate message IDs (e.g., from retries).
        if msg_id and msg_id in seen_ids:
            continue
        if msg_id:
            seen_ids.add(msg_id)

        # Skip internal-only events that shouldn't be in LLM context.
        meta = rec.get("message_meta")
        if isinstance(meta, dict):
            event = str(meta.get("event") or "")
            if event in {
                "stage_end",
                "stage_hard_turn_limit",
                "win_guard",
                "doom_loop",
                "global_budget_exhausted",
                "staleness_guard",
                "loop_end",
                "worker_stage_complete",
                "win_epilogue_complete",
                "session_resume",
                "worker_exception",
            }:
                continue

        role = str(msg.get("role") or "")
        if role not in {"system", "user", "assistant"}:
            continue

        messages.append(msg)

    return messages


def _load_action_frames(path: Path) -> List[Dict[str, Any]]:
    """Load game actions from action_frames.jsonl for environment replay."""
    actions: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = rec.get("action")
            result = rec.get("result")
            if isinstance(action, dict):
                entry: Dict[str, Any] = {
                    "name": str(action.get("name", "")),
                    "args": action.get("args", {}),
                }
                if isinstance(result, dict):
                    entry["status"] = str(result.get("status", ""))
                actions.append(entry)
    except Exception as exc:
        logger.warning("failed to read action_frames.jsonl: %s", exc)
    return actions


def _load_last_observation(path: Path) -> Dict[str, Any]:
    """Load the latest runtime observation."""
    last: Dict[str, Any] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            obs = rec.get("observation")
            if isinstance(obs, dict):
                last = obs
    except Exception as exc:
        logger.warning("failed to read runtime_observations.jsonl: %s", exc)
    return last


def _load_workspace_state(path: Path) -> Dict[str, bytes]:
    """Reconstruct workspace file contents from fs_versions.jsonl.

    Applies snapshots and patches in order to build the final state.
    """
    files: Dict[str, bytes] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = str(rec.get("kind") or "")
            if kind == "snapshot":
                # Full snapshot: replace everything.
                files.clear()
                for f in rec.get("files", []):
                    if not isinstance(f, dict):
                        continue
                    fpath = str(f.get("path", ""))
                    content_b64 = str(f.get("content_b64", ""))
                    if fpath and content_b64:
                        try:
                            files[fpath] = base64.b64decode(content_b64)
                        except Exception:
                            pass
            elif kind == "patch":
                # Incremental patch: apply ops.
                for op in rec.get("ops", []):
                    if not isinstance(op, dict):
                        continue
                    op_type = str(op.get("op", ""))
                    fpath = str(op.get("path", ""))
                    if op_type == "delete" and fpath in files:
                        del files[fpath]
                    elif op_type == "write" and fpath:
                        content_b64 = str(op.get("content_b64", ""))
                        if content_b64:
                            try:
                                files[fpath] = base64.b64decode(content_b64)
                            except Exception:
                                pass
    except Exception as exc:
        logger.warning("failed to read fs_versions.jsonl: %s", exc)
    return files


def _extract_stage_info(state: ResumeState) -> None:
    """Extract the last stage name and loop counter from messages."""
    # Walk messages backwards to find the latest stage_end or stage-prefixed source.
    for msg in reversed(state.messages):
        parts = msg.get("parts", [])
        if not isinstance(parts, list):
            continue
        for p in parts:
            if not isinstance(p, dict):
                continue
            text = str(p.get("text") or "")
            if "[stage_end]" in text:
                # Parse: [stage_end] stage=main reason=... turns_used=...
                for token in text.split():
                    if token.startswith("stage="):
                        state.stage_name = token.split("=", 1)[1]
                        break

    # Extract loop counter from manifest.
    state.loop_counter = int(state.manifest.get("loop_count", 0) or 0)


def _extract_last_compaction_summary(messages: List[Dict[str, Any]]) -> str:
    """Find the latest compaction handoff summary in messages."""
    for msg in reversed(messages):
        if str(msg.get("role") or "") != "user":
            continue
        parts = msg.get("parts", [])
        if not isinstance(parts, list):
            continue
        for p in parts:
            if not isinstance(p, dict):
                continue
            text = str(p.get("text") or "")
            if "<compaction_handoff>" in text:
                return text
    return ""


def restore_workspace(workspace: Path, files: Dict[str, bytes]) -> int:
    """Write workspace files from resume state. Returns count of files written."""
    count = 0
    for rel_path, content in files.items():
        target = workspace / rel_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            count += 1
        except Exception as exc:
            logger.warning("failed to restore workspace file %s: %s", rel_path, exc)
    return count


def build_resume_context_messages(
    state: ResumeState,
    system_content: str,
    max_context_tokens: int = 0,
) -> List[Dict[str, Any]]:
    """Build a message list suitable for resuming an LLM conversation.

    If the full message history exceeds max_context_tokens, we keep the
    system message + a compaction handoff + the recent tail.
    """
    from .compaction import estimate_message_tokens

    if not state.messages:
        return [{"role": "system", "parts": [{"text": system_content}]}]

    # Check if we already have a system message.
    has_system = (
        state.messages
        and str(state.messages[0].get("role") or "") == "system"
    )

    full: List[Dict[str, Any]] = []
    if has_system:
        # Replace old system with fresh one (config may have changed).
        full.append({"role": "system", "parts": [{"text": system_content}]})
        full.extend(state.messages[1:])
    else:
        full.append({"role": "system", "parts": [{"text": system_content}]})
        full.extend(state.messages)

    # If no budget constraint, return everything.
    if max_context_tokens <= 0:
        return full

    est = estimate_message_tokens(full)
    if est <= max_context_tokens:
        return full

    # Too large: use compaction summary + recent messages.
    result: List[Dict[str, Any]] = [
        {"role": "system", "parts": [{"text": system_content}]},
    ]
    if state.last_compaction_summary:
        result.append({
            "role": "user",
            "parts": [{"text": state.last_compaction_summary}],
        })

    # Add recent messages from the tail until we approach the budget.
    budget_remaining = max_context_tokens - estimate_message_tokens(result)
    recent: List[Dict[str, Any]] = []
    for msg in reversed(full[1:]):  # skip system
        msg_tokens = estimate_message_tokens([msg])
        if budget_remaining - msg_tokens < 0 and recent:
            break
        recent.insert(0, msg)
        budget_remaining -= msg_tokens

    result.extend(recent)
    return result

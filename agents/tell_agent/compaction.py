from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import get_env_float
from .claude_client import ClaudeClient
from .request_formatter import build_request_data

DEFAULT_COMPACTION_PROMPT = """
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions. This summary will be used as context when continuing the conversation, so preserve critical information including:

- What was accomplished
- Current work in progress
- Files modified and their purpose
- Next steps that were planned
- Key decisions made and their rationale
- Any errors encountered and how they were resolved
- User preferences or constraints mentioned

Do not call tools in the compaction step.
Produce the summary directly as text output.

Format the summary as a structured document that another instance of yourself could use to seamlessly continue the work.
""".strip()

DEFAULT_COMPACTION_USER_TEMPLATE = """
<compaction_handoff>
Task continuity:
- Keep solving the same game/task.
- Prioritize the latest user objective and runtime state.

Original user objective (first user message):
{FIRST_USER_MESSAGE}

Latest user-side context:
{LATEST_USER_MESSAGE}

Compacted history:
{COMPACTION_SUMMARY}
</compaction_handoff>
""".strip()


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool = False
    max_context_tokens: int = 800_000
    trigger_ratio: float = 0.8
    summary_max_tokens: int = 4096
    summary_prompt: str = DEFAULT_COMPACTION_PROMPT
    summary_user_template: str = DEFAULT_COMPACTION_USER_TEMPLATE
    multi_round_enabled: bool = True
    max_rounds: int = 4
    tool_names: Tuple[str, ...] = ("bash_exec",)
    pin_first_user_message: bool = False


def estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate token count for a list of messages.

    Uses char/4 heuristic for text and a fixed cost per inline image
    (images are resized/tokenized by the provider at ~258 tokens for a
    typical 512x512 image, but we budget 1200 to account for high-res).
    """
    IMAGE_TOKEN_ESTIMATE = 1200
    total_chars = 0
    image_tokens = 0
    for msg in messages:
        parts = msg.get("parts")
        if isinstance(parts, list):
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if isinstance(p.get("text"), str):
                    total_chars += len(p.get("text", ""))
                if isinstance(p.get("thoughtSignature"), str):
                    total_chars += len(p.get("thoughtSignature", ""))
                inline = p.get("inlineData")
                if isinstance(inline, dict):
                    # Use a fixed per-image estimate instead of counting
                    # base64 characters, which wildly overestimates tokens.
                    image_tokens += IMAGE_TOKEN_ESTIMATE
                if "functionCall" in p:
                    total_chars += len(_safe_json(p.get("functionCall")))
                if "functionResponse" in p:
                    total_chars += len(_safe_json(p.get("functionResponse")))
        else:
            total_chars += len(str(msg.get("content") or ""))

        function_calls = msg.get("function_calls")
        if isinstance(function_calls, list):
            for fc in function_calls:
                if isinstance(fc, dict):
                    total_chars += len(str(fc.get("name") or ""))
                    total_chars += len(_safe_json(fc.get("args")))
                    if isinstance(fc.get("thoughtSignature"), str):
                        total_chars += len(fc.get("thoughtSignature", ""))

        tool_results = msg.get("tool_results")
        if isinstance(tool_results, list):
            for tr in tool_results:
                if isinstance(tr, dict):
                    total_chars += len(str(tr.get("name") or ""))
                    total_chars += len(_safe_json(tr.get("result")))
    return max(0, total_chars // 4 + image_tokens)


def should_compact(messages: List[Dict[str, Any]], cfg: CompactionConfig) -> Tuple[bool, int]:
    est = estimate_message_tokens(messages)
    threshold = int(max(1, cfg.max_context_tokens) * max(0.0, min(1.0, cfg.trigger_ratio)))
    return (cfg.enabled and est >= threshold), est


def compact_messages(
    *,
    llm: ClaudeClient,
    messages: List[Dict[str, Any]],
    cfg: CompactionConfig,
    max_output_tokens: int,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_dispatch: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    overflow_mode: bool = False,
    round_logger: Optional[Callable[[int, Any], None]] = None,
    request_hook: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not messages:
        return messages, {"compacted": False, "reason": "empty_messages"}
    temperature = float(get_env_float("LLM_TEMPERATURE", 0.0))
    top_p_cfg = get_env_float("LLM_TOP_P", -1.0)
    top_p = top_p_cfg if 0.0 < float(top_p_cfg) <= 1.0 else None
    selected_tools = _select_compaction_tools(tools=tools, allowed_names=cfg.tool_names)
    candidates = _build_compaction_candidates(messages=messages, cfg=cfg, overflow_mode=overflow_mode)
    last_exc: Optional[Exception] = None

    for cand_idx, candidate in enumerate(candidates):
        try:
            system_msg = candidate[0] if candidate and str(candidate[0].get("role")) == "system" else None
            body = candidate[1:] if system_msg is not None else candidate[:]

            # --- Pin first user message ---
            # When enabled, the first user message (initial prompt with all
            # instructions) is preserved verbatim and excluded from
            # summarization.  After compaction the summary becomes an
            # assistant message, keeping proper user/assistant alternation.
            pinned_user_msg: Optional[Dict[str, Any]] = None
            summarize_body = body
            if cfg.pin_first_user_message:
                for i, msg in enumerate(body):
                    if str(msg.get("role") or "") == "user":
                        pinned_user_msg = msg
                        summarize_body = body[:i] + body[i + 1:]
                        break

            # Build the candidate for summarization: system + all body messages
            summarize_candidate: List[Dict[str, Any]] = []
            if system_msg is not None:
                summarize_candidate.append(system_msg)
            summarize_candidate.extend(summarize_body)

            summary_text, compaction_trace = _run_compaction_dialog(
                llm=llm,
                base_messages=summarize_candidate if summarize_candidate else candidate,
                prompt=cfg.summary_prompt,
                tools=selected_tools,
                tool_dispatch=tool_dispatch,
                temperature=temperature,
                top_p=top_p,
                max_tokens=min(max(1, cfg.summary_max_tokens), max(1, max_output_tokens)),
                multi_round_enabled=cfg.multi_round_enabled,
                max_rounds=max(1, int(cfg.max_rounds)),
                round_logger=round_logger,
                request_hook=request_hook,
            )
            if not summary_text.strip():
                summary_text = "Summary unavailable. Continue from current workspace files."

            compacted: List[Dict[str, Any]] = []
            if system_msg is not None:
                compacted.append(system_msg)

            if pinned_user_msg is not None:
                # Pinned mode: user(instructions) → assistant(summary) → user(continue)
                compacted.append(pinned_user_msg)
                compacted.append({
                    "role": "assistant",
                    "parts": [{"text": f"[Compaction — prior conversation compressed to free context space.]\n\n{summary_text.strip()}"}],
                })
                compacted.append({
                    "role": "user",
                    "parts": [{"text": "Continue."}],
                })
            else:
                # Default mode (no pin): user(handoff+continue)
                user_handoff = _render_summary_user_handoff(
                    template=cfg.summary_user_template,
                    summary=summary_text.strip(),
                    messages=candidate,
                )
                compacted.append(
                    {
                        "role": "user",
                        "parts": [{"text": f"{user_handoff}\n\nContinue."}],
                    }
                )

            return compacted, {
                "compacted": True,
                "summary_len": len(summary_text),
                "messages_before": len(messages),
                "messages_after": len(compacted),
                "compaction_turns": int(compaction_trace.get("turns", 0) or 0),
                "compaction_tool_calls": int(compaction_trace.get("tool_calls", 0) or 0),
                "overflow_mode": bool(overflow_mode),
                "candidate_index": int(cand_idx),
                "candidate_messages_before": len(candidate),
            }
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    return messages, {"compacted": False, "reason": "no_candidates"}



def _extract_summary(text: str) -> str:
    return text.strip()


def _render_summary_user_handoff(template: str, summary: str, messages: List[Dict[str, Any]]) -> str:
    tpl = (template or "").strip() or DEFAULT_COMPACTION_USER_TEMPLATE
    first_user = _truncate_text(_first_user_text(messages), 6000)
    latest_user = _truncate_text(_latest_user_text(messages), 6000)
    out = tpl
    out = out.replace("{COMPACTION_SUMMARY}", summary or "(empty)")
    out = out.replace("{FIRST_USER_MESSAGE}", first_user or "(empty)")
    out = out.replace("{LATEST_USER_MESSAGE}", latest_user or "(empty)")
    return out.strip()


def _first_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if str(msg.get("role") or "") != "user":
            continue
        text = _message_text(msg)
        if text.strip():
            return text.strip()
    return ""


def _latest_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role") or "") != "user":
            continue
        text = _message_text(msg)
        if text.strip():
            return text.strip()
    return ""


def _message_text(msg: Dict[str, Any]) -> str:
    parts = msg.get("parts")
    if isinstance(parts, list):
        chunks: List[str] = []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                chunks.append(p.get("text", ""))
        if chunks:
            return "\n".join(chunks)
    return str(msg.get("content") or "")


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    keep = max(128, limit - 64)
    omitted = len(text) - keep
    return f"{text[:keep]}\n...[truncated {omitted} chars]..."


def _build_compaction_candidates(
    *,
    messages: List[Dict[str, Any]],
    cfg: CompactionConfig,
    overflow_mode: bool,
) -> List[List[Dict[str, Any]]]:
    if not overflow_mode:
        return [messages]

    # For overflow recovery, keep the same model-summary compaction logic but
    # progressively reduce request size (especially inline media payloads).
    stripped = _strip_inline_media(messages)
    candidates: List[List[Dict[str, Any]]] = []
    seen: set[str] = set()

    def _push(candidate: List[Dict[str, Any]]) -> None:
        sig = f"{len(candidate)}::{estimate_message_tokens(candidate)}"
        if sig in seen:
            return
        seen.add(sig)
        candidates.append(candidate)

    _push(stripped)

    system_msg = stripped[0] if stripped and str(stripped[0].get("role")) == "system" else None
    body = stripped[1:] if system_msg is not None else stripped[:]
    base_keep = 6
    keep_windows = [
        max(base_keep * 8, 48),
        max(base_keep * 4, 24),
        max(base_keep * 2, 12),
        base_keep,
    ]
    for keep in keep_windows:
        recent = _select_recent_from_user_boundary(body, keep)
        candidate: List[Dict[str, Any]] = []
        if system_msg is not None:
            candidate.append(copy.deepcopy(system_msg))
        candidate.extend(copy.deepcopy(recent))
        if candidate:
            _push(candidate)
    return candidates


def _strip_inline_media(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = copy.deepcopy(messages)
    for msg in out:
        parts = msg.get("parts")
        if not isinstance(parts, list):
            continue
        new_parts: List[Dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            p = dict(part)
            inline = p.get("inlineData")
            if isinstance(inline, dict):
                mime = str(inline.get("mimeType") or "")
                p.pop("inlineData", None)
                if not isinstance(p.get("text"), str) or not str(p.get("text") or "").strip():
                    p["text"] = f"[inline_media_omitted:{mime}]"
            new_parts.append(p)
        msg["parts"] = new_parts
    return out



def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def _select_recent_from_user_boundary(body: List[Dict[str, Any]], keep_recent_messages: int) -> List[Dict[str, Any]]:
    """
    Keep a recent suffix but ensure the first message starts at a user boundary.
    This avoids handing off a dangling assistant/tool turn after compaction.
    """
    if not body or keep_recent_messages <= 0:
        return []
    start = max(0, len(body) - keep_recent_messages)
    while start > 0 and str(body[start].get("role") or "") != "user":
        start -= 1
    return body[start:]


def _select_compaction_tools(
    *,
    tools: Optional[Sequence[Dict[str, Any]]],
    allowed_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if not tools:
        return []
    allowed = {str(n).strip() for n in allowed_names if str(n).strip()}
    if not allowed:
        return []
    out: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if name and name in allowed:
            out.append(tool)
    return out


def _run_compaction_dialog(
    *,
    llm: ClaudeClient,
    base_messages: List[Dict[str, Any]],
    prompt: str,
    tools: Sequence[Dict[str, Any]],
    tool_dispatch: Optional[Callable[[str, Dict[str, Any]], Any]],
    temperature: float,
    top_p: Optional[float],
    max_tokens: int,
    multi_round_enabled: bool,
    max_rounds: int,
    round_logger: Optional[Callable[[int, Any], None]],
    request_hook: Optional[Callable[[str], None]],
) -> Tuple[str, Dict[str, Any]]:
    convo: List[Dict[str, Any]] = list(base_messages) + [{"role": "user", "content": prompt}]
    trace: Dict[str, Any] = {"turns": 0, "tool_calls": 0, "tool_call_names": []}
    rounds = 1 if not multi_round_enabled else max(1, int(max_rounds))
    for turn_idx in range(rounds):
        if request_hook is not None:
            request_hook("compaction")
        resp = llm.complete(
            messages=convo,
            tools=list(tools),
            max_tokens=max_tokens,
            temperature=temperature,
            request_data=build_request_data(
                messages=convo,
                tools=list(tools),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            ),
        )
        if round_logger is not None:
            try:
                round_logger(turn_idx, resp)
            except Exception:
                pass
        trace["turns"] = turn_idx + 1
        assistant_parts: List[Dict[str, Any]] = []
        text = str(getattr(resp, "text", "") or "")
        if text:
            assistant_parts.append({"text": text})
        tool_calls = list(getattr(resp, "tool_calls", []) or [])
        # Compaction should be summary-first; if no tools are configured, ignore
        # any spontaneous function calls and return text summary directly.
        if not tools:
            tool_calls = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            call_part: Dict[str, Any] = {
                "functionCall": {
                    "name": str(tc.get("name", "") or ""),
                    "args": tc.get("args", {}),
                }
            }
            thought_sig = tc.get("thoughtSignature")
            if thought_sig:
                call_part["thoughtSignature"] = thought_sig
            assistant_parts.append(call_part)
        if assistant_parts:
            convo.append({"role": "assistant", "parts": assistant_parts})

        if not tool_calls:
            return _extract_summary(text), trace

        if tool_dispatch is None:
            return _extract_summary(text), trace

        tool_results: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name", "") or "").strip()
            args = tc.get("args", {})
            try:
                result = tool_dispatch(name, args if isinstance(args, dict) else {})
            except Exception as exc:
                result = {"error": str(exc)}
            tool_results.append({"name": name, "result": result})
            trace["tool_calls"] = int(trace.get("tool_calls", 0) or 0) + 1
            names = trace.get("tool_call_names")
            if isinstance(names, list):
                names.append(name)
        convo.append({"role": "user", "tool_results": tool_results, "content": ""})

    # Reached round cap; return best effort from latest assistant text.
    for msg in reversed(convo):
        if str(msg.get("role", "")) != "assistant":
            continue
        parts = msg.get("parts")
        if not isinstance(parts, list):
            continue
        texts: List[str] = []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                texts.append(p.get("text", ""))
        if texts:
            return _extract_summary("\n".join(texts)), trace
    return "", trace

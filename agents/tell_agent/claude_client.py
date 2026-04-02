from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from .config import get_env_float, get_env_int, get_env_str
from .llm_response import LLMResponse
from .runtime_log_context import RuntimeLogContextMixin


class ClaudeClient(RuntimeLogContextMixin):
    """
    Claude client using the official Anthropic Python SDK.
    Accepts internal request_data format and converts to Anthropic Messages format.
    Supports both streaming and non-streaming modes.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
    ) -> None:
        super().__init__()
        self.model = model
        api_key = get_env_str("ANTHROPIC_API_KEY", "") or None
        base_url = get_env_str("ANTHROPIC_BASE_URL", "") or None
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
        )
        self._stream = str(
            get_env_str("TELL_LLM_STREAM", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"}

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del messages, tools
        internal_request = kwargs.get("request_data")
        if not isinstance(internal_request, dict):
            raise ValueError("ClaudeClient.complete requires request_data payload")

        request_payload = self._build_claude_request(internal_request)
        max_retries = max(1, get_env_int("LLM_ERROR_RETRIES", 8))
        retry_delay = max(0.0, get_env_float("LLM_ERROR_RETRY_DELAY", 16.0))
        retry_max_delay = max(retry_delay, get_env_float("LLM_ERROR_RETRY_MAX_DELAY", 120.0))
        timeout = float(kwargs.get("timeout", get_env_float("LLM_REQUEST_TIMEOUT", 120.0)))

        result: Optional[anthropic.types.Message] = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    sleep_s = min(retry_delay * (2 ** (attempt - 1)), retry_max_delay)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                result = self._request_once(request_payload, timeout=timeout)
                break
            except anthropic.BadRequestError as e:
                self._dump_failed_request(
                    request_payload=request_payload,
                    status=e.status_code,
                    error=str(e),
                    response_text=str(e.body) if hasattr(e, "body") else None,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                msg = self._build_bad_request_message(error=str(e), response_text=str(e.body) if hasattr(e, "body") else None)
                raise RuntimeError(msg) from e
            except Exception as e:
                status = getattr(e, "status_code", None)
                body = str(getattr(e, "body", "")) if hasattr(e, "body") else None
                self._dump_failed_request(
                    request_payload=request_payload,
                    status=status,
                    error=str(e),
                    response_text=body,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                if attempt == max_retries - 1:
                    raise

        if result is None:
            raise RuntimeError("Claude request failed after all retries")

        return self._parse_response(result, internal_request, request_payload)

    def _request_once(
        self,
        request_payload: Dict[str, Any],
        *,
        timeout: float,
    ) -> anthropic.types.Message:
        kwargs: Dict[str, Any] = {
            "model": request_payload["model"],
            "max_tokens": request_payload["max_tokens"],
            "messages": request_payload["messages"],
            "timeout": timeout,
        }
        if "system" in request_payload:
            kwargs["system"] = request_payload["system"]
        if "tools" in request_payload:
            kwargs["tools"] = request_payload["tools"]
        if "thinking" in request_payload:
            kwargs["thinking"] = request_payload["thinking"]
        if "temperature" in request_payload:
            kwargs["temperature"] = request_payload["temperature"]
        if "top_p" in request_payload:
            kwargs["top_p"] = request_payload["top_p"]

        if self._stream:
            return self._request_stream(kwargs)
        return self._client.messages.create(**kwargs)

    def _request_stream(self, kwargs: Dict[str, Any]) -> anthropic.types.Message:
        with self._client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        return response

    def _parse_response(
        self,
        message: anthropic.types.Message,
        internal_request: Dict[str, Any],
        request_payload: Dict[str, Any],
    ) -> LLMResponse:
        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in message.content:
            if block.type == "text":
                text_chunks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "args": block.input,
                    "id": block.id,
                })

        usage = {
            "prompt_tokens": message.usage.input_tokens,
            "completion_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }

        # Build raw dict for logging and downstream compatibility
        raw_dict = message.model_dump()
        # Inject candidates for state_machine._build_assistant_parts
        internal_parts = self._message_to_internal_parts(message)
        raw_dict["candidates"] = [{"content": {"parts": internal_parts}}]

        return LLMResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            usage=usage,
            raw=raw_dict,
            stop_reason=message.stop_reason or "",
            request_generation_config=self._extract_generation_cfg(internal_request, request_payload),
            raw_request=request_payload,
        )

    def _build_claude_request(self, internal_request: Dict[str, Any]) -> Dict[str, Any]:
        messages = self._convert_messages(internal_request)
        tools = self._convert_tools(internal_request.get("tools"))
        generation_cfg = internal_request.get("generationConfig") or {}
        max_tokens = int(generation_cfg.get("maxOutputTokens", 4096) or 4096)

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        system_text = self._extract_system_text(internal_request.get("systemInstruction"))
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = tools
        thinking_type = str(get_env_str("CLAUDE_THINKING_TYPE", "") or "").strip().lower()
        if thinking_type in {"enabled", "disabled", "adaptive"}:
            thinking: Dict[str, Any] = {"type": thinking_type}
            budget_raw = str(get_env_str("CLAUDE_THINKING_BUDGET_TOKENS", "") or "").strip()
            if budget_raw:
                try:
                    budget = int(budget_raw)
                    if budget > 0:
                        if budget >= max_tokens:
                            budget = max(1, max_tokens - 1)
                        thinking["budget_tokens"] = budget
                except Exception:
                    pass
            payload["thinking"] = thinking
        if "temperature" in generation_cfg:
            payload["temperature"] = float(generation_cfg.get("temperature", 0.0))
        elif "topP" in generation_cfg:
            payload["top_p"] = float(generation_cfg.get("topP", 1.0))
        return payload

    def _extract_generation_cfg(
        self, internal_request: Dict[str, Any], claude_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        out = dict(internal_request.get("generationConfig") or {})
        if "temperature" in claude_request:
            out["temperature"] = claude_request["temperature"]
        if "top_p" in claude_request:
            out["top_p"] = claude_request["top_p"]
        return out

    def _convert_tools(self, raw_tools: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(raw_tools, list):
            return out
        for t in raw_tools:
            if not isinstance(t, dict):
                continue
            decls = t.get("functionDeclarations")
            if not isinstance(decls, list):
                continue
            for d in decls:
                if not isinstance(d, dict):
                    continue
                out.append({
                    "name": str(d.get("name") or ""),
                    "description": str(d.get("description") or ""),
                    "input_schema": d.get("parameters", {}),
                })
        return out

    def _convert_messages(self, internal_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        tool_call_queue: List[Tuple[str, str]] = []
        contents = internal_request.get("contents")
        if not isinstance(contents, list):
            return out

        for content in contents:
            if not isinstance(content, dict):
                continue
            role = "assistant" if str(content.get("role") or "") == "model" else "user"
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue

            if role == "assistant":
                msg, queued = self._convert_assistant_parts(parts)
                tool_call_queue.extend(queued)
                if msg:
                    out.append(msg)
                continue

            user_tool_result_blocks: List[Dict[str, Any]] = []
            user_other_blocks: List[Dict[str, Any]] = []
            i = 0
            n = len(parts)
            while i < n:
                p = parts[i]
                i += 1
                if not isinstance(p, dict):
                    continue
                fr = p.get("functionResponse")
                if isinstance(fr, dict):
                    extra_blocks: List[Dict[str, Any]] = []
                    while i < n:
                        np = parts[i]
                        if not isinstance(np, dict):
                            i += 1
                            continue
                        if isinstance(np.get("functionResponse"), dict):
                            break
                        inline = np.get("inlineData")
                        if isinstance(inline, dict):
                            b = self._part_to_claude_block(np)
                            if b is not None:
                                extra_blocks.append(b)
                            i += 1
                            continue
                        break
                    user_tool_result_blocks.append(
                        self._build_tool_result_block(fr, tool_call_queue, extra_blocks=extra_blocks)
                    )
                    continue
                block = self._part_to_claude_block(p)
                if block is not None:
                    user_other_blocks.append(block)
            user_blocks = user_tool_result_blocks + user_other_blocks
            if user_blocks:
                out.append({"role": "user", "content": user_blocks})
        return out

    def _convert_assistant_parts(self, parts: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Tuple[str, str]]]:
        blocks: List[Dict[str, Any]] = []
        queued: List[Tuple[str, str]] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("thought") is True:
                thinking_text = str(p.get("text") or "")
                sig = str(p.get("thoughtSignature") or "")
                block: Dict[str, Any] = {"type": "thinking", "thinking": thinking_text}
                if sig:
                    block["signature"] = sig
                blocks.append(block)
                continue
            txt = p.get("text")
            if isinstance(txt, str) and txt:
                blocks.append({"type": "text", "text": txt})
            fc = p.get("functionCall")
            if isinstance(fc, dict):
                name = str(fc.get("name") or "")
                tool_id = f"toolu_{uuid.uuid4().hex[:12]}"
                blocks.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": fc.get("args", {}),
                })
                queued.append((name, tool_id))
        if not blocks:
            return None, queued
        return {"role": "assistant", "content": blocks}, queued

    def _build_tool_result_block(
        self,
        function_response: Dict[str, Any],
        queue: List[Tuple[str, str]],
        *,
        extra_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        name = str(function_response.get("name") or "")
        tool_id = ""
        if queue:
            idx = -1
            for i, (n, _) in enumerate(queue):
                if n == name:
                    idx = i
                    break
            if idx >= 0:
                _, tool_id = queue.pop(idx)
            else:
                _, tool_id = queue.pop(0)
        if not tool_id:
            tool_id = f"toolu_{uuid.uuid4().hex[:12]}"

        content_obj = {}
        resp = function_response.get("response")
        if isinstance(resp, dict):
            content_obj = resp.get("content", {})
        if isinstance(content_obj, str):
            content_text = content_obj
        else:
            content_text = json.dumps(content_obj, ensure_ascii=False)
        extras = [b for b in (extra_blocks or []) if isinstance(b, dict)]
        if extras:
            content_list: List[Dict[str, Any]] = [{"type": "text", "text": content_text}]
            content_list.extend(extras)
            return {"type": "tool_result", "tool_use_id": tool_id, "content": content_list}
        return {"type": "tool_result", "tool_use_id": tool_id, "content": content_text}

    def _part_to_claude_block(self, part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if isinstance(part.get("text"), str):
            return {"type": "text", "text": part.get("text", "")}
        inline = part.get("inlineData")
        if not isinstance(inline, dict):
            return None
        mime = str(inline.get("mimeType") or "").strip()
        data = str(inline.get("data") or "").strip()
        if not mime or not data:
            return None
        if mime.startswith("image/"):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": data,
                },
            }
        return {"type": "text", "text": f"[unsupported inline mime omitted: {mime}]"}

    def _extract_system_text(self, system_instruction: Any) -> str:
        if not isinstance(system_instruction, dict):
            return ""
        parts = system_instruction.get("parts")
        if not isinstance(parts, list):
            return ""
        out: List[str] = []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                out.append(p.get("text", ""))
        return "".join(out).strip()

    def _message_to_internal_parts(self, message: anthropic.types.Message) -> List[Dict[str, Any]]:
        """Convert Anthropic Message to internal-format parts for state_machine compatibility."""
        parts: List[Dict[str, Any]] = []
        for block in message.content:
            if block.type == "thinking":
                part: Dict[str, Any] = {
                    "text": block.thinking,
                    "thought": True,
                }
                if hasattr(block, "signature") and block.signature:
                    part["thoughtSignature"] = block.signature
                parts.append(part)
            elif block.type == "text":
                parts.append({"text": block.text})
            elif block.type == "tool_use":
                parts.append({
                    "functionCall": {
                        "name": block.name,
                        "args": block.input,
                    }
                })
        return parts

    def _dump_failed_request(
        self,
        *,
        request_payload: Dict[str, Any],
        status: Any,
        error: str,
        response_text: Optional[str],
        attempt: int,
        max_retries: int,
    ) -> None:
        path = self.get_runtime_log_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            run_id, replay_dir = self.get_runtime_log_context()
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "provider": "claude",
                "model": self.model,
                "run_id": run_id,
                "replay_dir": replay_dir,
                "attempt": int(attempt),
                "max_retries": int(max_retries),
                "status": status,
                "error": error,
                "request": request_payload,
                "response_text": response_text or "",
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _build_bad_request_message(self, *, error: str, response_text: Optional[str]) -> str:
        body = str(response_text or "")
        low = body.lower()
        if "token" in low and ("exceed" in low or "maximum" in low or "too many" in low):
            return f"INPUT_TOKENS_EXCEEDED: {error} | body={body}"
        return f"BAD_REQUEST: {error} | body={body}"

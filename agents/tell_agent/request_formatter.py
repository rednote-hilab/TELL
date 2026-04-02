from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional

from .config import get_env_str


_DATA_URL_RE = re.compile(r"(data:(?:image|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+)")


def build_request_data(
    *,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    contents: List[Dict[str, Any]] = []
    system_instruction: Optional[str] = None

    for msg in messages:
        role = str(msg.get("role") or "")
        if role == "system":
            if isinstance(msg.get("parts"), list):
                txts: List[str] = []
                for p in msg.get("parts", []):
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        txts.append(p.get("text", ""))
                system_instruction = "".join(txts)
            else:
                system_instruction = str(msg.get("content") or "")
            continue

        if isinstance(msg.get("parts"), list):
            parts = [p for p in msg.get("parts", []) if isinstance(p, dict)]
            if not parts:
                continue
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
            continue

        if role == "user":
            tool_results = msg.get("tool_results")
            if isinstance(tool_results, list) and tool_results:
                parts = build_tool_result_parts(tool_results)
                if parts:
                    contents.append({"role": "user", "parts": parts})
            else:
                content = str(msg.get("content", "") or "")
                parts = build_text_parts_with_inline_media(content)
                contents.append({"role": "user", "parts": parts})
            continue

        if role == "assistant":
            parts: List[Dict[str, Any]] = []
            content = str(msg.get("content", "") or "")
            if content:
                parts.append({"text": content})
            fcs = msg.get("function_calls")
            if isinstance(fcs, list):
                for fc in fcs:
                    if not isinstance(fc, dict):
                        continue
                    fc_part: Dict[str, Any] = {
                        "functionCall": {
                            "name": str(fc.get("name") or ""),
                            "args": fc.get("args", {}),
                        }
                    }
                    if fc.get("thoughtSignature"):
                        fc_part["thoughtSignature"] = fc["thoughtSignature"]
                    parts.append(fc_part)
            if not parts:
                parts = [{"text": ""}]
            contents.append({"role": "model", "parts": parts})

    request_data: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if top_p is not None:
        request_data["generationConfig"]["topP"] = float(top_p)

    include_thoughts_raw = str(get_env_str("LLM_INCLUDE_THOUGHTS", ""))
    include_thoughts = include_thoughts_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if include_thoughts:
        thinking_cfg: Dict[str, Any] = {"includeThoughts": True}
        thinking_budget = str(get_env_str("LLM_THINKING_BUDGET", "")).strip()
        if thinking_budget:
            try:
                thinking_cfg["thinkingBudget"] = int(thinking_budget)
            except Exception:
                pass
        request_data["generationConfig"]["thinkingConfig"] = thinking_cfg

    media_resolution = str(get_env_str("LLM_MEDIA_RESOLUTION", "")).strip()
    if media_resolution:
        request_data["generationConfig"]["mediaResolution"] = media_resolution

    if system_instruction:
        request_data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    if tools:
        request_data["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    }
                ]
            }
            for t in tools
            if isinstance(t, dict)
        ]
    return request_data


def build_text_parts_with_inline_media(text: str) -> List[Dict[str, Any]]:
    if not text:
        return [{"text": ""}]
    parts: List[Dict[str, Any]] = []
    pos = 0
    for m in _DATA_URL_RE.finditer(text):
        if m.start() > pos:
            chunk = text[pos : m.start()]
            if chunk:
                parts.append({"text": chunk})
        media = _data_url_to_part(m.group(1))
        if media:
            parts.append(media)
        else:
            parts.append({"text": m.group(1)})
        pos = m.end()
    if pos < len(text):
        tail = text[pos:]
        if tail:
            parts.append({"text": tail})
    if not parts:
        return [{"text": ""}]
    return parts


def build_tool_result_parts(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for tr in tool_results:
        if not isinstance(tr, dict):
            continue
        tr_name = str(tr.get("name") or "")
        raw_result = tr.get("result")
        parsed_result = _parse_tool_result(raw_result)
        compact_result = _compact_tool_result(parsed_result)
        parts.append(
            {
                "functionResponse": {
                    "name": tr_name,
                    "response": {"content": compact_result},
                }
            }
        )
        parts.extend(_extract_inline_media_parts(raw_result))
    return parts


def _parse_tool_result(result: Any) -> Any:
    if isinstance(result, str):
        s = result.strip()
        if s and s[0] in "{[":
            try:
                return json.loads(s)
            except Exception:
                return result
    return result


def _compact_tool_result(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in {"media", "image_b64", "image_data_url", "image_mime", "frames"}:
                continue
            out[k] = _compact_tool_result(v)
        return out
    if isinstance(obj, list):
        return [_compact_tool_result(x) for x in obj]
    if isinstance(obj, str):
        # Prevent data URLs from remaining inside functionResponse.content text payload.
        return _DATA_URL_RE.sub("[inline-media-omitted]", obj)
    return obj


def _extract_inline_media_parts(result: Any, max_images: int = 3) -> List[Dict[str, Any]]:
    medias: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def is_valid_media_b64(mime: str, b64: str) -> bool:
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception:
            return False
        if not raw:
            return False
        if mime == "image/png":
            return raw.startswith(b"\x89PNG\r\n\x1a\n")
        if mime == "image/gif":
            return raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a")
        if mime == "image/jpeg":
            return raw.startswith(b"\xff\xd8")
        if mime == "video/mp4":
            return len(raw) >= 12 and raw[4:8] == b"ftyp"
        return True

    def add_data_url(data_url: str) -> None:
        m = re.match(r"^data:((?:image|video)/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)$", data_url.strip())
        if not m or len(medias) >= max_images:
            return
        mime = m.group(1)
        b64 = m.group(2)
        key = (mime, b64)
        if is_valid_media_b64(mime, b64) and key not in seen:
            seen.add(key)
            medias.append({"mime": mime, "b64": b64})

    def walk(obj: Any) -> None:
        if len(medias) >= max_images:
            return
        if isinstance(obj, str):
            s = obj.strip()
            if s.startswith("data:image/") or s.startswith("data:video/"):
                add_data_url(s)
                return
            if s and s[0] in "{[":
                try:
                    walk(json.loads(s))
                    return
                except Exception:
                    pass
            for m in _DATA_URL_RE.finditer(s):
                add_data_url(m.group(1))
            return
        if isinstance(obj, dict):
            media = obj.get("media")
            if isinstance(media, dict):
                mime = str(media.get("mime") or "").strip()
                b64 = str(media.get("b64") or "").strip()
                if mime and b64:
                    key = (mime, b64)
                    if is_valid_media_b64(mime, b64) and key not in seen:
                        seen.add(key)
                        medias.append({"mime": mime, "b64": b64})
                    # If canonical `media` exists, treat it as source-of-truth
                    # and do not continue probing legacy duplicate fields.
                    return
            b64 = obj.get("image_b64")
            mime = str(obj.get("image_mime") or "").strip()
            if isinstance(b64, str) and b64.strip() and mime:
                b64n = b64.strip()
                key = (mime, b64n)
                if is_valid_media_b64(mime, b64n) and key not in seen:
                    seen.add(key)
                    medias.append({"mime": mime, "b64": b64n})
            data_url = obj.get("image_data_url")
            if isinstance(data_url, str):
                add_data_url(data_url)
            for v in obj.values():
                walk(v)
            return
        if isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(result)
    return [{"inlineData": {"mimeType": m["mime"], "data": m["b64"]}} for m in medias]


def _data_url_to_part(data_url: str) -> Optional[Dict[str, Any]]:
    m = re.match(r"^data:((?:image|video)/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)$", data_url.strip())
    if not m:
        return None
    mime = m.group(1)
    b64 = m.group(2)
    try:
        base64.b64decode(b64, validate=True)
    except Exception:
        return None
    return {"inlineData": {"mimeType": mime, "data": b64}}

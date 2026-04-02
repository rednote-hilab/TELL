from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class TELLToolHandlers:
    OUTPUT_LIMIT = 8000

    def __init__(
        self,
        workspace: Path,
        memory_root: Path,
        runtime_port: int,
        shell_timeout: float,
        output_limit: int = OUTPUT_LIMIT,
        workspace_size_limit_bytes: int = 0,
        workspace_hard_limited: bool = False,
        extra_read_paths: Optional[List[Path]] = None,
        subagent_runner: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ) -> None:
        self.workspace = workspace
        self.memory_root = memory_root
        self.runtime_port = runtime_port
        self.shell_timeout = shell_timeout
        self.output_limit = max(256, int(output_limit))
        self.workspace_size_limit_bytes = max(0, int(workspace_size_limit_bytes))
        self.workspace_hard_limited = bool(workspace_hard_limited)
        self._extra_read_prefixes = [
            Path(p).resolve()
            for p in (extra_read_paths or [])
            if str(p).strip()
        ]
        self.subagent_runner = subagent_runner
        self.todos: List[Dict[str, Any]] = []
        self._allowed_write_prefixes = [
            self.workspace.resolve(),
            Path("/tmp").resolve(),
            Path("/var/tmp").resolve(),
            Path("/dev").resolve(),
        ]

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        if not isinstance(args, dict):
            return "Error: tool args must be an object"
        if name == "bash_exec":
            return self.handle_bash_exec(args)
        if name == "screen_shot":
            return self.handle_screen_shot(args)
        if name == "read_file":
            return self.handle_read_file(args)
        if name == "grep_text":
            return self.handle_grep_text(args)
        if name == "list_dir":
            return self.handle_list_dir(args)
        if name == "write_memo":
            return self.handle_write_memo(args)
        if name == "write_file":
            return self.handle_write_file(args)
        if name == "todo_write":
            return self.handle_todo_write(args)
        if name == "run_grid_survey_task":
            return self.handle_run_subagent_task("grid_survey", args)
        if name == "run_deep_analysis":
            return self.handle_deep_analysis(args)
        return f"Error: unsupported tool {name}"

    def handle_screen_shot(self, args: Dict[str, Any]) -> str:
        _ = args  # no args by design
        base = f"http://127.0.0.1:{self.runtime_port}"
        try:
            req = urllib.request.Request(f"{base}/observations?index=-1", method="GET")
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            return f"Error: screen_shot request failed: {type(exc).__name__}: {exc}"
        obs_data = payload.get("observations") if isinstance(payload, dict) else None
        # API returns dict for single index, list for slice
        if isinstance(obs_data, dict):
            latest = obs_data
        elif isinstance(obs_data, list) and obs_data:
            latest = obs_data[-1] if isinstance(obs_data[-1], dict) else {}
        else:
            return json.dumps({"ok": False, "error": "no_observation"}, ensure_ascii=True)
        frames = latest.get("frames") if isinstance(latest.get("frames"), list) else []
        board = str(frames[-1]) if frames else ""
        if not board.strip():
            return json.dumps({"ok": False, "error": "empty_frames"}, ensure_ascii=True)
        data_url = self._render_ascii_board_png_data_url(board)
        out: Dict[str, Any] = {
            "ok": bool(data_url),
            "observation_index": latest.get("observation_index"),
            "total_actions": latest.get("total_actions"),
            "last_action": latest.get("last_action"),
            "current_level": latest.get("current_level"),
            "available_actions": latest.get("available_actions", []),
            "frame_count": latest.get("frame_count"),
            "media": {"mime": "image/png", "b64": data_url.split(",", 1)[1]} if data_url else {},
            "image_data_url": data_url,
        }
        if not data_url:
            out["error"] = "image_render_unavailable"
        return json.dumps(out, ensure_ascii=True)

    # Maximum memo size in bytes. Prevents context bloat from oversized memos.
    MEMO_SIZE_LIMIT = 16_384  # 16KB

    def handle_write_memo(self, args: Dict[str, Any]) -> str:
        memo = args.get("memo")
        if not isinstance(memo, str):
            return "Error: write_memo requires a string 'memo'"
        memo_bytes = len(memo.encode("utf-8"))
        if memo_bytes > self.MEMO_SIZE_LIMIT:
            return (
                f"Error: memo too large ({memo_bytes} bytes, limit {self.MEMO_SIZE_LIMIT} bytes). "
                f"Compress the memo: keep only verified facts, action-effect mappings, "
                f"and the current hypothesis. Remove verbose narratives and raw data."
            )
        memo_path = self.workspace / "MEMORY.md"
        try:
            memo_path.parent.mkdir(parents=True, exist_ok=True)
            memo_path.write_text(memo, encoding="utf-8")
            return json.dumps(
                {
                    "ok": True,
                    "path": "MEMORY.md",
                    "bytes": memo_bytes,
                },
                ensure_ascii=True,
            )
        except Exception as exc:
            return f"Error: write_memo failed: {type(exc).__name__}: {exc}"

    def handle_write_file(self, args: Dict[str, Any]) -> str:
        file_path = args.get("path") or args.get("file_path")
        content = args.get("content")
        if not isinstance(file_path, str) or not file_path.strip():
            return "Error: write_file requires a non-empty string 'path'"
        if not isinstance(content, str):
            return "Error: write_file requires a string 'content'"
        try:
            resolved = self._resolve_workspace_path(file_path.strip())
        except ValueError as exc:
            return f"Error: {exc}"
        content_bytes = len(content.encode("utf-8"))
        # Check workspace size limit before writing
        if self.workspace_size_limit_bytes > 0 and not self.workspace_hard_limited:
            current_usage = self._workspace_usage_bytes()
            # Subtract existing file size if overwriting
            existing_size = 0
            if resolved.is_file():
                try:
                    existing_size = resolved.stat().st_size
                except Exception:
                    pass
            projected = current_usage - existing_size + content_bytes
            if projected > self.workspace_size_limit_bytes:
                return (
                    f"Error: write would exceed workspace limit "
                    f"({self._format_bytes(projected)} > {self._format_bytes(self.workspace_size_limit_bytes)}). "
                    f"Free up space or reduce content size."
                )
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            rel_path = resolved.relative_to(self.workspace.resolve())
            return json.dumps(
                {"ok": True, "path": str(rel_path), "bytes": content_bytes},
                ensure_ascii=True,
            )
        except Exception as exc:
            return f"Error: write_file failed: {type(exc).__name__}: {exc}"

    def _render_ascii_board_png_data_url(self, board: str) -> str:
        if Image is None:
            return ""
        lines = [ln for ln in board.splitlines() if ln != ""]
        if not lines:
            return ""
        h = len(lines)
        w = max(len(ln) for ln in lines)
        scale = 8
        palette: Dict[str, tuple[int, int, int]] = {
            "W": (255, 255, 255),
            "w": (248, 244, 234),
            ".": (217, 217, 217),
            ":": (158, 158, 158),
            "d": (97, 97, 97),
            "K": (0, 0, 0),
            "M": (255, 0, 255),
            "p": (255, 154, 213),
            "R": (255, 0, 0),
            "B": (30, 64, 255),
            "b": (142, 203, 255),
            "Y": (255, 230, 0),
            "O": (255, 152, 0),
            "n": (128, 0, 0),
            "G": (0, 166, 81),
            "u": (126, 34, 206),
        }
        img = Image.new("RGB", (max(1, w * scale), max(1, h * scale)), (0, 0, 0))
        px = img.load()
        for y, row in enumerate(lines):
            for x in range(w):
                ch = row[x] if x < len(row) else "K"
                c = palette.get(ch, (0, 0, 0))
                for dy in range(scale):
                    for dx in range(scale):
                        px[x * scale + dx, y * scale + dy] = c
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def handle_run_subagent_task(self, kind: str, args: Dict[str, Any]) -> str:
        if self.subagent_runner is None:
            return f"Error: {kind} subagent runner is not configured"
        try:
            return str(self.subagent_runner(kind, args))
        except Exception as exc:
            return f"Error: {kind} subagent failed: {type(exc).__name__}: {exc}"

    def handle_deep_analysis(self, args: Dict[str, Any]) -> str:
        if self.subagent_runner is None:
            return "Error: deep analysis subagent runner is not configured"

        subagents_raw = args.get("subagents")
        if not isinstance(subagents_raw, list) or not subagents_raw:
            return "Error: run_deep_analysis requires a non-empty 'subagents' array"
        shared_task_spec = args.get("task_spec")
        if not isinstance(shared_task_spec, dict):
            shared_task_spec = {}

        subagents: List[Dict[str, Any]] = []
        for idx, raw in enumerate(subagents_raw):
            if not isinstance(raw, dict):
                continue
            objective = str(raw.get("objective") or "").strip()
            guidance = str(raw.get("guidance") or "").strip()
            if not objective or not guidance:
                continue
            subagent_id = str(raw.get("subagent_id") or raw.get("id") or f"subagent_{idx + 1}").strip()
            task_spec = dict(shared_task_spec)
            raw_spec = raw.get("task_spec")
            if isinstance(raw_spec, dict):
                task_spec.update(raw_spec)
            subagents.append(
                {
                    "subagent_id": subagent_id,
                    "objective": objective,
                    "guidance": guidance,
                    "task_spec": task_spec,
                }
            )
        if not subagents:
            return "Error: run_deep_analysis requires subagents with non-empty 'objective' and 'guidance'"

        parallelism = len(subagents)

        results: List[Optional[Dict[str, Any]]] = [None] * len(subagents)

        def run_subagent(index: int, spec: Dict[str, Any]) -> Dict[str, Any]:
            subagent_args: Dict[str, Any] = {
                "objective": str(spec["objective"]),
                "guidance": str(spec["guidance"]),
                "task_spec": dict(spec.get("task_spec") or {}),
                "subagent_id": str(spec["subagent_id"]),
            }
            try:
                result_text = str(self.subagent_runner("deep_analysis", subagent_args))
                return {
                    "ok": True,
                    "subagent_id": str(spec["subagent_id"]),
                    "objective": str(spec["objective"]),
                    "guidance": str(spec["guidance"]),
                    "result": result_text,
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "subagent_id": str(spec["subagent_id"]),
                    "objective": str(spec["objective"]),
                    "guidance": str(spec["guidance"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
            future_map = {
                executor.submit(run_subagent, idx, spec): idx for idx, spec in enumerate(subagents)
            }
            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    spec = subagents[idx]
                    results[idx] = {
                        "ok": False,
                        "subagent_id": str(spec["subagent_id"]),
                        "objective": str(spec["objective"]),
                        "guidance": str(spec["guidance"]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        payload = {
            "ok": all(isinstance(item, dict) and bool(item.get("ok")) for item in results),
            "subagent_count": len(subagents),
            "results": [item for item in results if isinstance(item, dict)],
        }
        return json.dumps(payload, ensure_ascii=False)

    def handle_todo_write(self, args: Dict[str, Any]) -> str:
        todos = args.get("todos", [])
        if not isinstance(todos, list):
            return "Error: 'todos' must be a list"
        merged = {str(t.get("id", "")): dict(t) for t in self.todos if isinstance(t, dict) and str(t.get("id", ""))}
        for t in todos:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id", ""))
            if not tid:
                continue
            merged[tid] = {
                "id": tid,
                "content": str(t.get("content", "")),
                "status": str(t.get("status", "pending")),
                "priority": str(t.get("priority", "medium")),
            }
        in_progress = [t for t in merged.values() if isinstance(t, dict) and t.get("status") == "in_progress"]
        if len(in_progress) > 1:
            return "Error: only one todo can be in_progress"
        self.todos = list(merged.values())
        return f"Todo list updated: {len(self.todos)} items"

    def handle_read_file(self, args: Dict[str, Any]) -> str:
        raw_path = str(args.get("path") or args.get("file_path") or "").strip()
        if not raw_path:
            return "Error: read_file requires 'path'"
        try:
            resolved = self._resolve_read_path(raw_path)
        except Exception as exc:
            return f"Error: invalid path: {exc}"
        rel_path = self._display_path(resolved)
        if not resolved.exists():
            return self._file_not_found_message(resolved)

        limit = max(1, min(2000, int(args.get("limit", 200) or 200)))
        offset = max(1, int(args.get("offset", 1) or 1))

        if resolved.is_dir():
            return self._render_directory_listing(resolved, offset=offset, limit=limit)

        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            return f"Error: read_file failed: {type(exc).__name__}: {exc}"

        total = len(lines)
        start = offset - 1
        chunk = lines[start : start + limit]
        if not chunk and total > 0:
            return (
                f"<path>{rel_path}</path>\n"
                f"<type>file</type>\n"
                f"<content>\n(No content at offset {offset}; file has {total} lines)\n</content>"
            )
        numbered = "\n".join(f"{start + idx + 1:>6}\t{line}" for idx, line in enumerate(chunk))
        truncated = start + len(chunk) < total
        footer = (
            f"(Showing lines {offset}-{start + len(chunk)} of {total}. Use offset={start + len(chunk) + 1} to continue.)"
            if truncated
            else f"(End of file - total {total} lines)"
        )
        return "\n".join(
            [
                f"<path>{rel_path}</path>",
                "<type>file</type>",
                "<content>",
                numbered,
                "",
                footer,
                "</content>",
            ]
        )

    def handle_list_dir(self, args: Dict[str, Any]) -> str:
        raw_path = str(args.get("path") or ".").strip() or "."
        try:
            resolved = self._resolve_read_path(raw_path)
        except Exception as exc:
            return f"Error: invalid path: {exc}"
        if not resolved.exists():
            return self._file_not_found_message(resolved)
        if not resolved.is_dir():
            return f"Error: path is not a directory: {self._display_path(resolved)}"
        limit = max(1, min(2000, int(args.get("limit", 200) or 200)))
        offset = max(1, int(args.get("offset", 1) or 1))
        return self._render_directory_listing(resolved, offset=offset, limit=limit)

    def handle_grep_text(self, args: Dict[str, Any]) -> str:
        pattern = str(args.get("pattern") or "").strip()
        if not pattern:
            return "Error: grep_text requires 'pattern'"
        raw_path = str(args.get("path") or ".").strip() or "."
        include = str(args.get("include") or "").strip()
        limit = max(1, min(500, int(args.get("limit", 100) or 100)))
        try:
            search_root = self._resolve_read_path(raw_path)
        except Exception as exc:
            return f"Error: invalid path: {exc}"
        if not search_root.exists():
            return self._file_not_found_message(search_root)

        rg_path = shutil.which("rg")
        if not rg_path:
            return "Error: grep_text requires ripgrep (rg) to be installed"
        cmd = [rg_path, "-nH", "--hidden", "--no-messages", "--regexp", pattern]
        if include:
            cmd.extend(["--glob", include])
        cmd.append(str(search_root))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.shell_timeout,
                check=False,
            )
        except Exception as exc:
            return f"Error: grep_text failed: {type(exc).__name__}: {exc}"

        raw = proc.stdout.strip()
        if proc.returncode not in {0, 1, 2}:
            return f"Error: grep_text failed: {proc.stderr.strip()}"
        if not raw:
            return f"Found 0 matches for {pattern!r}"

        lines = raw.splitlines()
        shown = lines[:limit]
        rendered: List[str] = []
        for line in shown:
            try:
                path_part, line_no, text = line.split(":", 2)
            except ValueError:
                rendered.append(line)
                continue
            try:
                rel = self._display_path(Path(path_part).resolve())
            except Exception:
                rel = path_part
            rendered.append(f"{rel}:{line_no}:{text}")
        suffix = ""
        if len(lines) > limit:
            suffix = f"\n(Showing first {limit} of {len(lines)} matches)"
        return f"Found {len(lines)} matches for {pattern!r}\n" + "\n".join(rendered) + suffix

    def _resolve_workspace_path(self, path_value: str) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.workspace / candidate).resolve()
        workspace_resolved = self.workspace.resolve()
        if workspace_resolved not in resolved.parents and resolved != workspace_resolved:
            raise ValueError("path escapes workspace")
        return resolved

    def _resolve_read_path(self, path_value: str) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.workspace / candidate).resolve()
        if self._is_under_workspace_or_extra_read_path(resolved):
            return resolved
        raise ValueError("path escapes workspace and configured read-only paths")

    def _is_under_workspace_or_extra_read_path(self, resolved: Path) -> bool:
        workspace_resolved = self.workspace.resolve()
        if resolved == workspace_resolved or resolved.is_relative_to(workspace_resolved):
            return True
        for prefix in self._extra_read_prefixes:
            if resolved == prefix or resolved.is_relative_to(prefix):
                return True
        return False

    def _display_path(self, resolved: Path) -> str:
        workspace_root = self.workspace.resolve()
        if resolved == workspace_root:
            return "."
        if resolved.is_relative_to(workspace_root):
            return resolved.relative_to(workspace_root).as_posix()
        return str(resolved)

    def _render_directory_listing(self, resolved: Path, *, offset: int, limit: int) -> str:
        try:
            entries = sorted(resolved.iterdir(), key=lambda item: item.name.lower())
        except Exception as exc:
            return f"Error: list_dir failed: {type(exc).__name__}: {exc}"
        start = offset - 1
        subset = entries[start : start + limit]
        items: List[str] = []
        for item in subset:
            suffix = "/" if item.is_dir() else ""
            mode = ""
            try:
                st = item.stat()
                mode = oct(stat.S_IMODE(st.st_mode))
            except Exception:
                mode = "?"
            items.append(f"{item.name}{suffix}\t{mode}")
        truncated = start + len(subset) < len(entries)
        footer = (
            f"(Showing entries {offset}-{start + len(subset)} of {len(entries)}. Use offset={start + len(subset) + 1} to continue.)"
            if truncated
            else f"({len(entries)} entries)"
        )
        rel = self._display_path(resolved)
        body = "\n".join(items)
        return "\n".join(
            [
                f"<path>{rel}</path>",
                "<type>directory</type>",
                "<entries>",
                body,
                "",
                footer,
                "</entries>",
            ]
        )

    def _file_not_found_message(self, resolved: Path) -> str:
        parent = resolved.parent
        base = resolved.name.lower()
        suggestions: List[str] = []
        try:
            if parent.exists() and parent.is_dir():
                for entry in sorted(parent.iterdir(), key=lambda item: item.name.lower()):
                    name = entry.name
                    if base in name.lower() or name.lower() in base:
                        suggestions.append(name)
                    if len(suggestions) >= 3:
                        break
        except Exception:
            suggestions = []
        if suggestions:
            return f"Error: file not found: {self._display_path(resolved)}\nDid you mean:\n" + "\n".join(suggestions)
        return f"Error: file not found: {self._display_path(resolved)}"

    def _is_allowed_write_path(self, raw_path: str, cwd: Path) -> bool:
        if not raw_path:
            return True
        p = Path(raw_path)
        try:
            resolved = (cwd / p).resolve() if not p.is_absolute() else p.resolve()
        except Exception:
            return False
        for prefix in self._allowed_write_prefixes:
            try:
                if resolved == prefix or resolved.is_relative_to(prefix):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _extract_strace_quoted_strings(line: str) -> List[str]:
        vals = re.findall(r'"((?:[^"\\]|\\.)*)"', line)
        out: List[str] = []
        for v in vals:
            try:
                out.append(bytes(v, "utf-8").decode("unicode_escape"))
            except Exception:
                out.append(v)
        return out

    def _check_strace_violation(self, line: str, cwd: Path) -> Optional[str]:
        # Network: only localhost:<runtime_port> is allowed for AF_INET/AF_INET6 connect.
        if " connect(" in f" {line}":
            if "AF_INET" in line or "AF_INET6" in line:
                host_ok = ('"127.0.0.1"' in line) or ('"::1"' in line) or ('"localhost"' in line)
                port_ok = f"htons({self.runtime_port})" in line
                if not (host_ok and port_ok):
                    return "network policy violation: only localhost runtime port is allowed"

        # Filesystem write policy: no writes outside workspace(/tmp,/var/tmp allowed).
        file_write_syscall = any(
            token in line
            for token in [
                " open(",
                " openat(",
                " creat(",
                " unlink(",
                " unlinkat(",
                " rename(",
                " renameat(",
                " renameat2(",
                " mkdir(",
                " rmdir(",
                " truncate(",
                " ftruncate(",
                " chmod(",
                " chown(",
            ]
        )
        if file_write_syscall:
            writes_via_open = (
                ((" open(" in f" {line}") or (" openat(" in f" {line}"))
                and any(flag in line for flag in ["O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"])
            )
            destructive_call = any(
                x in line
                for x in [" creat(", " unlink(", " unlinkat(", " rename(", " renameat(", " renameat2(", " mkdir(", " rmdir(", " truncate(", " chmod(", " chown("]
            )
            if writes_via_open or destructive_call:
                for path_text in self._extract_strace_quoted_strings(line):
                    if path_text.startswith("/") or path_text.startswith(".") or "/" in path_text:
                        if not self._is_allowed_write_path(path_text, cwd):
                            return f"filesystem policy violation: write outside workspace is blocked ({path_text})"
        return None

    def _run_supervised_command(self, command: str, cwd: Path) -> Dict[str, Any]:
        if not command.strip():
            return {"exit_code": 1, "stdout": "", "stderr": "Error: command is empty"}

        script_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".bash_exec_",
                suffix=".sh",
                delete=False,
            ) as script_file:
                script_file.write("exec 2>&1\n")
                script_file.write(command)
                if not command.endswith("\n"):
                    script_file.write("\n")
                script_path = Path(script_file.name)
        except Exception as exc:
            return {"exit_code": 1, "stdout": "", "stderr": f"Error: failed to prepare command script: {exc}"}

        strace_path = shutil.which("strace")
        supervision_enabled = bool(strace_path)
        if supervision_enabled:
            cmd = [strace_path or "strace", "-f", "-e", "trace=file,network", "-s", "1024", "bash", str(script_path)]
        else:
            cmd = ["bash", str(script_path)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", "/root"),
                "PYTHONPATH": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "RUNTIME_API_BASE": f"http://127.0.0.1:{self.runtime_port}",
            },
            bufsize=1,
        )

        stdout_chunks: List[str] = []
        strace_chunks: List[str] = []
        violation: Dict[str, str] = {}

        def _read_stdout() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                stdout_chunks.append(line)

        def _read_stderr_and_check() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                strace_chunks.append(line)
                if not supervision_enabled or "msg" in violation:
                    continue
                v = self._check_strace_violation(line, cwd)
                if v:
                    violation["msg"] = v
                    try:
                        os.killpg(proc.pid, 9)
                    except Exception:
                        pass

        t_out = threading.Thread(target=_read_stdout, daemon=True)
        t_err = threading.Thread(target=_read_stderr_and_check, daemon=True)
        t_out.start()
        t_err.start()
        try:
            proc.wait(timeout=self.shell_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except Exception:
                pass
            return {
                "exit_code": 124,
                "stdout": self._truncate_output("".join(stdout_chunks)),
                "stderr": self._truncate_output("Error: command timed out"),
            }
        finally:
            t_out.join(timeout=0.5)
            t_err.join(timeout=0.5)
            if script_path is not None:
                try:
                    script_path.unlink(missing_ok=True)
                except Exception:
                    pass

        if "msg" in violation:
            return {
                "exit_code": 126,
                "stdout": self._truncate_output("".join(stdout_chunks)),
                "stderr": self._truncate_output(violation["msg"]),
            }

        debug_stderr = ""
        if int(proc.returncode) != 0 and not stdout_chunks:
            debug_stderr = self._truncate_output("".join(strace_chunks))
        elif not supervision_enabled:
            debug_stderr = ""
        return {"exit_code": int(proc.returncode), "stdout": self._truncate_output("".join(stdout_chunks)), "stderr": debug_stderr}

    def handle_bash_exec(self, args: Dict[str, Any]) -> str:
        command = str(args.get("command", ""))
        exec_dir = str(args.get("exec_dir", ".")).strip() or "."
        try:
            resolved_dir = self._resolve_workspace_path(exec_dir)
        except Exception as exc:
            return f"Error: invalid exec_dir: {exc}"
        if not self.workspace_hard_limited:
            over_limit_before, current_bytes = self._workspace_over_limit()
            if over_limit_before and not self._is_cleanup_or_inspection_command(command):
                return self._format_shell_result(
                    {
                        "exit_code": 122,
                        "stdout": "",
                        "stderr": self._workspace_limit_message(current_bytes),
                    }
                )
        out = self._run_supervised_command(command, resolved_dir)
        if self._looks_like_space_error(out):
            usage = self._workspace_usage_bytes()
            warning = self._workspace_limit_message(usage)
            stderr = str(out.get("stderr", "") or "")
            out["stderr"] = f"{stderr}\n{warning}".strip() if stderr else warning
        elif not self.workspace_hard_limited:
            over_limit_after, after_bytes = self._workspace_over_limit()
            if over_limit_after:
                warning = self._workspace_limit_message(after_bytes)
                stdout = str(out.get("stdout", "") or "")
                out["exit_code"] = 122
                out["stdout"] = stdout
                out["stderr"] = warning
        return self._format_shell_result(out)

    @staticmethod
    def _format_shell_result(out: Dict[str, Any]) -> str:
        exit_code = int(out.get("exit_code", 1))
        stdout = str(out.get("stdout", "") or "")
        stderr = str(out.get("stderr", "") or "")
        return (
            f"exit_code: {exit_code}\n"
            "stdout:\n"
            f"{stdout}\n"
            "stderr:\n"
            f"{stderr}"
        )

    def _truncate_output(self, text: str) -> str:
        if len(text) <= self.output_limit:
            return text
        keep_each = max(128, self.output_limit // 2)
        if keep_each * 2 >= len(text):
            return text[: self.output_limit]
        head = text[:keep_each]
        tail = text[-keep_each:]
        omitted = len(text) - (keep_each * 2)
        return f"{head}\n...[truncated {omitted} chars]...\n{tail}"

    def _workspace_over_limit(self) -> tuple[bool, int]:
        current = self._workspace_usage_bytes()
        if self.workspace_size_limit_bytes <= 0:
            return False, current
        return current > self.workspace_size_limit_bytes, current

    def _workspace_usage_bytes(self) -> int:
        total = 0
        for root, dirs, files in os.walk(self.workspace, topdown=True, followlinks=False):
            filtered_dirs: List[str] = []
            for dirname in dirs:
                path = Path(root) / dirname
                try:
                    st = path.lstat()
                except OSError:
                    continue
                if stat.S_ISLNK(st.st_mode):
                    continue
                filtered_dirs.append(dirname)
            dirs[:] = filtered_dirs
            for filename in files:
                path = Path(root) / filename
                try:
                    st = path.lstat()
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    total += int(st.st_size)
        return total

    def _largest_workspace_entries(self, limit: int = 5) -> List[tuple[str, int]]:
        items: List[tuple[str, int]] = []
        for root, _, files in os.walk(self.workspace, topdown=True, followlinks=False):
            for filename in files:
                path = Path(root) / filename
                try:
                    st = path.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                try:
                    rel = path.relative_to(self.workspace).as_posix()
                except ValueError:
                    rel = path.name
                items.append((rel, int(st.st_size)))
        items.sort(key=lambda item: item[1], reverse=True)
        return items[: max(1, limit)]

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        value = float(max(0, int(num_bytes)))
        units = ["B", "KB", "MB", "GB", "TB"]
        unit = units[0]
        for candidate in units:
            unit = candidate
            if value < 1024.0 or candidate == units[-1]:
                break
            value /= 1024.0
        if unit == "B":
            return f"{int(value)}{unit}"
        return f"{value:.1f}{unit}"

    def _workspace_limit_message(self, current_bytes: int) -> str:
        if self.workspace_size_limit_bytes <= 0:
            return "Workspace storage limit reached. Free up space, reduce new content, or delete existing files before writing again."
        top_entries = self._largest_workspace_entries()
        lines = [
            "Workspace storage limit reached.",
            "Free up space, reduce new content, or delete existing files before writing again.",
            f"current_usage={self._format_bytes(current_bytes)} limit={self._format_bytes(self.workspace_size_limit_bytes)}",
        ]
        if top_entries:
            lines.append("largest_files:")
            for rel, size in top_entries:
                lines.append(f"- {rel} ({self._format_bytes(size)})")
        return "\n".join(lines)

    def workspace_budget_status(self) -> str:
        """Return a short workspace budget summary string.

        Used by the state machine to inject a workspace-usage reminder
        after file-modifying tool calls.
        """
        current = self._workspace_usage_bytes()
        limit = self.workspace_size_limit_bytes
        if limit <= 0:
            return f"workspace_usage={self._format_bytes(current)} limit=unlimited"
        remaining = max(0, limit - current)
        pct = min(100.0, (current / limit) * 100.0) if limit > 0 else 0.0
        return (
            f"workspace_usage={self._format_bytes(current)}/{self._format_bytes(limit)} "
            f"remaining={self._format_bytes(remaining)} ({100.0 - pct:.0f}% free)"
        )

    @staticmethod
    def _is_cleanup_or_inspection_command(command: str) -> bool:
        text = str(command or "").strip()
        if not text:
            return False
        lowered = " ".join(text.lower().split())
        if any(token in lowered for token in [">", ">>", "| tee", " tee ", "cp ", "mv ", "mkdir ", "touch ", "truncate ", "dd "]):
            return False
        allowed_prefixes = (
            "rm ",
            "find ",
            "du ",
            "ls",
            "pwd",
            "rg ",
            "grep ",
            "wc ",
            "head ",
            "tail ",
            "sed ",
            "cat ",
            "stat ",
            "tree",
            "git status",
            "git diff",
        )
        return lowered.startswith(allowed_prefixes)

    @staticmethod
    def _looks_like_space_error(out: Dict[str, Any]) -> bool:
        joined = "\n".join(
            [
                str(out.get("stdout", "") or ""),
                str(out.get("stderr", "") or ""),
            ]
        ).lower()
        if not joined.strip():
            return False
        patterns = [
            "no space left on device",
            "disk quota exceeded",
            "[errno 28]",
            "[errno 122]",
            "file too large",
        ]
        return any(pattern in joined for pattern in patterns)

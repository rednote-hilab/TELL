import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from agents.tell_agent import tools as tell_tools
from agents.tell_agent.config import get_workspace_size_limit_bytes
from agents.tell_agent.tool_handlers import TELLToolHandlers
from agents.tell_agent import workspace_volume as workspace_volume_module
from agents.tell_agent.workspace_volume import create_workspace_volume


def test_deep_analysis_tool_fans_out_subagents_with_objective_and_guidance():
    calls = []

    def fake_runner(kind, args):
        calls.append((kind, args))
        return f"done:{args['objective']}"

    handler = TELLToolHandlers(
        workspace=Path("."),
        memory_root=Path("."),
        runtime_port=8000,
        shell_timeout=30.0,
        subagent_runner=fake_runner,
    )

    raw = handler.handle_deep_analysis(
        {
            "subagents": [
                {"subagent_id": "s1", "objective": "alpha", "guidance": "inspect left branch"},
                {
                    "subagent_id": "s2",
                    "objective": "beta",
                    "guidance": "inspect right branch",
                    "task_spec": {"focus": "delta"},
                },
            ],
        }
    )
    payload = json.loads(raw)

    assert payload["subagent_count"] == 2
    assert payload["ok"] is True
    assert {item["subagent_id"] for item in payload["results"]} == {"s1", "s2"}
    assert len(calls) == 2
    seen = {str(args["subagent_id"]): args for kind, args in calls if kind == "deep_analysis"}
    assert seen["s1"]["objective"] == "alpha"
    assert seen["s1"]["guidance"] == "inspect left branch"
    assert seen["s2"]["objective"] == "beta"


def test_load_tools_merges_yaml_tools_with_defaults(monkeypatch):
    monkeypatch.setattr(
        tell_tools,
        "get_tools_config",
        lambda: [
            {"name": "bash_exec", "description": "yaml override", "parameters": {"type": "object"}},
            {"name": "custom_yaml_tool", "description": "custom", "parameters": {"type": "object"}},
        ],
    )

    tools = tell_tools.load_tools()
    names = [str(tool.get("name") or "") for tool in tools]

    assert names.count("bash_exec") == 1
    assert "custom_yaml_tool" in names
    assert "run_deep_analysis" in names


def test_workspace_size_limit_parser(monkeypatch):
    monkeypatch.setattr(
        "agents.tell_agent.config.get_workspace_config",
        lambda: {"size_limit": "1.5MB"},
    )
    assert get_workspace_size_limit_bytes(0) == int(1.5 * 1024 * 1024)


def test_bash_exec_reports_when_workspace_crosses_size_limit(tmp_path):
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
        workspace_size_limit_bytes=10,
    )

    raw = handler.handle_bash_exec({"command": "printf '12345678901' > over.txt"})

    assert "Workspace storage limit reached." in raw
    assert (tmp_path / "over.txt").exists()


def test_bash_exec_blocks_new_write_when_workspace_already_over_limit(tmp_path):
    (tmp_path / "existing.txt").write_text("12345678901", encoding="utf-8")
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
        workspace_size_limit_bytes=10,
    )

    raw = handler.handle_bash_exec({"command": "printf 'x' > blocked.txt"})

    assert "exit_code: 122" in raw
    assert "Workspace storage limit reached." in raw
    assert not (tmp_path / "blocked.txt").exists()


def test_bash_exec_hard_limit_mode_maps_enospc_to_friendly_message(tmp_path, monkeypatch):
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
        workspace_size_limit_bytes=10,
        workspace_hard_limited=True,
    )
    monkeypatch.setattr(
        handler,
        "_run_supervised_command",
        lambda command, cwd: {
            "exit_code": 1,
            "stdout": "OSError: [Errno 28] No space left on device",
            "stderr": "",
        },
    )

    out = handler.handle_bash_exec({"command": "python3 -c 'print(1)' "})

    assert "Workspace storage limit reached." in out
    assert "No space left on device" in out


def test_bash_exec_temp_script_is_created_outside_workspace(tmp_path, monkeypatch):
    handler = TELLToolHandlers(
        workspace=tmp_path,
        memory_root=tmp_path,
        runtime_port=8000,
        shell_timeout=30.0,
    )
    seen = {}
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def fake_named_temporary_file(*args, **kwargs):
        seen["dir"] = kwargs.get("dir")
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr("agents.tell_agent.tool_handlers.tempfile.NamedTemporaryFile", fake_named_temporary_file)

    out = handler._run_supervised_command("printf ok", tmp_path)

    assert out["exit_code"] == 0
    assert out["stdout"] == "ok"
    assert seen["dir"] is None


def test_create_workspace_volume_directory_backend(tmp_path):
    volume = create_workspace_volume(
        base_root=tmp_path,
        workspace_name="demo",
        size_limit_bytes=1024,
        backend="directory",
        preserve_image=False,
        cleanup_stale=False,
    )

    assert volume.backend == "directory"
    assert volume.hard_limited is False
    assert volume.workspace == tmp_path / "demo"
    assert volume.workspace.is_dir()


def test_create_workspace_volume_linux_loop_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace_volume_module.sys, "platform", "linux")
    monkeypatch.setattr(workspace_volume_module.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        _ = kwargs
        if cmd[0] == "mkfs.ext4":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["mount", "-o", "loop"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(workspace_volume_module.subprocess, "run", fake_run)

    volume = create_workspace_volume(
        base_root=tmp_path,
        workspace_name="linux_demo",
        size_limit_bytes=1024 * 1024,
        backend="disk_image",
        preserve_image=False,
        cleanup_stale=False,
    )

    assert volume.backend == "linux_loop"
    assert volume.hard_limited is True
    assert volume.workspace == tmp_path / "_workspace_mounts" / "linux_demo"
    assert volume.image_path == tmp_path / "_workspace_images" / "linux_demo.img"
    assert volume.metadata_path == tmp_path / "_workspace_meta" / "linux_demo.json"
    assert volume.workspace.is_dir()
    assert volume.image_path is not None and volume.image_path.exists()
    assert volume.metadata_path is not None and volume.metadata_path.exists()


def test_create_workspace_volume_linux_loop_failure_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace_volume_module.sys, "platform", "linux")
    monkeypatch.setattr(workspace_volume_module.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        _ = kwargs
        if cmd[0] == "mkfs.ext4":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["mount", "-o", "loop"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="operation not permitted")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(workspace_volume_module.subprocess, "run", fake_run)

    volume = create_workspace_volume(
        base_root=tmp_path,
        workspace_name="linux_fallback",
        size_limit_bytes=1024 * 1024,
        backend="auto",
        preserve_image=False,
        cleanup_stale=False,
    )

    assert volume.backend == "directory"
    assert volume.hard_limited is False
    assert volume.workspace == tmp_path / "linux_fallback"
    assert volume.workspace.is_dir()

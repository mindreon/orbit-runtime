"""Running a verification command inside a workspace, per backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from orbit_worker import workspace_exec
from orbit_worker.workspace import (
    DockerWorkspaceAdapter,
    LocalWorkspaceAdapter,
    OpenSandboxWorkspaceAdapter,
    PersistentWorkspaceAdapter,
    WorkspaceError,
    WorkspaceLease,
)
from orbit_worker.workspace_exec import run_command


class _NoStore:
    async def acquire_workspace_lease(self, **kwargs: Any) -> None: ...
    async def renew_workspace_lease(self, **kwargs: Any) -> None: ...
    async def release_workspace_lease(self, **kwargs: Any) -> None: ...


async def test_local_command_runs_in_the_workspace_and_reports_the_exit_code(tmp_path: Path) -> None:
    adapter = LocalWorkspaceAdapter(tmp_path)
    lease = await adapter.acquire("tenant-a", "task-a")
    (tmp_path / "tenant-a" / lease.workspace_id / "data.txt").write_text("seven", encoding="utf-8")
    ok = await run_command(adapter, lease, "cat data.txt", timeout_s=10)
    assert (ok.exit_code, ok.output.strip(), ok.timed_out) == (0, "seven", False)
    bad = await run_command(adapter, lease, "echo boom >&2; exit 3", timeout_s=10)
    assert (bad.exit_code, bad.output.strip()) == (3, "boom")
    await adapter.release(lease)


async def test_local_command_does_not_see_the_workers_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORBIT_CHECKPOINT_FERNET_KEY", "super-secret")
    adapter = LocalWorkspaceAdapter(tmp_path)
    lease = await adapter.acquire("tenant-a", "task-a")
    outcome = await run_command(adapter, lease, "env", timeout_s=10)
    assert "super-secret" not in outcome.output
    await adapter.release(lease)


async def test_local_command_that_runs_too_long_is_killed(tmp_path: Path) -> None:
    adapter = LocalWorkspaceAdapter(tmp_path)
    lease = await adapter.acquire("tenant-a", "task-a")
    marker = tmp_path / "tenant-a" / lease.workspace_id / "child-survived"
    started = asyncio.get_running_loop().time()
    outcome = await run_command(adapter, lease, f"(sleep 3; touch {marker}) & sleep 30", timeout_s=1)
    assert outcome.timed_out is True and outcome.exit_code is None
    assert asyncio.get_running_loop().time() - started < 10
    await asyncio.sleep(3.5)
    assert not marker.exists(), "the whole process group must be killed, not just the shell"
    await adapter.release(lease)


async def test_persistent_wrapper_is_unwrapped(tmp_path: Path) -> None:
    adapter = PersistentWorkspaceAdapter(LocalWorkspaceAdapter(tmp_path), _NoStore())
    lease = await adapter.acquire("tenant-a", "task-a", holder="verify")
    outcome = await run_command(adapter, lease, "echo hi", timeout_s=10)
    assert outcome.exit_code == 0 and outcome.output.strip() == "hi"
    await adapter.release(lease)


class _FakeProc:
    def __init__(self, returncode: int, output: bytes, hang: bool = False) -> None:
        self.returncode, self._output, self._hang = returncode, output, hang
        self.killed = False
        self.pid = 4242

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(60)
        return self._output, b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


async def test_docker_command_execs_in_the_named_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        calls.append(args)
        return _FakeProc(1, b"failed\n")

    monkeypatch.setattr(workspace_exec.asyncio, "create_subprocess_exec", fake_exec)
    adapter = DockerWorkspaceAdapter(tmp_path, "image")
    lease = WorkspaceLease("ws_abc", "task-a", "tenant-a", "docker", False, 9e12)
    outcome = await run_command(adapter, lease, "pytest -q", timeout_s=5)
    assert calls == [("docker", "exec", "-w", "/workspace", "orbit-ws_abc", "sh", "-c", "pytest -q")]
    assert (outcome.exit_code, outcome.output.strip()) == (1, "failed")


async def test_docker_command_timeout_kills_the_exec_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProc(0, b"", hang=True)

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(workspace_exec.asyncio, "create_subprocess_exec", fake_exec)
    adapter = DockerWorkspaceAdapter(tmp_path, "image")
    lease = WorkspaceLease("ws_abc", "task-a", "tenant-a", "docker", False, 9e12)
    outcome = await run_command(adapter, lease, "sleep 99", timeout_s=1)
    assert outcome.timed_out is True and proc.killed is True


class _FakeExecution:
    def __init__(self, exit_code: int | None, text: str, error: Any = None) -> None:
        self.exit_code, self.text, self.error = exit_code, text, error


class _FakeCommands:
    def __init__(self, execution: _FakeExecution) -> None:
        self.execution, self.calls = execution, []

    async def run(self, command: str, *, opts: Any = None) -> _FakeExecution:
        self.calls.append((command, opts))
        return self.execution


def _opensandbox(tmp_path: Path, execution: _FakeExecution) -> tuple[OpenSandboxWorkspaceAdapter, WorkspaceLease, _FakeCommands]:
    adapter = OpenSandboxWorkspaceAdapter(connection_config=object(), image="img")
    lease = WorkspaceLease("sbx-1", "task-a", "tenant-a", "opensandbox", False, 9e12)
    commands = _FakeCommands(execution)
    adapter._leases["sbx-1"] = lease
    adapter._sandboxes["sbx-1"] = type("Sandbox", (), {"commands": commands})()
    return adapter, lease, commands


async def test_opensandbox_command_uses_the_workspace_directory_and_timeout(tmp_path: Path) -> None:
    adapter, lease, commands = _opensandbox(tmp_path, _FakeExecution(0, "green\n"))
    outcome = await run_command(adapter, lease, "pytest -q", timeout_s=30)
    assert (outcome.exit_code, outcome.output.strip()) == (0, "green")
    command, opts = commands.calls[0]
    assert command == "pytest -q"
    assert opts.working_directory == "/workspace" and opts.timeout.total_seconds() == 30


async def test_opensandbox_execution_error_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    error = type("Err", (), {"name": "ExecError", "value": "cannot start"})()
    adapter, lease, _ = _opensandbox(tmp_path, _FakeExecution(None, "", error))
    outcome = await run_command(adapter, lease, "pytest -q", timeout_s=30)
    assert outcome.exit_code not in (0, None) or "cannot start" in outcome.output
    assert outcome.failures("pytest -q")


async def test_unknown_backend_cannot_run_commands() -> None:
    lease = WorkspaceLease("ws", "t", "tenant", "x", False, 9e12)
    with pytest.raises(WorkspaceError):
        await run_command(object(), lease, "true", timeout_s=1)  # type: ignore[arg-type]


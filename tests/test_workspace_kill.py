"""Killing a sandbox by the ids of its lease, for the lease reaper (17 G6)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from orbit_worker.workspace import (
    DockerWorkspaceAdapter,
    LocalWorkspaceAdapter,
    OpenSandboxWorkspaceAdapter,
    WorkspaceError,
    keep_lease_alive,
)

WORKSPACE = "ws_" + "a" * 32


async def test_local_kill_removes_the_workspace_a_crashed_worker_left(tmp_path: Path) -> None:
    lease = await LocalWorkspaceAdapter(tmp_path).acquire("tenant-a", "task-a")
    (tmp_path / "tenant-a" / lease.workspace_id / "scratch.txt").write_text("x")
    # A different adapter instance, like the worker that started after the crash: no in-process state.
    successor = LocalWorkspaceAdapter(tmp_path)
    await successor.kill("tenant-a", lease.workspace_id)
    assert not (tmp_path / "tenant-a" / lease.workspace_id).exists()
    assert not (tmp_path / "tenant-a" / f"{lease.workspace_id}.lease").exists()
    await successor.kill("tenant-a", lease.workspace_id)  # already gone: fine


@pytest.mark.parametrize("workspace_id", ["../x", "ws_short", "task-1", "ws_" + "A" * 32, ""])
async def test_kill_refuses_an_id_that_is_not_a_workspace(tmp_path: Path, workspace_id: str) -> None:
    with pytest.raises(WorkspaceError):
        await LocalWorkspaceAdapter(tmp_path).kill("tenant-a", workspace_id)


async def test_kill_stays_inside_the_root(tmp_path: Path) -> None:
    keep = tmp_path / "keep"
    keep.mkdir()
    with pytest.raises(WorkspaceError):
        await LocalWorkspaceAdapter(tmp_path / "root").kill("..", WORKSPACE)
    assert keep.exists()


class _Proc:
    def __init__(self, code: int, error: bytes = b"") -> None:
        self.returncode = code
        self._error = error

    async def communicate(self, _input=None):
        return b"", self._error

    async def wait(self):
        return self.returncode


async def test_docker_kill_removes_the_container_by_name(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*args, **_kwargs):
        calls.append(args)
        return _Proc(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await DockerWorkspaceAdapter(tmp_path, "img").kill("tenant-a", WORKSPACE)
    assert calls == [("docker", "rm", "-f", f"orbit-{WORKSPACE}")]


async def test_docker_kill_ignores_a_missing_container_and_reports_other_failures(
    tmp_path: Path, monkeypatch
) -> None:
    outcomes = iter([_Proc(1, b"Error: No such container: orbit-x"), _Proc(1, b"permission denied")])

    async def fake_exec(*_args, **_kwargs):
        return next(outcomes)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    adapter = DockerWorkspaceAdapter(tmp_path, "img")
    await adapter.kill("tenant-a", WORKSPACE)
    with pytest.raises(WorkspaceError, match="permission denied"):
        await adapter.kill("tenant-a", WORKSPACE)


async def test_docker_leases_are_recorded_as_docker(tmp_path: Path, monkeypatch) -> None:
    async def fake_exec(*_args, **_kwargs):
        proc = _Proc(0)

        async def communicate(_input=None):
            return b"container-id\n", b""

        proc.communicate = communicate  # type: ignore[method-assign]
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    lease = await DockerWorkspaceAdapter(tmp_path, "img").acquire("tenant-a", "task-a")
    assert lease.mode == "docker"


class _Manager:
    def __init__(self, error: Exception | None = None) -> None:
        self.killed: list[str] = []
        self.error = error

    async def kill_sandbox(self, sandbox_id: str) -> None:
        self.killed.append(sandbox_id)
        if self.error:
            raise self.error


def _opensandbox(manager: _Manager) -> OpenSandboxWorkspaceAdapter:
    adapter = OpenSandboxWorkspaceAdapter(connection_config=object(), image="img")
    adapter._manager = manager
    return adapter


async def test_opensandbox_kill_goes_through_the_manager_by_id() -> None:
    manager = _Manager()
    await _opensandbox(manager).kill("tenant-a", "sandbox-42")
    assert manager.killed == ["sandbox-42"]


async def test_opensandbox_kill_ignores_404_and_reports_the_rest() -> None:
    class NotFound(Exception):
        status_code = 404

    class Boom(Exception):
        status_code = 500

    await _opensandbox(_Manager(NotFound())).kill("tenant-a", "gone")
    with pytest.raises(Boom):
        await _opensandbox(_Manager(Boom())).kill("tenant-a", "broken")


async def test_keep_lease_alive_renews_until_cancelled() -> None:
    renewals: list[int] = []

    class Adapter:
        async def renew(self, lease, ttl_s):
            renewals.append(ttl_s)
            return lease

    task = asyncio.create_task(keep_lease_alive(Adapter(), object(), 0.03))  # type: ignore[arg-type]
    await asyncio.sleep(0.12)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(renewals) >= 3


async def test_keep_lease_alive_stops_when_a_renewal_fails() -> None:
    class Adapter:
        calls = 0

        async def renew(self, lease, ttl_s):
            self.calls += 1
            raise WorkspaceError("lease is gone")

    adapter = Adapter()
    lease = type("Lease", (), {"workspace_id": WORKSPACE})()
    await asyncio.wait_for(keep_lease_alive(adapter, lease, 0.03), 1)  # type: ignore[arg-type]
    assert adapter.calls == 1

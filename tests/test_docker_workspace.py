"""The Docker workspace backend against a fake `docker` CLI (17 G5).

The fake keeps the containers apart from the adapter, so a second adapter instance behaves like the worker that
started after a crash: it has no in-process state and has to find the containers by their names.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from orbit_worker.workspace import (
    DockerLimits,
    DockerWorkspaceAdapter,
    WorkspaceError,
    WorkspaceLease,
)

WORKSPACE = "ws_" + "a" * 32
NAME = f"orbit-{WORKSPACE}"


class _Proc:
    def __init__(self, fake: FakeDocker, args: tuple[str, ...]) -> None:
        self._fake, self._args = fake, args
        self.returncode = 0

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        code, out, err = self._fake.handle(self._args, input)
        self.returncode = code
        return out, err

    async def wait(self) -> int:
        await self.communicate()
        return self.returncode


class FakeDocker:
    """Just enough of `docker run/inspect/start/exec/rm` for the adapter, with containers that outlive adapters."""

    def __init__(self) -> None:
        self.containers: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, ...]] = []
        self.run_error: str | None = None

    async def __call__(self, *args: str, **_kwargs: Any) -> _Proc:
        self.calls.append(args)
        return _Proc(self, args)

    def verbs(self) -> list[str]:
        return [call[1] for call in self.calls]

    def handle(self, args: tuple[str, ...], stdin: bytes | None) -> tuple[int, bytes, bytes]:
        verb = args[1]
        return getattr(self, f"_{verb}")(args, stdin)

    def _run(self, args: tuple[str, ...], _stdin: bytes | None) -> tuple[int, bytes, bytes]:
        if self.run_error:
            self.containers[args[args.index("--name") + 1]] = {"id": "half", "status": "created", "labels": {}}
            return 125, b"", self.run_error.encode()
        name = args[args.index("--name") + 1]
        labels = dict(a.split("=", 1) for i, a in enumerate(args) if i and args[i - 1] == "--label")
        self.containers[name] = {
            "id": f"id-{name}", "status": "running", "labels": labels, "args": args, "archive": b"", "started": 0,
        }
        return 0, f"id-{name}\n".encode(), b""

    def _inspect(self, args: tuple[str, ...], _stdin: bytes | None) -> tuple[int, bytes, bytes]:
        name = args[-1]
        found = self.containers.get(name)
        if found is None:
            return 1, b"[]\n", f"Error: No such container: {name}".encode()
        body = [{"Id": found["id"], "State": {"Status": found["status"]}, "Config": {"Labels": found["labels"]}}]
        return 0, json.dumps(body).encode(), b""

    def _start(self, args: tuple[str, ...], _stdin: bytes | None) -> tuple[int, bytes, bytes]:
        self.containers[args[-1]]["status"] = "running"
        self.containers[args[-1]]["started"] += 1
        return 0, args[-1].encode(), b""

    def _exec(self, args: tuple[str, ...], stdin: bytes | None) -> tuple[int, bytes, bytes]:
        name = next(a for a in args if a.startswith("orbit-"))
        found = self.containers.get(name)
        if found is None:
            return 1, b"", f"Error response from daemon: No such container: {name}".encode()
        if found["status"] != "running":
            return 1, b"", f"container {found['id']} is not running".encode()
        if "czf" in args:
            return 0, found["archive"], b""
        found["archive"] = stdin or b""
        return 0, b"", b""

    def _rm(self, args: tuple[str, ...], _stdin: bytes | None) -> tuple[int, bytes, bytes]:
        if self.containers.pop(args[-1], None) is None:
            return 1, b"", f"Error: No such container: {args[-1]}".encode()
        return 0, args[-1].encode(), b""


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)
    return fake


def _lease(read_only: bool = False, expires_at: float | None = None) -> WorkspaceLease:
    return WorkspaceLease(
        WORKSPACE, "task-a", "tenant-a", "docker", read_only, expires_at if expires_at else time.time() + 300
    )


def _seed(docker: FakeDocker, **fields: Any) -> None:
    docker.containers[NAME] = {
        "id": "id-1", "status": "running", "archive": b"payload", "started": 0,
        "labels": {"orbit.workspace_id": WORKSPACE, "orbit.tenant_id": "tenant-a", "orbit.task_id": "task-a"},
        **fields,
    }


def _run_args(docker: FakeDocker) -> tuple[str, ...]:
    return next(call for call in docker.calls if call[1] == "run")


async def test_run_isolates_the_network_and_caps_resources(tmp_path: Path, docker: FakeDocker) -> None:
    limits = DockerLimits(cpus=1.5, memory="768m", pids_limit=100)
    lease = await DockerWorkspaceAdapter(tmp_path, "img", limits=limits).acquire("tenant-a", "task-a")
    args = _run_args(docker)
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--cpus") + 1] == "1.5"
    assert args[args.index("--memory") + 1] == "768m"
    assert args[args.index("--pids-limit") + 1] == "100"
    assert args[args.index("--name") + 1] == f"orbit-{lease.workspace_id}"
    assert "--read-only" not in args
    assert lease.mode == "docker"


async def test_run_has_default_limits_and_labels_the_owner(tmp_path: Path, docker: FakeDocker) -> None:
    lease = await DockerWorkspaceAdapter(tmp_path, "img").acquire("tenant-a", "task-a")
    args = _run_args(docker)
    for flag in ("--cpus", "--memory", "--pids-limit"):
        assert flag in args
    labels = docker.containers[f"orbit-{lease.workspace_id}"]["labels"]
    assert labels == {
        "orbit.workspace_id": lease.workspace_id, "orbit.tenant_id": "tenant-a", "orbit.task_id": "task-a",
    }


async def test_workspace_directory_exists_for_images_that_lack_it(tmp_path: Path, docker: FakeDocker) -> None:
    await DockerWorkspaceAdapter(tmp_path, "img").acquire("tenant-a", "task-a")
    args = _run_args(docker)
    image_at = args.index("img")
    command = " ".join(args[image_at + 1:])
    assert "mkdir -p /workspace" in command and "sleep infinity" in command


async def test_read_only_replica_keeps_a_writable_workspace_on_tmpfs(tmp_path: Path, docker: FakeDocker) -> None:
    await DockerWorkspaceAdapter(tmp_path, "img").acquire("tenant-a", "task-a", read_only=True)
    args = _run_args(docker)
    assert "--read-only" in args
    tmpfs = args[args.index("--tmpfs") + 1]
    assert tmpfs.startswith("/workspace:") and "rw" in tmpfs


async def test_failed_run_removes_what_it_left_and_reports_docker_output(tmp_path: Path, docker: FakeDocker) -> None:
    docker.run_error = "invalid memory"
    adapter = DockerWorkspaceAdapter(tmp_path, "img")
    with pytest.raises(WorkspaceError, match="invalid memory"):
        await adapter.acquire("tenant-a", "task-a")
    assert docker.containers == {}
    assert list((tmp_path / "tenant-a").glob("*.lease")) == []


async def test_a_fresh_process_takes_over_the_running_container(tmp_path: Path, docker: FakeDocker) -> None:
    first = DockerWorkspaceAdapter(tmp_path, "img")
    lease = await first.acquire("tenant-a", "task-a")
    docker.containers[f"orbit-{lease.workspace_id}"]["archive"] = b"work in progress"

    restarted = DockerWorkspaceAdapter(tmp_path, "img")
    assert restarted._containers == {} and restarted._leases == {}
    assert await restarted.get_archive(lease) == b"work in progress"
    assert "inspect" in docker.verbs()
    assert "start" not in docker.verbs()


async def test_a_fresh_process_can_put_snapshot_renew_and_release(tmp_path: Path, docker: FakeDocker) -> None:
    lease = await DockerWorkspaceAdapter(tmp_path, "img").acquire("tenant-a", "task-a")
    restarted = DockerWorkspaceAdapter(tmp_path, "img")
    await restarted.put_archive(lease, b"restored")
    assert docker.containers[f"orbit-{lease.workspace_id}"]["archive"] == b"restored"
    renewed = await restarted.renew(lease, 600)
    assert renewed.expires_at > lease.expires_at
    assert (await restarted.snapshot(renewed)).startswith("sha256:")
    await restarted.release(renewed)
    assert docker.containers == {}
    assert restarted._leases == {} and restarted._containers == {}


async def test_the_container_is_looked_up_once_per_process(tmp_path: Path, docker: FakeDocker) -> None:
    lease = await DockerWorkspaceAdapter(tmp_path, "img").acquire("tenant-a", "task-a")
    restarted = DockerWorkspaceAdapter(tmp_path, "img")
    await restarted.get_archive(lease)
    await restarted.get_archive(lease)
    assert docker.verbs().count("inspect") == 1


async def test_a_stopped_container_is_started_again_and_keeps_its_files(tmp_path: Path, docker: FakeDocker) -> None:
    _seed(docker, status="exited")
    assert await DockerWorkspaceAdapter(tmp_path, "img").get_archive(_lease()) == b"payload"
    assert docker.containers[NAME]["started"] == 1


async def test_a_stopped_read_only_container_lost_its_tmpfs_and_is_not_started(
    tmp_path: Path, docker: FakeDocker
) -> None:
    _seed(docker, status="exited")
    with pytest.raises(WorkspaceError, match="stopped"):
        await DockerWorkspaceAdapter(tmp_path, "img").get_archive(_lease(read_only=True))
    assert "start" not in docker.verbs()


async def test_a_container_in_another_state_is_an_error(tmp_path: Path, docker: FakeDocker) -> None:
    _seed(docker, status="dead")
    with pytest.raises(WorkspaceError, match="dead"):
        await DockerWorkspaceAdapter(tmp_path, "img").get_archive(_lease())


async def test_a_missing_container_is_a_clear_error_for_every_use(tmp_path: Path, docker: FakeDocker) -> None:
    adapter = DockerWorkspaceAdapter(tmp_path, "img")
    lease = _lease()
    for call in (adapter.get_archive(lease), adapter.put_archive(lease, b"x"), adapter.renew(lease, 60)):
        with pytest.raises(WorkspaceError, match=f"container {NAME} does not exist"):
            await call
    assert adapter._leases == {} and adapter._containers == {}


async def test_a_container_of_another_tenant_or_task_is_not_taken_over(tmp_path: Path, docker: FakeDocker) -> None:
    _seed(docker, labels={"orbit.workspace_id": WORKSPACE, "orbit.tenant_id": "tenant-b", "orbit.task_id": "task-a"})
    with pytest.raises(WorkspaceError, match="does not belong"):
        await DockerWorkspaceAdapter(tmp_path, "img").get_archive(_lease())
    _seed(docker, labels={})
    with pytest.raises(WorkspaceError, match="does not belong"):
        await DockerWorkspaceAdapter(tmp_path, "img").get_archive(_lease())


async def test_an_expired_lease_is_not_taken_over(tmp_path: Path, docker: FakeDocker) -> None:
    _seed(docker)
    with pytest.raises(WorkspaceError, match="expired"):
        await DockerWorkspaceAdapter(tmp_path, "img").get_archive(_lease(expires_at=time.time() - 1))
    assert "inspect" not in docker.verbs()


async def test_an_id_that_is_not_a_workspace_is_refused_before_docker_is_called(
    tmp_path: Path, docker: FakeDocker
) -> None:
    lease = WorkspaceLease("../evil", "task-a", "tenant-a", "docker", False, time.time() + 60)
    with pytest.raises(WorkspaceError, match="not a workspace"):
        await DockerWorkspaceAdapter(tmp_path, "img").get_archive(lease)
    assert docker.calls == []


async def test_a_docker_failure_other_than_not_found_is_reported(tmp_path: Path, docker: FakeDocker) -> None:
    def broken(_args: tuple[str, ...], _stdin: bytes | None) -> tuple[int, bytes, bytes]:
        return 1, b"", b"Cannot connect to the Docker daemon"

    docker._inspect = broken  # type: ignore[method-assign]
    with pytest.raises(WorkspaceError, match="Cannot connect"):
        await DockerWorkspaceAdapter(tmp_path, "img").get_archive(_lease())


async def test_release_removes_the_container_by_name_without_process_state(tmp_path: Path, docker: FakeDocker) -> None:
    lease = await DockerWorkspaceAdapter(tmp_path, "img").acquire("tenant-a", "task-a")
    lock = tmp_path / "tenant-a" / f"{lease.workspace_id}.lease"
    assert lock.exists()
    await DockerWorkspaceAdapter(tmp_path, "img").release(lease)
    assert docker.containers == {}
    assert not lock.exists()


async def test_release_of_a_container_that_is_already_gone_succeeds(tmp_path: Path, docker: FakeDocker) -> None:
    await DockerWorkspaceAdapter(tmp_path, "img").release(_lease())
    assert docker.calls[-1] == ("docker", "rm", "-f", NAME)


async def test_release_reports_a_docker_failure(tmp_path: Path, docker: FakeDocker) -> None:
    docker._rm = lambda *_: (1, b"", b"permission denied")  # type: ignore[method-assign]
    with pytest.raises(WorkspaceError, match="permission denied"):
        await DockerWorkspaceAdapter(tmp_path, "img").release(_lease())


async def test_kill_by_name_needs_no_process_state_and_forgets_the_cache(tmp_path: Path, docker: FakeDocker) -> None:
    adapter = DockerWorkspaceAdapter(tmp_path, "img")
    lease = await adapter.acquire("tenant-a", "task-a")
    await DockerWorkspaceAdapter(tmp_path, "img").kill("tenant-a", lease.workspace_id)
    assert docker.containers == {}
    await adapter.kill("tenant-a", lease.workspace_id)  # already gone: fine
    assert adapter._containers == {} and adapter._leases == {}

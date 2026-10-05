"""The Docker backend against a real Docker daemon: a worker that restarts finds its workspace again (17 G5).

Skipped when there is no Docker or the small image cannot be had; the CI database gate (ORBIT_REQUIRE_PG_TESTS) does
not require it.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from orbit_worker.workspace import DockerLimits, DockerWorkspaceAdapter, WorkspaceError

IMAGE = os.environ.get("ORBIT_TEST_DOCKER_IMAGE", "alpine:3.20")


def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)


def _image_ready() -> bool:
    if shutil.which("docker") is None or _docker("info", timeout=20).returncode:
        return False
    return _docker("image", "inspect", IMAGE).returncode == 0 or _docker("pull", "-q", IMAGE).returncode == 0


pytestmark = pytest.mark.skipif(not _image_ready(), reason=f"no Docker daemon or image {IMAGE}")


def _archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return output.getvalue()


def _files(archive: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return {
            member.name.removeprefix("./"): tar.extractfile(member).read()  # type: ignore[union-attr]
            for member in tar.getmembers()
            if member.isfile()
        }


def _adapter(root: Path) -> DockerWorkspaceAdapter:
    return DockerWorkspaceAdapter(root, IMAGE, limits=DockerLimits(cpus=0.5, memory="128m", pids_limit=64))


@pytest.fixture
async def workspaces(tmp_path: Path) -> AsyncIterator[list[str]]:
    made: list[str] = []
    yield made
    for workspace_id in made:
        await _adapter(tmp_path).kill("tenant-a", workspace_id)


async def test_a_restarted_worker_finds_the_workspace_it_left(tmp_path: Path, workspaces: list[str]) -> None:
    before = _adapter(tmp_path)
    lease = await before.acquire("tenant-a", "task-a")
    workspaces.append(lease.workspace_id)
    await before.put_archive(lease, _archive({"notes/result.txt": b"written before the crash"}))

    # A new adapter instance is the worker after the restart: nothing in memory, only the container's name.
    after = _adapter(tmp_path)
    assert after._containers == {}
    assert _files(await after.get_archive(lease))["notes/result.txt"] == b"written before the crash"
    await after.put_archive(lease, _archive({"more.txt": b"and after"}))
    assert set(_files(await after.get_archive(lease))) == {"notes/result.txt", "more.txt"}

    await after.release(lease)
    assert _docker("inspect", f"orbit-{lease.workspace_id}").returncode != 0
    with pytest.raises(WorkspaceError, match="does not exist"):
        await _adapter(tmp_path).get_archive(lease)


async def test_the_container_has_no_network_and_the_configured_limits(tmp_path: Path, workspaces: list[str]) -> None:
    lease = await _adapter(tmp_path).acquire("tenant-a", "task-a")
    workspaces.append(lease.workspace_id)
    host = json.loads(_docker("inspect", f"orbit-{lease.workspace_id}").stdout)[0]["HostConfig"]
    assert host["NetworkMode"] == "none"
    assert host["Memory"] == 128 * 1024 * 1024
    assert host["NanoCpus"] == 500_000_000
    assert host["PidsLimit"] == 64
    interfaces = _docker("exec", f"orbit-{lease.workspace_id}", "ls", "/sys/class/net").stdout.split()
    assert interfaces == ["lo"]


async def test_a_stopped_container_is_started_again_with_its_files(tmp_path: Path, workspaces: list[str]) -> None:
    lease = await _adapter(tmp_path).acquire("tenant-a", "task-a")
    workspaces.append(lease.workspace_id)
    await _adapter(tmp_path).put_archive(lease, _archive({"kept.txt": b"survives a stop"}))
    assert _docker("stop", "-t", "1", f"orbit-{lease.workspace_id}").returncode == 0

    assert _files(await _adapter(tmp_path).get_archive(lease))["kept.txt"] == b"survives a stop"


async def test_a_read_only_replica_has_a_readable_workspace(tmp_path: Path, workspaces: list[str]) -> None:
    lease = await _adapter(tmp_path).acquire("tenant-a", "task-a", read_only=True)
    workspaces.append(lease.workspace_id)
    assert _files(await _adapter(tmp_path).get_archive(lease)) == {}
    # A replica is a copy of its own (a member of a team stage works in one): /workspace takes writes, nothing else does.
    await _adapter(tmp_path).put_archive(lease, _archive({"x": b"y"}))
    assert _files(await _adapter(tmp_path).get_archive(lease)) == {"x": b"y"}
    written = _docker("exec", f"orbit-{lease.workspace_id}", "sh", "-c", "touch /etc/x")
    assert written.returncode != 0  # the root file system is read-only


async def test_kill_removes_the_container_of_a_worker_that_is_gone(tmp_path: Path) -> None:
    lease = await _adapter(tmp_path).acquire("tenant-a", "task-a")
    await _adapter(tmp_path).kill("tenant-a", lease.workspace_id)
    assert _docker("inspect", f"orbit-{lease.workspace_id}").returncode != 0

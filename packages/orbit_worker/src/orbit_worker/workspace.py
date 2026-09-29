"""Workspace leases and archive based snapshots.

The worker talks to this small protocol only.  Local and Docker backends are
dependency free; the OpenSandbox backend accepts an SDK client from the
process that owns the SDK version, keeping the activity code stable.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import tarfile
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceLease:
    workspace_id: str
    task_id: str
    tenant_id: str
    mode: str
    read_only: bool
    expires_at: float


class WorkspaceAdapter(Protocol):
    async def acquire(self, tenant_id: str, task_id: str, *, read_only: bool = False) -> WorkspaceLease: ...
    async def renew(self, lease: WorkspaceLease, ttl_s: int = 300) -> WorkspaceLease: ...
    async def snapshot(self, lease: WorkspaceLease) -> str: ...
    async def restore(self, lease: WorkspaceLease, snapshot_ref: str) -> None: ...
    async def release(self, lease: WorkspaceLease) -> None: ...
    async def get_archive(self, lease: WorkspaceLease) -> bytes: ...
    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None: ...


class SandboxKiller(Protocol):
    """What the lease reaper needs from a backend (17 G6): destroy the sandbox of a lease by its ids alone, so it
    works for a lease that another worker, or an earlier run of this one, acquired. Killing what is already gone
    succeeds."""

    backend: str

    async def kill(self, tenant_id: str, workspace_id: str) -> None: ...


class LeaseStore(Protocol):
    async def acquire_workspace_lease(self, **kwargs: Any) -> None: ...
    async def renew_workspace_lease(self, **kwargs: Any) -> None: ...
    async def release_workspace_lease(self, **kwargs: Any) -> None: ...


class SnapshotStore(Protocol):
    async def put_snapshot(self, *, tenant_id: str, digest: str, payload: bytes) -> None: ...
    async def get_snapshot(self, *, tenant_id: str, digest: str) -> bytes: ...


class PersistentWorkspaceAdapter:
    """Persist workspace lease state while preserving the adapter protocol."""

    def __init__(self, adapter: WorkspaceAdapter, store: LeaseStore) -> None:
        self.adapter = adapter
        self.store = store
        # How long a lease lives without a renewal; the activity renews it well inside this.
        self.ttl_s: int = getattr(adapter, "ttl_s", 300)

    async def acquire(
        self, tenant_id: str, task_id: str, *, read_only: bool = False, holder: str | None = None
    ) -> WorkspaceLease:
        lease = await self.adapter.acquire(tenant_id, task_id, read_only=read_only)
        try:
            await self.store.acquire_workspace_lease(
                lease_id=lease.workspace_id,
                tenant_id=tenant_id,
                lease_key=f"{task_id}/ro/{lease.workspace_id}" if read_only else task_id,
                lease_mode="read" if read_only else "write",
                backend=lease.mode,
                holder_attempt=holder or task_id,
                expires_at=lease.expires_at,
                sandbox_id=lease.workspace_id if lease.mode == "opensandbox" else None,
            )
        except Exception as exc:
            await self.adapter.release(lease)
            raise WorkspaceError("workspace lease could not be persisted") from exc
        return lease

    async def renew(self, lease: WorkspaceLease, ttl_s: int = 300) -> WorkspaceLease:
        renewed = await self.adapter.renew(lease, ttl_s)
        await self.store.renew_workspace_lease(lease_id=renewed.workspace_id, tenant_id=renewed.tenant_id, expires_at=renewed.expires_at)
        return renewed

    async def snapshot(self, lease: WorkspaceLease) -> str:
        return await self.adapter.snapshot(lease)

    async def restore(self, lease: WorkspaceLease, snapshot_ref: str) -> None:
        await self.adapter.restore(lease, snapshot_ref)

    async def release(self, lease: WorkspaceLease) -> None:
        try:
            await self.adapter.release(lease)
        finally:
            await self.store.release_workspace_lease(lease_id=lease.workspace_id, tenant_id=lease.tenant_id)

    async def get_archive(self, lease: WorkspaceLease) -> bytes:
        return await self.adapter.get_archive(lease)

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        await self.adapter.put_archive(lease, archive)


async def keep_lease_alive(adapter: WorkspaceAdapter, lease: WorkspaceLease, ttl_s: int) -> None:
    """Renew `lease` every third of its ttl until cancelled. This is what tells the lease reaper (17 G6) that the
    holder is alive, so a turn that runs longer than the ttl keeps its sandbox. If a renewal fails the lease is lost
    or the store is down: it is logged and renewing stops, and the reaper decides."""
    while True:
        await asyncio.sleep(max(ttl_s / 3, 0.01))
        try:
            lease = await adapter.renew(lease, ttl_s)
        except Exception:
            logger.exception("workspace lease could not be renewed", workspace_id=lease.workspace_id)
            return


def _pack(root: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root))
    return output.getvalue()


def _unpack(root: Path, archive: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for member in source.getmembers():
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root.resolve()) + os.sep):
                raise WorkspaceError("archive contains a path outside the workspace")
        try:
            source.extractall(root, filter="data")
        except TypeError:  # Python 3.11/3.12 compatibility
            source.extractall(root)


_WORKSPACE_ID = re.compile(r"^(?:ws|ro)_[0-9a-f]{32}$")


class LocalWorkspaceAdapter:
    """Filesystem backend used by dev and tests."""

    backend = "local"

    def __init__(self, root: str | Path, *, ttl_s: int = 300) -> None:
        self.root = Path(root)
        self.ttl_s = ttl_s
        self._leases: dict[str, WorkspaceLease] = {}
        self._locks: dict[str, Path] = {}
        self._snapshots = self.root / "snapshots"

    def _path(self, lease: WorkspaceLease) -> Path:
        return self.root / lease.tenant_id / lease.workspace_id

    async def acquire(self, tenant_id: str, task_id: str, *, read_only: bool = False) -> WorkspaceLease:
        workspace_id = f"ws_{uuid.uuid4().hex}"
        if read_only:
            workspace_id = f"ro_{workspace_id[3:]}"
        lease = WorkspaceLease(workspace_id, task_id, tenant_id, "local", read_only, time.time() + self.ttl_s)
        path = self._path(lease)
        path.mkdir(parents=True, exist_ok=False)
        if not read_only:
            lock = path.with_suffix(".lease")
            try:
                lock.touch(exist_ok=False)
            except FileExistsError as exc:
                raise WorkspaceError("workspace already has a writer") from exc
            self._locks[workspace_id] = lock
        self._leases[workspace_id] = lease
        return lease

    async def renew(self, lease: WorkspaceLease, ttl_s: int = 300) -> WorkspaceLease:
        self._require(lease)
        renewed = WorkspaceLease(lease.workspace_id, lease.task_id, lease.tenant_id, lease.mode, lease.read_only, time.time() + ttl_s)
        self._leases[lease.workspace_id] = renewed
        return renewed

    async def snapshot(self, lease: WorkspaceLease) -> str:
        self._require(lease)
        payload = await self.get_archive(lease)
        digest = __import__("hashlib").sha256(payload).hexdigest()
        path = self._snapshots / lease.tenant_id / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return f"sha256:{digest}"

    async def restore(self, lease: WorkspaceLease, snapshot_ref: str) -> None:
        self._require(lease)
        if not snapshot_ref.startswith("sha256:"):
            raise WorkspaceError("snapshot reference must be content addressed")
        path = self._snapshots / lease.tenant_id / snapshot_ref.removeprefix("sha256:")
        if not path.is_file():
            raise WorkspaceError("snapshot not found")
        await self.put_archive(lease, path.read_bytes())

    async def release(self, lease: WorkspaceLease) -> None:
        self._require(lease)
        self._leases.pop(lease.workspace_id, None)
        lock = self._locks.pop(lease.workspace_id, None)
        if lock:
            lock.unlink(missing_ok=True)

    async def get_archive(self, lease: WorkspaceLease) -> bytes:
        self._require(lease)
        return _pack(self._path(lease))

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        self._require(lease)
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        _unpack(self._path(lease), archive)

    async def reap_expired(self) -> int:
        expired = [lease for lease in self._leases.values() if lease.expires_at < time.time()]
        for lease in expired:
            await self.release(lease)
        return len(expired)

    async def kill(self, tenant_id: str, workspace_id: str) -> None:
        """Remove the workspace directory and its lock, whoever created them."""
        if not _WORKSPACE_ID.fullmatch(workspace_id) or "/" in tenant_id or tenant_id in {"", ".", ".."}:
            raise WorkspaceError("not a workspace of this backend")
        path = self.root / tenant_id / workspace_id
        self._leases.pop(workspace_id, None)
        self._locks.pop(workspace_id, None)
        path.with_suffix(".lease").unlink(missing_ok=True)
        await asyncio.to_thread(shutil.rmtree, path, True)

    def _require(self, lease: WorkspaceLease) -> None:
        current = self._leases.get(lease.workspace_id)
        if current is None or current.expires_at < time.time():
            raise WorkspaceError("workspace lease is missing or expired")


DEFAULT_DOCKER_CPUS = 1.0
DEFAULT_DOCKER_MEMORY = "1g"
DEFAULT_DOCKER_PIDS_LIMIT = 256
# The workspace of a read-only replica: the root file system is read-only, so the directory is a tmpfs.
_READ_ONLY_TMPFS = "/workspace:rw,size=256m,mode=1777"
_SANDBOX_DIR = "/workspace"
# `mkdir -p` because the image is not ours and need not have the directory.
_KEEP_ALIVE = f"mkdir -p {_SANDBOX_DIR} && exec sleep infinity"
_LABEL_WORKSPACE, _LABEL_TENANT, _LABEL_TASK = "orbit.workspace_id", "orbit.tenant_id", "orbit.task_id"


@dataclass(frozen=True)
class DockerLimits:
    """What one sandbox container may use; `docker run --cpus/--memory/--pids-limit`."""

    cpus: float = DEFAULT_DOCKER_CPUS
    memory: str = DEFAULT_DOCKER_MEMORY
    pids_limit: int = DEFAULT_DOCKER_PIDS_LIMIT


class DockerWorkspaceAdapter(LocalWorkspaceAdapter):
    """Docker backend with the same archive contract as the local backend.

    The container of a workspace is named `orbit-<workspace_id>` and labelled with its tenant and task, so any
    process can find it: `_containers` and `_leases` are only a cache of what this process has already checked.
    A lease this process does not know (a worker that started after a crash, or another worker) is taken over by
    looking the container up by name (17 G5). The container has no network (it must reach neither control, the
    database, the object store nor the metadata service) and has CPU, memory and process limits.
    """

    backend = "docker"

    def __init__(
        self, root: str | Path, image: str, *, ttl_s: int = 300, limits: DockerLimits | None = None
    ) -> None:
        super().__init__(root, ttl_s=ttl_s)
        self.image = image
        self.limits = limits or DockerLimits()
        self._containers: dict[str, str] = {}

    async def acquire(self, tenant_id: str, task_id: str, *, read_only: bool = False) -> WorkspaceLease:
        lease = await super().acquire(tenant_id, task_id, read_only=read_only)
        # The lease record says which backend to kill the sandbox with (17 G6), so it must not say `local`.
        lease = replace(lease, mode="docker")
        self._leases[lease.workspace_id] = lease
        code, output, error = await _docker(*self._run_args(lease))
        if code:
            # A container that was created but did not start keeps its name; do not leave it behind.
            await _docker("docker", "rm", "-f", _container_name(lease.workspace_id))
            await super().release(lease)
            raise WorkspaceError(_failure("docker failed to start workspace", error))
        self._containers[lease.workspace_id] = output.decode().strip()
        return lease

    def _run_args(self, lease: WorkspaceLease) -> list[str]:
        args = [
            "docker", "run", "-d", "--name", _container_name(lease.workspace_id),
            "--label", f"{_LABEL_WORKSPACE}={lease.workspace_id}",
            "--label", f"{_LABEL_TENANT}={lease.tenant_id}",
            "--label", f"{_LABEL_TASK}={lease.task_id}",
            "--network", "none",
            "--cpus", str(self.limits.cpus),
            "--memory", self.limits.memory,
            "--pids-limit", str(self.limits.pids_limit),
        ]
        if lease.read_only:
            args.extend(["--read-only", "--tmpfs", _READ_ONLY_TMPFS])
        args.extend([self.image, "sh", "-c", _KEEP_ALIVE])
        return args

    async def renew(self, lease: WorkspaceLease, ttl_s: int = 300) -> WorkspaceLease:
        await self._attach(lease)
        return await super().renew(lease, ttl_s)

    async def snapshot(self, lease: WorkspaceLease) -> str:
        await self._attach(lease)
        return await super().snapshot(lease)

    async def restore(self, lease: WorkspaceLease, snapshot_ref: str) -> None:
        await self._attach(lease)
        await super().restore(lease, snapshot_ref)

    async def release(self, lease: WorkspaceLease) -> None:
        """Remove the container by name, whether or not this process started it or has ever seen the lease."""
        await self._remove(lease.workspace_id)
        self._leases.pop(lease.workspace_id, None)
        lock = self._locks.pop(lease.workspace_id, None) or self._lock_path(lease)
        lock.unlink(missing_ok=True)

    async def kill(self, tenant_id: str, workspace_id: str) -> None:
        """`docker rm -f` the container by its name, which is derived from the workspace id, so no in-process state is
        needed. A container that is already gone is fine."""
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise WorkspaceError("not a workspace of this backend")
        await self._remove(workspace_id)
        await super().kill(tenant_id, workspace_id)

    async def get_archive(self, lease: WorkspaceLease) -> bytes:
        await self._attach(lease)
        code, output, error = await self._exec(
            lease, "tar", "czf", "-", "-C", _SANDBOX_DIR, "."
        )
        if code:
            raise WorkspaceError(_failure("docker archive failed", error))
        return output

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        await self._attach(lease)
        code, _, error = await self._exec(
            lease, "tar", "xzf", "-", "-C", _SANDBOX_DIR, stdin=archive, interactive=True
        )
        if code:
            raise WorkspaceError(_failure("docker restore failed", error))

    async def _exec(
        self, lease: WorkspaceLease, *command: str, stdin: bytes | None = None, interactive: bool = False
    ) -> tuple[int, bytes, bytes]:
        flags = ["-i"] if interactive else []
        result = await _docker("docker", "exec", *flags, _container_name(lease.workspace_id), *command, stdin=stdin)
        if result[0] and b"No such container" in result[2]:
            self._containers.pop(lease.workspace_id, None)
        return result

    def _lock_path(self, lease: WorkspaceLease) -> Path:
        return self._path(lease).with_suffix(".lease")

    async def _remove(self, workspace_id: str) -> None:
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise WorkspaceError("not a workspace of this backend")
        code, _, error = await _docker("docker", "rm", "-f", _container_name(workspace_id))
        if code and b"No such container" not in error:
            raise WorkspaceError(_failure("docker rm failed", error))
        self._containers.pop(workspace_id, None)

    async def _attach(self, lease: WorkspaceLease) -> None:
        """Make `lease` usable in this process: find its container by name, check it is the lease's own, start it if
        it stopped, and remember both. A lease and container this process already knows are not looked up again.

        A stopped writer container is started because its file system, and so the workspace, survives a stop (a
        daemon or host restart). A stopped read-only one is not: its workspace is a tmpfs, which a stop empties,
        and handing out an empty workspace as if it were the replica would be silent data loss."""
        workspace_id = lease.workspace_id
        if workspace_id in self._leases and workspace_id in self._containers:
            self._require(lease)
            return
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise WorkspaceError("not a workspace of this backend")
        if lease.expires_at < time.time():
            raise WorkspaceError("workspace lease is missing or expired")
        name = _container_name(workspace_id)
        found = await self._inspect(name)
        labels = found.get("Config", {}).get("Labels") or {}
        owner = (labels.get(_LABEL_WORKSPACE), labels.get(_LABEL_TENANT), labels.get(_LABEL_TASK))
        if owner != (workspace_id, lease.tenant_id, lease.task_id):
            raise WorkspaceError(f"docker container {name} does not belong to this lease")
        state = found.get("State", {}).get("Status")
        if state in {"exited", "created"}:
            if lease.read_only:
                raise WorkspaceError(f"docker container {name} is stopped and its read-only workspace is gone")
            code, _, error = await _docker("docker", "start", name)
            if code:
                raise WorkspaceError(_failure(f"docker container {name} could not be started", error))
            logger.warning("stopped workspace container started again", workspace_id=workspace_id)
        elif state != "running":
            raise WorkspaceError(f"docker container {name} is {state}, not running")
        self._leases[workspace_id] = lease
        self._containers[workspace_id] = str(found.get("Id", name))
        if not lease.read_only:
            self._locks[workspace_id] = self._lock_path(lease)

    async def _inspect(self, name: str) -> dict[str, Any]:
        code, output, error = await _docker("docker", "inspect", "--type", "container", name)
        if code:
            if b"No such" in error:
                raise WorkspaceError(f"docker container {name} does not exist")
            raise WorkspaceError(_failure("docker inspect failed", error))
        try:
            return json.loads(output)[0]
        except (ValueError, IndexError, TypeError) as exc:
            raise WorkspaceError(f"docker inspect of {name} returned something unreadable") from exc


def _container_name(workspace_id: str) -> str:
    return f"orbit-{workspace_id}"


def _failure(what: str, error: bytes) -> str:
    detail = error.decode(errors="replace").strip()
    return f"{what}: {detail}" if detail else what


async def _docker(*args: str, stdin: bytes | None = None) -> tuple[int, bytes, bytes]:
    """Run one docker CLI command; the exit code, stdout and stderr."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise WorkspaceError(f"the docker CLI could not be run: {exc}") from exc
    output, error = await proc.communicate(stdin)
    return proc.returncode or 0, output, error


class OpenSandboxWorkspaceAdapter:
    """OpenSandbox 0.1.x workspace backed by the official Python SDK."""

    backend = "opensandbox"

    def __init__(
        self,
        *,
        connection_config: Any,
        image: str,
        snapshot_store: SnapshotStore | None = None,
        ttl_s: int = 300,
    ) -> None:
        self.connection_config = connection_config
        self.image = image
        self.snapshot_store = snapshot_store
        self.ttl_s = ttl_s
        self._leases: dict[str, WorkspaceLease] = {}
        self._sandboxes: dict[str, Any] = {}
        self._manager: Any | None = None

    async def _sdk(self) -> tuple[Any, Any, Any]:
        try:
            from opensandbox import Sandbox, SandboxManager
            from opensandbox.models.sandboxes import SandboxFilter, SandboxState
        except ImportError as exc:  # pragma: no cover - exercised in deployment image.
            raise WorkspaceError("opensandbox is required for the opensandbox workspace backend") from exc
        if self._manager is None:
            self._manager = await SandboxManager.create(self.connection_config)
        return Sandbox, SandboxFilter, SandboxState

    async def acquire(self, tenant_id: str, task_id: str, *, read_only: bool = False) -> WorkspaceLease:
        Sandbox, SandboxFilter, SandboxState = await self._sdk()
        metadata = {
            "orbit.tenant_id": tenant_id,
            "orbit.task_id": task_id,
            "orbit.workspace_mode": "read" if read_only else "write",
        }
        infos = await self._manager.list_sandbox_infos(
            SandboxFilter(metadata=metadata, page_size=100)
        )
        active = next(
            (
                info
                for info in infos.sandbox_infos
                if info.status.state in {SandboxState.RUNNING, SandboxState.PAUSED}
            ),
            None,
        )
        if active is None:
            sandbox = await Sandbox.create(
                image=self.image,
                timeout=None,
                metadata=metadata,
                connection_config=self.connection_config,
            )
        elif active.status.state == SandboxState.PAUSED:
            sandbox = await Sandbox.resume(active.id, self.connection_config)
        else:
            sandbox = await Sandbox.connect(active.id, self.connection_config)
        workspace_id = str(sandbox.id)
        lease = WorkspaceLease(
            workspace_id,
            task_id,
            tenant_id,
            "opensandbox",
            read_only,
            time.time() + self.ttl_s,
        )
        self._leases[workspace_id] = lease
        self._sandboxes[workspace_id] = sandbox
        return lease

    async def renew(self, lease: WorkspaceLease, ttl_s: int = 300) -> WorkspaceLease:
        sandbox = self._require_sandbox(lease)
        from datetime import timedelta

        await sandbox.renew(timedelta(seconds=ttl_s))
        renewed = WorkspaceLease(
            lease.workspace_id,
            lease.task_id,
            lease.tenant_id,
            lease.mode,
            lease.read_only,
            time.time() + ttl_s,
        )
        self._leases[lease.workspace_id] = renewed
        return renewed

    async def snapshot(self, lease: WorkspaceLease) -> str:
        archive = await self.get_archive(lease)
        digest = __import__("hashlib").sha256(archive).hexdigest()
        if self.snapshot_store is None:
            raise WorkspaceError("opensandbox snapshots require a snapshot store")
        await self.snapshot_store.put_snapshot(
            tenant_id=lease.tenant_id, digest=digest, payload=archive
        )
        return f"sha256:{digest}"

    async def restore(self, lease: WorkspaceLease, snapshot_ref: str) -> None:
        self._require(lease)
        if self.snapshot_store is None:
            raise WorkspaceError("opensandbox snapshots require a snapshot store")
        if not snapshot_ref.startswith("sha256:"):
            raise WorkspaceError("snapshot reference must be content addressed")
        archive = await self.snapshot_store.get_snapshot(
            tenant_id=lease.tenant_id,
            digest=snapshot_ref.removeprefix("sha256:"),
        )
        await self.put_archive(lease, archive)

    async def release(self, lease: WorkspaceLease) -> None:
        sandbox = self._require_sandbox(lease)
        await sandbox.pause()
        self._leases.pop(lease.workspace_id, None)
        self._sandboxes.pop(lease.workspace_id, None)

    async def get_archive(self, lease: WorkspaceLease) -> bytes:
        sandbox = self._require_sandbox(lease)
        await sandbox.commands.run(
            "tar -czf /tmp/orbit-workspace.tar.gz -C /workspace ."
        )
        stream = sandbox.files.read_bytes_stream("/tmp/orbit-workspace.tar.gz")
        return b"".join([chunk async for chunk in stream])

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        sandbox = self._require_sandbox(lease)
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        await sandbox.files.write_file("/tmp/orbit-workspace.tar.gz", archive)
        execution = await sandbox.commands.run(
            "tar -xzf /tmp/orbit-workspace.tar.gz -C /workspace"
        )
        if execution.exit_code not in (None, 0):
            raise WorkspaceError(f"opensandbox restore failed with exit code {execution.exit_code}")

    def _require(self, lease: WorkspaceLease) -> None:
        current = self._leases.get(lease.workspace_id)
        if current is None or current.expires_at < time.time():
            raise WorkspaceError("workspace lease is missing or expired")

    def _require_sandbox(self, lease: WorkspaceLease) -> Any:
        self._require(lease)
        sandbox = self._sandboxes.get(lease.workspace_id)
        if sandbox is None:
            raise WorkspaceError("opensandbox instance is not connected")
        return sandbox

    async def kill(self, tenant_id: str, workspace_id: str) -> None:
        """Kill the sandbox by its id through the manager (AgentScope only pauses, 08 §1). The workspace id of this
        backend is the sandbox id, and a sandbox that no longer exists is not an error."""
        await self._sdk()
        try:
            await self._manager.kill_sandbox(workspace_id)
        except Exception as exc:
            if not _is_not_found(exc):
                raise
        self._leases.pop(workspace_id, None)
        self._sandboxes.pop(workspace_id, None)

    async def reap_expired(self) -> int:
        expired = [lease for lease in self._leases.values() if lease.expires_at < time.time()]
        for lease in expired:
            sandbox = self._sandboxes.get(lease.workspace_id)
            if sandbox is not None:
                await sandbox.kill()
            self._leases.pop(lease.workspace_id, None)
            self._sandboxes.pop(lease.workspace_id, None)
        return len(expired)


def _is_not_found(exc: Exception) -> bool:
    """The OpenSandbox SDK raises SandboxApiException with a status code; 404 means the sandbox is already gone."""
    return getattr(exc, "status_code", None) == 404

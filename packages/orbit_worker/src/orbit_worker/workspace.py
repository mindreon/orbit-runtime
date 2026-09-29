"""Workspace leases and archive based snapshots.

The worker talks to this small protocol only.  Local and Docker backends are
dependency free; the OpenSandbox backend accepts an SDK client from the
process that owns the SDK version, keeping the activity code stable.
"""

from __future__ import annotations

import asyncio
import io
import os
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


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


class LocalWorkspaceAdapter:
    """Filesystem backend used by dev and tests."""

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

    def _require(self, lease: WorkspaceLease) -> None:
        current = self._leases.get(lease.workspace_id)
        if current is None or current.expires_at < time.time():
            raise WorkspaceError("workspace lease is missing or expired")


class DockerWorkspaceAdapter(LocalWorkspaceAdapter):
    """Docker backend with the same archive contract as the local backend."""

    def __init__(self, root: str | Path, image: str, *, ttl_s: int = 300) -> None:
        super().__init__(root, ttl_s=ttl_s)
        self.image = image
        self._containers: dict[str, str] = {}

    async def acquire(self, tenant_id: str, task_id: str, *, read_only: bool = False) -> WorkspaceLease:
        lease = await super().acquire(tenant_id, task_id, read_only=read_only)
        name = f"orbit-{lease.workspace_id}"
        args = ["docker", "run", "-d", "--name", name]
        if read_only:
            args.append("--read-only")
        args.extend([self.image, "sleep", "infinity"])
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE)
        output, _ = await proc.communicate()
        if proc.returncode:
            await super().release(lease)
            raise WorkspaceError("docker failed to start workspace")
        self._containers[lease.workspace_id] = output.decode().strip()
        return lease

    async def release(self, lease: WorkspaceLease) -> None:
        container = self._containers.pop(lease.workspace_id, None)
        if container:
            proc = await asyncio.create_subprocess_exec("docker", "rm", "-f", container)
            await proc.wait()
        await super().release(lease)

    async def get_archive(self, lease: WorkspaceLease) -> bytes:
        self._require(lease)
        container = self._containers.get(lease.workspace_id)
        if not container:
            raise WorkspaceError("docker container is missing")
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container, "tar", "czf", "-", "-C", "/workspace", ".",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        output, error = await proc.communicate()
        if proc.returncode:
            raise WorkspaceError(error.decode(errors="replace") or "docker archive failed")
        return output

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        self._require(lease)
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        container = self._containers.get(lease.workspace_id)
        if not container:
            raise WorkspaceError("docker container is missing")
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", container, "tar", "xzf", "-", "-C", "/workspace",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, error = await proc.communicate(archive)
        if proc.returncode:
            raise WorkspaceError(error.decode(errors="replace") or "docker restore failed")


class OpenSandboxWorkspaceAdapter:
    """OpenSandbox 0.1.x workspace backed by the official Python SDK."""

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

    async def reap_expired(self) -> int:
        expired = [lease for lease in self._leases.values() if lease.expires_at < time.time()]
        for lease in expired:
            sandbox = self._sandboxes.get(lease.workspace_id)
            if sandbox is not None:
                await sandbox.kill()
            self._leases.pop(lease.workspace_id, None)
            self._sandboxes.pop(lease.workspace_id, None)
        return len(expired)

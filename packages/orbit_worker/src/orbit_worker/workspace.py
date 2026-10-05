"""Workspace leases and archive based snapshots.

The worker talks to this small protocol only.  Local and Docker backends are
dependency free; the OpenSandbox backend accepts an SDK client from the
process that owns the SDK version, keeping the activity code stable.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import posixpath
import re
import shlex
import shutil
import signal
import tarfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class WorkspaceError(RuntimeError):
    pass


class SnapshotNotFound(WorkspaceError):
    """The snapshot a task points at is not in the store (the backend lost it, e.g. a worker without a persistent root)."""


class WorkspaceLost(WorkspaceError):
    """The lease ran out or the sandbox behind it is gone (reclaimed, killed, removed). What was in it is lost; a
    caller that wants a workspace takes a new lease."""


@dataclass(frozen=True)
class WorkspaceLease:
    workspace_id: str
    task_id: str
    tenant_id: str
    mode: str
    read_only: bool
    expires_at: float


# Where a workspace is, as the agent and the commands see it, on every backend. A path in the protocol below is this
# directory or something under it, or relative to it.
WORKSPACE_DIR = "/workspace"
_OUTPUT_KEPT = 64 * 1024
_VIRTUAL_ROOT = re.compile(r"(?<![\w/.-])" + re.escape(WORKSPACE_DIR) + r"(?![\w.-])")
# Room for the SDK call to return after the sandbox itself enforced the command timeout.
_SDK_GRACE_S = 30


@dataclass(frozen=True)
class ExecResult:
    """How one program ended in a workspace. `exit_code` is None when it did not finish in time."""

    exit_code: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False


class WorkspaceAdapter(Protocol):
    async def acquire(self, tenant_id: str, task_id: str, *, read_only: bool = False) -> WorkspaceLease: ...
    async def renew(self, lease: WorkspaceLease, ttl_s: int = 300) -> WorkspaceLease: ...
    async def snapshot(self, lease: WorkspaceLease) -> str: ...
    async def restore(self, lease: WorkspaceLease, snapshot_ref: str) -> None: ...
    async def release(self, lease: WorkspaceLease) -> None: ...
    async def get_archive(self, lease: WorkspaceLease) -> bytes: ...
    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None: ...
    # The archive a snapshot reference names, without a workspace to put it in.
    async def load_snapshot(self, tenant_id: str, snapshot_ref: str) -> bytes: ...
    # What an agent's tools need of a workspace: run a program, read a file, write a file.
    async def exec(
        self,
        lease: WorkspaceLease,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout_s: float = 60,
        stdin: bytes | None = None,
    ) -> ExecResult: ...
    async def read_file(self, lease: WorkspaceLease, path: str) -> bytes: ...
    async def write_file(self, lease: WorkspaceLease, path: str, data: bytes) -> None: ...


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

    async def load_snapshot(self, tenant_id: str, snapshot_ref: str) -> bytes:
        return await self.adapter.load_snapshot(tenant_id, snapshot_ref)

    async def exec(
        self,
        lease: WorkspaceLease,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout_s: float = 60,
        stdin: bytes | None = None,
    ) -> ExecResult:
        return await self.adapter.exec(lease, argv, cwd=cwd, timeout_s=timeout_s, stdin=stdin)

    async def read_file(self, lease: WorkspaceLease, path: str) -> bytes:
        return await self.adapter.read_file(lease, path)

    async def write_file(self, lease: WorkspaceLease, path: str, data: bytes) -> None:
        await self.adapter.write_file(lease, path, data)


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
# A tenant id is a directory name of the local backend: nothing in it may climb out of the root or name another path.
_TENANT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _require_tenant(tenant_id: str) -> None:
    if not _TENANT_ID.fullmatch(tenant_id) or ".." in tenant_id:
        raise WorkspaceError("not a tenant id of this backend")


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
        _require_tenant(tenant_id)
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
        await self.put_archive(lease, await self.load_snapshot(lease.tenant_id, snapshot_ref))

    async def load_snapshot(self, tenant_id: str, snapshot_ref: str) -> bytes:
        _require_tenant(tenant_id)
        if not snapshot_ref.startswith("sha256:"):
            raise WorkspaceError("snapshot reference must be content addressed")
        path = self._snapshots / tenant_id / snapshot_ref.removeprefix("sha256:")
        if not path.is_file():
            raise SnapshotNotFound("snapshot not found")
        return path.read_bytes()

    async def release(self, lease: WorkspaceLease) -> None:
        """Give the workspace back and delete its directory. What it held is in the snapshot taken before this: a
        workspace is a lease, not a place that outlives it. Releasing a lease that is already lost cleans what is left."""
        self._leases.pop(lease.workspace_id, None)
        lock = self._locks.pop(lease.workspace_id, None)
        if lock:
            lock.unlink(missing_ok=True)
        await asyncio.to_thread(shutil.rmtree, self._path(lease), True)

    async def get_archive(self, lease: WorkspaceLease) -> bytes:
        self._require(lease)
        return _pack(self._path(lease))

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        self._require(lease)
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        _unpack(self._path(lease), archive)

    def _host_text(self, lease: WorkspaceLease, text: str) -> str:
        """`text` with the workspace's place on the host where it says /workspace. This backend has no mount of its own, so
        a command line that names /workspace/x has to be told where that is; a container does not need it."""
        return _VIRTUAL_ROOT.sub(str(self._path(lease)), text)

    def _virtual_text(self, lease: WorkspaceLease, text: str) -> str:
        return text.replace(str(self._path(lease)), WORKSPACE_DIR)

    def _inside(self, lease: WorkspaceLease, path: str) -> Path:
        """The host path of `path`, which must be the workspace or under it. The workspace is a directory of the host
        here, so this is the whole confinement of the file calls: nothing outside it is read or written, links
        included. A command line is not confined (it is a shell of the host): this backend is for development."""
        relative = posixpath.relpath(path, WORKSPACE_DIR) if posixpath.isabs(path) else path
        root = os.path.realpath(self._path(lease))
        target = os.path.realpath(os.path.join(root, relative))
        if target != root and not target.startswith(root + os.sep):
            raise WorkspaceError(f"{path} is outside the workspace")
        return Path(target)

    async def exec(
        self,
        lease: WorkspaceLease,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout_s: float = 60,
        stdin: bytes | None = None,
    ) -> ExecResult:
        self._require(lease)
        root = self._path(lease)
        # Not the worker's environment: it holds the database URL and the checkpoint key.
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root), "LANG": "C.UTF-8"}
        proc = await asyncio.create_subprocess_exec(
            *(self._host_text(lease, part) for part in argv),
            cwd=str(self._inside(lease, cwd or ".")), env=env, start_new_session=True,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(stdin), timeout_s)
        except TimeoutError:
            # A new session, so the whole group goes: `sh -c "a & b"` leaves children the shell does not wait for.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            return ExecResult(exit_code=None, timed_out=True)
        return ExecResult(
            exit_code=proc.returncode,
            stdout=self._virtual_text(lease, stdout[-_OUTPUT_KEPT:].decode("utf-8", "replace")).encode(),
            stderr=self._virtual_text(lease, stderr[-_OUTPUT_KEPT:].decode("utf-8", "replace")).encode(),
        )

    async def read_file(self, lease: WorkspaceLease, path: str) -> bytes:
        self._require(lease)
        return await asyncio.to_thread(self._inside(lease, path).read_bytes)

    async def write_file(self, lease: WorkspaceLease, path: str, data: bytes) -> None:
        self._require(lease)
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        target = self._inside(lease, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def reap_expired(self) -> int:
        expired = [lease for lease in self._leases.values() if lease.expires_at < time.time()]
        for lease in expired:
            await self.release(lease)
        return len(expired)

    async def kill(self, tenant_id: str, workspace_id: str) -> None:
        """Remove the workspace directory and its lock, whoever created them."""
        _require_tenant(tenant_id)
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise WorkspaceError("not a workspace of this backend")
        path = self.root / tenant_id / workspace_id
        self._leases.pop(workspace_id, None)
        self._locks.pop(workspace_id, None)
        path.with_suffix(".lease").unlink(missing_ok=True)
        await asyncio.to_thread(shutil.rmtree, path, True)

    def _require_lease(self, lease: WorkspaceLease) -> None:
        current = self._leases.get(lease.workspace_id)
        if current is None or current.expires_at < time.time():
            raise WorkspaceLost("workspace lease is missing or expired")

    def _require(self, lease: WorkspaceLease) -> None:
        self._require_lease(lease)
        if not self._path(lease).is_dir():
            raise WorkspaceLost("the workspace is gone")


DEFAULT_DOCKER_CPUS = 1.0
DEFAULT_DOCKER_MEMORY = "1g"
DEFAULT_DOCKER_PIDS_LIMIT = 256
# The workspace of a read-only replica: the root file system is read-only, so the directory is a tmpfs.
_READ_ONLY_TMPFS = "/workspace:rw,size=256m,mode=1777"
# `mkdir -p` because the image is not ours and need not have the directory.
_KEEP_ALIVE = f"mkdir -p {WORKSPACE_DIR} && exec sleep infinity"
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
    looking the container up by name (17 G5). The container has CPU, memory and process limits, and its network
    comes from `network` (`none` by default: it must reach neither control, the database, the object store nor the
    metadata service; `bridge` gives outbound internet for skills that call external APIs).
    """

    backend = "docker"

    def __init__(
        self, root: str | Path, image: str, *, ttl_s: int = 300, limits: DockerLimits | None = None, network: str = "none"
    ) -> None:
        super().__init__(root, ttl_s=ttl_s)
        self.image = image
        self.limits = limits or DockerLimits()
        self.network = network
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
            "--network", self.network,
            "--cpus", str(self.limits.cpus),
            "--memory", self.limits.memory,
            "--pids-limit", str(self.limits.pids_limit),
        ]
        if lease.read_only:
            args.extend(["--read-only", "--tmpfs", _READ_ONLY_TMPFS])
        args.extend([self.image, "sh", "-c", _KEEP_ALIVE])
        return args

    def _require(self, lease: WorkspaceLease) -> None:
        # The workspace is in the container, not in the host directory the local base made.
        self._require_lease(lease)

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
        # The host directory the local base made is not the container's workspace; it only has to go.
        await asyncio.to_thread(shutil.rmtree, self._path(lease), True)

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
            lease, "tar", "czf", "-", "-C", WORKSPACE_DIR, "."
        )
        if code:
            raise WorkspaceError(_failure("docker archive failed", error))
        return output

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        await self._attach(lease)
        code, _, error = await self._exec(
            lease, "tar", "xzf", "-", "-C", WORKSPACE_DIR, stdin=archive, interactive=True
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

    async def exec(
        self,
        lease: WorkspaceLease,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout_s: float = 60,
        stdin: bytes | None = None,
    ) -> ExecResult:
        await self._attach(lease)
        flags = ["-i"] if stdin is not None else []
        workdir = posixpath.join(WORKSPACE_DIR, cwd) if cwd else WORKSPACE_DIR
        command = ["docker", "exec", *flags, "-w", workdir, _container_name(lease.workspace_id), *argv]
        try:
            code, stdout, stderr = await asyncio.wait_for(_docker(*command, stdin=stdin), timeout_s)
        except TimeoutError:
            # This ends the client; the container, and the process in it, are removed when the lease is released.
            return ExecResult(exit_code=None, timed_out=True)
        if code and b"No such container" in stderr:
            self._containers.pop(lease.workspace_id, None)
            raise WorkspaceLost(f"docker container {_container_name(lease.workspace_id)} is gone")
        return ExecResult(exit_code=code, stdout=stdout[-_OUTPUT_KEPT:], stderr=stderr[-_OUTPUT_KEPT:])

    async def read_file(self, lease: WorkspaceLease, path: str) -> bytes:
        result = await self.exec(lease, ["cat", "--", path])
        if result.exit_code:
            raise WorkspaceError(_failure(f"cannot read {path}", result.stderr))
        return result.stdout

    async def write_file(self, lease: WorkspaceLease, path: str, data: bytes) -> None:
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        script = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        result = await self.exec(lease, ["sh", "-c", script, "sh", path], stdin=data)
        if result.exit_code:
            raise WorkspaceError(_failure(f"cannot write {path}", result.stderr))

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
            raise WorkspaceLost("workspace lease is missing or expired")
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
                raise WorkspaceLost(f"docker container {name} does not exist")
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
    """Run one docker CLI command; the exit code, stdout and stderr. If the caller stops waiting, the client is killed."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise WorkspaceError(f"the docker CLI could not be run: {exc}") from exc
    try:
        output, error = await (proc.communicate(stdin) if stdin is not None else proc.communicate())
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
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
        await self.put_archive(lease, await self.load_snapshot(lease.tenant_id, snapshot_ref))

    async def load_snapshot(self, tenant_id: str, snapshot_ref: str) -> bytes:
        if self.snapshot_store is None:
            raise WorkspaceError("opensandbox snapshots require a snapshot store")
        if not snapshot_ref.startswith("sha256:"):
            raise WorkspaceError("snapshot reference must be content addressed")
        try:
            return await self.snapshot_store.get_snapshot(tenant_id=tenant_id, digest=snapshot_ref.removeprefix("sha256:"))
        except FileNotFoundError as exc:
            raise SnapshotNotFound("snapshot not found") from exc
        except Exception as exc:
            if getattr(exc, "code", None) == "NoSuchKey":
                raise SnapshotNotFound("snapshot not found") from exc
            raise

    async def release(self, lease: WorkspaceLease) -> None:
        sandbox = self._require_sandbox(lease)
        await sandbox.pause()
        self._leases.pop(lease.workspace_id, None)
        self._sandboxes.pop(lease.workspace_id, None)

    async def get_archive(self, lease: WorkspaceLease) -> bytes:
        sandbox = self._require_sandbox(lease)
        await sandbox.commands.run(
            f"tar -czf /tmp/orbit-workspace.tar.gz -C {WORKSPACE_DIR} ."
        )
        stream = sandbox.files.read_bytes_stream("/tmp/orbit-workspace.tar.gz")
        return b"".join([chunk async for chunk in stream])

    async def put_archive(self, lease: WorkspaceLease, archive: bytes) -> None:
        sandbox = self._require_sandbox(lease)
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        await sandbox.files.write_file("/tmp/orbit-workspace.tar.gz", archive)
        execution = await sandbox.commands.run(
            f"tar -xzf /tmp/orbit-workspace.tar.gz -C {WORKSPACE_DIR}"
        )
        if execution.exit_code not in (None, 0):
            raise WorkspaceError(f"opensandbox restore failed with exit code {execution.exit_code}")

    async def exec(
        self,
        lease: WorkspaceLease,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout_s: float = 60,
        stdin: bytes | None = None,
    ) -> ExecResult:
        from opensandbox.models.execd import RunCommandOpts

        if stdin is not None:
            raise WorkspaceError("opensandbox commands take no standard input")
        sandbox = self._require_sandbox(lease)
        workdir = posixpath.join(WORKSPACE_DIR, cwd) if cwd else WORKSPACE_DIR
        opts = RunCommandOpts(working_directory=workdir, timeout=timedelta(seconds=timeout_s))
        try:
            execution = await asyncio.wait_for(sandbox.commands.run(shlex.join(argv), opts=opts), timeout_s + _SDK_GRACE_S)
        except TimeoutError:
            return ExecResult(exit_code=None, timed_out=True)
        error = getattr(execution, "error", None)
        text = str(getattr(execution, "text", "") or "").encode()[-_OUTPUT_KEPT:]
        problem = b"" if error is None else f"{getattr(error, 'name', 'error')}: {getattr(error, 'value', '')}".encode()
        code = getattr(execution, "exit_code", None)
        if code is None:
            # No exit code: it is a pass only if the sandbox reported no error either.
            code = 0 if error is None else 1
        return ExecResult(exit_code=code, stdout=text, stderr=problem)

    async def read_file(self, lease: WorkspaceLease, path: str) -> bytes:
        sandbox = self._require_sandbox(lease)
        return b"".join([chunk async for chunk in sandbox.files.read_bytes_stream(_in_workspace(path))])

    async def write_file(self, lease: WorkspaceLease, path: str, data: bytes) -> None:
        sandbox = self._require_sandbox(lease)
        if lease.read_only:
            raise WorkspaceError("read-only workspace cannot be modified")
        await sandbox.files.write_file(_in_workspace(path), data)

    def _require(self, lease: WorkspaceLease) -> None:
        current = self._leases.get(lease.workspace_id)
        if current is None or current.expires_at < time.time():
            raise WorkspaceLost("workspace lease is missing or expired")

    def _require_sandbox(self, lease: WorkspaceLease) -> Any:
        self._require(lease)
        sandbox = self._sandboxes.get(lease.workspace_id)
        if sandbox is None:
            raise WorkspaceLost("opensandbox instance is not connected")
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


def _in_workspace(path: str) -> str:
    """`path` as an absolute path of the sandbox, which must be the workspace or under it."""
    full = posixpath.normpath(posixpath.join(WORKSPACE_DIR, path))
    if full != WORKSPACE_DIR and not full.startswith(WORKSPACE_DIR + "/"):
        raise WorkspaceError(f"{path} is outside the workspace")
    return full


def _is_not_found(exc: Exception) -> bool:
    """The OpenSandbox SDK raises SandboxApiException with a status code; 404 means the sandbox is already gone."""
    return getattr(exc, "status_code", None) == 404

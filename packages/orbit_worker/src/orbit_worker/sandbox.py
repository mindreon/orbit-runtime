"""The sandbox as a resource of an attempt, not the place the agent runs.

The agent runs in the worker. Its tools (AgentScope's Bash, Read, Write and Edit) reach the task's workspace through a
`LeaseBackend`, which is what makes a tool call happen *in* the sandbox. The workspace is taken when a tool first needs
it, so an attempt that never touches it never holds the task's writer lease, and it is given back at the end:

    open -> (first tool call: lease, restore the task's last snapshot, stage skills) -> close: snapshot, release

The snapshot is recorded with the attempt's manifest: the next attempt of the task, on any worker, starts from it. The files in
the workspace at the end are the attempt's artifacts. What the agent says is conversation, never an artifact.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import io
import mimetypes
import tarfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol, TypeVar

import structlog
from agentscope.tool import BackendBase, Bash, Edit, ExecResult, Read, ToolBase, Write

from orbit_worker.workspace import (
    WORKSPACE_DIR,
    SnapshotNotFound,
    WorkspaceAdapter,
    WorkspaceError,
    WorkspaceLease,
    WorkspaceLost,
    keep_lease_alive,
)

logger = structlog.get_logger(__name__)

MAX_FILES = 100
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
# A command that names no limit of its own still ends: nothing in a sandbox may run for ever.
MAX_EXEC_S = 600
SKILLS_DIR = ".skills"
# What the members of a team stage left, for its leader to read and merge (07 §8): `.team/<role>/<file>`. Like the skills, it is
# data of the attempt and is taken out before the workspace is snapshotted.
TEAM_DIR = ".team"
_DEFAULT_MEDIA_TYPE = "application/octet-stream"


class SnapshotSource(Protocol):
    async def latest_workspace_snapshot(
        self, *, tenant_id: str, task_id: str, before_attempt: str | None = None
    ) -> str | None: ...
    async def attempt_workspace_snapshot(self, *, tenant_id: str, task_id: str, attempt_id: str) -> str | None: ...


@dataclass(frozen=True)
class SandboxFile:
    name: str
    media_type: str
    payload: bytes


_T = TypeVar("_T")

RESET_MESSAGE = (
    "The workspace was reclaimed while you were using it (its lease ran out). It is a new one, restored to the state this "
    "attempt started with: the files you wrote during this attempt are gone. This call did not run; do what you still "
    "need again."
)


class WorkspaceReset(WorkspaceError):
    """The workspace under a tool call was lost and has been replaced. It is what the agent is told."""


class SandboxSession:
    """One attempt's use of its task's workspace."""

    def __init__(
        self,
        adapter: WorkspaceAdapter,
        store: SnapshotSource,
        *,
        tenant_id: str,
        task_id: str,
        holder: str,
        restore_from: str | None = None,
        replica: bool = False,
    ) -> None:
        # `restore_from` names the snapshot the workspace starts from instead of the task's latest one, and makes the session
        # a scratch one: it is released at the end and nothing is saved from it (a verifier looks at an attempt's files).
        # `replica` is a scratch session too, on a read lease (08 §1): a member of a team stage works in its own copy of the
        # task's head snapshot, beside the others and the leader, and what it leaves there comes back as files, not as the head.
        self._restore_from = restore_from
        self._replica = replica
        self._base_ref: str | None = None
        self._team_files: list[tuple[str, bytes]] = []
        self._adapter = adapter
        self._store = store
        self._tenant_id = tenant_id
        self._task_id = task_id
        self._holder = holder
        self._lease: WorkspaceLease | None = None
        self._keepalive: asyncio.Task[None] | None = None
        self._taking = asyncio.Lock()
        self._skills: dict[str, Path] = {}
        self.backend = LeaseBackend(self)

    @property
    def adapter(self) -> WorkspaceAdapter:
        return self._adapter

    @property
    def taken(self) -> bool:
        return self._lease is not None

    def tools(self) -> list[ToolBase]:
        """What an agent does in a workspace. Search is `find` and `grep` through Bash: the dedicated search tools need
        a helper or ripgrep inside the sandbox image."""
        backend = self.backend
        return [Bash(cwd=WORKSPACE_DIR, backend=backend), Read(backend=backend), Write(backend=backend), Edit(backend=backend)]

    def offer_team_files(self, role: str, files: list[tuple[str, bytes]]) -> None:
        """Say that what the member `role` left is to be in the workspace, under `.team/<role>/`, when it is taken."""
        self._team_files.extend((f"{TEAM_DIR}/{role}/{name}", payload) for name, payload in files)

    def offer_skill(self, name: str, host_dir: Path) -> str:
        """Say that the skill staged at `host_dir` is to be in the workspace, and where. It is copied in when the workspace
        is taken and left out of the snapshot, so a skill is data of the attempt, not of the task."""
        self._skills[name] = host_dir
        return f"{WORKSPACE_DIR}/{SKILLS_DIR}/{name}"

    async def lease(self) -> WorkspaceLease:
        async with self._taking:
            if self._lease is None:
                self._lease = await self._take()
            return self._lease

    async def replace(self, lost: WorkspaceLease) -> NoReturn:
        """The lease `lost` is gone: take a new one, restored to the last snapshot, and tell the caller that what the
        attempt wrote since is lost. Always raises: a call that found its workspace gone has not run. Calls that failed
        for the same loss find the new lease already there and are told the same."""
        async with self._taking:
            if self._lease is lost:
                self._lease = None
                if self._keepalive is not None:
                    self._keepalive.cancel()
                    self._keepalive = None
                logger.warning("the workspace was lost and is taken again", workspace_id=lost.workspace_id)
                with contextlib.suppress(Exception):
                    await self._adapter.release(lost)
                self._lease = await self._take()
        raise WorkspaceReset(RESET_MESSAGE)

    async def _take(self) -> WorkspaceLease:
        adapter = self._adapter
        lease = await adapter.acquire(  # type: ignore[call-arg]
            self._tenant_id, self._task_id, holder=self._holder, **({"read_only": True} if self._replica else {})
        )
        try:
            last = self._restore_from or await self._store.latest_workspace_snapshot(
                tenant_id=self._tenant_id, task_id=self._task_id
            )
            self._base_ref = last
            if last:
                try:
                    await adapter.restore(lease, last)
                except SnapshotNotFound:
                    # The task's earlier files are gone from the store: go on with an empty workspace rather than fail the attempt.
                    logger.warning("the task's workspace snapshot is missing; starting empty", snapshot=last, task_id=self._task_id)
            for name, host_dir in self._skills.items():
                await adapter.put_archive(lease, await asyncio.to_thread(_skill_archive, name, host_dir))
            if self._team_files:
                await adapter.put_archive(lease, await asyncio.to_thread(_files_archive, self._team_files))
        except BaseException:
            await adapter.release(lease)
            raise
        self._keepalive = asyncio.create_task(keep_lease_alive(adapter, lease, getattr(adapter, "ttl_s", 300)))
        return lease

    async def files(self) -> list[SandboxFile]:
        """The files this attempt added or changed: the ones in the workspace now that were not, or were different, when
        the attempt found it. If this run never took the workspace they are still the attempt's when an earlier run of the
        attempt did (it stopped for an approval, and what it wrote is in its snapshot): then the workspace is taken to
        read them. Otherwise there are none. A replica's are the ones that differ from the snapshot it was seeded with."""
        if self._replica:
            if self._lease is None:
                return []
            now = files_in_archive(await self._adapter.get_archive(self._lease))
            before: dict[str, bytes] = {}
            if self._base_ref:
                try:
                    before = {file.name: file.payload for file in files_in_archive(await self._adapter.load_snapshot(self._tenant_id, self._base_ref))}
                except SnapshotNotFound:
                    before = {}
            return [file for file in now if before.get(file.name) != file.payload]
        if self._lease is None:
            earlier = await self._store.attempt_workspace_snapshot(
                tenant_id=self._tenant_id, task_id=self._task_id, attempt_id=self._holder
            )
            if earlier is None:
                return []
            await self.lease()
        assert self._lease is not None
        now = files_in_archive(await self._adapter.get_archive(self._lease))
        found = await self._store.latest_workspace_snapshot(
            tenant_id=self._tenant_id, task_id=self._task_id, before_attempt=self._holder
        )
        if found is None:
            return now
        try:
            earlier = await self._adapter.load_snapshot(self._tenant_id, found)
        except SnapshotNotFound:
            return now
        before = {file.name: file.payload for file in files_in_archive(earlier)}
        return [file for file in now if before.get(file.name) != file.payload]

    async def close(self) -> str | None:
        """Snapshot the workspace and give the lease back. The reference of the snapshot, if there was a workspace; the
        caller records it. The lease is released even when the snapshot fails."""
        lease, self._lease = self._lease, None
        if self._keepalive is not None:
            self._keepalive.cancel()
            self._keepalive = None
        if lease is None:
            return None
        if self._restore_from is not None or self._replica:
            await self._adapter.release(lease)
            return None
        try:
            if self._skills:
                await self._adapter.exec(lease, ["rm", "-rf", "--", SKILLS_DIR])
            if self._team_files:
                await self._adapter.exec(lease, ["rm", "-rf", "--", TEAM_DIR])
            return await self._adapter.snapshot(lease)
        finally:
            await self._adapter.release(lease)


class LeaseBackend(BackendBase):
    """The task's workspace, for AgentScope's tools: the three primitives of a backend, on the workspace adapter."""

    def __init__(self, session: SandboxSession) -> None:
        self._session = session

    async def getcwd(self) -> str:
        return WORKSPACE_DIR

    async def _run(self, call: Callable[[WorkspaceLease], Awaitable[_T]]) -> _T:
        """Make `call` on the workspace; if the workspace turns out to be gone, replace it (see `replace`)."""
        lease = await self._session.lease()
        try:
            return await call(lease)
        except WorkspaceLost:
            await self._session.replace(lease)

    async def exec_shell(self, command: list[str], *, cwd: str | None = None, timeout: float | None = None) -> ExecResult:
        adapter = self._session.adapter
        timeout_s = min(timeout or MAX_EXEC_S, MAX_EXEC_S)
        result = await self._run(lambda lease: adapter.exec(lease, command, cwd=cwd, timeout_s=timeout_s))
        # -1 is what a backend says for "did not finish".
        return ExecResult(exit_code=-1 if result.timed_out else int(result.exit_code or 0), stdout=result.stdout, stderr=result.stderr)

    async def read_file(self, path: str) -> bytes:
        return await self._run(lambda lease: self._session.adapter.read_file(lease, path))

    async def write_file(self, path: str, data: bytes) -> None:
        await self._run(lambda lease: self._session.adapter.write_file(lease, path, data))


# The workspace of the attempt that is running, for the code that assembles its agent.
_current: contextvars.ContextVar[SandboxSession | None] = contextvars.ContextVar("orbit_sandbox", default=None)


def bind_sandbox(session: SandboxSession | None) -> contextvars.Token[SandboxSession | None]:
    return _current.set(session)


def unbind_sandbox(token: contextvars.Token[SandboxSession | None]) -> None:
    _current.reset(token)


def current_sandbox() -> SandboxSession | None:
    return _current.get()


def _skill_archive(name: str, host_dir: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for path in sorted(host_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                tar.add(path, arcname=f"{SKILLS_DIR}/{name}/{path.relative_to(host_dir).as_posix()}")
    return output.getvalue()


def _files_archive(files: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name, payload in files:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return output.getvalue()


# What an artifact's type is for the common files an agent leaves. Explicit, so it does not depend on the host's mime table (macOS
# has no entry for .md, Linux does): the same file has the same type in dev and in the cluster. `mimetypes` is only the fallback.
_MEDIA_TYPES = {
    ".md": "text/markdown", ".markdown": "text/markdown", ".txt": "text/plain", ".log": "text/plain",
    ".json": "application/json", ".csv": "text/csv", ".tsv": "text/tab-separated-values",
    ".yaml": "application/yaml", ".yml": "application/yaml", ".toml": "application/toml", ".xml": "application/xml",
    ".html": "text/html", ".htm": "text/html", ".css": "text/css",
    ".py": "text/x-python", ".js": "text/javascript", ".mjs": "text/javascript", ".ts": "text/typescript",
    ".tsx": "text/typescript", ".jsx": "text/javascript", ".sql": "application/sql", ".sh": "application/x-sh",
    ".go": "text/x-go", ".java": "text/x-java", ".rs": "text/x-rust", ".c": "text/x-c", ".h": "text/x-c",
    ".pdf": "application/pdf", ".zip": "application/zip",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def media_type_of(name: str) -> str:
    known = _MEDIA_TYPES.get(Path(name).suffix.lower())
    if known:
        return known
    guessed, _ = mimetypes.guess_type(name)
    return guessed or _DEFAULT_MEDIA_TYPE


def _visible(name: str) -> bool:
    parts = name.split("/")
    return bool(name) and all(part not in ("", ".", "..") and not part.startswith(".") for part in parts)


def files_in_archive(archive: bytes) -> list[SandboxFile]:
    """The regular, visible files of a workspace archive, sorted by name. Too many files, or too many bytes, keeps only
    what fits: an attempt is never failed because its workspace is large."""
    found: list[SandboxFile] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
        for member in sorted(tar.getmembers(), key=lambda item: item.name):
            name = member.name.removeprefix("./")
            if not member.isreg() or not _visible(name):
                continue
            if member.size > MAX_FILE_BYTES or total + member.size > MAX_TOTAL_BYTES or len(found) >= MAX_FILES:
                logger.warning("sandbox file left out of the artifacts: over a limit", name=name)
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            found.append(SandboxFile(name=name, media_type=media_type_of(name), payload=handle.read()))
            total += member.size
    return found


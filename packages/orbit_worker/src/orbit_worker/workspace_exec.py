"""Run one shell command inside a leased workspace, for the verification checks (04 §5).

Kept apart from the adapters: the adapter protocol is about leases and archives, and only verification runs commands.
Each backend runs it where the workspace is: a process group on the host for `local` (dev and tests only), `docker
exec` in the container for `docker`, the SDK's command API for `opensandbox`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from datetime import timedelta
from pathlib import Path
from typing import Any

from orbit_worker.verify import CommandOutcome
from orbit_worker.workspace import (
    DockerWorkspaceAdapter,
    LocalWorkspaceAdapter,
    OpenSandboxWorkspaceAdapter,
    PersistentWorkspaceAdapter,
    WorkspaceError,
    WorkspaceLease,
)

_OUTPUT_KEPT = 64 * 1024
_SANDBOX_DIR = "/workspace"
# Room for the SDK call to return after the sandbox itself enforced the command timeout.
_SDK_GRACE_S = 30


async def run_command(adapter: Any, lease: WorkspaceLease, command: str, *, timeout_s: int) -> CommandOutcome:
    """Run `command` in the workspace behind `lease` and say how it ended. A command that outlives `timeout_s` is
    killed and reported as timed out."""
    backend = adapter.adapter if isinstance(adapter, PersistentWorkspaceAdapter) else adapter
    # DockerWorkspaceAdapter extends the local one, so it has to be tested first.
    if isinstance(backend, DockerWorkspaceAdapter):
        return await _docker(lease, command, timeout_s)
    if isinstance(backend, LocalWorkspaceAdapter):
        return await _local(backend, lease, command, timeout_s)
    if isinstance(backend, OpenSandboxWorkspaceAdapter):
        return await _opensandbox(backend, lease, command, timeout_s)
    raise WorkspaceError(f"{type(backend).__name__} cannot run commands")


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")[-_OUTPUT_KEPT:]


async def _local(backend: LocalWorkspaceAdapter, lease: WorkspaceLease, command: str, timeout_s: int) -> CommandOutcome:
    root = Path(backend.root) / lease.tenant_id / lease.workspace_id
    # Not the worker's environment: it holds the database URL and the checkpoint key.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root), "LANG": "C.UTF-8"}
    proc = await asyncio.create_subprocess_exec(
        "sh", "-c", command,
        cwd=str(root), env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        # A new session, so the whole group goes: `sh -c "a & b"` leaves children the shell does not wait for.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
        return CommandOutcome(exit_code=None, output="", timed_out=True)
    return CommandOutcome(exit_code=proc.returncode, output=_decode(output), timed_out=False)


async def _docker(lease: WorkspaceLease, command: str, timeout_s: int) -> CommandOutcome:
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "-w", _SANDBOX_DIR, f"orbit-{lease.workspace_id}", "sh", "-c", command,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        # This ends the client; the container, and the process in it, are removed when the lease is released.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return CommandOutcome(exit_code=None, output="", timed_out=True)
    return CommandOutcome(exit_code=proc.returncode, output=_decode(output), timed_out=False)


async def _opensandbox(
    backend: OpenSandboxWorkspaceAdapter, lease: WorkspaceLease, command: str, timeout_s: int
) -> CommandOutcome:
    from opensandbox.models.execd import RunCommandOpts

    sandbox = backend._require_sandbox(lease)
    opts = RunCommandOpts(working_directory=_SANDBOX_DIR, timeout=timedelta(seconds=timeout_s))
    try:
        execution = await asyncio.wait_for(sandbox.commands.run(command, opts=opts), timeout_s + _SDK_GRACE_S)
    except TimeoutError:
        return CommandOutcome(exit_code=None, output="", timed_out=True)
    error = getattr(execution, "error", None)
    output = str(getattr(execution, "text", "") or "")
    if error is not None:
        output = f"{output}\n{getattr(error, 'name', 'error')}: {getattr(error, 'value', '')}".strip()
    exit_code = getattr(execution, "exit_code", None)
    if exit_code is None:
        # No exit code: it is a pass only if the sandbox reported no error either.
        exit_code = 0 if error is None else 1
    return CommandOutcome(exit_code=exit_code, output=output[-_OUTPUT_KEPT:], timed_out=False)

"""Run one shell command inside a leased workspace, for the verification checks (04 §5).

Every backend runs it where the workspace is (see `WorkspaceAdapter.exec`): a process group on the host for `local`
(dev and tests only), `docker exec` in the container for `docker`, the SDK's command API for `opensandbox`.
"""

from __future__ import annotations

from typing import Any

from orbit_worker.verify import CommandOutcome
from orbit_worker.workspace import WorkspaceLease

_OUTPUT_KEPT = 64 * 1024


async def run_command(adapter: Any, lease: WorkspaceLease, command: str, *, timeout_s: int) -> CommandOutcome:
    """Run `command` in the workspace behind `lease` and say how it ended. A command that outlives `timeout_s` is
    killed and reported as timed out."""
    result = await adapter.exec(lease, ["sh", "-c", command], timeout_s=timeout_s)
    if result.timed_out:
        return CommandOutcome(exit_code=None, output="", timed_out=True)
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")[-_OUTPUT_KEPT:]
    return CommandOutcome(exit_code=result.exit_code, output=output, timed_out=False)

"""Temporal activities for completion verification (04 §5).

`verify_schema`, `verify_artifacts` and `verify_completion` are short io activities. `verify_command` runs the node's
test command in a sandbox, so it is on the agent queue with heartbeats and a timeout of its own. The checks
themselves are in `orbit_worker.verify`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from orbit_worker import verify
from orbit_worker.workspace import WorkspaceError
from orbit_worker.workspace_exec import run_command

HEARTBEAT_INTERVAL_S = 10

_registry: verify.SchemaRegistry | None = None


def set_schema_registry(registry: verify.SchemaRegistry | None) -> None:
    global _registry
    _registry = registry


def _bad_payload(message: str) -> ApplicationError:
    """A payload that is malformed is a bug in the caller: retrying the same payload cannot help."""
    return ApplicationError(message, type="BAD_VERIFY_PAYLOAD", non_retryable=True)


def _require(payload: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if not payload.get(key)]
    if missing:
        raise _bad_payload(f"verification payload is missing {', '.join(missing)}")


class _StorePorts:
    """The task store as the artifact check's manifest table and blob store."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def get_manifest(self, tenant_id: str, manifest_id: str) -> dict[str, Any] | None:
        return await self._store.get_manifest(tenant_id=tenant_id, manifest_id=manifest_id)  # type: ignore[no-any-return]

    async def blob_digest(self, tenant_id: str, blob_ref: str) -> tuple[str, int] | None:
        return await self._store.artifact_blob_digest(tenant_id=tenant_id, blob_ref=blob_ref)  # type: ignore[no-any-return]


def _ports() -> verify.VerifyPorts:
    from orbit_worker.task_activities import get_task_store

    return _StorePorts(get_task_store())


def _verdict(failures: list[verify.Failure]) -> dict[str, Any]:
    return {"ok": not failures, "failures": failures}


@activity.defn(name="verify_schema")
async def verify_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Check `output` against the schema `schema_ref` names (04 §5, step 1)."""
    _require(payload, "schema_ref")
    return _verdict(await verify.verify_schema(_registry, str(payload["schema_ref"]), dict(payload.get("output") or {})))


@activity.defn(name="verify_artifacts")
async def verify_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    """Check the required artifacts and that every blob of the manifest is stored intact (04 §5, step 2)."""
    _require(payload, "tenant_id")
    outcome = await verify.verify_artifacts(
        _ports(),
        str(payload["tenant_id"]),
        payload.get("manifest_id"),
        list(payload.get("required_artifacts") or []),
    )
    return {**_verdict(outcome.failures), "workspace_snapshot_ref": outcome.workspace_snapshot_ref}


@activity.defn(name="verify_completion")
async def verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    """The io checks of a completion proposal, composed from the node's `completion_contract`. The payload is the
    proposal plus `tenant_id`, `task_id` and the node's contract; the answer lists every failure it found."""
    _require(payload, "tenant_id", "node_id")
    return await verify.verify_completion(
        payload, dict(payload.get("completion_contract") or {}), registry=_registry, ports=_ports()
    )


async def _heartbeats() -> None:
    while True:
        activity.heartbeat()
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)


@activity.defn(name="verify_command")
async def verify_command(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one command check on the workspace snapshot the completion was made with (04 §5, step 3).

    The snapshot is restored into a scratch workspace that exists only for this activity, so the check cannot change
    what the attempt produced and two checks on one completion see the same files."""
    _require(payload, "tenant_id", "task_id", "attempt_id", "command")
    spec = verify.command_spec(_as_verification(payload))
    if isinstance(spec, dict):
        raise _bad_payload(spec["message"])
    snapshot = payload.get("workspace_snapshot_ref")
    if not snapshot:
        return _verdict([verify.failure(
            "command", "workspace_snapshot_unavailable",
            "the completion has no workspace snapshot to run the command on", command=spec.command,
        )])
    from orbit_worker.task_activities import get_workspace_adapter

    workspace = get_workspace_adapter()
    if workspace is None:
        raise ApplicationError("no workspace adapter is installed", type="NO_WORKSPACE", non_retryable=True)
    heartbeat = asyncio.create_task(_heartbeats())
    try:
        # A lease that cannot be taken is an error the activity retry handles, not a verdict on the completion.
        lease = await workspace.acquire(str(payload["tenant_id"]), str(payload["task_id"]), holder=f"verify:{payload['attempt_id']}")
        try:
            try:
                await workspace.restore(lease, str(snapshot))
            except (WorkspaceError, OSError) as exc:
                return _verdict([verify.failure(
                    "command", "workspace_snapshot_unavailable",
                    f"workspace snapshot {snapshot} cannot be restored: {exc}", command=spec.command,
                )])
            outcome = await run_command(workspace, lease, spec.command, timeout_s=spec.timeout_s)
        finally:
            await workspace.release(lease)
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
    return {**_verdict(outcome.failures(spec.command)), "exit_code": outcome.exit_code}


def _as_verification(payload: dict[str, Any]) -> Any:
    from orbit_contracts.v3.nodes import Verification

    spec = {"command": payload["command"]}
    if "timeout_s" in payload:
        spec["timeout_s"] = payload["timeout_s"]
    return Verification(kind="command", spec=spec)


VERIFY_IO_ACTIVITIES = [verify_completion, verify_schema, verify_artifacts]
VERIFY_AGENT_ACTIVITIES = [verify_command]

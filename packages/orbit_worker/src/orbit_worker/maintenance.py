"""Short maintenance activities invoked by Temporal maintenance schedules."""

from __future__ import annotations

from typing import Any

from temporalio import activity

from orbit_worker.attempt_cleanup import TemporalAttemptProbe, cleanup_attempts
from orbit_worker.lease_reaper import reap_leases
from orbit_worker.task_store import TaskStore
from orbit_worker.workspace import SandboxKiller

_store: TaskStore | None = None
_workspaces: SandboxKiller | None = None


def set_maintenance_store(store: TaskStore) -> None:
    global _store
    _store = store


def set_maintenance_workspaces(workspaces: SandboxKiller) -> None:
    global _workspaces
    _workspaces = workspaces


@activity.defn(name="maintenance_tick")
async def maintenance_tick(payload: dict[str, Any]) -> dict[str, Any]:
    if _store is None:
        raise RuntimeError("task store is not installed")
    operation = str(payload.get("operation", ""))
    tenant_id = str(payload.get("tenant_id", "default"))
    if operation == "cleanup_attempts":
        # Temporal, not the row's age, says whether an attempt is orphaned (17 G2). `removed` counts rows corrected.
        removed = await cleanup_attempts(_store, tenant_id, TemporalAttemptProbe(activity.client()))
    elif operation == "reap_leases":
        if _workspaces is None:
            raise RuntimeError("workspace backend is not installed")
        removed = await reap_leases(_store, _workspaces, tenant_id)
    else:
        removed = await _store.maintenance(operation, tenant_id=tenant_id)
    return {"operation": operation, "tenant_id": tenant_id, "removed": removed}


MAINTENANCE_ACTIVITIES = [maintenance_tick]

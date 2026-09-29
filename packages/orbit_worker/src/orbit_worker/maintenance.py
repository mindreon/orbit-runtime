"""Short maintenance activities invoked by Temporal maintenance schedules."""

from __future__ import annotations

from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator
from temporalio import activity

from orbit_worker.activity_input import parse_input
from orbit_worker.attempt_cleanup import TemporalAttemptProbe, cleanup_attempts
from orbit_worker.lease_reaper import reap_leases
from orbit_worker.task_store import TaskStore
from orbit_worker.workspace import SandboxKiller

logger = structlog.get_logger(__name__)

_store: TaskStore | None = None
_workspaces: SandboxKiller | None = None


def set_maintenance_store(store: TaskStore) -> None:
    global _store
    _store = store


def set_maintenance_workspaces(workspaces: SandboxKiller) -> None:
    global _workspaces
    _workspaces = workspaces


class MaintenanceInput(BaseModel):
    """What a maintenance schedule sends: an operation, and either one tenant or `all_tenants` (17 G7). A payload
    that names neither is a mistake, so it is rejected instead of quietly running for some default tenant."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["reap_leases", "gc_checkpoints", "cleanup_attempts"]
    tenant_id: str | None = Field(None, min_length=1)
    all_tenants: bool = False
    # Read by RuntimeMaintenanceWorkflow, which sends the same dict on.
    task_queue: str | None = None

    @model_validator(mode="after")
    def _one_scope(self) -> MaintenanceInput:
        if (self.tenant_id is None) == (not self.all_tenants):
            raise ValueError("name exactly one of tenant_id and all_tenants")
        return self


class MaintenanceError(RuntimeError):
    """The operation failed for some tenants. The others were done, and running it again is safe."""


async def _run_for_tenant(store: TaskStore, operation: str, tenant_id: str) -> int:
    if operation == "cleanup_attempts":
        # Temporal, not the row's age, says whether an attempt is orphaned (17 G2). Counts rows corrected.
        return await cleanup_attempts(store, tenant_id, TemporalAttemptProbe(activity.client()))
    if operation == "reap_leases":
        assert _workspaces is not None  # checked by the tick before it visits any tenant
        return await reap_leases(store, _workspaces, tenant_id)
    return await store.maintenance(operation, tenant_id=tenant_id)


@activity.defn(name="maintenance_tick")
async def maintenance_tick(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one operation for one tenant, or for every tenant that exists when the tick runs, each in a transaction of
    its own (so RLS holds) and each independent of the others' failures."""
    request = parse_input(MaintenanceInput, payload)
    if _store is None:
        raise RuntimeError("task store is not installed")
    if request.operation == "reap_leases" and _workspaces is None:
        raise RuntimeError("workspace backend is not installed")
    tenants = [request.tenant_id] if request.tenant_id else await _store.list_tenants()
    removed: dict[str, int] = {}
    failed: list[str] = []
    for tenant_id in tenants:
        try:
            removed[tenant_id] = await _run_for_tenant(_store, request.operation, tenant_id)
        except Exception:
            logger.exception("maintenance failed", operation=request.operation, tenant_id=tenant_id)
            failed.append(tenant_id)
    if failed:
        raise MaintenanceError(
            f"{request.operation} failed for {len(failed)} of {len(tenants)} tenants: {', '.join(failed)}"
        )
    return {
        "operation": request.operation,
        "tenants": len(tenants),
        "tenant_id": request.tenant_id,
        "removed": sum(removed.values()),
    }


MAINTENANCE_ACTIVITIES = [maintenance_tick]

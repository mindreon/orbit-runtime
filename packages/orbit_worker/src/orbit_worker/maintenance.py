"""Short maintenance activities invoked by Temporal maintenance schedules."""

from __future__ import annotations

from typing import Any

from temporalio import activity

from orbit_worker.task_store import TaskStore

_store: TaskStore | None = None


def set_maintenance_store(store: TaskStore) -> None:
    global _store
    _store = store


@activity.defn(name="maintenance_tick")
async def maintenance_tick(payload: dict[str, Any]) -> dict[str, Any]:
    if _store is None:
        raise RuntimeError("task store is not installed")
    operation = str(payload.get("operation", ""))
    tenant_id = str(payload.get("tenant_id", "default"))
    return {
        "operation": operation,
        "tenant_id": tenant_id,
        "removed": await _store.maintenance(operation, tenant_id=tenant_id),
    }


MAINTENANCE_ACTIVITIES = [maintenance_tick]

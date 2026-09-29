"""Maintenance over every tenant against the real schema (17 G7): a tenant created after the worker started is
covered by the next tick, and each tenant is worked on under its own RLS."""

from __future__ import annotations

from orbit_worker.maintenance import (
    maintenance_tick,
    set_maintenance_store,
    set_maintenance_workspaces,
)

PAST = 946684800.0


class FakeSandboxes:
    backend = "local"

    def __init__(self) -> None:
        self.killed: list[tuple[str, str]] = []

    async def kill(self, tenant_id: str, workspace_id: str) -> None:
        self.killed.append((tenant_id, workspace_id))


async def _expired_lease(store, tenant: str, lease_id: str) -> None:
    await store.acquire_workspace_lease(
        lease_id=lease_id,
        tenant_id=tenant,
        lease_key=f"{tenant}/task",
        lease_mode="write",
        backend="local",
        holder_attempt="att_1",
        expires_at=PAST,
    )


async def test_a_tenant_created_after_startup_is_reaped_on_the_next_tick(clean_db, task_store) -> None:
    sandboxes = FakeSandboxes()
    set_maintenance_store(task_store)
    set_maintenance_workspaces(sandboxes)
    await _expired_lease(task_store, "tenant-a", "ws_a")
    payload = {"operation": "reap_leases", "all_tenants": True}
    assert (await maintenance_tick(payload))["removed"] == 1

    conn = await clean_db.owner()
    try:
        await conn.execute("INSERT INTO tenants(id) VALUES ('tenant-late')")
    finally:
        await conn.close()
    await _expired_lease(task_store, "tenant-late", "ws_late")
    result = await maintenance_tick(payload)
    assert result["removed"] == 1
    assert sandboxes.killed == [("tenant-a", "ws_a"), ("tenant-late", "ws_late")]


async def test_a_single_tenant_payload_still_works(clean_db, task_store) -> None:
    sandboxes = FakeSandboxes()
    set_maintenance_store(task_store)
    set_maintenance_workspaces(sandboxes)
    await _expired_lease(task_store, "tenant-a", "ws_a")
    await _expired_lease(task_store, "tenant-b", "ws_b")
    result = await maintenance_tick({"operation": "reap_leases", "tenant_id": "tenant-b"})
    assert result["removed"] == 1 and sandboxes.killed == [("tenant-b", "ws_b")]

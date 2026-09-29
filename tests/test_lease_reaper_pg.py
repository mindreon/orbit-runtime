"""reap_leases against the real schema (17 G6, A33 dev part): expired leases lose their sandbox, the rest is left."""

from __future__ import annotations

import pytest
from orbit_worker.lease_reaper import LeaseReapError, reap_leases

TENANT = "tenant-a"
OTHER = "tenant-b"
PAST = 946684800.0  # 2000-01-01
FUTURE = 4102444800.0  # 2100-01-01


class FakeSandboxes:
    def __init__(self, backend: str = "local", failing: set[str] | None = None) -> None:
        self.backend = backend
        self.killed: list[tuple[str, str]] = []
        self.failing = failing or set()

    async def kill(self, tenant_id: str, workspace_id: str) -> None:
        if workspace_id in self.failing:
            raise RuntimeError("sandbox api is down")
        self.killed.append((tenant_id, workspace_id))


async def _lease(store, lease_id: str, *, expires_at: float, tenant: str = TENANT, mode: str = "write",
                 backend: str = "local", sandbox_id: str | None = None) -> None:
    await store.acquire_workspace_lease(
        lease_id=lease_id,
        tenant_id=tenant,
        lease_key=f"{tenant}/{lease_id}" if mode == "write" else f"{tenant}/t/ro/{lease_id}",
        lease_mode=mode,
        backend=backend,
        holder_attempt="att_1",
        expires_at=expires_at,
        sandbox_id=sandbox_id,
    )


async def _released(db, lease_id: str) -> bool:
    conn = await db.owner()
    try:
        return await conn.fetchval("SELECT released_at IS NOT NULL FROM workspace_leases WHERE lease_id=$1", lease_id)
    finally:
        await conn.close()


async def test_an_expired_lease_loses_its_sandbox_and_is_released(clean_db, task_store) -> None:
    await _lease(task_store, "ws_expired", expires_at=PAST)
    await _lease(task_store, "ws_live", expires_at=FUTURE)
    sandboxes = FakeSandboxes()
    assert await reap_leases(task_store, sandboxes, TENANT) == 1
    assert sandboxes.killed == [(TENANT, "ws_expired")]
    assert await _released(clean_db, "ws_expired") is True
    assert await _released(clean_db, "ws_live") is False
    # Reaped once: the next tick has nothing to do.
    assert await reap_leases(task_store, sandboxes, TENANT) == 0
    assert len(sandboxes.killed) == 1


async def test_the_sandbox_id_of_the_lease_is_what_gets_killed(clean_db, task_store) -> None:
    await _lease(
        task_store, "ws_lease", expires_at=PAST, backend="opensandbox", sandbox_id="sandbox-77"
    )
    sandboxes = FakeSandboxes("opensandbox")
    await reap_leases(task_store, sandboxes, TENANT)
    assert sandboxes.killed == [(TENANT, "sandbox-77")]


async def test_read_copy_leases_are_reaped_too(clean_db, task_store) -> None:
    await _lease(task_store, "ro_copy", expires_at=PAST, mode="read")
    sandboxes = FakeSandboxes()
    assert await reap_leases(task_store, sandboxes, TENANT) == 1


async def test_other_tenants_leases_are_not_touched(clean_db, task_store) -> None:
    await _lease(task_store, "ws_other", expires_at=PAST, tenant=OTHER)
    sandboxes = FakeSandboxes()
    assert await reap_leases(task_store, sandboxes, TENANT) == 0
    assert sandboxes.killed == []
    assert await _released(clean_db, "ws_other") is False


async def test_a_sandbox_that_cannot_be_killed_keeps_its_lease_row(clean_db, task_store) -> None:
    await _lease(task_store, "ws_bad", expires_at=PAST)
    await _lease(task_store, "ws_good", expires_at=PAST - 1)
    sandboxes = FakeSandboxes(failing={"ws_bad"})
    with pytest.raises(LeaseReapError, match="ws_bad"):
        await reap_leases(task_store, sandboxes, TENANT)
    assert await _released(clean_db, "ws_good") is True
    assert await _released(clean_db, "ws_bad") is False
    # Retried on the next tick, and succeeds once the backend is back.
    sandboxes.failing.clear()
    assert await reap_leases(task_store, sandboxes, TENANT) == 1
    assert await _released(clean_db, "ws_bad") is True


async def test_a_lease_of_another_backend_is_reported_not_hidden(clean_db, task_store) -> None:
    await _lease(task_store, "ws_docker", expires_at=PAST, backend="docker")
    sandboxes = FakeSandboxes("local")
    with pytest.raises(LeaseReapError, match="ws_docker"):
        await reap_leases(task_store, sandboxes, TENANT)
    assert sandboxes.killed == []
    assert await _released(clean_db, "ws_docker") is False


async def test_a_lease_renewed_between_listing_and_release_is_kept(clean_db, task_store) -> None:
    await _lease(task_store, "ws_race", expires_at=PAST)
    await task_store.renew_workspace_lease(lease_id="ws_race", tenant_id=TENANT, expires_at=FUTURE)
    assert await task_store.mark_leases_released(tenant_id=TENANT, lease_ids=["ws_race"]) == 0
    assert await _released(clean_db, "ws_race") is False

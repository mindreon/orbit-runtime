"""TaskStore against the real orbit-control schema, as `orbit_worker` (RLS, column grants, unique keys)."""

from __future__ import annotations

import json

import asyncpg
import pytest
from orbit_worker.task_store import TaskStore

TENANT = "tenant-a"
OTHER = "tenant-b"


async def _count(db, sql: str, *args) -> int:
    conn = await db.owner()
    try:
        return int(await conn.fetchval(sql, *args))
    finally:
        await conn.close()


async def _put(
    store: TaskStore, *, attempt: str, seq: int, payload: bytes, tenant: str = TENANT
) -> str:
    return await store.put_checkpoint(
        tenant_id=tenant,
        task_id="task-1",
        node_id="n_1",
        attempt_id=attempt,
        seq=seq,
        kind="agent_state",
        payload=payload,
    )


async def test_checkpoint_round_trip_is_scoped_to_the_tenant(clean_db, task_store) -> None:
    ref = await _put(task_store, attempt="att_1", seq=1, payload=b"state-1")
    assert ref.startswith("sha256:")
    found = await task_store.latest_checkpoint(
        tenant_id=TENANT, attempt_id="att_1", kind="agent_state"
    )
    assert found == b"state-1"
    # RLS: the same attempt id under another tenant sees nothing.
    other = await task_store.latest_checkpoint(
        tenant_id=OTHER, attempt_id="att_1", kind="agent_state"
    )
    assert other is None


async def test_the_same_seq_is_written_once(clean_db, task_store) -> None:
    await _put(task_store, attempt="att_1", seq=1, payload=b"first")
    await _put(task_store, attempt="att_1", seq=1, payload=b"first")
    assert await _count(clean_db, "SELECT count(*) FROM checkpoints WHERE attempt_id='att_1'") == 1


async def test_a_row_for_another_tenant_is_refused_by_rls(clean_db, task_store) -> None:
    pool = task_store.pool
    assert pool is not None
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
        with pytest.raises(asyncpg.PostgresError):
            await conn.execute(
                "INSERT INTO idempotency_ledger(scope, key, tenant_id, request_hash, status)"
                " VALUES ('side_effect', 'k', $1, 'h', 'started')",
                OTHER,
            )


async def test_ledger_claims_once_and_rejects_a_different_request(clean_db, task_store) -> None:
    def claim(request_hash: str, owner: str):
        return task_store.claim_ledger(
            tenant_id=TENANT, scope="side_effect", key="att_1:c1", request_hash=request_hash, owner=owner
        )

    first = await claim("h1", "w1")
    again = await claim("h1", "w2")
    assert first["claimed"] is True and again["claimed"] is False
    with pytest.raises(ValueError, match="different request"):
        await claim("h2", "w1")
    await task_store.finish_ledger(
        tenant_id=TENANT,
        scope="side_effect",
        key="att_1:c1",
        status="succeeded",
        result_ref={"ok": 1},
    )
    done = await claim("h1", "w3")
    assert done["status"] == "succeeded" and json.loads(done["result_ref"]) == {"ok": 1}


async def test_events_are_published_to_the_outbox_once(clean_db, task_store) -> None:
    event = {"tenant_id": TENANT, "task_id": "task-1", "event_id": "evt_1", "type": "x"}
    await task_store.publish_events([event])
    await task_store.publish_events([event])
    assert await _count(clean_db, "SELECT count(*) FROM runtime_outbox WHERE event_id='evt_1'") == 1


async def test_lease_lifecycle_keeps_one_writer(clean_db, task_store) -> None:
    lease = {
        "tenant_id": TENANT,
        "lease_key": f"{TENANT}/task-1",
        "lease_mode": "write",
        "backend": "local",
        "sandbox_id": "sb-1",
        "expires_at": 4102444800.0,
    }
    await task_store.acquire_workspace_lease(lease_id="l1", holder_attempt="att_1", **lease)
    with pytest.raises(asyncpg.UniqueViolationError):
        await task_store.acquire_workspace_lease(lease_id="l2", holder_attempt="att_2", **lease)
    await task_store.release_workspace_lease(lease_id="l1", tenant_id=TENANT)
    await task_store.acquire_workspace_lease(lease_id="l2", holder_attempt="att_2", **lease)


async def test_list_tenants_sees_every_tenant_and_only_ids(clean_db, task_store) -> None:
    assert {"default", TENANT, OTHER} <= set(await task_store.list_tenants())
    conn = await clean_db.owner()
    try:
        await conn.execute("INSERT INTO tenants(id, name) VALUES ('tenant-created-later', 'secret name')")
    finally:
        await conn.close()
    assert "tenant-created-later" in await task_store.list_tenants()
    pool = task_store.pool
    assert pool is not None
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        await pool.fetch("SELECT name FROM tenants")

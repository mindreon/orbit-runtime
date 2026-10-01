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
    store: TaskStore,
    *,
    attempt: str,
    seq: int,
    payload: bytes,
    tenant: str = TENANT,
    kind: str = "agent_state",
) -> str:
    return await store.put_checkpoint(
        tenant_id=tenant,
        task_id="task-1",
        node_id="n_1",
        attempt_id=attempt,
        seq=seq,
        kind=kind,
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


async def test_other_kinds_at_the_same_seq_are_all_kept(clean_db, task_store) -> None:
    """17 G16: the key is (attempt_id, kind, seq); a second kind must not be dropped by the conflict clause."""
    kinds = ("agent_state", "sop_run_state", "workspace_snapshot", "plan")
    for kind in kinds:
        await _put(task_store, attempt="att_k", seq=1, payload=f"{kind}-payload".encode(), kind=kind)
    assert await _count(clean_db, "SELECT count(*) FROM checkpoints WHERE attempt_id='att_k'") == len(kinds)
    for kind in kinds:
        found = await task_store.latest_checkpoint(tenant_id=TENANT, attempt_id="att_k", kind=kind)
        assert found == f"{kind}-payload".encode()


async def test_a_repeated_write_of_one_kind_stays_idempotent_next_to_another_kind(
    clean_db, task_store
) -> None:
    await _put(task_store, attempt="att_k", seq=1, payload=b"state", kind="agent_state")
    await _put(task_store, attempt="att_k", seq=1, payload=b"run", kind="sop_run_state")
    # A retried activity writes both again, and one of them with different content: the first write of each stays.
    await _put(task_store, attempt="att_k", seq=1, payload=b"state", kind="agent_state")
    await _put(task_store, attempt="att_k", seq=1, payload=b"run-retried", kind="sop_run_state")
    assert await _count(clean_db, "SELECT count(*) FROM checkpoints WHERE attempt_id='att_k'") == 2
    kept = await task_store.latest_checkpoint(tenant_id=TENANT, attempt_id="att_k", kind="sop_run_state")
    assert kept == b"run"


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


MANIFEST = "man_01J00000000000000000000001"
SNAPSHOT = "sha256:" + "7" * 64
ENTRIES = [{"name": "result.txt", "media_type": "text/plain", "size_bytes": 2, "blob_ref": "sha256:" + "a" * 64}]


async def _put_manifest(store: TaskStore, *, tenant: str = TENANT, entries=None, snapshot: str | None = SNAPSHOT) -> None:
    await store.put_manifest(
        tenant_id=tenant,
        task_id="task-1",
        attempt_id="att_1",
        manifest_id=MANIFEST,
        entries=ENTRIES if entries is None else entries,
        manifest_hash="sha256:" + "b" * 64,
        workspace_snapshot_ref=snapshot,
    )


async def test_a_manifest_the_worker_wrote_is_read_back_with_its_snapshot(clean_db, task_store) -> None:
    await _put_manifest(task_store)
    assert await task_store.get_manifest(tenant_id=TENANT, manifest_id=MANIFEST) == {
        "entries": ENTRIES,
        "workspace_snapshot_ref": SNAPSHOT,
    }
    # RLS: another tenant does not see it.
    assert await task_store.get_manifest(tenant_id=OTHER, manifest_id=MANIFEST) is None


async def test_a_manifest_without_a_snapshot_reads_back_without_one(clean_db, task_store) -> None:
    await _put_manifest(task_store, snapshot=None)
    found = await task_store.get_manifest(tenant_id=TENANT, manifest_id=MANIFEST)
    assert found is not None and found["workspace_snapshot_ref"] is None


async def test_a_manifest_is_immutable_so_a_second_write_changes_nothing(clean_db, task_store) -> None:
    await _put_manifest(task_store)
    await _put_manifest(task_store, entries=[], snapshot=None)
    found = await task_store.get_manifest(tenant_id=TENANT, manifest_id=MANIFEST)
    assert found == {"entries": ENTRIES, "workspace_snapshot_ref": SNAPSHOT}
    assert await _count(clean_db, "SELECT count(*) FROM artifact_manifests") == 1


async def _register_profile(db, *, tenant: str, ref: str, spec: dict) -> None:
    profile_id, _, version = ref.rpartition("@")
    conn = await db.owner()
    try:
        await conn.execute(
            "INSERT INTO agent_profiles(tenant_id, profile_id, version, spec) VALUES ($1, $2, $3, $4::jsonb)",
            tenant,
            profile_id,
            int(version),
            json.dumps(spec),
        )
    finally:
        await conn.close()


async def test_agent_config_is_read_from_the_tenants_profile_version(clean_db, task_store) -> None:
    spec = {
        "instructions": "Be brief.",
        "mcp_connectors": [{"id": "mcp_docs", "name": "Docs", "command": "orbit-mcp-docs"}],
    }
    await _register_profile(clean_db, tenant=TENANT, ref="writer@2", spec=spec)
    config = await task_store.agent_config(tenant_id=TENANT, profile_ref="writer@2")
    assert config.instructions == "Be brief."
    assert [item["id"] for item in config.mcp_connectors] == ["mcp_docs"]
    # Another version, another tenant, or no profile at all: the default Agent, never an error.
    for tenant, ref in ((TENANT, "writer@1"), (OTHER, "writer@2"), (TENANT, "")):
        empty = await task_store.agent_config(tenant_id=tenant, profile_ref=ref)
        assert (empty.instructions, empty.mcp_connectors) == ("", ())

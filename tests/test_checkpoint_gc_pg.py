"""Checkpoint commit and GC against the real schema (17 G1): what history references is kept, garbage is collected,
and a blob shared by digest goes only when no row references it any more."""

from __future__ import annotations

import asyncio

from orbit_worker.task_store import TaskStore

TENANT = "tenant-a"
OTHER = "tenant-b"


async def _put(
    store: TaskStore,
    attempt: str,
    seq: int,
    payload: bytes,
    *,
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


async def _age(db, hours: int = 25) -> None:
    conn = await db.owner()
    try:
        await conn.execute(f"UPDATE checkpoints SET created_at = now() - interval '{hours} hours'")
    finally:
        await conn.close()


async def _rows(db, attempt: str) -> dict[int, bool]:
    conn = await db.owner()
    try:
        found = await conn.fetch(
            "SELECT seq, committed_in_history FROM checkpoints WHERE attempt_id=$1", attempt
        )
    finally:
        await conn.close()
    return {row["seq"]: row["committed_in_history"] for row in found}


def _blob_exists(store: TaskStore, tenant: str, ref: str) -> bool:
    return (store.root / tenant / ref.removeprefix("sha256:")).exists()


async def test_a_parked_attempt_keeps_its_checkpoint_after_a_day(clean_db, task_store) -> None:
    ref = await _put(task_store, "att_parked", 1, b"parked-state")
    assert await task_store.commit_checkpoints(
        tenant_id=TENANT, attempt_id="att_parked", checkpoint_ref=ref
    ) == 1
    await _age(clean_db, 25 * 24)
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 0
    assert await _rows(clean_db, "att_parked") == {1: True}
    assert _blob_exists(task_store, TENANT, ref)


async def test_the_newest_checkpoint_of_an_attempt_is_never_collected(clean_db, task_store) -> None:
    """Even without a commit (the commit activity failed, or the attempt is still running)."""
    first = await _put(task_store, "att_run", 1, b"one")
    newest = await _put(task_store, "att_run", 2, b"two")
    await _age(clean_db)
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 1
    assert await _rows(clean_db, "att_run") == {2: False}
    assert not _blob_exists(task_store, TENANT, first)
    assert _blob_exists(task_store, TENANT, newest)


async def test_uncommitted_checkpoints_of_a_finished_attempt_are_collected(
    clean_db, task_store
) -> None:
    refs = [await _put(task_store, "att_done", seq, f"s{seq}".encode()) for seq in (1, 2, 3)]
    await task_store.commit_checkpoints(
        tenant_id=TENANT, attempt_id="att_done", checkpoint_ref=refs[2]
    )
    await _age(clean_db)
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 2
    assert await _rows(clean_db, "att_done") == {3: True}
    assert [_blob_exists(task_store, TENANT, ref) for ref in refs] == [False, False, True]


async def test_checkpoints_younger_than_a_day_stay(clean_db, task_store) -> None:
    await _put(task_store, "att_new", 1, b"one")
    await _put(task_store, "att_new", 2, b"two")
    await _age(clean_db, 23)
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 0
    assert set(await _rows(clean_db, "att_new")) == {1, 2}


async def test_a_blob_shared_by_digest_goes_with_its_last_row(clean_db, task_store) -> None:
    shared = await _put(task_store, "att_x", 1, b"identical")
    await _put(task_store, "att_x", 2, b"x-newest")
    assert await _put(task_store, "att_y", 1, b"identical") == shared
    await _age(clean_db)
    # att_x/1 is garbage, but att_y/1 (the newest of att_y) still references the blob.
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 1
    assert await _rows(clean_db, "att_x") == {2: False}
    assert _blob_exists(task_store, TENANT, shared)
    assert await task_store.get_checkpoint(tenant_id=TENANT, digest=shared) == b"identical"
    # Once att_y moves on, its old row is garbage too and the blob is not referenced by anything.
    await _put(task_store, "att_y", 2, b"y-newest")
    await _age(clean_db)
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 1
    assert not _blob_exists(task_store, TENANT, shared)


async def test_a_committed_row_keeps_the_blob_a_garbage_row_shared(clean_db, task_store) -> None:
    shared = await _put(task_store, "att_x", 1, b"identical")
    await _put(task_store, "att_x", 2, b"x-newest")
    await _put(task_store, "att_kept", 1, b"identical")
    await task_store.commit_checkpoints(
        tenant_id=TENANT, attempt_id="att_kept", checkpoint_ref=shared
    )
    await _age(clean_db)
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 1
    assert _blob_exists(task_store, TENANT, shared)


async def test_gc_and_commit_stay_inside_the_tenant(clean_db, task_store) -> None:
    await _put(task_store, "att_a", 1, b"a1", tenant=TENANT)
    await _put(task_store, "att_a", 2, b"a2", tenant=TENANT)
    other_ref = await _put(task_store, "att_b", 1, b"b1", tenant=OTHER)
    await _put(task_store, "att_b", 2, b"b2", tenant=OTHER)
    await _age(clean_db)
    assert await task_store.commit_checkpoints(
        tenant_id=TENANT, attempt_id="att_b", checkpoint_ref=other_ref
    ) == 0
    assert await task_store.maintenance("gc_checkpoints", tenant_id=TENANT) == 1
    assert set(await _rows(clean_db, "att_b")) == {1, 2}
    assert _blob_exists(task_store, OTHER, other_ref)


async def test_commit_marks_the_ref_and_the_newest_of_each_kind_once(clean_db, task_store) -> None:
    await _put(task_store, "att_c", 1, b"one")
    await _put(task_store, "att_c", 2, b"two")
    # The two kinds use different seq ranges to keep their newest rows apart.
    await _put(task_store, "att_c", 11, b"run-1", kind="sop_run_state")
    run_ref = await _put(task_store, "att_c", 12, b"run-2", kind="sop_run_state")
    assert await task_store.commit_checkpoints(
        tenant_id=TENANT, attempt_id="att_c", checkpoint_ref=run_ref
    ) == 2
    assert await task_store.commit_checkpoints(
        tenant_id=TENANT, attempt_id="att_c", checkpoint_ref=run_ref
    ) == 0
    conn = await clean_db.owner()
    try:
        marked = await conn.fetch(
            "SELECT kind, seq FROM checkpoints WHERE attempt_id='att_c' AND committed_in_history"
            " ORDER BY kind"
        )
    finally:
        await conn.close()
    assert [(row["kind"], row["seq"]) for row in marked] == [
        ("agent_state", 2),
        ("sop_run_state", 12),
    ]


async def test_a_write_racing_gc_of_the_same_digest_keeps_its_blob(clean_db, task_store) -> None:
    """The blob of a garbage row is written again by a new checkpoint at the same moment."""
    for round_no in range(15):
        payload = f"same-{round_no}".encode()
        await _put(task_store, f"att_old_{round_no}", 1, payload)
        await _put(task_store, f"att_old_{round_no}", 2, b"newest-%d" % round_no)
        await _age(clean_db)
        _, ref = await asyncio.gather(
            task_store.maintenance("gc_checkpoints", tenant_id=TENANT),
            _put(task_store, f"att_new_{round_no}", 1, payload),
        )
        assert await task_store.get_checkpoint(tenant_id=TENANT, digest=ref) == payload

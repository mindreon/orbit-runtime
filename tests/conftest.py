from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import control_db as cdb
import pytest
from orbit_worker.task_store import TaskStore


@pytest.fixture(scope="session")
def control_db() -> Iterator[cdb.ControlDb]:
    reason = cdb.unavailable_reason()
    if reason is not None:
        if cdb.required():
            pytest.fail(reason)
        pytest.skip(reason)
    db = cdb.create_control_db()
    try:
        yield db
    finally:
        cdb.drop_control_db(db)


@pytest.fixture
async def clean_db(control_db: cdb.ControlDb) -> AsyncIterator[cdb.ControlDb]:
    """The migrated database with every runtime table emptied (tenants stay)."""
    owner = await control_db.owner()
    try:
        await owner.execute(
            "TRUNCATE checkpoints, stage_attempts, workspace_leases, idempotency_ledger, runtime_outbox, artifact_manifests"
        )
    finally:
        await owner.close()
    yield control_db


@pytest.fixture
async def task_store(
    clean_db: cdb.ControlDb, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[TaskStore]:
    """A TaskStore connected as `orbit_worker`, with checkpoint blobs on the local disk."""
    monkeypatch.delenv("ORBIT_OBJECT_STORE_ENDPOINT", raising=False)
    monkeypatch.delenv("ORBIT_CHECKPOINT_FERNET_KEY", raising=False)
    store = TaskStore(url=clean_db.url("orbit_worker"), root=str(tmp_path / "blobs"))
    await store.start()
    try:
        yield store
    finally:
        await store.close()

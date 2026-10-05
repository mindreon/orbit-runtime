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


@pytest.fixture(autouse=True)
def _dump_asyncio_tasks_when_a_test_is_stuck(request: pytest.FixtureRequest) -> Iterator[None]:
    """A test that runs for minutes is stuck: print what every asyncio task is awaiting, which the faulthandler's thread
    dump cannot show. The pytest-asyncio loop is not running yet when this fixture sets up, so the watchdog is a thread
    that asks the running loop (found by its tasks) to print them."""
    import sys
    import threading

    done = threading.Event()

    def watch() -> None:
        if done.wait(180):
            return
        import asyncio
        import gc

        print(f"\n=== {request.node.nodeid} has run for 180s: asyncio tasks ===", file=sys.stderr, flush=True)
        for obj in gc.get_objects():
            if isinstance(obj, asyncio.Task) and not obj.done():
                print(f"--- {obj.get_name()}: {obj.get_coro()!r}", file=sys.stderr, flush=True)
                try:
                    obj.print_stack(limit=6, file=sys.stderr)
                except Exception as exc:  # noqa: BLE001 - a diagnostic must not fail the run
                    print(f"(no stack: {exc})", file=sys.stderr, flush=True)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()

"""Attempts of one task run side by side, each in a workspace of its own; only saving into the task's head is exclusive.

How it can go wrong, written down before the code:
  - the second attempt of a task cannot get a workspace because the first one holds the task's writer lease for its whole run;
  - the second attempt to save writes its whole workspace as the head and the first one's files are gone;
  - an attempt that deleted a file puts it back, or one that did not touch a file takes the other's version of it away;
  - a saver that finds the lock taken fails at once, or waits for ever, or says nothing about why it gave up;
  - the head is read before the previous saver's manifest is written, so two savers merge onto the same head;
  - a task that is saved by one attempt at a time is changed by the merge.
"""

from __future__ import annotations

import asyncio
import io
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest
from orbit_worker.sandbox import SandboxSession
from orbit_worker.workspace import (
    LocalWorkspaceAdapter,
    PersistentWorkspaceAdapter,
    WorkspaceBusy,
    WorkspaceError,
)
from orbit_worker.workspace_merge import merge_archives
from structlog.testing import capture_logs

TENANT, TASK = "tenant-a", "task-a"


class _Store:
    """What the worker needs of Postgres: the manifests (the head is the newest with a snapshot), and the lease rows with the
    unique index of one live write lease per key."""

    def __init__(self) -> None:
        self.manifests: list[tuple[str, str, str]] = []  # (task, attempt, snapshot)
        self.leases: dict[str, dict[str, Any]] = {}

    def _live_writer(self, key: str) -> bool:
        return any(
            row["lease_key"] == key and row["lease_mode"] == "write" and not row["released"] for row in self.leases.values()
        )

    async def latest_workspace_snapshot(self, *, tenant_id: str, task_id: str, before_attempt: str | None = None) -> str | None:
        found = [snap for task, attempt, snap in self.manifests if task == task_id and attempt != before_attempt]
        return found[-1] if found else None

    async def attempt_workspace_snapshot(self, *, tenant_id: str, task_id: str, attempt_id: str) -> str | None:
        found = [snap for task, attempt, snap in self.manifests if task == task_id and attempt == attempt_id]
        return found[-1] if found else None

    async def acquire_workspace_lease(self, **row: Any) -> None:
        if row["lease_mode"] == "write" and self._live_writer(row["lease_key"]):
            raise RuntimeError('duplicate key value violates unique constraint "workspace_leases_one_writer_idx"')
        self.leases[row["lease_id"]] = {**row, "released": False}

    async def try_acquire_commit_lease(self, **row: Any) -> bool:
        if self._live_writer(row["lease_key"]):
            return False
        self.leases[row["lease_id"]] = {**row, "lease_mode": "write", "released": False}
        return True

    async def renew_workspace_lease(self, **kwargs: Any) -> None: ...

    async def release_workspace_lease(self, *, lease_id: str, tenant_id: str) -> None:
        self.leases[lease_id]["released"] = True


def _session(adapter: Any, store: _Store, attempt: str) -> SandboxSession:
    return SandboxSession(adapter, store, tenant_id=TENANT, task_id=TASK, holder=attempt)


async def _save(session: SandboxSession, store: _Store, attempt: str) -> str | None:
    async def record(ref: str) -> None:
        store.manifests.append((TASK, attempt, ref))

    return await session.close(on_saved=record)


async def _head_files(adapter: Any, store: _Store) -> dict[str, str]:
    """The files of the task's head, restored into a fresh workspace."""
    session = _session(adapter, store, "reader")
    lease = await session.lease()
    root = adapter.adapter.root / lease.tenant_id / lease.workspace_id
    files = {str(path.relative_to(root)): path.read_text() for path in root.rglob("*") if path.is_file()}
    await adapter.release(lease)
    return files


def _adapter(tmp_path: Path, store: _Store, **kwargs: Any) -> PersistentWorkspaceAdapter:
    return PersistentWorkspaceAdapter(LocalWorkspaceAdapter(tmp_path), store, **kwargs)


async def _write(adapter: Any, session: SandboxSession, name: str, text: str) -> None:
    await adapter.write_file(await session.lease(), name, text.encode())


async def test_two_attempts_of_a_task_hold_a_workspace_at_once_and_both_sets_of_files_survive(tmp_path: Path) -> None:
    store = _Store()
    adapter = _adapter(tmp_path, store)
    backend, frontend = _session(adapter, store, "att-backend"), _session(adapter, store, "att-frontend")
    await _write(adapter, backend, "api/main.py", "api")  # the second one used to fail here: one writer per task
    await _write(adapter, frontend, "web/app.tsx", "web")
    assert len([row for row in store.leases.values() if row["lease_mode"] == "write"]) == 2
    await asyncio.gather(_save(backend, store, "att-backend"), _save(frontend, store, "att-frontend"))
    assert await _head_files(adapter, store) == {"api/main.py": "api", "web/app.tsx": "web"}
    assert all(row["released"] for row in store.leases.values() if row["lease_id"].startswith("ws_")), "every lease given back"


async def test_the_second_to_save_keeps_the_first_ones_files_whatever_the_order(tmp_path: Path) -> None:
    for first in ("att-a", "att-b"):
        store = _Store()
        adapter = _adapter(tmp_path / first, store)
        a, b = _session(adapter, store, "att-a"), _session(adapter, store, "att-b")
        await _write(adapter, a, "a.txt", "from a")
        await _write(adapter, b, "b.txt", "from b")
        order = [("att-a", a), ("att-b", b)] if first == "att-a" else [("att-b", b), ("att-a", a)]
        for name, session in order:  # one after the other: the second starts from the base it found, not from the head
            await _save(session, store, name)
        assert await _head_files(adapter, store) == {"a.txt": "from a", "b.txt": "from b"}


async def test_a_deletion_is_kept_and_a_file_nobody_touched_is_not_lost(tmp_path: Path) -> None:
    store = _Store()
    adapter = _adapter(tmp_path, store)
    seed = _session(adapter, store, "att-seed")
    await _write(adapter, seed, "old.txt", "old")
    await _write(adapter, seed, "keep.txt", "keep")
    await _save(seed, store, "att-seed")
    cleaner, writer = _session(adapter, store, "att-clean"), _session(adapter, store, "att-write")
    lease = await cleaner.lease()
    await adapter.exec(lease, ["rm", "old.txt"])
    await _write(adapter, writer, "new.txt", "new")
    await _write(adapter, writer, "keep.txt", "keep, edited")
    await _save(writer, store, "att-write")
    await _save(cleaner, store, "att-clean")
    assert await _head_files(adapter, store) == {"keep.txt": "keep, edited", "new.txt": "new"}


async def test_an_attempt_that_changed_nothing_leaves_the_head_as_it_is(tmp_path: Path) -> None:
    store = _Store()
    adapter = _adapter(tmp_path, store)
    seed = _session(adapter, store, "att-seed")
    await _write(adapter, seed, "a.txt", "a")
    await _save(seed, store, "att-seed")
    idle, busy = _session(adapter, store, "att-idle"), _session(adapter, store, "att-busy")
    await idle.lease()
    await _write(adapter, busy, "b.txt", "b")
    await _save(busy, store, "att-busy")
    head = await store.latest_workspace_snapshot(tenant_id=TENANT, task_id=TASK)
    assert await _save(idle, store, "att-idle") == head, "no new snapshot for a merge that adds nothing"


async def test_attempts_one_after_the_other_are_saved_as_they_were(tmp_path: Path) -> None:
    store = _Store()
    adapter = _adapter(tmp_path, store)
    first = _session(adapter, store, "att-1")
    await _write(adapter, first, "a.txt", "a")
    ref = await _save(first, store, "att-1")
    second = _session(adapter, store, "att-2")
    await _write(adapter, second, "b.txt", "b")
    assert ref == await store.latest_workspace_snapshot(tenant_id=TENANT, task_id=TASK)
    with capture_logs() as logs:
        await _save(second, store, "att-2")
    # The head had not moved since it started: the plain snapshot of its workspace, as before, no merge.
    assert not any(log["event"].startswith("workspace merged") for log in logs)
    assert await _head_files(adapter, store) == {"a.txt": "a", "b.txt": "b"}


async def test_the_next_saver_reads_the_head_the_previous_one_wrote_because_the_manifest_is_written_under_the_lock(
    tmp_path: Path,
) -> None:
    store = _Store()
    adapter = _adapter(tmp_path, store)
    sessions = {f"att-{n}": _session(adapter, store, f"att-{n}") for n in range(5)}
    for name, session in sessions.items():
        await _write(adapter, session, f"{name}.txt", name)
    await asyncio.gather(*(_save(session, store, name) for name, session in sessions.items()))
    assert await _head_files(adapter, store) == {f"{name}.txt": name for name in sessions}


async def test_a_saver_waits_for_the_lock_and_then_goes_on(tmp_path: Path) -> None:
    store = _Store()
    adapter = _adapter(tmp_path, store, commit_wait_s=5)
    other_worker = _adapter(tmp_path, store, commit_wait_s=5)  # shares only the database
    released = asyncio.Event()

    async def hold() -> None:
        async with other_worker.commit_lock(TENANT, TASK, "att-holder"):
            await released.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    entered = asyncio.Event()

    async def wait() -> None:
        async with adapter.commit_lock(TENANT, TASK, "att-waiter"):
            entered.set()

    waiter = asyncio.create_task(wait())
    await asyncio.sleep(0.3)
    assert not entered.is_set(), "taken by the other worker: wait"
    released.set()
    await asyncio.wait_for(asyncio.gather(holder, waiter), 5)
    assert entered.is_set()


async def test_a_lock_held_past_the_wait_budget_fails_cleanly_and_says_why(tmp_path: Path) -> None:
    store = _Store()
    holder_adapter = _adapter(tmp_path, store)
    waiter_adapter = _adapter(tmp_path, store, commit_wait_s=0.3)
    release = asyncio.Event()

    async def hold() -> None:
        async with holder_adapter.commit_lock(TENANT, TASK, "att-holder"):
            await release.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    started = time.monotonic()
    with capture_logs() as logs, pytest.raises(WorkspaceBusy, match="being saved by another attempt"):
        async with waiter_adapter.commit_lock(TENANT, TASK, "att-waiter"):
            pytest.fail("the lock was not free")
    assert 0.25 < time.monotonic() - started < 3
    assert any(log["event"].startswith("workspace commit lock still held") and log["log_level"] == "error" for log in logs)
    release.set()
    await holder
    async with waiter_adapter.commit_lock(TENANT, TASK, "att-waiter"):
        pass  # given back: the next one gets it


async def test_attempts_of_one_worker_are_kept_apart_without_the_database(tmp_path: Path) -> None:
    adapter = PersistentWorkspaceAdapter(LocalWorkspaceAdapter(tmp_path), _Store(), commit_wait_s=0.2)
    inside = 0
    overlap = 0

    async def save(name: str) -> None:
        nonlocal inside, overlap
        async with adapter.commit_lock(TENANT, TASK, name):
            inside += 1
            overlap = max(overlap, inside)
            await asyncio.sleep(0.01)
            inside -= 1

    await asyncio.gather(*(save(f"att-{n}") for n in range(4)))
    assert overlap == 1 and adapter._commit_locks == {}, "one at a time, and nothing is kept for a task that is done"


async def test_a_lease_that_cannot_be_persisted_logs_the_cause(tmp_path: Path) -> None:
    class Broken(_Store):
        async def acquire_workspace_lease(self, **row: Any) -> None:
            raise RuntimeError("duplicate key value violates unique constraint")

    adapter = _adapter(tmp_path, Broken())
    with capture_logs() as logs, pytest.raises(WorkspaceError, match="could not be persisted") as raised:
        await adapter.acquire(TENANT, TASK, holder="att-1")
    assert isinstance(raised.value.__cause__, RuntimeError)
    entry = next(log for log in logs if log["event"] == "workspace lease could not be persisted")
    assert entry["task_id"] == TASK and entry["exc_info"] is True


# ---- the merge itself --------------------------------------------------------------------------------------------------


def _tar(files: dict[str, str], dirs: tuple[str, ...] = ()) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name in dirs:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    return output.getvalue()


def _names(archive: bytes) -> dict[str, str]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return {m.name: (tar.extractfile(m).read().decode() if m.isreg() else "") for m in tar.getmembers()}  # type: ignore[union-attr]


def test_merge_applies_only_what_the_attempt_changed() -> None:
    base = _tar({"a": "1", "b": "1", "c": "1"})
    mine = _tar({"a": "1", "b": "2", "d": "new"})  # changed b, deleted c, added d
    head = _tar({"a": "other", "c": "1", "e": "theirs"})  # someone changed a, added e
    merged = merge_archives(base, mine, head)
    assert merged is not None
    assert _names(merged) == {"a": "other", "b": "2", "d": "new", "e": "theirs"}


def test_merge_with_no_base_puts_everything_in_and_keeps_empty_directories() -> None:
    merged = merge_archives(None, _tar({"x": "1"}, dirs=("emptydir",)), _tar({"y": "2"}))
    assert merged is not None and _names(merged) == {"emptydir": "", "x": "1", "y": "2"}


def test_merge_that_changes_nothing_returns_none() -> None:
    same = _tar({"a": "1"})
    assert merge_archives(same, same, _tar({"a": "1", "b": "2"})) is None
    assert merge_archives(same, _tar({"a": "1", "z": "3"}), _tar({"a": "1", "z": "3"})) is None, "the head already has it"

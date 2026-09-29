"""Agent session state lives in checkpoints: any worker reads what another wrote."""

from __future__ import annotations

from typing import Any

import pytest
from cryptography.fernet import InvalidToken
from orbit_contracts.models import OpenSessionInput, ResolveApprovalInput, RunTurnInput
from orbit_worker.checkpoint_state import SESSION_SEQ_BASE, CheckpointStateStore, session_seq
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import SessionBlob, StateUnreadableError
from orbit_worker.task_stream import TaskStreamContext, streaming_for

ATTEMPT = "att_01J9Z3K4M5N6P7Q8R9S0T1V2W3"
CONTEXT = TaskStreamContext(
    tenant_id="tenant-a", task_id="task-1", attempt_id=ATTEMPT, activity_attempt=1, node_id="n_1"
)


class FakeTaskStore:
    """The checkpoint half of TaskStore, shared by every 'worker' of a test."""

    def __init__(self) -> None:
        self.pool = object()
        self.rows: list[dict[str, Any]] = []
        self.unreadable = False

    async def put_checkpoint(self, **row: Any) -> str:
        if not any(
            r["attempt_id"] == row["attempt_id"] and r["seq"] == row["seq"] for r in self.rows
        ):
            self.rows.append(row)
        return "sha256:0"

    async def latest_checkpoint(
        self, *, tenant_id: str, attempt_id: str, kind: str, min_seq: int = 0
    ) -> bytes | None:
        if self.unreadable:
            raise InvalidToken
        rows = [
            r
            for r in self.rows
            if r["tenant_id"] == tenant_id
            and r["attempt_id"] == attempt_id
            and r["kind"] == kind
            and r["seq"] >= min_seq
        ]
        return max(rows, key=lambda r: r["seq"])["payload"] if rows else None


def _blob(version: int = 1) -> SessionBlob:
    return SessionBlob(
        session_id=ATTEMPT,
        task_id="task-1",
        state_version=version,
        agent_state={"context": "conversation"},
        permission_preset="workspace-write",
    )


@pytest.mark.asyncio
async def test_outside_an_attempt_the_state_stays_in_memory() -> None:
    tasks = FakeTaskStore()
    store = CheckpointStateStore(tasks)  # type: ignore[arg-type]
    await store.put(_blob())
    assert tasks.rows == []
    assert (await store.get(ATTEMPT)) is not None


@pytest.mark.asyncio
async def test_a_new_store_reads_the_latest_checkpoint_of_the_attempt() -> None:
    tasks = FakeTaskStore()
    with streaming_for(CONTEXT):
        first = CheckpointStateStore(tasks)  # type: ignore[arg-type]
        await first.put(_blob(1))
        await first.put(_blob(2))
        loaded = await CheckpointStateStore(tasks).get(ATTEMPT)  # type: ignore[arg-type]
    assert loaded is not None and loaded.state_version == 2
    assert {row["kind"] for row in tasks.rows} == {"agent_state"}
    assert all(row["seq"] >= SESSION_SEQ_BASE for row in tasks.rows)
    assert (tasks.rows[0]["tenant_id"], tasks.rows[0]["node_id"]) == ("tenant-a", "n_1")


def test_seq_grows_with_every_write_of_a_session() -> None:
    blob = _blob(1)
    before = session_seq(blob)
    blob.idempotency["turn-1:runTurn"] = {}
    assert session_seq(blob) > before
    blob.state_version = 2
    assert session_seq(blob) > session_seq(_blob(1))


@pytest.mark.asyncio
async def test_a_checkpoint_the_key_cannot_read_is_unreadable() -> None:
    tasks = FakeTaskStore()
    tasks.unreadable = True
    with streaming_for(CONTEXT), pytest.raises(StateUnreadableError):
        await CheckpointStateStore(tasks).get(ATTEMPT)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_parked_approval_is_decided_by_a_worker_that_never_saw_it() -> None:
    tasks = FakeTaskStore()
    with streaming_for(CONTEXT):
        first = AgentRuntime(CheckpointStateStore(tasks))  # type: ignore[arg-type]
        opened = await first.open_session(OpenSessionInput(room_id="task-1", turn_id="open"))
        assert opened.session_id == ATTEMPT
        parked = await first.run_turn(
            RunTurnInput(
                room_id="task-1",
                session_id=opened.session_id,
                turn_id="turn-1",
                message="echo:hello",
                state_version=opened.state_version,
            )
        )
        assert parked.status == "needs_approval" and parked.approval is not None
        # Opening again after a retry finds the session instead of forking it.
        again = await AgentRuntime(CheckpointStateStore(tasks)).open_session(  # type: ignore[arg-type]
            OpenSessionInput(room_id="task-1", turn_id="open")
        )
        assert again.state_version == parked.state_version
        resumed = await AgentRuntime(CheckpointStateStore(tasks)).resolve_approval(  # type: ignore[arg-type]
            ResolveApprovalInput(
                room_id="task-1",
                session_id=ATTEMPT,
                turn_id="approve-1",
                approval_request_id=parked.approval.approval_request_id,
                outcome="allowed-once",
            )
        )
    assert resumed.status == "completed"

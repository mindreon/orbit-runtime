"""Agent session state as checkpoints (08 §3).

An attempt's session is written as an `agent_state` checkpoint after every turn, into the same table and object store as
every other checkpoint, and read back from there. The session id of an attempt is its attempt id, so any worker that
gets the next activity of the attempt finds the state. Outside an attempt (no task context, or no database) the state
lives in memory: SOP steps and local development open throwaway sessions that never outlive one call.

Turn-end checkpoints take `seq` from `SESSION_SEQ_BASE` up, clear of the per-batch checkpoints the policy middleware
writes below it; within a session `seq` grows with every write, a failed turn's cached result included.
"""

from __future__ import annotations

from cryptography.fernet import InvalidToken
from pydantic import ValidationError

from orbit_worker.secrets import reject_secret_values
from orbit_worker.store import MemoryStateStore, SessionBlob, StateUnreadableError
from orbit_worker.task_store import TaskStore
from orbit_worker.task_stream import current_task_context

SESSION_SEQ_BASE = 1_000_000_000
_KIND = "agent_state"


def session_seq(blob: SessionBlob) -> int:
    return SESSION_SEQ_BASE + blob.state_version * 100 + len(blob.idempotency)


class CheckpointStateStore:
    def __init__(self, tasks: TaskStore) -> None:
        self._tasks = tasks
        self._memory = MemoryStateStore()

    def _durable(self):
        context = current_task_context()
        return context if context is not None and self._tasks.pool is not None else None

    async def put(self, blob: SessionBlob) -> None:
        reject_secret_values(blob.agent_state)
        context = self._durable()
        if context is None:
            await self._memory.put(blob)
            return
        await self._tasks.put_checkpoint(
            tenant_id=context.tenant_id,
            task_id=context.task_id,
            node_id=context.node_id,
            attempt_id=context.attempt_id,
            seq=session_seq(blob),
            kind=_KIND,
            payload=blob.model_dump_json().encode("utf-8"),
        )

    async def get(self, session_id: str) -> SessionBlob | None:
        context = self._durable()
        if context is None:
            return await self._memory.get(session_id)
        try:
            raw = await self._tasks.latest_checkpoint(
                tenant_id=context.tenant_id,
                attempt_id=session_id,
                kind=_KIND,
                min_seq=SESSION_SEQ_BASE,
            )
        except InvalidToken:
            raise StateUnreadableError("checkpoint does not decrypt with the current key") from None
        if raw is None:
            return None
        try:
            return SessionBlob.model_validate_json(raw)
        except ValidationError:
            raise StateUnreadableError("checkpoint is not a session") from None

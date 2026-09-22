"""In-process stand-in for the AgentState store (ADR-010).

Production replaces this with Postgres. The interface is the same: load the
latest blob, save the next version. The blob is opaque JSON to everyone
except the worker.
"""

from pydantic import BaseModel, Field


class SessionBlob(BaseModel):
    session_id: str
    room_id: str
    state_version: int
    agent_state: dict
    permission_preset: str
    closed: bool = False
    idempotency: dict[str, dict] = Field(default_factory=dict)


class MemoryStateStore:
    """Process-local store. A worker restart drops it; tests share one instance."""

    def __init__(self) -> None:
        self._rows: dict[str, SessionBlob] = {}

    async def put(self, blob: SessionBlob) -> None:
        self._rows[blob.session_id] = blob

    async def get(self, session_id: str) -> SessionBlob | None:
        row = self._rows.get(session_id)
        return row.model_copy(deep=True) if row is not None else None

    async def find_by_idempotency(self, room_id: str, key: str) -> SessionBlob | None:
        for row in self._rows.values():
            if row.room_id == room_id and key in row.idempotency:
                return row.model_copy(deep=True)
        return None

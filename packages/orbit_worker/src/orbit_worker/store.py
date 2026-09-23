"""AgentState blobs (ADR-010).

The worker owns the blob. Control stores only the session id and the
current state version. A blob is opaque JSON plus bookkeeping the worker
needs to retry an Activity safely.
"""

from typing import Protocol

from pydantic import BaseModel, Field

from orbit_worker.secrets import reject_secret_values


class SessionBlob(BaseModel):
    session_id: str
    room_id: str
    state_version: int
    agent_state: dict
    permission_preset: str
    closed: bool = False
    idempotency: dict[str, dict] = Field(default_factory=dict)
    isolation_mode: str = "local"
    share_net: bool = False
    backend: str = "local"
    runtime: str = "agentscope"
    runtime_version: str = "2.0.8"


class StateStore(Protocol):
    async def put(self, blob: SessionBlob) -> None: ...

    async def get(self, session_id: str) -> SessionBlob | None: ...

    async def find_by_idempotency(self, room_id: str, key: str) -> SessionBlob | None: ...


class MemoryStateStore:
    """Process-local store. Tests and a worker without Postgres use it."""

    def __init__(self) -> None:
        self._rows: dict[str, SessionBlob] = {}

    async def put(self, blob: SessionBlob) -> None:
        reject_secret_values(blob.agent_state)
        current = self._rows.get(blob.session_id)
        if current is not None and current.state_version > blob.state_version:
            raise ValueError(
                f"state version {blob.state_version} is older than {current.state_version}"
            )
        self._rows[blob.session_id] = blob.model_copy(deep=True)

    async def get(self, session_id: str) -> SessionBlob | None:
        row = self._rows.get(session_id)
        return row.model_copy(deep=True) if row is not None else None

    async def find_by_idempotency(self, room_id: str, key: str) -> SessionBlob | None:
        for row in self._rows.values():
            if row.room_id == room_id and key in row.idempotency:
                return row.model_copy(deep=True)
        return None

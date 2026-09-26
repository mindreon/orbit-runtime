"""AgentState blobs (ADR-010).

The worker owns the blob. Control stores only the session id and the
current state version. A blob is opaque JSON plus bookkeeping the worker
needs to retry an Activity safely.
"""

from typing import Protocol

from orbit_contracts.models import AgentRef
from pydantic import BaseModel, Field

from orbit_worker.secrets import reject_secret_values

# Product text for TurnErrorCode.STATE_UNREADABLE: the session's agent state
# is void. Never built from the blob, the key, or the exception.
STATE_UNREADABLE_MESSAGE = "这个会话的运行状态已无法读取，无法继续对话；历史记录仍可查看。"


class StateUnreadableError(Exception):
    """The saved blob exists but this worker may not or cannot read it.

    ``str(exc)`` is ``STATE_UNREADABLE_MESSAGE``. ``reason`` names the case
    (never key material or blob bytes) and is for worker logs only.
    ``state_version`` is the row's stored version, which is kept in clear.
    """

    def __init__(self, reason: str, state_version: int = 0) -> None:
        super().__init__(STATE_UNREADABLE_MESSAGE)
        self.reason = reason
        self.state_version = state_version


class SessionBlob(BaseModel):
    session_id: str
    room_id: str
    state_version: int
    agent_state: dict
    permission_preset: str
    # Identity stamped on every event of this session.
    agent: AgentRef = Field(default_factory=AgentRef)
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

"""Postgres state store. Blobs are encrypted when ORBIT_STATE_KEY is set."""

import json
from collections.abc import Awaitable, Callable

from cryptography.fernet import Fernet

from orbit_worker.secrets import reject_secret_values
from orbit_worker.store import SessionBlob

_DDL = """
CREATE TABLE IF NOT EXISTS orbit_agent_state (
    session_id text PRIMARY KEY,
    room_id text NOT NULL,
    state_version bigint NOT NULL,
    runtime text NOT NULL,
    runtime_version text NOT NULL,
    permission_preset text NOT NULL,
    closed boolean NOT NULL,
    blob bytea NOT NULL,
    idempotency jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS orbit_agent_idempotency (
    room_id text NOT NULL,
    idem_key text NOT NULL,
    session_id text NOT NULL,
    PRIMARY KEY (room_id, idem_key)
);
"""


def encode_blob(blob: SessionBlob, key: str) -> bytes:
    raw = blob.model_dump_json().encode("utf-8")
    if not key:
        return b"plain:" + raw
    token = Fernet(key.encode("utf-8")).encrypt(raw)
    return b"fernet:" + token


def decode_blob(payload: bytes, key: str) -> SessionBlob:
    if payload.startswith(b"fernet:"):
        if not key:
            raise ValueError("encrypted blob requires ORBIT_STATE_KEY")
        raw = Fernet(key.encode("utf-8")).decrypt(payload.removeprefix(b"fernet:"))
    elif payload.startswith(b"plain:"):
        raw = payload.removeprefix(b"plain:")
    else:
        raw = payload
    return SessionBlob.model_validate_json(raw)


class PostgresStateStore:
    """Versioned AgentState rows. An older version is rejected on write."""

    def __init__(self, connect: Callable[[], Awaitable[object]], key: str = "") -> None:
        self._connect = connect
        self._key = key

    async def ensure_schema(self) -> None:
        conn = await self._connect()
        try:
            for statement in _DDL.split(";"):
                sql = statement.strip()
                if sql:
                    await conn.execute(sql)  # type: ignore[attr-defined]
        finally:
            await conn.close()  # type: ignore[attr-defined]

    async def put(self, blob: SessionBlob) -> None:
        reject_secret_values(blob.agent_state)
        payload = encode_blob(blob, self._key)
        conn = await self._connect()
        try:
            current = await conn.fetchrow(  # type: ignore[attr-defined]
                "SELECT state_version FROM orbit_agent_state WHERE session_id = $1",
                blob.session_id,
            )
            if current is not None and int(current["state_version"]) > blob.state_version:
                raise ValueError(
                    f"state version {blob.state_version} is older than {current['state_version']}"
                )
            await conn.execute(  # type: ignore[attr-defined]
                """
                INSERT INTO orbit_agent_state (
                    session_id, room_id, state_version, runtime, runtime_version,
                    permission_preset, closed, blob, idempotency
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                ON CONFLICT (session_id) DO UPDATE SET
                    room_id = EXCLUDED.room_id,
                    state_version = EXCLUDED.state_version,
                    runtime = EXCLUDED.runtime,
                    runtime_version = EXCLUDED.runtime_version,
                    permission_preset = EXCLUDED.permission_preset,
                    closed = EXCLUDED.closed,
                    blob = EXCLUDED.blob,
                    idempotency = EXCLUDED.idempotency
                """,
                blob.session_id,
                blob.room_id,
                blob.state_version,
                blob.runtime,
                blob.runtime_version,
                blob.permission_preset,
                blob.closed,
                payload,
                json.dumps(blob.idempotency),
            )
            for idem_key in blob.idempotency:
                await conn.execute(  # type: ignore[attr-defined]
                    """
                    INSERT INTO orbit_agent_idempotency (room_id, idem_key, session_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (room_id, idem_key) DO UPDATE SET session_id = EXCLUDED.session_id
                    """,
                    blob.room_id,
                    idem_key,
                    blob.session_id,
                )
        finally:
            await conn.close()  # type: ignore[attr-defined]

    async def get(self, session_id: str) -> SessionBlob | None:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(  # type: ignore[attr-defined]
                "SELECT blob FROM orbit_agent_state WHERE session_id = $1",
                session_id,
            )
        finally:
            await conn.close()
        if row is None:
            return None
        return decode_blob(bytes(row["blob"]), self._key)

    async def find_by_idempotency(self, room_id: str, key: str) -> SessionBlob | None:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(  # type: ignore[attr-defined]
                """
                SELECT session_id FROM orbit_agent_idempotency
                WHERE room_id = $1 AND idem_key = $2
                """,
                room_id,
                key,
            )
        finally:
            await conn.close()
        if row is None:
            return None
        return await self.get(str(row["session_id"]))

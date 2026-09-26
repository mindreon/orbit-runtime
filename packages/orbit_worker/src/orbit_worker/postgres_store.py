"""Postgres state store. Blobs are Fernet-encrypted with ORBIT_STATE_KEY.

Production is the default. Plaintext blobs are written and read only when
``ORBIT_ALLOW_PLAINTEXT_STATE`` is exactly ``1``. There is no migration: in
production a ``plain:`` blob, or a ``fernet:`` blob the current key cannot
decrypt, is unreadable and its session's agent state is void.
"""

import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from pydantic import ValidationError

from orbit_worker.secrets import reject_secret_values
from orbit_worker.store import SessionBlob, StateUnreadableError

KEY_VAR = "ORBIT_STATE_KEY"
PLAINTEXT_VAR = "ORBIT_ALLOW_PLAINTEXT_STATE"

_PLAIN = b"plain:"
_FERNET = b"fernet:"
# Fernet.generate_key(): 32 bytes, url-safe base64, one "=" of padding.
# base64 decoding alone skips stray characters, so the form is checked first.
_FERNET_KEY = re.compile(r"[A-Za-z0-9_-]{43}=")

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


class StateConfigError(RuntimeError):
    """The state store configuration is unusable. The message names variables only."""


@dataclass(frozen=True)
class StateCipher:
    """How blobs are sealed. ``fernet`` is None only when plaintext is allowed."""

    fernet: Fernet | None
    allow_plaintext: bool


def resolve_state_cipher(env: Mapping[str, str] | None = None) -> StateCipher:
    """Read the state key at worker startup.

    Raises ``StateConfigError`` when production has no key, or when a key is
    set but is not a Fernet key. The error never carries the key.
    """

    source = os.environ if env is None else env
    allow_plaintext = source.get(PLAINTEXT_VAR) == "1"
    raw = source.get(KEY_VAR, "")
    if not raw:
        if allow_plaintext:
            return StateCipher(fernet=None, allow_plaintext=True)
        raise StateConfigError(
            f"{KEY_VAR} is not set. The Postgres state store needs a Fernet key "
            f"(only {PLAINTEXT_VAR}=1 allows plaintext state, for local development)."
        )
    try:
        if not _FERNET_KEY.fullmatch(raw):
            raise ValueError
        fernet = Fernet(raw.encode("ascii"))
    # Any error here, including the base64 decoder's, may quote the key.
    except Exception:  # noqa: BLE001
        raise StateConfigError(
            f"{KEY_VAR} is not a valid Fernet key (32 url-safe base64-encoded bytes)."
        ) from None
    return StateCipher(fernet=fernet, allow_plaintext=allow_plaintext)


def encode_blob(blob: SessionBlob, cipher: StateCipher) -> bytes:
    raw = blob.model_dump_json().encode("utf-8")
    if cipher.fernet is not None:
        return _FERNET + cipher.fernet.encrypt(raw)
    if cipher.allow_plaintext:
        return _PLAIN + raw
    raise StateConfigError(f"{KEY_VAR} is required to write state")


def decode_blob(payload: bytes, cipher: StateCipher) -> SessionBlob:
    """Raise ``StateUnreadableError`` for any blob this cipher may not read."""

    if payload.startswith(_FERNET):
        if cipher.fernet is None:
            raise StateUnreadableError("encrypted blob and no state key")
        try:
            raw = cipher.fernet.decrypt(payload.removeprefix(_FERNET))
        except InvalidToken:
            raise StateUnreadableError(
                "encrypted blob does not decrypt with the current key"
            ) from None
    elif payload.startswith(_PLAIN):
        if not cipher.allow_plaintext:
            raise StateUnreadableError("plaintext blob and plaintext state is not allowed")
        raw = payload.removeprefix(_PLAIN)
    else:
        raise StateUnreadableError("blob has no known format prefix")
    try:
        return SessionBlob.model_validate_json(raw)
    # The validation error quotes blob content.
    except ValidationError:
        raise StateUnreadableError("blob content is not a session blob") from None


class PostgresStateStore:
    """Versioned AgentState rows. An older version is rejected on write."""

    def __init__(self, connect: Callable[[], Awaitable[object]], cipher: StateCipher) -> None:
        self._connect = connect
        self._cipher = cipher

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
        payload = encode_blob(blob, self._cipher)
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
        return decode_blob(bytes(row["blob"]), self._cipher)

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

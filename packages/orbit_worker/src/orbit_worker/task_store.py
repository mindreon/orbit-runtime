"""Worker owned persistence for task runtime writes.

The worker receives a least-privilege Postgres URL.  Every transaction sets
``app.tenant_id`` before touching tenant tables, so Postgres RLS remains the
last line of isolation even when a query is accidentally under-scoped.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
from cryptography.fernet import Fernet
from orbit_contracts.v3 import Failure, Policy
from orbit_orch.plan_engine import deterministic_id

from orbit_worker.agent_config import AgentConfig, agent_config_from_spec
from orbit_worker.policy import from_profile_spec, merge
from orbit_worker.settings import StoreSettings
from orbit_worker.sop import Step

try:
    from minio import Minio
except ImportError:  # pragma: no cover - dependency is installed in worker images.
    Minio = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class StaleAttempt:
    """A stage_attempts row still STARTING or RUNNING after a day (17 G2)."""

    attempt_id: str
    task_id: str
    node_id: str
    attempt_no: int


@dataclass(frozen=True)
class ExpiredLease:
    """A workspace lease nobody renewed: its holder is gone, its sandbox may still be there (17 G6)."""

    lease_id: str
    tenant_id: str
    backend: str
    sandbox_id: str | None


@dataclass(frozen=True)
class Closure:
    """How a stale attempt row is closed out: a terminal status and why."""

    attempt_id: str
    status: str
    failure: Failure


REPLAY_APPROVED = "replay-approved"
EXTENSION_TOOL = "orbit_request_budget_extension"


class TaskStore:
    def __init__(
        self,
        url: str | None = None,
        root: str | None = None,
        settings: StoreSettings | None = None,
    ) -> None:
        settings = settings or StoreSettings()
        self.url = url or settings.control_worker_db_url
        self.root = Path(root or settings.checkpoint_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool: asyncpg.Pool | None = None
        self._fernet = self._load_cipher(settings)
        self._bucket = settings.object_store_bucket
        self._object_store = self._build_object_store(settings)

    async def start(self) -> None:
        if self.url:
            self.pool = await asyncpg.create_pool(self.url, min_size=1, max_size=8)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()

    @contextlib.asynccontextmanager
    async def _tenant_tx(self, tenant_id: str) -> AsyncIterator[asyncpg.Connection]:
        """The only way to reach a tenant table: one transaction whose first statement binds the tenant GUC, so
        row-level security applies to everything run on the connection. Callers must have checked `self.pool`."""
        assert self.pool is not None
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            yield conn

    async def publish_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        if self.pool is None:
            raise RuntimeError(
                "durable events are written through runtime_outbox; set ORBIT_CONTROL_WORKER_DB_URL"
            )
        async with self.pool.acquire() as conn, conn.transaction():
            by_tenant: dict[str, list[dict[str, Any]]] = {}
            for event in events:
                by_tenant.setdefault(str(event.get("tenant_id", "default")), []).append(event)
            for tenant_id, tenant_events in by_tenant.items():
                await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
                for event in tenant_events:
                    await conn.execute(
                        """
                        INSERT INTO runtime_outbox (tenant_id, task_id, event_id, body)
                        VALUES ($1, $2, $3, $4::jsonb)
                        ON CONFLICT (event_id) DO NOTHING
                        """,
                        tenant_id,
                        event["task_id"],
                        event["event_id"],
                        json.dumps(event),
                    )

    async def claim_ledger(
        self,
        *,
        tenant_id: str,
        scope: str,
        key: str,
        request_hash: str,
        owner: str,
        intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.pool is None:
            return {"status": "started", "result_ref": None, "claimed": True}
        async with self._tenant_tx(tenant_id) as conn:
            row = await conn.fetchrow(
                "SELECT request_hash, status, result_ref, owner FROM idempotency_ledger WHERE scope=$1 AND key=$2",
                scope,
                key,
            )
            if row is not None:
                if row["request_hash"] != request_hash:
                    raise ValueError("idempotency key reused with a different request")
                return {
                    "status": row["status"],
                    "result_ref": row["result_ref"],
                    "owner": row["owner"],
                    "claimed": False,
                }
            await conn.execute(
                """
                INSERT INTO idempotency_ledger(scope, key, tenant_id, request_hash, status, owner, result_ref)
                VALUES ($1, $2, $3, $4, 'started', $5, $6::jsonb)
                """,
                scope,
                key,
                tenant_id,
                request_hash,
                owner,
                json.dumps(intent) if intent is not None else None,
            )
            return {"status": "started", "result_ref": None, "claimed": True}

    async def unknown_side_effects(
        self, *, tenant_id: str, attempt_id: str
    ) -> list[dict[str, Any]]:
        """Non-read-only calls of this attempt that started and never recorded an outcome, and were not yet cleared."""
        if self.pool is None:
            return []
        async with self._tenant_tx(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT key, result_ref FROM idempotency_ledger
                 WHERE scope='side_effect' AND status='started' AND key LIKE $1
                   AND COALESCE(owner, '') <> $2
                 ORDER BY first_seen
                """,
                f"{attempt_id}:%",
                REPLAY_APPROVED,
            )
        calls = []
        for row in rows:
            intent = json.loads(row["result_ref"]) if row["result_ref"] else {}
            if not intent.get("read_only", False):
                calls.append({"key": row["key"], "tool": intent.get("tool", "a tool")})
        return calls

    async def succeeded_side_effects(
        self, *, tenant_id: str, attempt_ids: list[str], limit: int = 50
    ) -> list[str]:
        """What the given attempts did that is done and was not just a read, one short line per call (the tool, a digest of
        its arguments and a look at its result), the newest `limit` of them in the order they ran. It is what an agent that
        takes over from them (another model, 11 §3) is told, so it does not do again what was already done."""
        if self.pool is None or not attempt_ids:
            return []
        async with self._tenant_tx(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT request_hash, result_ref FROM idempotency_ledger
                 WHERE scope='side_effect' AND status='succeeded' AND key LIKE ANY($1::text[])
                 ORDER BY first_seen DESC
                 LIMIT $2
                """,
                [f"{attempt_id}:%" for attempt_id in attempt_ids],
                limit * 4,  # reads are filtered out below, so look at more than will be kept
            )
        lines: list[str] = []
        for row in rows:
            result = json.loads(row["result_ref"]) if isinstance(row["result_ref"], str) else (row["result_ref"] or {})
            if result.get("read_only") or result.get("tool") == EXTENSION_TOOL:
                continue
            text = " ".join(str(result.get("text", "")).split())[:80]
            lines.append(f"{result.get('tool', 'a tool')} [{str(row['request_hash'])[:8]}]" + (f": {text}" if text else ""))
            if len(lines) == limit:
                break
        return lines[::-1]

    async def count_side_effects(self, *, tenant_id: str, attempt_id: str) -> int:
        """Tool calls this attempt has made, not counting requests for more budget."""
        if self.pool is None:
            return 0
        async with self._tenant_tx(tenant_id) as conn:
            return int(
                await conn.fetchval(
                    """
                    SELECT count(*) FROM idempotency_ledger
                     WHERE scope='side_effect' AND key LIKE $1
                       AND COALESCE(result_ref->>'tool', '') <> $2
                    """,
                    f"{attempt_id}:%",
                    EXTENSION_TOOL,
                )
            )

    async def granted_extensions(self, *, tenant_id: str, attempt_id: str) -> int:
        """How many requests for more exploration budget a person allowed for this attempt."""
        if self.pool is None:
            return 0
        async with self._tenant_tx(tenant_id) as conn:
            return int(
                await conn.fetchval(
                    """
                    SELECT count(*) FROM idempotency_ledger
                     WHERE scope='side_effect' AND key LIKE $1 AND status='succeeded' AND result_ref->>'tool' = $2
                    """,
                    f"{attempt_id}:%",
                    EXTENSION_TOOL,
                )
            )

    async def effective_policy(
        self, *, tenant_id: str, profile_ref: str, task_policy: Policy
    ) -> Policy:
        """Tenant, task and profile layers combined, each only tightening the one before (05 §6)."""
        tenant = await self._tenant_policy(tenant_id)
        profile = from_profile_spec(await self._profile_spec(tenant_id, profile_ref))
        return merge(tenant, task_policy, profile)

    async def agent_config(self, *, tenant_id: str, profile_ref: str) -> AgentConfig:
        """The instructions, model and MCP connectors a profile gives the attempt's Agent (15 T8.3)."""
        config = agent_config_from_spec(await self._profile_spec(tenant_id, profile_ref))
        return dataclasses.replace(config, bundle_ref=profile_ref) if config.bundle_skills else config

    async def _tenant_policy(self, tenant_id: str) -> Policy:
        if self.pool is None:
            return Policy()
        async with self._tenant_tx(tenant_id) as conn:
            raw = await conn.fetchval(
                "SELECT spec FROM tenant_policy WHERE tenant_id=$1", tenant_id
            )
        return Policy.model_validate(json.loads(raw)) if raw else Policy()

    async def _profile_spec(self, tenant_id: str, profile_ref: str) -> dict[str, Any]:
        if self.pool is None:
            return {}
        profile_id, _, version = profile_ref.rpartition("@")
        if not profile_id or not version.isdigit():
            return {}
        async with self._tenant_tx(tenant_id) as conn:
            raw = await conn.fetchval(
                "SELECT spec FROM agent_profiles WHERE tenant_id=$1 AND profile_id=$2 AND version=$3",
                tenant_id,
                profile_id,
                int(version),
            )
        return json.loads(raw) if raw else {}

    async def approve_replay(self, *, tenant_id: str, keys: list[str]) -> None:
        """A person approved running these unknown-outcome calls again."""
        if self.pool is None or not keys:
            return
        async with self._tenant_tx(tenant_id) as conn:
            await conn.execute(
                "UPDATE idempotency_ledger SET owner=$1, last_seen=now() WHERE scope='side_effect' AND key = ANY($2::text[])",
                REPLAY_APPROVED,
                keys,
            )

    async def finish_ledger(
        self,
        *,
        tenant_id: str,
        scope: str,
        key: str,
        status: str,
        result_ref: dict[str, Any] | None,
    ) -> None:
        if self.pool is None:
            return
        async with self._tenant_tx(tenant_id) as conn:
            await conn.execute(
                "UPDATE idempotency_ledger SET status=$1, result_ref=$2::jsonb, last_seen=now() WHERE scope=$3 AND key=$4",
                status,
                json.dumps(result_ref) if result_ref is not None else None,
                scope,
                key,
            )

    async def get_sop_steps(self, *, tenant_id: str, sop_ref: str) -> tuple[Step, ...] | None:
        """The steps of `sop_id@version`, or None when control has no such definition."""
        if self.pool is None:
            return None
        sop_id, _, version = sop_ref.rpartition("@")
        if not sop_id or not version.isdigit():
            return None
        async with self._tenant_tx(tenant_id) as conn:
            raw = await conn.fetchval(
                "SELECT steps FROM sop_definitions WHERE tenant_id=$1 AND sop_id=$2 AND version=$3",
                tenant_id,
                sop_id,
                int(version),
            )
        return tuple(Step.parse(item) for item in json.loads(raw)) if raw is not None else None

    async def get_sop_definition(self, *, tenant_id: str, sop_ref: str) -> dict[str, Any] | None:
        """`sop_id@version` as the definition control stored: `{name, description, steps}` with the steps as written (v1
        or v2; `orbit_contracts.v3.sop` gives them their defaults). None when there is no such definition."""
        if self.pool is None:
            return None
        sop_id, _, version = sop_ref.rpartition("@")
        if not sop_id or not version.isdigit():
            return None
        async with self._tenant_tx(tenant_id) as conn:
            row = await conn.fetchrow(
                "SELECT name, description, steps FROM sop_definitions WHERE tenant_id=$1 AND sop_id=$2 AND version=$3",
                tenant_id,
                sop_id,
                int(version),
            )
        if row is None:
            return None
        return {"name": row["name"] or sop_id, "description": row["description"] or "", "steps": json.loads(row["steps"])}

    async def put_checkpoint(
        self,
        *,
        tenant_id: str,
        task_id: str,
        node_id: str,
        attempt_id: str,
        seq: int,
        kind: str,
        payload: bytes,
    ) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        blob = self._fernet.encrypt(payload) if self._fernet else payload
        if self.pool is None:
            await self._write_checkpoint_blob(tenant_id, digest, blob)
            return f"sha256:{digest}"
        async with self._tenant_tx(tenant_id) as conn:
            # The blob is written and the row inserted under the digest's lock, so the GC, which removes a blob only
            # under the same lock and only when no row references it, never removes the blob of a row being added.
            await self._lock_digest(conn, tenant_id, digest)
            await self._write_checkpoint_blob(tenant_id, digest, blob)
            await conn.execute(
                """
                INSERT INTO checkpoints(checkpoint_id, tenant_id, task_id, node_id, attempt_id, seq,
                                        kind, blob_ref, size_bytes, encryption, schema_version, agentscope_version)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, 'orbit.checkpoint/1', '2.0.9')
                ON CONFLICT (attempt_id, kind, seq) DO NOTHING
                """,
                # One row per (attempt, seq, kind), whatever the content: blobs are shared by digest, rows are not.
                deterministic_id(f"{tenant_id}:{task_id}:{attempt_id}:{seq}:{kind}", "ckpt"),
                tenant_id,
                task_id,
                node_id,
                attempt_id,
                seq,
                kind,
                f"sha256:{digest}",
                len(payload),
                json.dumps({"alg": "fernet" if self._fernet else "none"}),
            )
        return f"sha256:{digest}"

    @staticmethod
    async def _lock_digest(conn: asyncpg.Connection, tenant_id: str, digest: str) -> None:
        """Held until the transaction ends. Callers that take several take them in sorted order."""
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"checkpoint:{tenant_id}/{digest}"
        )

    async def _write_checkpoint_blob(self, tenant_id: str, digest: str, blob: bytes) -> None:
        if self._object_store is not None:
            await asyncio.to_thread(self._put_object, f"checkpoints/{tenant_id}/{digest}", blob)
            return
        path = self.root / tenant_id / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(blob)

    async def _remove_checkpoint_blob(self, tenant_id: str, digest: str) -> None:
        if self._object_store is not None:
            await asyncio.to_thread(self._remove_object, f"checkpoints/{tenant_id}/{digest}")
        else:
            (self.root / tenant_id / digest).unlink(missing_ok=True)

    async def commit_checkpoints(
        self, *, tenant_id: str, attempt_id: str, checkpoint_ref: str | None = None
    ) -> int:
        """Mark what the workflow history now references as committed (08 §3); returns the rows marked.

        Called once an activity result of the attempt is in history. That is the checkpoint whose digest the result
        names, and the newest one of each kind of the attempt: a resume reads the newest session (`agent_state`) or
        run state (`sop_run_state`), and the result carries the matching session id and state version. Committed rows
        are never collected."""
        if self.pool is None:
            return 0
        async with self._tenant_tx(tenant_id) as conn:
            result = await conn.execute(
                """
                WITH newest AS (
                    SELECT DISTINCT ON (kind) checkpoint_id FROM checkpoints
                     WHERE attempt_id = $1 ORDER BY kind, seq DESC
                )
                UPDATE checkpoints SET committed_in_history = true
                 WHERE attempt_id = $1 AND NOT committed_in_history
                   AND (blob_ref = $2 OR checkpoint_id IN (SELECT checkpoint_id FROM newest))
                """,
                attempt_id,
                checkpoint_ref,
            )
        return int(result.rsplit(" ", 1)[-1])

    async def get_checkpoint(self, *, tenant_id: str, digest: str) -> bytes:
        """The checkpoint's content. Raises `cryptography.fernet.InvalidToken` when it was written with another key."""
        digest = digest.removeprefix("sha256:")
        if self._object_store is not None:
            blob = await asyncio.to_thread(self._get_object, f"checkpoints/{tenant_id}/{digest}")
        else:
            blob = (self.root / tenant_id / digest).read_bytes()
        return self._fernet.decrypt(blob) if self._fernet else blob

    async def latest_checkpoint_ref(
        self, *, tenant_id: str, attempt_id: str, kind: str, min_seq: int = 0
    ) -> str | None:
        """The reference of the newest checkpoint of an attempt (highest `seq` from `min_seq`), or None. Needs the database."""
        if self.pool is None:
            return None
        async with self._tenant_tx(tenant_id) as conn:
            return await conn.fetchval(
                """
                SELECT blob_ref FROM checkpoints
                 WHERE tenant_id=$1 AND attempt_id=$2 AND kind=$3 AND seq >= $4
                 ORDER BY seq DESC LIMIT 1
                """,
                tenant_id,
                attempt_id,
                kind,
                min_seq,
            )

    async def latest_checkpoint(
        self, *, tenant_id: str, attempt_id: str, kind: str, min_seq: int = 0
    ) -> bytes | None:
        """The newest checkpoint of an attempt (highest `seq` from `min_seq`), or None. Needs the database."""
        ref = await self.latest_checkpoint_ref(tenant_id=tenant_id, attempt_id=attempt_id, kind=kind, min_seq=min_seq)
        if ref is None:
            return None
        return await self.get_checkpoint(tenant_id=tenant_id, digest=ref)

    async def put_artifact_blob(self, *, tenant_id: str, task_id: str, payload: bytes) -> str:
        """Store one content addressed artifact and return its sha256 reference.

        With an object store the blob goes to `artifacts/<tenant>/<digest>` and control only signs a URL for it (08 §2).
        Without one, local development keeps the same reference format under the task store directory.
        """
        digest = hashlib.sha256(payload).hexdigest()
        if self._object_store is not None:
            await asyncio.to_thread(self._put_object, f"artifacts/{tenant_id}/{digest}", payload)
            return f"sha256:{digest}"
        path = self.root / "artifacts" / tenant_id / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return f"sha256:{digest}"

    async def get_artifact_blob(self, *, tenant_id: str, blob_ref: str) -> bytes:
        """The bytes of an artifact blob `put_artifact_blob` stored (a team's leader reads what its members left)."""
        digest = blob_ref.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("an artifact blob is named by its sha256")
        if self._object_store is not None:
            return await asyncio.to_thread(self._get_object, f"artifacts/{tenant_id}/{digest}")
        return await asyncio.to_thread((self.root / "artifacts" / tenant_id / digest).read_bytes)

    async def put_snapshot(self, *, tenant_id: str, digest: str, payload: bytes) -> None:
        """Store a workspace archive by content address for cross-worker restore."""
        if self._object_store is not None:
            await asyncio.to_thread(self._put_object, f"snapshots/{tenant_id}/{digest}", payload)
            return
        path = self.root / "snapshots" / tenant_id / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)

    async def get_snapshot(self, *, tenant_id: str, digest: str) -> bytes:
        digest = digest.removeprefix("sha256:")
        if self._object_store is not None:
            return await asyncio.to_thread(self._get_object, f"snapshots/{tenant_id}/{digest}")
        return (self.root / "snapshots" / tenant_id / digest).read_bytes()

    async def latest_workspace_snapshot(
        self, *, tenant_id: str, task_id: str, before_attempt: str | None = None
    ) -> str | None:
        """The snapshot a task's workspace was last left in: the newest manifest of the task that has one. Writes are
        serial (one writer lease per task), so the next attempt, on any worker, starts from it. With `before_attempt`,
        the newest one that is not that attempt's own: the workspace as that attempt found it."""
        if self.pool is None:
            return None
        async with self._tenant_tx(tenant_id) as conn:
            return await conn.fetchval(
                """
                SELECT workspace_snapshot_id FROM artifact_manifests
                 WHERE tenant_id = $1 AND task_id = $2 AND workspace_snapshot_id IS NOT NULL
                   AND ($3::text IS NULL OR attempt_id <> $3)
                 ORDER BY created_at DESC LIMIT 1
                """,
                tenant_id,
                task_id,
                before_attempt,
            )

    async def attempt_workspace_snapshot(self, *, tenant_id: str, task_id: str, attempt_id: str) -> str | None:
        """The snapshot an earlier run of this attempt left (it stopped for an approval or an answer), or None."""
        if self.pool is None:
            return None
        async with self._tenant_tx(tenant_id) as conn:
            return await conn.fetchval(
                """
                SELECT workspace_snapshot_id FROM artifact_manifests
                 WHERE tenant_id = $1 AND task_id = $2 AND attempt_id = $3 AND workspace_snapshot_id IS NOT NULL
                 ORDER BY created_at DESC LIMIT 1
                """,
                tenant_id,
                task_id,
                attempt_id,
            )

    async def get_manifest(self, *, tenant_id: str, manifest_id: str) -> dict[str, Any] | None:
        """The entries of one artifact manifest and the workspace snapshot it was taken with (03 §9), or None."""
        if self.pool is None:
            return None
        async with self._tenant_tx(tenant_id) as conn:
            row = await conn.fetchrow(
                "SELECT entries, workspace_snapshot_id FROM artifact_manifests WHERE tenant_id=$1 AND manifest_id=$2",
                tenant_id,
                manifest_id,
            )
        if row is None:
            return None
        entries = json.loads(row["entries"]) if isinstance(row["entries"], str) else row["entries"]
        return {"entries": entries, "workspace_snapshot_ref": row["workspace_snapshot_id"]}

    async def put_manifest(
        self,
        *,
        tenant_id: str,
        task_id: str,
        attempt_id: str,
        manifest_id: str,
        entries: list[dict[str, Any]],
        manifest_hash: str,
        workspace_snapshot_ref: str | None,
    ) -> None:
        """Write the manifest of a finished attempt before the workflow's `artifact.manifest_created` event reaches
        control (03 §9). The completion check reads the manifest and the workspace snapshot the command checks run on
        as soon as the attempt reports, and the projection would only insert the row later, without the snapshot.
        Manifests are immutable: an existing row (a retried activity, the projection) is left as it is."""
        if self.pool is None:
            raise RuntimeError("artifact manifests are stored in Postgres; set ORBIT_CONTROL_WORKER_DB_URL")
        async with self._tenant_tx(tenant_id) as conn:
            await conn.execute(
                """
                INSERT INTO artifact_manifests(manifest_id, tenant_id, task_id, attempt_id, workspace_snapshot_id,
                                               entries, manifest_hash)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                ON CONFLICT (manifest_id) DO NOTHING
                """,
                manifest_id,
                tenant_id,
                task_id,
                attempt_id,
                workspace_snapshot_ref,
                json.dumps(entries),
                manifest_hash,
            )

    async def artifact_blob_digest(self, *, tenant_id: str, blob_ref: str) -> tuple[str, int] | None:
        """The sha256 reference and size of the bytes stored for an artifact blob, read back, or None if absent."""
        digest = blob_ref.removeprefix("sha256:")
        if self._object_store is not None:
            return await asyncio.to_thread(self._hash_object, f"artifacts/{tenant_id}/{digest}")
        path = self.root / "artifacts" / tenant_id / digest
        return await asyncio.to_thread(self._hash_file, path)

    @staticmethod
    def _hash_file(path: Path) -> tuple[str, int] | None:
        if not path.is_file():
            return None
        hasher, size = hashlib.sha256(), 0
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                hasher.update(chunk)
                size += len(chunk)
        return f"sha256:{hasher.hexdigest()}", size

    def _hash_object(self, name: str) -> tuple[str, int] | None:
        from minio.error import S3Error

        try:
            response = self._object_store.get_object(self._bucket, name)
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject"}:
                return None
            raise
        hasher, size = hashlib.sha256(), 0
        try:
            for chunk in response.stream(1 << 20):
                hasher.update(chunk)
                size += len(chunk)
        finally:
            response.close()
            response.release_conn()
        return f"sha256:{hasher.hexdigest()}", size

    async def acquire_workspace_lease(
        self,
        *,
        lease_id: str,
        tenant_id: str,
        lease_key: str,
        lease_mode: str,
        backend: str,
        holder_attempt: str,
        expires_at: float,
        sandbox_id: str | None = None,
    ) -> None:
        if self.pool is None:
            return
        async with self._tenant_tx(tenant_id) as conn:
            # The same attempt coming back after a crash takes over its own lease; an expired one is stale.
            await conn.execute(
                """
                UPDATE workspace_leases SET released_at = now()
                 WHERE tenant_id = $1 AND lease_key = $2 AND lease_mode = $3 AND released_at IS NULL
                   AND (holder_attempt = $4 OR expires_at < now())
                """,
                tenant_id,
                lease_key,
                lease_mode,
                holder_attempt,
            )
            await conn.execute(
                """
                INSERT INTO workspace_leases
                    (lease_id, tenant_id, lease_key, lease_mode, backend, sandbox_id, holder_attempt, expires_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,to_timestamp($8))
                """,
                lease_id,
                tenant_id,
                lease_key,
                lease_mode,
                backend,
                sandbox_id,
                holder_attempt,
                expires_at,
            )

    async def renew_workspace_lease(
        self, *, lease_id: str, tenant_id: str, expires_at: float
    ) -> None:
        if self.pool is None:
            return
        async with self._tenant_tx(tenant_id) as conn:
            await conn.execute(
                "UPDATE workspace_leases SET expires_at=to_timestamp($1) WHERE tenant_id=$2 AND lease_id=$3 AND released_at IS NULL",
                expires_at,
                tenant_id,
                lease_id,
            )

    async def release_workspace_lease(self, *, lease_id: str, tenant_id: str) -> None:
        if self.pool is None:
            return
        async with self._tenant_tx(tenant_id) as conn:
            await conn.execute(
                "UPDATE workspace_leases SET released_at=now() WHERE tenant_id=$1 AND lease_id=$2 AND released_at IS NULL",
                tenant_id,
                lease_id,
            )

    async def maintenance(self, operation: str, *, tenant_id: str = "default") -> int:
        if self.pool is None:
            return 0
        async with self._tenant_tx(tenant_id) as conn:
            if operation == "reap_leases":
                raise ValueError("reap_leases needs a workspace backend: use lease_reaper.reap_leases")
            if operation == "gc_checkpoints":
                return await self._gc_checkpoints(conn, tenant_id)
            if operation == "cleanup_attempts":
                raise ValueError("cleanup_attempts needs Temporal: use attempt_cleanup.cleanup_attempts")
            raise ValueError(f"unknown maintenance operation: {operation}")

    async def list_tenants(self) -> list[str]:
        """Every tenant id, for the maintenance schedules (17 G7). `tenants` has no RLS and the worker may read its id
        column only (control migration 00018), so this is the one query that sees across tenants, and it returns
        nothing but ids. Each tenant is then worked on in a transaction of its own."""
        if self.pool is None:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM tenants ORDER BY id")
        return [row["id"] for row in rows]

    async def expired_leases(self, *, tenant_id: str, limit: int = 200) -> list[ExpiredLease]:
        """Leases past `expires_at` that were never released, oldest first."""
        if self.pool is None:
            return []
        async with self._tenant_tx(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT lease_id, backend, sandbox_id FROM workspace_leases
                 WHERE released_at IS NULL AND expires_at < now() ORDER BY expires_at LIMIT $1
                """,
                limit,
            )
        return [ExpiredLease(r["lease_id"], tenant_id, r["backend"], r["sandbox_id"]) for r in rows]

    async def mark_leases_released(self, *, tenant_id: str, lease_ids: Sequence[str]) -> int:
        """Set `released_at` on leases whose sandbox is gone. A lease that was renewed or released meanwhile is left
        alone. Returns the rows changed."""
        if self.pool is None or not lease_ids:
            return 0
        async with self._tenant_tx(tenant_id) as conn:
            result = await conn.execute(
                """
                UPDATE workspace_leases SET released_at = now()
                 WHERE lease_id = ANY($1::text[]) AND released_at IS NULL AND expires_at < now()
                """,
                list(lease_ids),
            )
        return int(result.rsplit(" ", 1)[-1])

    async def stale_attempts(self, *, tenant_id: str, limit: int = 500) -> list[StaleAttempt]:
        """Attempts still STARTING or RUNNING after 24 hours, oldest first. Age only nominates them: whether one is
        really stale is for Temporal to say (attempt_cleanup)."""
        if self.pool is None:
            return []
        async with self._tenant_tx(tenant_id) as conn:
            rows = await conn.fetch(
                """
                SELECT attempt_id, task_id, node_id, attempt_no FROM stage_attempts
                 WHERE status IN ('STARTING', 'RUNNING') AND started_at < now() - interval '24 hours'
                 ORDER BY started_at LIMIT $1
                """,
                limit,
            )
        return [
            StaleAttempt(r["attempt_id"], r["task_id"], r["node_id"], r["attempt_no"]) for r in rows
        ]

    async def close_out_attempts(self, *, tenant_id: str, closures: Sequence[Closure]) -> int:
        """Move the given attempts to their terminal status in one transaction. A row that has left STARTING or
        RUNNING meanwhile (the projector caught up) is left as it is. Never deletes. Returns the rows changed."""
        if self.pool is None or not closures:
            return 0
        changed = 0
        async with self._tenant_tx(tenant_id) as conn:
            for closure in closures:
                result = await conn.execute(
                    """
                    UPDATE stage_attempts
                       SET status = $2, failure = $3::jsonb, finished_at = now(), entity_version = entity_version + 1
                     WHERE attempt_id = $1 AND status IN ('STARTING', 'RUNNING')
                    """,
                    closure.attempt_id,
                    closure.status,
                    closure.failure.model_dump_json(),
                )
                changed += int(result.rsplit(" ", 1)[-1])
        return changed

    async def _gc_checkpoints(self, conn: asyncpg.Connection, tenant_id: str) -> int:
        """Delete checkpoints that nothing can resume from (08 §3, 17 G1); returns the rows deleted.

        A row is garbage when history never committed it, it is over a day old (longer than any activity can run,
        so no activity in flight can still hand it to history), and a newer checkpoint of the same attempt and kind
        exists (the newest one of a running or parked attempt is what it resumes from, so it stays whatever its age).
        A blob goes with its last row: it is content addressed and shared, so it is removed only when, after the
        delete and under the digest's lock, no row of the tenant references it."""
        candidates = await conn.fetch(
            """
            SELECT checkpoint_id, blob_ref FROM (
                SELECT checkpoint_id, blob_ref, committed_in_history, created_at,
                       row_number() OVER (PARTITION BY attempt_id, kind ORDER BY seq DESC) AS newer_rank
                  FROM checkpoints
            ) ranked
             WHERE newer_rank > 1 AND NOT committed_in_history AND created_at < now() - interval '24 hours'
            """
        )
        digests = sorted({str(row["blob_ref"]).removeprefix("sha256:") for row in candidates})
        for digest in digests:
            await self._lock_digest(conn, tenant_id, digest)
        deleted = await conn.fetch(
            """
            DELETE FROM checkpoints WHERE checkpoint_id = ANY($1::text[]) AND NOT committed_in_history
            RETURNING checkpoint_id
            """,
            [row["checkpoint_id"] for row in candidates],
        )
        for digest in digests:
            if not await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM checkpoints WHERE blob_ref = $1)", f"sha256:{digest}"
            ):
                await self._remove_checkpoint_blob(tenant_id, digest)
        return len(deleted)

    @staticmethod
    def _build_object_store(settings: StoreSettings):
        endpoint = settings.object_store_endpoint
        if not endpoint:
            return None
        if Minio is None:
            raise RuntimeError(
                "ORBIT_OBJECT_STORE_ENDPOINT is configured but the MinIO client is not installed"
            )
        endpoint = endpoint.removeprefix("http://").removeprefix("https://")
        return Minio(
            endpoint,
            access_key=settings.object_store_access_key,
            secret_key=settings.object_store_secret_key,
            secure=settings.object_store_secure,
            region=settings.object_store_region,
        )

    def _put_object(self, name: str, payload: bytes) -> None:
        from io import BytesIO

        self._object_store.put_object(
            self._bucket,
            name,
            BytesIO(payload),
            len(payload),
            content_type="application/octet-stream",
        )

    def _get_object(self, name: str) -> bytes:
        response = self._object_store.get_object(self._bucket, name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def _remove_object(self, name: str) -> None:
        self._object_store.remove_object(self._bucket, name)

    @staticmethod
    def _load_cipher(settings: StoreSettings) -> Fernet | None:
        key = settings.checkpoint_fernet_key
        return Fernet(key.encode("ascii")) if key else None

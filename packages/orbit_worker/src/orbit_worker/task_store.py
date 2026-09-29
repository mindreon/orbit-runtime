"""Worker owned persistence for task runtime writes.

The worker receives a least-privilege Postgres URL.  Every transaction sets
``app.tenant_id`` before touching tenant tables, so Postgres RLS remains the
last line of isolation even when a query is accidentally under-scoped.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import asyncpg
from cryptography.fernet import Fernet
from orbit_contracts.v3 import Policy
from orbit_orch.plan_engine import deterministic_id

from orbit_worker.policy import from_profile_spec, merge
from orbit_worker.sop import Step

try:
    from minio import Minio
except ImportError:  # pragma: no cover - dependency is installed in worker images.
    Minio = None  # type: ignore[assignment,misc]


REPLAY_APPROVED = "replay-approved"
EXTENSION_TOOL = "orbit_request_budget_extension"


class TaskStore:
    def __init__(self, url: str | None = None, root: str | None = None) -> None:
        self.url = url or os.environ.get("ORBIT_CONTROL_WORKER_DB_URL", "")
        self.root = Path(root or os.environ.get("ORBIT_CHECKPOINT_DIR", ".orbit-checkpoints"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool: asyncpg.Pool | None = None
        self._fernet = self._load_cipher()
        self._bucket = os.environ.get("ORBIT_OBJECT_STORE_BUCKET", "orbit")
        self._object_store = self._build_object_store()

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
            raise RuntimeError("durable events are written through runtime_outbox; set ORBIT_CONTROL_WORKER_DB_URL")
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

    async def unknown_side_effects(self, *, tenant_id: str, attempt_id: str) -> list[dict[str, Any]]:
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

    async def effective_policy(self, *, tenant_id: str, profile_ref: str, task_policy: Policy) -> Policy:
        """Tenant, task and profile layers combined, each only tightening the one before (05 §6)."""
        tenant = await self._tenant_policy(tenant_id)
        profile = from_profile_spec(await self._profile_spec(tenant_id, profile_ref))
        return merge(tenant, task_policy, profile)

    async def _tenant_policy(self, tenant_id: str) -> Policy:
        if self.pool is None:
            return Policy()
        async with self._tenant_tx(tenant_id) as conn:
            raw = await conn.fetchval("SELECT spec FROM tenant_policy WHERE tenant_id=$1", tenant_id)
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
        self, *, tenant_id: str, scope: str, key: str, status: str, result_ref: dict[str, Any] | None
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
        path = self.root / tenant_id / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._object_store is not None:
            object_name = f"checkpoints/{tenant_id}/{digest}"
            await asyncio.to_thread(self._put_object, object_name, blob)
        elif not path.exists():
            path.write_bytes(blob)
        if self.pool is not None:
            async with self._tenant_tx(tenant_id) as conn:
                await conn.execute(
                    """
                    INSERT INTO checkpoints(checkpoint_id, tenant_id, task_id, node_id, attempt_id, seq,
                                            kind, blob_ref, size_bytes, encryption, schema_version, agentscope_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, 'orbit.checkpoint/1', '2.0.9')
                    ON CONFLICT DO NOTHING
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

    async def get_checkpoint(self, *, tenant_id: str, digest: str) -> bytes:
        digest = digest.removeprefix("sha256:")
        if self._object_store is not None:
            return await asyncio.to_thread(self._get_object, f"checkpoints/{tenant_id}/{digest}")
        return (self.root / tenant_id / digest).read_bytes()

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

    async def put_snapshot(self, *, tenant_id: str, digest: str, payload: bytes) -> None:
        """Store a workspace archive by content address for cross-worker restore."""
        if self._object_store is not None:
            await asyncio.to_thread(
                self._put_object, f"snapshots/{tenant_id}/{digest}", payload
            )
            return
        path = self.root / "snapshots" / tenant_id / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)

    async def get_snapshot(self, *, tenant_id: str, digest: str) -> bytes:
        digest = digest.removeprefix("sha256:")
        if self._object_store is not None:
            return await asyncio.to_thread(
                self._get_object, f"snapshots/{tenant_id}/{digest}"
            )
        return (self.root / "snapshots" / tenant_id / digest).read_bytes()

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
                tenant_id, lease_key, lease_mode, holder_attempt,
            )
            await conn.execute(
                """
                INSERT INTO workspace_leases
                    (lease_id, tenant_id, lease_key, lease_mode, backend, sandbox_id, holder_attempt, expires_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,to_timestamp($8))
                """,
                lease_id, tenant_id, lease_key, lease_mode, backend, sandbox_id, holder_attempt, expires_at,
            )

    async def renew_workspace_lease(self, *, lease_id: str, tenant_id: str, expires_at: float) -> None:
        if self.pool is None:
            return
        async with self._tenant_tx(tenant_id) as conn:
            await conn.execute(
                "UPDATE workspace_leases SET expires_at=to_timestamp($1) WHERE tenant_id=$2 AND lease_id=$3 AND released_at IS NULL",
                expires_at, tenant_id, lease_id,
            )

    async def release_workspace_lease(self, *, lease_id: str, tenant_id: str) -> None:
        if self.pool is None:
            return
        async with self._tenant_tx(tenant_id) as conn:
            await conn.execute(
                "UPDATE workspace_leases SET released_at=now() WHERE tenant_id=$1 AND lease_id=$2 AND released_at IS NULL",
                tenant_id, lease_id,
            )

    async def maintenance(self, operation: str, *, tenant_id: str = "default") -> int:
        if self.pool is None:
            return 0
        async with self._tenant_tx(tenant_id) as conn:
            if operation == "reap_leases":
                result = await conn.execute(
                    "UPDATE workspace_leases SET released_at=now() WHERE released_at IS NULL AND expires_at < now()"
                )
                return int(result.rsplit(" ", 1)[-1])
            if operation == "gc_checkpoints":
                rows = await conn.fetch(
                    "SELECT tenant_id, blob_ref FROM checkpoints WHERE committed_in_history=false AND created_at < now() - interval '24 hours'"
                )
                await conn.execute(
                    "DELETE FROM checkpoints WHERE committed_in_history=false AND created_at < now() - interval '24 hours'"
                )
                for row in rows:
                    digest = str(row["blob_ref"]).removeprefix("sha256:")
                    if self._object_store is not None:
                        await asyncio.to_thread(self._remove_object, f"checkpoints/{row['tenant_id']}/{digest}")
                    else:
                        (self.root / str(row["tenant_id"]) / digest).unlink(missing_ok=True)
                return len(rows)
            if operation == "cleanup_attempts":
                result = await conn.execute(
                    "DELETE FROM stage_attempts WHERE status IN ('STARTING','RUNNING') AND started_at < now() - interval '24 hours'"
                )
                return int(result.rsplit(" ", 1)[-1])
            raise ValueError(f"unknown maintenance operation: {operation}")

    def _build_object_store(self):
        endpoint = os.environ.get("ORBIT_OBJECT_STORE_ENDPOINT", "")
        if not endpoint:
            return None
        if Minio is None:
            raise RuntimeError("ORBIT_OBJECT_STORE_ENDPOINT is configured but the MinIO client is not installed")
        endpoint = endpoint.removeprefix("http://").removeprefix("https://")
        return Minio(
            endpoint,
            access_key=os.environ.get("ORBIT_OBJECT_STORE_ACCESS_KEY", ""),
            secret_key=os.environ.get("ORBIT_OBJECT_STORE_SECRET_KEY", ""),
            secure=os.environ.get("ORBIT_OBJECT_STORE_SECURE", "0") == "1",
            # A fixed region skips the GetBucketLocation probe, which the worker's prefix-limited policy denies.
            region=os.environ.get("ORBIT_OBJECT_STORE_REGION", "us-east-1"),
        )

    def _put_object(self, name: str, payload: bytes) -> None:
        from io import BytesIO

        self._object_store.put_object(self._bucket, name, BytesIO(payload), len(payload), content_type="application/octet-stream")

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
    def _load_cipher() -> Fernet | None:
        raw = os.environ.get("ORBIT_CHECKPOINT_FERNET_KEY", "")
        if not raw:
            return None
        return Fernet(raw.encode("ascii"))

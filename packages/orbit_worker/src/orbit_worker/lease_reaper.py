"""Reclaim the sandboxes of expired workspace leases (08 §1, 17 G6).

A lease lives `ttl_s` seconds and the holding activity renews it on its heartbeats, so a lease past its expiry belongs
to a holder that is gone: a crashed worker, or an activity that ended without releasing. Its sandbox may still exist
(AgentScope's OpenSandbox backend only pauses one on close), so it is killed first, by the ids in the lease row, and
only then is the row marked released. A sandbox that could not be killed keeps its row, so the next tick tries again."""

from __future__ import annotations

import structlog

from orbit_worker.task_store import TaskStore
from orbit_worker.workspace import SandboxKiller

logger = structlog.get_logger(__name__)


class LeaseReapError(RuntimeError):
    """Some expired leases could not be reclaimed. The others were, and their rows are released."""


async def reap_leases(store: TaskStore, workspaces: SandboxKiller, tenant_id: str) -> int:
    """Kill the sandboxes of this tenant's expired leases and release them; returns the leases released."""
    reclaimed: list[str] = []
    failed: list[str] = []
    for lease in await store.expired_leases(tenant_id=tenant_id):
        if lease.backend != workspaces.backend:
            # Another backend's sandbox: this worker cannot destroy it, and marking it released would hide it.
            logger.error(
                "expired lease belongs to another backend",
                lease_id=lease.lease_id,
                lease_backend=lease.backend,
                worker_backend=workspaces.backend,
            )
            failed.append(lease.lease_id)
            continue
        try:
            await workspaces.kill(lease.tenant_id, lease.sandbox_id or lease.lease_id)
        except Exception:
            logger.exception("could not kill the sandbox of an expired lease", lease_id=lease.lease_id)
            failed.append(lease.lease_id)
            continue
        reclaimed.append(lease.lease_id)
    released = await store.mark_leases_released(tenant_id=tenant_id, lease_ids=reclaimed)
    if failed:
        raise LeaseReapError(
            f"{len(failed)} expired lease(s) of {tenant_id} could not be reclaimed: {', '.join(failed)}"
        )
    return released

"""Storing the artifact manifest of a turn: what it left in the workspace, and the workspace snapshot it ended in."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from orbit_orch.plan_engine import deterministic_id


class ManifestStore(Protocol):
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
    ) -> None: ...


async def record_manifest(store: ManifestStore, payload: dict[str, Any], outcome: dict[str, Any]) -> None:
    """Write the manifest of a turn once the workspace snapshot exists. The completion check of 17 G15 runs as soon as
    the attempt reports and reads the manifest from the table; the entries and the snapshot must be there by then, and
    the projection of `artifact.manifest_created` (which has no snapshot) comes later.

    A turn that completed writes the manifest the workflow announces. A turn that stopped to wait (for an approval or an
    answer) or failed writes none to announce, only its snapshot, under an id of its own: it is how the next attempt, or
    the same attempt resumed, finds the files the workspace holds."""
    snapshot = outcome.get("workspace_snapshot_ref")
    announced = outcome.get("status") == "completed" and outcome.get("manifest_id")
    if not announced and not snapshot:
        return
    task_id, attempt_id = str(payload["task_id"]), str(payload["attempt_id"])
    entries = list(outcome.get("manifest_entries", [])) if announced else []
    manifest_id = (
        str(outcome["manifest_id"])
        if announced
        else deterministic_id(f"{task_id}:{attempt_id}:workspace:{payload.get('state_version', 0)}", "man")
    )
    manifest_hash = (
        str(outcome["manifest_hash"])
        if announced
        else "sha256:" + hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    )
    await store.put_manifest(
        tenant_id=str(payload["tenant_id"]),
        task_id=task_id,
        attempt_id=attempt_id,
        manifest_id=manifest_id,
        entries=entries,
        manifest_hash=manifest_hash,
        workspace_snapshot_ref=snapshot,
    )

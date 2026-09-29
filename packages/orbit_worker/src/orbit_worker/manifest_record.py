"""Storing the artifact manifest of a completed turn."""

from __future__ import annotations

from typing import Any, Protocol


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
    """Write the manifest of a turn that completed, once the workspace snapshot exists. The completion check of
    17 G15 runs as soon as the attempt reports and reads the manifest from the table; the entries and the snapshot
    must be there by then, and the projection of `artifact.manifest_created` (which has no snapshot) comes later."""
    manifest_id = outcome.get("manifest_id")
    if outcome.get("status") != "completed" or not manifest_id:
        return
    await store.put_manifest(
        tenant_id=str(payload["tenant_id"]),
        task_id=str(payload["task_id"]),
        attempt_id=str(payload["attempt_id"]),
        manifest_id=str(manifest_id),
        entries=list(outcome.get("manifest_entries", [])),
        manifest_hash=str(outcome["manifest_hash"]),
        workspace_snapshot_ref=outcome.get("workspace_snapshot_ref"),
    )

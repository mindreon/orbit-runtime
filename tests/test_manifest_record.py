"""The manifest of a completed turn is stored before the attempt reports, with the snapshot it was taken on."""

from __future__ import annotations

from typing import Any

import pytest
from orbit_worker.manifest_record import record_manifest

PAYLOAD = {"tenant_id": "tenant-a", "task_id": "task_1", "attempt_id": "att_1", "node_id": "n_1"}
ENTRIES = [{"name": "result.txt", "media_type": "text/plain", "size_bytes": 2, "blob_ref": "sha256:" + "a" * 64}]


class _Store:
    def __init__(self) -> None:
        self.written: list[dict[str, Any]] = []

    async def put_manifest(self, **kwargs: Any) -> None:
        self.written.append(kwargs)


def _completed(**extra: Any) -> dict[str, Any]:
    return {
        "status": "completed", "manifest_id": "man_01J00000000000000000000001", "manifest_entries": ENTRIES,
        "manifest_hash": "sha256:" + "b" * 64, **extra,
    }


async def test_a_completed_turn_stores_its_manifest_with_the_workspace_snapshot() -> None:
    store = _Store()
    await record_manifest(store, PAYLOAD, _completed(workspace_snapshot_ref="sha256:" + "7" * 64))
    assert store.written == [{
        "tenant_id": "tenant-a", "task_id": "task_1", "attempt_id": "att_1",
        "manifest_id": "man_01J00000000000000000000001", "entries": ENTRIES,
        "manifest_hash": "sha256:" + "b" * 64, "workspace_snapshot_ref": "sha256:" + "7" * 64,
    }]


async def test_a_turn_without_a_workspace_has_a_manifest_without_a_snapshot() -> None:
    store = _Store()
    await record_manifest(store, PAYLOAD, _completed())
    assert store.written[0]["workspace_snapshot_ref"] is None


@pytest.mark.parametrize("outcome", [{"status": "failed", "error": "x"}, {"status": "parked_input"}, {"status": "completed"}])
async def test_only_a_completed_turn_with_a_manifest_stores_one(outcome: dict[str, Any]) -> None:
    store = _Store()
    await record_manifest(store, PAYLOAD, outcome)
    assert store.written == []

"""The verification activities, with a real local workspace and a real local blob store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from orbit_worker import task_activities, verify_activities
from orbit_worker.task_store import TaskStore
from orbit_worker.verify import DirectorySchemaRegistry
from orbit_worker.workspace import LocalWorkspaceAdapter, PersistentWorkspaceAdapter
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

TENANT = "tenant-a"


class _ManifestStore(TaskStore):
    """A TaskStore on the local blob store whose manifest table is a dict."""

    def __init__(self, root: Path) -> None:
        super().__init__(url="", root=str(root))
        self.manifests: dict[str, dict[str, Any]] = {}

    async def get_manifest(self, *, tenant_id: str, manifest_id: str) -> dict[str, Any] | None:
        return self.manifests.get(manifest_id)


class _LeaseLog:
    async def acquire_workspace_lease(self, **kwargs: Any) -> None: ...
    async def renew_workspace_lease(self, **kwargs: Any) -> None: ...
    async def release_workspace_lease(self, **kwargs: Any) -> None: ...


@pytest.fixture
def store(tmp_path: Path) -> _ManifestStore:
    instance = _ManifestStore(tmp_path / "store")
    task_activities.set_task_store(instance)
    return instance


@pytest.fixture
def workspace(tmp_path: Path) -> LocalWorkspaceAdapter:
    adapter = LocalWorkspaceAdapter(tmp_path / "workspaces")
    # As the worker installs it: leases are recorded through the persistent wrapper.
    task_activities.set_workspace_adapter(PersistentWorkspaceAdapter(adapter, _LeaseLog()))
    return adapter


@pytest.fixture(autouse=True)
def schemas(tmp_path: Path) -> None:
    root = tmp_path / "schemas"
    (root / "report").mkdir(parents=True)
    (root / "report" / "1.json").write_text(
        json.dumps({"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}})
    )
    verify_activities.set_schema_registry(DirectorySchemaRegistry(root))
    yield
    verify_activities.set_schema_registry(None)


async def _run(activity_fn: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return await ActivityEnvironment().run(activity_fn, payload)


async def _snapshot_with(workspace: LocalWorkspaceAdapter, files: dict[str, str]) -> str:
    lease = await workspace.acquire(TENANT, "task_1")
    for name, body in files.items():
        (Path(workspace.root) / TENANT / lease.workspace_id / name).write_text(body, encoding="utf-8")
    ref = await workspace.snapshot(lease)
    await workspace.release(lease)
    return ref


async def test_verify_schema_activity() -> None:
    ok = await _run(verify_activities.verify_schema, {"schema_ref": "schema://report/1", "output": {"summary": "x"}})
    bad = await _run(verify_activities.verify_schema, {"schema_ref": "schema://report/1", "output": {}})
    assert ok == {"ok": True, "failures": []}
    assert bad["ok"] is False and bad["failures"][0]["code"] == "schema_violation"


async def test_verify_artifacts_activity_reads_the_stored_bytes(store: _ManifestStore) -> None:
    ref = await store.put_artifact_blob(tenant_id=TENANT, task_id="task_1", payload=b"report body")
    store.manifests["man_1"] = {
        "entries": [{"name": "report.md", "media_type": "text/markdown", "size_bytes": 11, "blob_ref": ref}],
        "workspace_snapshot_ref": None,
    }
    required = [{"name": "report.md", "media_type": "text/markdown", "min_count": 1}]
    payload = {"tenant_id": TENANT, "manifest_id": "man_1", "required_artifacts": required}
    assert (await _run(verify_activities.verify_artifacts, payload))["ok"] is True
    # Someone changes the stored bytes behind the reference.
    (store.root / "artifacts" / TENANT / ref.removeprefix("sha256:")).write_bytes(b"tampered")
    result = await _run(verify_activities.verify_artifacts, payload)
    assert result["ok"] is False
    assert [item["code"] for item in result["failures"]] == ["artifact_hash_mismatch"]
    # ... or removes them.
    (store.root / "artifacts" / TENANT / ref.removeprefix("sha256:")).unlink()
    gone = await _run(verify_activities.verify_artifacts, payload)
    assert [item["code"] for item in gone["failures"]] == ["artifact_blob_missing"]


def _completion(**overrides: Any) -> dict[str, Any]:
    base = {
        "tenant_id": TENANT, "task_id": "task_1", "node_id": "n_1", "attempt_id": "att_1",
        "command_id": "cmd_1", "output": {"summary": "done"}, "artifact_manifest_id": None,
        "checkpoint_ref": "sha256:" + "1" * 64,
        "completion_contract": {"output_schema_ref": "schema://report/1"},
    }
    return {**base, **overrides}


async def test_verify_completion_activity_composes_the_contract(store: _ManifestStore) -> None:
    passed = await _run(verify_activities.verify_completion, _completion())
    assert passed["ok"] is True and passed["checkpoint_ref"] == "sha256:" + "1" * 64
    failed = await _run(verify_activities.verify_completion, _completion(output={"summary": 1}))
    assert failed["ok"] is False and failed["failures"][0]["check"] == "output_schema"


async def test_verify_completion_without_a_contract_accepts(store: _ManifestStore) -> None:
    payload = _completion()
    del payload["completion_contract"]
    assert (await _run(verify_activities.verify_completion, payload))["ok"] is True


@pytest.mark.parametrize("missing", ["tenant_id", "node_id"])
async def test_malformed_payload_is_not_retried(store: _ManifestStore, missing: str) -> None:
    payload = _completion()
    del payload[missing]
    with pytest.raises(ApplicationError) as raised:
        await _run(verify_activities.verify_completion, payload)
    assert raised.value.non_retryable is True


def _command(snapshot: str | None, command: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "tenant_id": TENANT, "task_id": "task_1", "node_id": "n_1", "attempt_id": "att_1",
        "workspace_snapshot_ref": snapshot, "command": command, "timeout_s": 20,
    }
    return {**base, **overrides}


async def test_verify_command_runs_on_the_snapshot_and_releases_the_workspace(
    store: _ManifestStore, workspace: LocalWorkspaceAdapter
) -> None:
    snapshot = await _snapshot_with(workspace, {"check.sh": "grep -q ready state.txt", "state.txt": "ready"})
    passed = await _run(verify_activities.verify_command, _command(snapshot, "sh check.sh"))
    assert passed["ok"] is True and passed["failures"] == []
    assert workspace._leases == {}  # the scratch workspace is gone

    broken = await _snapshot_with(workspace, {"check.sh": "grep -q ready state.txt", "state.txt": "not yet"})
    failed = await _run(verify_activities.verify_command, _command(broken, "sh check.sh"))
    assert failed["ok"] is False
    assert failed["failures"][0]["code"] == "command_failed"
    assert failed["failures"][0]["detail"]["exit_code"] == 1
    assert workspace._leases == {}


async def test_verify_command_sees_only_the_snapshot_not_later_changes(
    store: _ManifestStore, workspace: LocalWorkspaceAdapter
) -> None:
    snapshot = await _snapshot_with(workspace, {"state.txt": "v1"})
    await _snapshot_with(workspace, {"state.txt": "v2"})  # a later state of the same task
    result = await _run(verify_activities.verify_command, _command(snapshot, "grep -q v1 state.txt"))
    assert result["ok"] is True


async def test_verify_command_timeout(store: _ManifestStore, workspace: LocalWorkspaceAdapter) -> None:
    snapshot = await _snapshot_with(workspace, {"a": "1"})
    result = await _run(verify_activities.verify_command, _command(snapshot, "sleep 30", timeout_s=1))
    assert result["ok"] is False and result["failures"][0]["code"] == "command_timeout"
    assert workspace._leases == {}


async def test_verify_command_without_a_usable_snapshot_fails(
    store: _ManifestStore, workspace: LocalWorkspaceAdapter
) -> None:
    none_given = await _run(verify_activities.verify_command, _command(None, "true"))
    assert [item["code"] for item in none_given["failures"]] == ["workspace_snapshot_unavailable"]
    unknown = await _run(verify_activities.verify_command, _command("sha256:" + "9" * 64, "true"))
    assert [item["code"] for item in unknown["failures"]] == ["workspace_snapshot_unavailable"]
    assert workspace._leases == {}


async def test_verify_command_rejects_a_bad_spec(store: _ManifestStore, workspace: LocalWorkspaceAdapter) -> None:
    with pytest.raises(ApplicationError) as raised:
        await _run(verify_activities.verify_command, _command("sha256:" + "1" * 64, "  "))
    assert raised.value.non_retryable is True


def test_activities_are_registered_on_the_right_queues() -> None:
    io = {fn.__temporal_activity_definition.name for fn in task_activities.IO_ACTIVITIES}
    agent = {fn.__temporal_activity_definition.name for fn in task_activities.AGENT_ACTIVITIES}
    assert {"verify_completion", "verify_schema", "verify_artifacts"} <= io
    assert "verify_command" in agent
    assert "verify_command" not in io

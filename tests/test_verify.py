"""Completion verification (04 §5): schema, artifacts, commands and their composition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from orbit_worker.verify import (
    CommandOutcome,
    DirectorySchemaRegistry,
    verify_artifacts,
    verify_completion,
    verify_schema,
)

TENANT = "tenant-a"
REPORT_SCHEMA = {
    "type": "object",
    "required": ["summary", "score"],
    "properties": {"summary": {"type": "string"}, "score": {"type": "integer", "minimum": 0}},
    "additionalProperties": False,
}


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@pytest.fixture
def registry(tmp_path: Path) -> DirectorySchemaRegistry:
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "1.json").write_text(json.dumps(REPORT_SCHEMA))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "1.json").write_text('{"type": "not-a-type"}')
    (tmp_path / "garbled").mkdir()
    (tmp_path / "garbled" / "1.json").write_text("{nope")
    return DirectorySchemaRegistry(tmp_path)


class FakePorts:
    """The manifest table and the blob store, in memory."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.manifests: dict[str, dict[str, Any]] = {}

    def add_blob(self, data: bytes) -> str:
        ref = _sha(data)
        self.blobs[ref] = data
        return ref

    def add_manifest(self, manifest_id: str, entries: list[dict[str, Any]], snapshot: str | None = None) -> None:
        self.manifests[manifest_id] = {"entries": entries, "workspace_snapshot_ref": snapshot}

    async def get_manifest(self, tenant_id: str, manifest_id: str) -> dict[str, Any] | None:
        assert tenant_id == TENANT
        return self.manifests.get(manifest_id)

    async def blob_digest(self, tenant_id: str, blob_ref: str) -> tuple[str, int] | None:
        assert tenant_id == TENANT
        data = self.blobs.get(blob_ref)
        return None if data is None else (_sha(data), len(data))


def _entry(name: str, data: bytes, media_type: str = "text/plain", *, blob_ref: str | None = None) -> dict[str, Any]:
    return {"name": name, "media_type": media_type, "size_bytes": len(data), "blob_ref": blob_ref or _sha(data)}


def _codes(failures: list[dict[str, Any]]) -> list[str]:
    return [item["code"] for item in failures]


# ---- verify_schema ----------------------------------------------------------


async def test_schema_accepts_matching_output(registry: DirectorySchemaRegistry) -> None:
    assert await verify_schema(registry, "schema://report/1", {"summary": "ok", "score": 3}) == []


async def test_schema_rejects_output_with_every_violation_listed(registry: DirectorySchemaRegistry) -> None:
    failures = await verify_schema(registry, "schema://report/1", {"summary": 5, "extra": 1})
    assert {item["check"] for item in failures} == {"output_schema"}
    assert _codes(failures) == ["schema_violation"] * len(failures)
    paths = {item["detail"]["path"] for item in failures}
    assert "summary" in paths  # wrong type
    assert "" in paths  # missing "score" and the unexpected "extra" are reported on the object itself
    assert all(item["message"] for item in failures)


async def test_schema_without_a_ref_is_not_checked(registry: DirectorySchemaRegistry) -> None:
    assert await verify_schema(registry, None, {"anything": 1}) == []


@pytest.mark.parametrize(
    ("ref", "code"),
    [
        ("schema://missing/1", "schema_unknown"),
        ("schema://report/2", "schema_unknown"),
        ("schema://broken/1", "schema_invalid"),
        ("schema://garbled/1", "schema_invalid"),
        ("schema://../etc/1", "schema_unknown"),
        ("http://report/1", "schema_unknown"),
    ],
)
async def test_unresolvable_or_invalid_schema_fails_closed(
    registry: DirectorySchemaRegistry, ref: str, code: str
) -> None:
    failures = await verify_schema(registry, ref, {})
    assert _codes(failures) == [code]
    assert failures[0]["detail"]["schema_ref"] == ref


async def test_schema_ref_with_no_registry_fails_closed() -> None:
    failures = await verify_schema(None, "schema://report/1", {})
    assert _codes(failures) == ["schema_unknown"]


# ---- verify_artifacts -------------------------------------------------------


async def test_artifacts_pass_when_present_and_hash_matches() -> None:
    ports = FakePorts()
    data = b"hello"
    ports.add_blob(data)
    ports.add_manifest("man_1", [_entry("result.txt", data)], snapshot="sha256:" + "a" * 64)
    outcome = await verify_artifacts(
        ports, TENANT, "man_1", [{"name": "result.txt", "media_type": "text/plain", "min_count": 1}]
    )
    assert outcome.failures == []
    assert outcome.workspace_snapshot_ref == "sha256:" + "a" * 64


async def test_required_artifact_missing_from_manifest() -> None:
    ports = FakePorts()
    data = b"hello"
    ports.add_blob(data)
    ports.add_manifest("man_1", [_entry("result.txt", data)])
    outcome = await verify_artifacts(
        ports, TENANT, "man_1", [{"name": "report.pdf", "media_type": "application/pdf", "min_count": 1}]
    )
    assert _codes(outcome.failures) == ["artifact_missing"]
    assert outcome.failures[0]["detail"] == {
        "name": "report.pdf", "media_type": "application/pdf", "min_count": 1, "found": 0,
    }


async def test_min_count_and_media_type_are_enforced() -> None:
    ports = FakePorts()
    entries = []
    for index in range(2):
        data = f"page {index}".encode()
        ports.add_blob(data)
        entries.append(_entry(f"page-{index}.png", data, "image/png"))
    other = b"note"
    ports.add_blob(other)
    entries.append(_entry("page-note.png", other, "text/plain"))
    ports.add_manifest("man_1", entries)
    enough = await verify_artifacts(ports, TENANT, "man_1", [{"name": "page-*.png", "media_type": "image/png", "min_count": 2}])
    assert enough.failures == []
    short = await verify_artifacts(ports, TENANT, "man_1", [{"name": "page-*.png", "media_type": "image/png", "min_count": 3}])
    assert _codes(short.failures) == ["artifact_missing"]
    assert short.failures[0]["detail"]["found"] == 2


async def test_blob_missing_from_the_object_store() -> None:
    ports = FakePorts()
    ports.add_manifest("man_1", [_entry("result.txt", b"gone")])
    outcome = await verify_artifacts(ports, TENANT, "man_1", [])
    assert _codes(outcome.failures) == ["artifact_blob_missing"]
    assert outcome.failures[0]["detail"]["name"] == "result.txt"


async def test_blob_content_that_does_not_match_its_reference() -> None:
    ports = FakePorts()
    claimed = _sha(b"what the manifest says")
    ports.blobs[claimed] = b"what is really stored"
    ports.add_manifest("man_1", [_entry("result.txt", b"what the manifest says")])
    outcome = await verify_artifacts(ports, TENANT, "man_1", [])
    assert _codes(outcome.failures) == ["artifact_hash_mismatch"]
    assert outcome.failures[0]["detail"]["expected"] == claimed


async def test_blob_size_that_does_not_match_the_manifest() -> None:
    ports = FakePorts()
    data = b"12345"
    ports.add_blob(data)
    entry = _entry("result.txt", data)
    entry["size_bytes"] = 99
    ports.add_manifest("man_1", [entry])
    outcome = await verify_artifacts(ports, TENANT, "man_1", [])
    assert _codes(outcome.failures) == ["artifact_size_mismatch"]


async def test_entry_with_a_malformed_blob_ref() -> None:
    ports = FakePorts()
    ports.add_manifest("man_1", [_entry("result.txt", b"x", blob_ref="not-a-digest")])
    outcome = await verify_artifacts(ports, TENANT, "man_1", [])
    assert _codes(outcome.failures) == ["artifact_blob_ref_invalid"]


async def test_unknown_manifest_fails_only_when_something_is_required() -> None:
    ports = FakePorts()
    nothing_required = await verify_artifacts(ports, TENANT, None, [])
    assert nothing_required.failures == []
    missing = await verify_artifacts(ports, TENANT, "man_nope", [])
    assert _codes(missing.failures) == ["manifest_missing"]
    required = [{"name": "a", "media_type": "text/plain", "min_count": 1}]
    absent_id = await verify_artifacts(ports, TENANT, None, required)
    assert _codes(absent_id.failures) == ["manifest_missing"]


# ---- verify_completion ------------------------------------------------------


def _proposal(**overrides: Any) -> dict[str, Any]:
    base = {
        "tenant_id": TENANT,
        "task_id": "task_1",
        "node_id": "n_1",
        "attempt_id": "att_1",
        "output": {"summary": "done", "score": 1},
        "artifact_manifest_id": None,
        "checkpoint_ref": "sha256:" + "1" * 64,
    }
    return {**base, **overrides}


async def test_empty_contract_passes_and_echoes_the_checkpoint(registry: DirectorySchemaRegistry) -> None:
    result = await verify_completion(_proposal(), {}, registry=registry, ports=FakePorts())
    assert result == {
        "ok": True, "failures": [], "checkpoint_ref": "sha256:" + "1" * 64, "workspace_snapshot_ref": None,
    }


async def test_composes_schema_and_artifacts_and_reports_every_failure(registry: DirectorySchemaRegistry) -> None:
    contract = {
        "output_schema_ref": "schema://report/1",
        "required_artifacts": [{"name": "report.pdf", "media_type": "application/pdf", "min_count": 1}],
        "verifications": [],
    }
    result = await verify_completion(
        _proposal(output={"summary": 1}), contract, registry=registry, ports=FakePorts()
    )
    assert result["ok"] is False
    checks = {item["check"] for item in result["failures"]}
    assert checks == {"output_schema", "manifest"}


async def test_schema_kind_verification_uses_its_own_ref(registry: DirectorySchemaRegistry) -> None:
    contract = {"verifications": [{"kind": "schema", "spec": {"schema_ref": "schema://report/1"}}]}
    good = await verify_completion(_proposal(), contract, registry=registry, ports=FakePorts())
    bad = await verify_completion(_proposal(output={}), contract, registry=registry, ports=FakePorts())
    assert good["ok"] is True
    assert bad["ok"] is False and bad["failures"][0]["check"] == "output_schema"


async def test_schema_kind_without_a_ref_or_contract_schema_is_a_failure(registry: DirectorySchemaRegistry) -> None:
    contract = {"verifications": [{"kind": "schema", "spec": {}}]}
    result = await verify_completion(_proposal(), contract, registry=registry, ports=FakePorts())
    assert _codes(result["failures"]) == ["verification_spec_invalid"]


async def test_human_verification_is_not_silently_accepted(registry: DirectorySchemaRegistry) -> None:
    contract = {"verifications": [{"kind": "human", "spec": {}}]}
    result = await verify_completion(_proposal(), contract, registry=registry, ports=FakePorts())
    assert result["ok"] is False
    assert _codes(result["failures"]) == ["verification_unsupported"]


async def test_sop_verifier_was_already_checked_inside_the_step(registry: DirectorySchemaRegistry) -> None:
    contract = {"verifications": [{"kind": "sop_verifier", "spec": {}}]}
    assert (await verify_completion(_proposal(), contract, registry=registry, ports=FakePorts()))["ok"] is True


async def test_workspace_snapshot_is_returned_for_the_command_checks(registry: DirectorySchemaRegistry) -> None:
    ports = FakePorts()
    ports.add_manifest("man_1", [], snapshot="sha256:" + "b" * 64)
    contract = {"verifications": [{"kind": "command", "spec": {"command": "true"}}]}
    result = await verify_completion(
        _proposal(artifact_manifest_id="man_1"), contract, registry=registry, ports=ports
    )
    assert result["ok"] is True
    assert result["workspace_snapshot_ref"] == "sha256:" + "b" * 64


def test_command_outcome_failure_reasons() -> None:
    passed = CommandOutcome(exit_code=0, output="ok", timed_out=False)
    assert passed.failures("pytest -q") == []
    failed = CommandOutcome(exit_code=2, output="x" * 10_000, timed_out=False)
    [item] = failed.failures("pytest -q")
    assert item["code"] == "command_failed" and item["detail"]["exit_code"] == 2
    assert len(item["detail"]["output_tail"]) <= 4096
    assert item["detail"]["output_tail"] == "x" * 4096
    [timeout] = CommandOutcome(exit_code=None, output="", timed_out=True).failures("sleep 99")
    assert timeout["code"] == "command_timeout"

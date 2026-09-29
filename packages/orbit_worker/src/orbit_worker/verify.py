"""Completion verification (04 §5).

`verify_completion` runs the checks that fit a short io activity, in the order of the design, and reports every
failure it finds as a structured reason. The command checks (`verify_command`) run in a sandbox and are a separate
activity on the agent queue; this module holds their spec validation and their outcome type.

Nothing here talks to Temporal, Postgres or the object store directly: the manifest table and the blob store come in
through `VerifyPorts`, so the checks are plain functions over data.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jsonschema
from orbit_contracts.v3.nodes import ArtifactRequirement, CompletionContract, Verification

DEFAULT_COMMAND_TIMEOUT_S = 600
MAX_COMMAND_TIMEOUT_S = 3600
OUTPUT_TAIL_CHARS = 4096
MAX_SCHEMA_VIOLATIONS = 20
_BLOB_CONCURRENCY = 8

_SCHEMA_REF = re.compile(
    r"^schema://(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*)/(?P<version>[1-9][0-9]*)$"
)
_BLOB_REF = re.compile(r"^sha256:[0-9a-f]{64}$")

Failure = dict[str, Any]


def failure(check: str, code: str, message: str, **detail: Any) -> Failure:
    """One structured reason a completion was refused. `check` names the step, `code` is stable for callers."""
    return {"check": check, "code": code, "message": message, "detail": detail}


class VerifyPorts(Protocol):
    """What the artifact check reads: the manifest table and the blob store."""

    async def get_manifest(self, tenant_id: str, manifest_id: str) -> dict[str, Any] | None:
        """`{"entries": [...], "workspace_snapshot_ref": str | None}`, or None when there is no such manifest."""
        ...

    async def blob_digest(self, tenant_id: str, blob_ref: str) -> tuple[str, int] | None:
        """The sha256 reference and the size of what is stored for `blob_ref`, or None when nothing is."""
        ...


class SchemaRegistry(Protocol):
    def get(self, schema_ref: str) -> dict[str, Any] | None:
        """The JSON Schema behind `schema://<name>/<version>`, or None when it is not registered."""
        ...


class SchemaInvalidError(ValueError):
    """A registered schema that cannot be read or is not a valid JSON Schema."""


class DirectorySchemaRegistry:
    """`schema://<name>/<version>` is the file `<root>/<name>/<version>.json`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get(self, schema_ref: str) -> dict[str, Any] | None:
        match = _SCHEMA_REF.match(schema_ref)
        if match is None:
            return None
        path = self.root / match["name"] / f"{match['version']}.json"
        if not path.is_file():
            return None
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(schema)
        except (OSError, ValueError, jsonschema.SchemaError) as exc:
            raise SchemaInvalidError(str(exc)) from exc
        return schema  # type: ignore[no-any-return]


async def verify_schema(
    registry: SchemaRegistry | None, schema_ref: str | None, output: dict[str, Any]
) -> list[Failure]:
    """Step 1: validate `output` against `schema_ref`. No ref means there is nothing to check; a ref that cannot be
    resolved is a failure, never a pass."""
    if schema_ref is None:
        return []
    try:
        schema = None if registry is None else await asyncio.to_thread(registry.get, schema_ref)
    except SchemaInvalidError as exc:
        return [failure("output_schema", "schema_invalid", f"schema {schema_ref} is not usable: {exc}", schema_ref=schema_ref)]
    if schema is None:
        return [failure("output_schema", "schema_unknown", f"schema {schema_ref} is not registered", schema_ref=schema_ref)]
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(output), key=lambda error: (list(map(str, error.absolute_path)), error.message))
    return [
        failure(
            "output_schema",
            "schema_violation",
            error.message,
            schema_ref=schema_ref,
            path="/".join(str(part) for part in error.absolute_path),
        )
        for error in errors[:MAX_SCHEMA_VIOLATIONS]
    ]


@dataclass(frozen=True)
class ArtifactsOutcome:
    failures: list[Failure]
    # The workspace snapshot the manifest was taken with: the command checks run on it.
    workspace_snapshot_ref: str | None


async def verify_artifacts(
    ports: VerifyPorts,
    tenant_id: str,
    manifest_id: str | None,
    required: list[Any],
) -> ArtifactsOutcome:
    """Step 2: every required artifact is in the manifest, and every blob the manifest names exists in the object
    store with the content its reference says."""
    requirements = [ArtifactRequirement.model_validate(item) for item in required]
    if manifest_id is None:
        if requirements:
            return ArtifactsOutcome([failure("manifest", "manifest_missing", "the completion names no artifact manifest")], None)
        return ArtifactsOutcome([], None)
    manifest = await ports.get_manifest(tenant_id, manifest_id)
    if manifest is None:
        return ArtifactsOutcome(
            [failure("manifest", "manifest_missing", f"manifest {manifest_id} does not exist", manifest_id=manifest_id)], None
        )
    entries = [dict(entry) for entry in manifest.get("entries", [])]
    failures = [item for requirement in requirements for item in _requirement_failures(requirement, entries)]
    limit = asyncio.Semaphore(_BLOB_CONCURRENCY)

    async def check(entry: dict[str, Any]) -> list[Failure]:
        async with limit:
            return await _entry_failures(ports, tenant_id, entry)

    for found in await asyncio.gather(*(check(entry) for entry in entries)):
        failures.extend(found)
    return ArtifactsOutcome(failures, manifest.get("workspace_snapshot_ref"))


def _requirement_failures(requirement: ArtifactRequirement, entries: list[dict[str, Any]]) -> list[Failure]:
    """`name` and `media_type` are exact, or glob patterns (`page-*.png`, `image/*`)."""
    found = sum(
        1
        for entry in entries
        if fnmatch.fnmatchcase(str(entry.get("name", "")), requirement.name)
        and fnmatch.fnmatchcase(str(entry.get("media_type", "")).lower(), requirement.media_type.lower())
    )
    if found >= requirement.min_count:
        return []
    return [
        failure(
            "required_artifact",
            "artifact_missing",
            f"{found} of {requirement.min_count} required artifacts named {requirement.name} ({requirement.media_type})",
            name=requirement.name,
            media_type=requirement.media_type,
            min_count=requirement.min_count,
            found=found,
        )
    ]


async def _entry_failures(ports: VerifyPorts, tenant_id: str, entry: dict[str, Any]) -> list[Failure]:
    name, blob_ref = str(entry.get("name", "")), str(entry.get("blob_ref", ""))
    if not _BLOB_REF.match(blob_ref):
        return [failure("artifact_blob", "artifact_blob_ref_invalid", f"artifact {name} has no valid blob reference", name=name)]
    stored = await ports.blob_digest(tenant_id, blob_ref)
    if stored is None:
        return [failure("artifact_blob", "artifact_blob_missing", f"artifact {name} is not in the object store", name=name, blob_ref=blob_ref)]
    digest, size = stored
    if digest != blob_ref:
        return [
            failure(
                "artifact_blob", "artifact_hash_mismatch", f"artifact {name} does not have the content its reference names",
                name=name, expected=blob_ref, actual=digest,
            )
        ]
    declared = entry.get("size_bytes")
    if isinstance(declared, int) and declared != size:
        return [
            failure(
                "artifact_blob", "artifact_size_mismatch", f"artifact {name} is {size} bytes, the manifest says {declared}",
                name=name, expected=declared, actual=size,
            )
        ]
    return []


@dataclass(frozen=True)
class CommandSpec:
    command: str
    timeout_s: int


def command_spec(verification: Verification) -> CommandSpec | Failure:
    """The `command` verification's settings, or the reason they are unusable."""
    command = verification.spec.get("command")
    timeout = verification.spec.get("timeout_s", DEFAULT_COMMAND_TIMEOUT_S)
    if not isinstance(command, str) or not command.strip():
        return failure("verification", "verification_spec_invalid", "a command verification needs spec.command")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_COMMAND_TIMEOUT_S:
        return failure(
            "verification", "verification_spec_invalid",
            f"spec.timeout_s must be an integer from 1 to {MAX_COMMAND_TIMEOUT_S}", command=command,
        )
    return CommandSpec(command, timeout)


@dataclass(frozen=True)
class CommandOutcome:
    """How one command check ended in the sandbox."""

    exit_code: int | None
    output: str
    timed_out: bool

    def failures(self, command: str) -> list[Failure]:
        tail = self.output[-OUTPUT_TAIL_CHARS:]
        if self.timed_out:
            return [failure("command", "command_timeout", f"{command} did not finish in time", command=command, output_tail=tail)]
        if self.exit_code != 0:
            return [
                failure(
                    "command", "command_failed", f"{command} exited with {self.exit_code}",
                    command=command, exit_code=self.exit_code, output_tail=tail,
                )
            ]
        return []


async def verify_completion(
    proposal: dict[str, Any],
    contract: dict[str, Any],
    *,
    registry: SchemaRegistry | None,
    ports: VerifyPorts,
) -> dict[str, Any]:
    """The io part of 04 §5, driven by the node's completion contract: the output schema, the required artifacts and
    what is in the manifest, and the non-command verifications. It returns every failure, not the first, plus the
    workspace snapshot the command checks must run on.

    Not covered: `claimed_side_effects` in the ledger (04 §5 step 4) and the completion Approval (step 5)."""
    completion = CompletionContract.model_validate(contract)
    output = dict(proposal.get("output") or {})
    failures = await verify_schema(registry, completion.output_schema_ref, output)
    checked = {completion.output_schema_ref}
    artifacts = await verify_artifacts(
        ports, str(proposal["tenant_id"]), proposal.get("artifact_manifest_id"), completion.required_artifacts
    )
    failures.extend(artifacts.failures)
    for verification in completion.verifications:
        failures.extend(await _verification_failures(verification, completion, output, registry, checked))
    return {
        "ok": not failures,
        "failures": failures,
        "checkpoint_ref": proposal.get("checkpoint_ref"),
        "workspace_snapshot_ref": artifacts.workspace_snapshot_ref,
    }


async def _verification_failures(
    verification: Verification,
    completion: CompletionContract,
    output: dict[str, Any],
    registry: SchemaRegistry | None,
    checked: set[str | None],
) -> list[Failure]:
    if verification.kind == "schema":
        ref = verification.spec.get("schema_ref") or completion.output_schema_ref
        if not isinstance(ref, str):
            return [failure("verification", "verification_spec_invalid", "a schema verification needs spec.schema_ref or an output_schema_ref")]
        if ref in checked:
            return []
        checked.add(ref)
        return await verify_schema(registry, ref, output)
    if verification.kind == "command":
        spec = command_spec(verification)
        return [spec] if isinstance(spec, dict) else []
    if verification.kind == "sop_verifier":
        return []  # the SOP step's verifier agent has already judged the step (06 §2)
    return [
        failure(
            "verification", "verification_unsupported",
            f"{verification.kind} verification is not implemented, so the completion cannot be accepted",
            kind=verification.kind,
        )
    ]

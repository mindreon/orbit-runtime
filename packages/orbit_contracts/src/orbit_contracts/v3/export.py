"""Write schema/v3/contracts.json and schema/v3/examples.json.

``contracts.json`` is one JSON Schema bundle: every definition sits in
``$defs``; ``x-models`` lists the top-level models, ``x-unions`` the
discriminated unions (each with ``x-go-type``) and ``x-enums`` the named
string sets. orbit-control generates its Go types from this file and
decodes ``examples.json`` in its tests.
"""

import json
from pathlib import Path
from typing import Any, get_args

from pydantic import TypeAdapter
from pydantic.json_schema import models_json_schema

from orbit_contracts.v3 import common, messages, nodes, plan
from orbit_contracts.v3.events import Event
from orbit_contracts.v3.examples import contract_examples
from orbit_contracts.v3.messages import (
    ApprovalDecidedSignal,
    AttemptFinishedSignal,
    AttemptParkedSignal,
    CompletionProposal,
    CompletionResult,
    DecideApprovalInput,
    DecideApprovalResult,
    DeliverMessagesSignal,
    ExternalEventSignal,
    GrantBudgetInput,
    GrantBudgetResult,
    RequestProfileSwitchInput,
    RequestProfileSwitchResult,
    SendMessageInput,
    SendMessageResult,
    TaskControlInput,
    TaskControlResult,
    UpdateTaskConfigInput,
    UpdateTaskConfigResult,
)
from orbit_contracts.v3.nodes import TaskNodeDraft
from orbit_contracts.v3.plan import PlanChangeCommand, PlanChangeResult, PlanOp
from orbit_contracts.v3.views import InboxView, PlanView, TaskView, TaskWorkflowInput

REF_TEMPLATE = "#/$defs/{model}"

MODELS = (
    TaskWorkflowInput,
    PlanChangeCommand,
    CompletionProposal,
    SendMessageInput,
    SendMessageResult,
    DecideApprovalInput,
    DecideApprovalResult,
    TaskControlInput,
    TaskControlResult,
    GrantBudgetInput,
    GrantBudgetResult,
    RequestProfileSwitchInput,
    RequestProfileSwitchResult,
    UpdateTaskConfigInput,
    UpdateTaskConfigResult,
    ExternalEventSignal,
    AttemptFinishedSignal,
    AttemptParkedSignal,
    ApprovalDecidedSignal,
    DeliverMessagesSignal,
    TaskView,
    PlanView,
    InboxView,
)
UNIONS: dict[str, Any] = {
    "TaskNodeDraft": TaskNodeDraft,
    "PlanOp": PlanOp,
    "PlanChangeResult": PlanChangeResult,
    "CompletionResult": CompletionResult,
    "Event": Event,
}
ENUMS: dict[str, Any] = {
    "TaskStatus": common.TaskStatus,
    "NodeStatus": common.NodeStatus,
    "AttemptStatus": common.AttemptStatus,
    "FailureClass": common.FailureClass,
    "NodeType": nodes.NodeType,
    "WorkspaceAccess": nodes.WorkspaceAccess,
    "PlanRejectCode": plan.PlanRejectCode,
    "UpdateRejectCode": messages.UpdateRejectCode,
    "ApprovalSubjectKind": messages.ApprovalSubjectKind,
    "Delivery": messages.Delivery,
}


def _merge_defs(target: dict[str, Any], defs: dict[str, Any]) -> None:
    for name, schema in defs.items():
        if name in target and target[name] != schema:
            raise ValueError(f"two different definitions named {name}")
        target[name] = schema


def build_bundle() -> dict[str, Any]:
    defs: dict[str, Any] = {}
    _, top = models_json_schema([(m, "validation") for m in MODELS], ref_template=REF_TEMPLATE)
    _merge_defs(defs, top["$defs"])
    unions: dict[str, Any] = {}
    for name, union in UNIONS.items():
        schema = TypeAdapter(union).json_schema(ref_template=REF_TEMPLATE)
        _merge_defs(defs, schema.pop("$defs", {}))
        unions[name] = schema
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mindreon.com/orbit/contracts/v3/contracts.json",
        "title": "Orbit contracts v3 (TaskWorkflow)",
        "$defs": dict(sorted(defs.items())),
        "x-models": sorted(m.__name__ for m in MODELS),
        "x-unions": unions,
        "x-enums": {name: list(get_args(value)) for name, value in ENUMS.items()},
    }


def build_examples() -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for name, (adapter, items) in contract_examples().items():
        out[name] = [
            adapter.dump_python(item, mode="json", exclude_none=True, by_alias=True)
            for item in items
        ]
    return out


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", "utf-8")


def export(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = [directory / "contracts.json", directory / "examples.json"]
    _write(written[0], build_bundle())
    _write(written[1], build_examples())
    return written

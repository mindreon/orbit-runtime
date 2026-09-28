"""Plan change commands and their verdicts (B-prime, 05 §2)."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from orbit_contracts.v3.common import (
    Actor,
    Budget,
    CommandId,
    ContractModel,
    NodeId,
    NodeRef,
    Sha256Ref,
    TaskId,
    go_union,
)
from orbit_contracts.v3.nodes import TaskNodeDraft

PlanRejectCode = Literal[
    "VERSION_CONFLICT",
    "FROZEN_NODE",
    "TYPE_NOT_ALLOWED",
    "POLICY_DENIED",
    "BUDGET_EXCEEDED",
    "CYCLE",
    "DEPTH_EXCEEDED",
    "TOO_MANY_OPS",
    "VISIBILITY",
    "SCHEMA_INVALID",
    "STALE_ATTEMPT",
]


class NodePatch(ContractModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    # Merged into the node spec, then validated against the node type.
    spec: dict[str, JsonValue] | None = None
    budget: Budget | None = None


class AddNodeOp(ContractModel):
    op: Literal["add_node"] = "add_node"
    node: TaskNodeDraft


class UpdateNodeOp(ContractModel):
    op: Literal["update_node"] = "update_node"
    node_id: NodeRef
    patch: NodePatch


class RemoveNodeOp(ContractModel):
    op: Literal["remove_node"] = "remove_node"
    node_id: NodeRef


class AddEdgeOp(ContractModel):
    op: Literal["add_edge"] = "add_edge"
    from_node: NodeRef = Field(alias="from")
    to: NodeRef


class RemoveEdgeOp(ContractModel):
    op: Literal["remove_edge"] = "remove_edge"
    from_node: NodeRef = Field(alias="from")
    to: NodeRef


class DeclareBlockedOp(ContractModel):
    """An agent says its own node cannot proceed (05 §2, agent rights)."""

    op: Literal["declare_blocked"] = "declare_blocked"
    node_id: NodeRef
    reason: str = Field(min_length=1)


PlanOp = Annotated[
    AddNodeOp | UpdateNodeOp | RemoveNodeOp | AddEdgeOp | RemoveEdgeOp | DeclareBlockedOp,
    go_union("PlanOp", "op"),
]


class PlanChangeCommand(ContractModel):
    """Update ``submitPlanChange``. The whole command applies or none of it does."""

    schema_version: Literal["orbit.plan_change/1"] = "orbit.plan_change/1"
    command_id: CommandId
    task_id: TaskId
    base_plan_version: int = Field(ge=1)
    actor: Actor
    ops: list[PlanOp] = Field(min_length=1)
    reason: str = ""


class PlanChangeAccepted(ContractModel):
    status: Literal["accepted"] = "accepted"
    plan_version: int = Field(ge=1)
    id_map: dict[str, NodeId] = Field(default_factory=dict)


class PlanChangeRejected(ContractModel):
    """No full graph here: Update results land in history (04 §4.1)."""

    status: Literal["rejected"] = "rejected"
    code: PlanRejectCode
    detail: str
    latest_plan_version: int = Field(ge=1)
    latest_hash: Sha256Ref


PlanChangeResult = Annotated[
    PlanChangeAccepted | PlanChangeRejected,
    go_union("PlanChangeResult", "status"),
]

"""Workflow start input and Query results (03 §2, 04 §4.3)."""

from typing import Literal

from pydantic import Field, JsonValue

from orbit_contracts.v3.common import (
    Actor,
    ApprovalId,
    AttemptId,
    Budget,
    ContractModel,
    NodeId,
    NodeStatus,
    Policy,
    Sha256Ref,
    TaskConfig,
    TaskId,
    TaskStatus,
    Usage,
    VersionedRef,
)
from orbit_contracts.v3.messages import InboxMessage
from orbit_contracts.v3.nodes import NodeType, WorkspaceAccess

TaskMode = Literal["single", "multi", "long"]


class TaskWorkflowInput(ContractModel):
    """Start input of ``TaskWorkflow``; id ``task/{tenant_id}/{task_id}``."""

    task_id: TaskId
    tenant_id: str = Field(min_length=1)
    created_by: Actor
    title: str = Field(min_length=1, max_length=200)
    goal: str = Field(min_length=1)
    mode: TaskMode = "single"
    profile: VersionedRef
    sop: VersionedRef | None = None
    node_type_registry_version: int = Field(ge=1)
    budgets: Budget
    # The task layer of the policy (05 §6); the tenant and profile layers are read where they apply.
    policy: Policy = Field(default_factory=Policy)
    # What the task runs with (15 M8). Changes after creation arrive through `updateTaskConfig`.
    config: TaskConfig = Field(default_factory=TaskConfig)
    # Internal continuation payload. It is written only by Continue-As-New and
    # never exposed by the control API.
    carry: dict[str, JsonValue] | None = None


class PlanEdge(ContractModel):
    from_node: NodeId = Field(alias="from")
    to: NodeId


class NodeView(ContractModel):
    node_id: NodeId
    type: NodeType
    title: str
    status: NodeStatus
    depends_on: list[NodeId] = Field(default_factory=list)
    workspace_access: WorkspaceAccess
    owner_profile: VersionedRef
    frozen: bool = False
    current_attempt_id: AttemptId | None = None
    attempt_count: int = Field(default=0, ge=0)


class PlanView(ContractModel):
    """Query ``getPlan``: the graph as the asking actor may see it."""

    plan_version: int = Field(ge=1)
    hash: Sha256Ref
    nodes: list[NodeView] = Field(default_factory=list)
    edges: list[PlanEdge] = Field(default_factory=list)


class TaskView(ContractModel):
    """Query ``getTaskView``."""

    task_id: TaskId
    status: TaskStatus
    plan_version: int = Field(ge=1)
    pending_approvals: list[ApprovalId] = Field(default_factory=list)
    budgets: Budget
    usage: Usage = Usage()
    config: TaskConfig = Field(default_factory=TaskConfig)


class InboxView(ContractModel):
    """Query ``getInbox(after_seq)``."""

    messages: list[InboxMessage] = Field(default_factory=list)

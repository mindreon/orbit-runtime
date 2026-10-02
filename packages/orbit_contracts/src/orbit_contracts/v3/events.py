"""The ``orbit.event/3`` envelope and payloads (09 §1, §2).

Retention is fixed by the event type. Durable events get a per-task ``seq``
from the Projector; ephemeral ones carry ``after_seq`` instead and live only
in the control ring buffer.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from orbit_contracts.v3.common import (
    Actor,
    ApprovalId,
    AttemptId,
    Budget,
    CheckpointId,
    CommandId,
    ConfigMode,
    ContractModel,
    EventId,
    Failure,
    ManifestId,
    NodeId,
    NodeStatus,
    Sha256Ref,
    TaskId,
    TaskStatus,
    Usage,
    VersionedRef,
    go_union,
)
from orbit_contracts.v3.messages import (
    ApprovalSubject,
    Attachment,
    AttemptOutcome,
    Delivery,
    ParkReason,
)
from orbit_contracts.v3.plan import PlanRejectCode

EntityKind = Literal[
    "task", "plan", "node", "attempt", "approval", "message", "checkpoint", "artifact"
]
SourceKind = Literal["workflow", "worker", "control"]
ToolCallState = Literal["success", "error", "denied", "interrupted"]


class EventSource(ContractModel):
    kind: SourceKind
    id: str = Field(min_length=1)
    attempt_id: AttemptId | None = None


class EntityRef(ContractModel):
    """Consumers keep, per entity, only the highest version they have seen (09 §1)."""

    kind: EntityKind
    id: str = Field(min_length=1)
    version: int = Field(ge=0)


class _EventBase(ContractModel):
    schema_: Literal["orbit.event/3"] = Field(default="orbit.event/3", alias="schema")
    event_id: EventId
    task_id: TaskId
    seq: int | None = Field(default=None, ge=1)
    after_seq: int | None = Field(default=None, ge=0)
    source: EventSource
    entity: EntityRef
    visibility: Literal["tenant", "internal"] = "tenant"
    occurred_at: datetime


class _Durable(_EventBase):
    retention: Literal["durable"] = "durable"


class _Ephemeral(_EventBase):
    retention: Literal["ephemeral"] = "ephemeral"


# ---- payloads ---------------------------------------------------------------


class TaskCreatedPayload(ContractModel):
    title: str
    goal: str
    profile: VersionedRef
    created_by: Actor


class TaskStatusChangedPayload(ContractModel):
    from_status: TaskStatus
    to_status: TaskStatus
    reason: str = ""


class TaskClosedPayload(ContractModel):
    reason: str = ""
    failure: Failure | None = None


class PlanVersionCommittedPayload(ContractModel):
    plan_version: int = Field(ge=1)
    parent_version: int = Field(ge=0)
    hash: Sha256Ref
    change_command_id: CommandId | None = None
    actor: Actor


class PlanChangeRejectedPayload(ContractModel):
    command_id: CommandId
    code: PlanRejectCode
    detail: str
    actor: Actor


class NodeStatusChangedPayload(ContractModel):
    node_id: NodeId
    from_status: NodeStatus
    to_status: NodeStatus
    reason: str = ""


class AttemptStartedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    attempt_no: int = Field(ge=1)
    profile: VersionedRef
    # The task configuration (15 M8) this attempt runs with, fixed at its start. Older histories carry none.
    config_version: int | None = Field(default=None, ge=1)


class AttemptResumedPayload(ContractModel):
    """A Temporal retry inside the same attempt, e.g. after a worker crash."""

    node_id: NodeId
    attempt_id: AttemptId
    attempt_no: int = Field(ge=1)
    activity_attempt: int = Field(ge=2)


class AttemptParkedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    reason: ParkReason
    question: str | None = None


class AttemptFinishedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    outcome: AttemptOutcome
    failure: Failure | None = None
    usage: Usage = Usage()


class ApprovalRequestedPayload(ContractModel):
    approval_id: ApprovalId
    node_id: NodeId | None = None
    attempt_id: AttemptId | None = None
    tool_call_id: str | None = None
    subject: ApprovalSubject


class ApprovalDecidedPayload(ContractModel):
    approval_id: ApprovalId
    status: Literal["APPROVED", "REJECTED", "CANCELLED", "TAKEN_OVER"]
    decided_by: str = ""
    comment: str = ""
    # The approval also allowed its rule for the rest of the task.
    always: bool = False


class CheckpointCommittedPayload(ContractModel):
    checkpoint_id: CheckpointId
    attempt_id: AttemptId
    kind: Literal["agent_state", "sop_run_state", "workspace_snapshot", "plan", "team_snapshot"]
    blob_ref: Sha256Ref


class ManifestCreatedPayload(ContractModel):
    manifest_id: ManifestId
    attempt_id: AttemptId
    entry_count: int = Field(ge=0)
    manifest_hash: Sha256Ref


class UserMessagePayload(ContractModel):
    message_seq: int = Field(ge=1)
    client_message_id: CommandId
    text: str
    attachments: list[Attachment] = Field(default_factory=list)
    delivery: Delivery = "queue"


class AgentFinalMessagePayload(ContractModel):
    attempt_id: AttemptId
    text: str


class BudgetExhaustedPayload(ContractModel):
    scope: Literal["task", "node", "exploration"]
    node_id: NodeId | None = None


class BudgetGrantedPayload(ContractModel):
    command_id: CommandId
    delta: Budget


class TaskConfigChangedPayload(ContractModel):
    """What the task runs with from its next attempt on. Connector ids only: the launch targets stay out of events."""

    config_version: int = Field(ge=2)
    expert: VersionedRef | None = None
    skills: list[str] | None = None
    connector_ids: list[str] | None = None
    mode: ConfigMode


class ProfileSwitchedPayload(ContractModel):
    node_id: NodeId
    from_profile: VersionedRef
    to_profile: VersionedRef
    reason: str
    approval_id: ApprovalId | None = None


class ToolCallStartedPayload(ContractModel):
    attempt_id: AttemptId
    tool_call_id: str
    tool_name: str
    args_preview: str = ""


class ToolCallFinishedPayload(ToolCallStartedPayload):
    state: ToolCallState
    result_preview: str = ""


class UsagePayload(ContractModel):
    attempt_id: AttemptId
    usage: Usage


class TextDeltaPayload(ContractModel):
    attempt_id: AttemptId
    text: str
    # One model round streams one block: what a client groups the deltas by.
    block_id: str | None = None


class ToolProgressPayload(ContractModel):
    attempt_id: AttemptId
    tool_call_id: str
    text: str


class ExecOutputPayload(ContractModel):
    attempt_id: AttemptId
    stream: Literal["stdout", "stderr"]
    text: str


class HeartbeatPayload(ContractModel):
    pass


# ---- durable events ---------------------------------------------------------


class TaskCreatedEvent(_Durable):
    type: Literal["task.created"] = "task.created"
    payload: TaskCreatedPayload


class TaskStatusChangedEvent(_Durable):
    type: Literal["task.status_changed"] = "task.status_changed"
    payload: TaskStatusChangedPayload


class TaskCompletedEvent(_Durable):
    type: Literal["task.completed"] = "task.completed"
    payload: TaskClosedPayload


class TaskFailedEvent(_Durable):
    type: Literal["task.failed"] = "task.failed"
    payload: TaskClosedPayload


class TaskCancelledEvent(_Durable):
    type: Literal["task.cancelled"] = "task.cancelled"
    payload: TaskClosedPayload


class PlanVersionCommittedEvent(_Durable):
    type: Literal["plan.version_committed"] = "plan.version_committed"
    payload: PlanVersionCommittedPayload


class PlanChangeRejectedEvent(_Durable):
    type: Literal["plan.change_rejected"] = "plan.change_rejected"
    payload: PlanChangeRejectedPayload


class NodeStatusChangedEvent(_Durable):
    type: Literal["node.status_changed"] = "node.status_changed"
    payload: NodeStatusChangedPayload


class AttemptStartedEvent(_Durable):
    type: Literal["attempt.started"] = "attempt.started"
    payload: AttemptStartedPayload


class AttemptResumedEvent(_Durable):
    type: Literal["attempt.resumed"] = "attempt.resumed"
    payload: AttemptResumedPayload


class AttemptParkedEvent(_Durable):
    type: Literal["attempt.parked"] = "attempt.parked"
    payload: AttemptParkedPayload


class AttemptFinishedEvent(_Durable):
    type: Literal["attempt.finished"] = "attempt.finished"
    payload: AttemptFinishedPayload


class ApprovalRequestedEvent(_Durable):
    type: Literal["approval.requested"] = "approval.requested"
    payload: ApprovalRequestedPayload


class ApprovalDecidedEvent(_Durable):
    type: Literal["approval.decided"] = "approval.decided"
    payload: ApprovalDecidedPayload


class CheckpointCommittedEvent(_Durable):
    type: Literal["checkpoint.committed"] = "checkpoint.committed"
    payload: CheckpointCommittedPayload


class ManifestCreatedEvent(_Durable):
    type: Literal["artifact.manifest_created"] = "artifact.manifest_created"
    payload: ManifestCreatedPayload


class UserMessageEvent(_Durable):
    type: Literal["message.user"] = "message.user"
    payload: UserMessagePayload


class AgentFinalMessageEvent(_Durable):
    type: Literal["message.agent_final"] = "message.agent_final"
    payload: AgentFinalMessagePayload


class BudgetExhaustedEvent(_Durable):
    type: Literal["budget.exhausted"] = "budget.exhausted"
    payload: BudgetExhaustedPayload


class BudgetGrantedEvent(_Durable):
    type: Literal["budget.granted"] = "budget.granted"
    payload: BudgetGrantedPayload


class TaskConfigChangedEvent(_Durable):
    type: Literal["task.config_changed"] = "task.config_changed"
    payload: TaskConfigChangedPayload


class ProfileSwitchedEvent(_Durable):
    type: Literal["profile.switched"] = "profile.switched"
    payload: ProfileSwitchedPayload


class ToolCallFinishedEvent(_Durable):
    type: Literal["tool.call_finished"] = "tool.call_finished"
    payload: ToolCallFinishedPayload


class UsageRecordedEvent(_Durable):
    type: Literal["usage.recorded"] = "usage.recorded"
    payload: UsagePayload


# ---- ephemeral events -------------------------------------------------------


class TokenDeltaEvent(_Ephemeral):
    type: Literal["agent.token_delta"] = "agent.token_delta"
    payload: TextDeltaPayload


class ThinkingDeltaEvent(_Ephemeral):
    type: Literal["agent.thinking_delta"] = "agent.thinking_delta"
    payload: TextDeltaPayload


class ToolCallStartedEvent(_Ephemeral):
    type: Literal["tool.call_started"] = "tool.call_started"
    payload: ToolCallStartedPayload


class ToolProgressEvent(_Ephemeral):
    type: Literal["tool.progress"] = "tool.progress"
    payload: ToolProgressPayload


class UsageDeltaEvent(_Ephemeral):
    type: Literal["usage.delta"] = "usage.delta"
    payload: UsagePayload


class ExecOutputEvent(_Ephemeral):
    type: Literal["workspace.exec_output"] = "workspace.exec_output"
    payload: ExecOutputPayload


class HeartbeatEvent(_Ephemeral):
    type: Literal["heartbeat"] = "heartbeat"
    payload: HeartbeatPayload = HeartbeatPayload()


DURABLE_EVENTS = (
    TaskCreatedEvent,
    TaskStatusChangedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskCancelledEvent,
    PlanVersionCommittedEvent,
    PlanChangeRejectedEvent,
    NodeStatusChangedEvent,
    AttemptStartedEvent,
    AttemptResumedEvent,
    AttemptParkedEvent,
    AttemptFinishedEvent,
    ApprovalRequestedEvent,
    ApprovalDecidedEvent,
    CheckpointCommittedEvent,
    ManifestCreatedEvent,
    UserMessageEvent,
    AgentFinalMessageEvent,
    BudgetExhaustedEvent,
    BudgetGrantedEvent,
    ProfileSwitchedEvent,
    TaskConfigChangedEvent,
    ToolCallFinishedEvent,
    UsageRecordedEvent,
)
EPHEMERAL_EVENTS = (
    TokenDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallStartedEvent,
    ToolProgressEvent,
    UsageDeltaEvent,
    ExecOutputEvent,
    HeartbeatEvent,
)
DURABLE_EVENT_TYPES = frozenset(cls.model_fields["type"].default for cls in DURABLE_EVENTS)
EPHEMERAL_EVENT_TYPES = frozenset(cls.model_fields["type"].default for cls in EPHEMERAL_EVENTS)

Event = Annotated[
    TaskCreatedEvent
    | TaskStatusChangedEvent
    | TaskCompletedEvent
    | TaskFailedEvent
    | TaskCancelledEvent
    | PlanVersionCommittedEvent
    | PlanChangeRejectedEvent
    | NodeStatusChangedEvent
    | AttemptStartedEvent
    | AttemptResumedEvent
    | AttemptParkedEvent
    | AttemptFinishedEvent
    | ApprovalRequestedEvent
    | ApprovalDecidedEvent
    | CheckpointCommittedEvent
    | ManifestCreatedEvent
    | UserMessageEvent
    | AgentFinalMessageEvent
    | BudgetExhaustedEvent
    | BudgetGrantedEvent
    | ProfileSwitchedEvent
    | TaskConfigChangedEvent
    | ToolCallFinishedEvent
    | UsageRecordedEvent
    | TokenDeltaEvent
    | ThinkingDeltaEvent
    | ToolCallStartedEvent
    | ToolProgressEvent
    | UsageDeltaEvent
    | ExecOutputEvent
    | HeartbeatEvent,
    go_union("Event", "type"),
]

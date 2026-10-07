"""The ``orbit.event/3`` envelope and payloads (09 §1, §2).

Retention is fixed by the event type. Durable events get a per-task ``seq``
from the Projector; ephemeral ones carry ``after_seq`` instead and live only
in the control ring buffer.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue

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
    PermissionSpec,
    Sha256Ref,
    TaskId,
    TaskStatus,
    Usage,
    VersionedRef,
    go_union,
)
from orbit_contracts.v3.messages import (
    ApprovalSubject,
    AskUserQuestion,
    Attachment,
    AttemptOutcome,
    Delivery,
    ParkReason,
)
from orbit_contracts.v3.nodes import NodeType, SopStepInfo, TeamStageInfo, WorkspaceAccess
from orbit_contracts.v3.plan import PlanRejectCode

EntityKind = Literal[
    "task", "plan", "node", "attempt", "approval", "message", "checkpoint", "artifact", "team"
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
    # Why the system committed it ("compaction", "sop expansion"); a person's or agent's change has none.
    reason: str = ""
    # The `sop_stage` node a version that expanded an SOP compiled into a subgraph.
    sop_node_id: NodeId | None = None
    # The workflow still sends the command id as `command_id` too, for consumers from before `change_command_id`.


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
    # What the node is after the change, so a projection can create or update its row from this event alone (the first
    # change of a node is its first event; nodes are not announced when they are added). Present from
    # `task-event-payloads-v2`; older histories carry none.
    node_type: NodeType | None = None
    title: str | None = None
    workspace_access: WorkspaceAccess | None = None
    owner_profile: VersionedRef | None = None
    depends_on: list[NodeId] | None = None
    frozen: bool | None = None
    attempt_count: int | None = Field(default=None, ge=0)
    current_attempt_id: AttemptId | None = None
    # The node this one is nested under (an SOP's steps under the `sop_stage` node, a team member's work under the
    # leader's), and for the nodes of a compiled SOP which step they are.
    parent_node_id: NodeId | None = None
    sop_step: SopStepInfo | None = None
    # In a task with a team: the role (and label) the node belongs to.
    owner_role: str | None = None
    owner_label: str | None = None
    # Which round of leader reviews the node is (a review node), and the limits of its stage (a `team_stage` node).
    review_round: int | None = Field(default=None, ge=1)
    team: TeamStageInfo | None = None


class PlanReviewLimitReachedPayload(ContractModel):
    """The leader's reviews reached `Policy.max_review_rounds` (05 §7): the tasks created in the last round are done and
    nobody reviewed them. The task asks for a review of its own (`task.status_changed`)."""

    node_id: NodeId
    round: int = Field(ge=1)
    max_rounds: int = Field(ge=1)
    children: int = Field(ge=0)


class AttemptStartedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    attempt_no: int = Field(ge=1)
    profile: VersionedRef
    # The task configuration (15 M8) this attempt runs with, fixed at its start. Older histories carry none.
    config_version: int | None = Field(default=None, ge=1)
    # The profile the node ran as before this attempt, when a profile switch took effect for it (11 §3).
    switched_from: VersionedRef | None = None
    # What the parent reserved for the attempt from the task budget (05 §4).
    budget_reserved: Budget | None = None
    # AgentScope's version, the runtime image digest and the contract schema version are known to the worker only, and
    # are not part of this event: the workflow cannot state them without doing I/O.


class AttemptResumedPayload(ContractModel):
    """A Temporal retry inside the same attempt, e.g. after a worker crash, or an activity that carries on after an approval or
    an answer. The worker emits it (entity.version 0), so the entity version cannot order it against the workflow's events of
    the attempt. The rule for a consumer: events of one attempt are ordered by (`activity_attempt`, `state_version`), and
    against the workflow's events by `occurred_at`; a resumed event never changes the attempt's status by itself, it only
    says the attempt is running again."""

    node_id: NodeId
    attempt_id: AttemptId
    attempt_no: int = Field(ge=1)
    activity_attempt: int = Field(ge=1)
    # The attempt's session state version when this activity started (it grows with every turn of the attempt).
    state_version: int | None = Field(default=None, ge=0)


class AttemptParkedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    reason: ParkReason
    question: str | None = None
    # The structured form of `question` (at most 4), when the agent asked with options.
    questions: list[AskUserQuestion] | None = Field(default=None, max_length=4)


class AttemptFinishedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    # Which attempt of the node and what it ran as: what a projection needs to create the attempt row from this event.
    attempt_no: int | None = Field(default=None, ge=1)
    profile: VersionedRef | None = None
    config_version: int | None = Field(default=None, ge=1)
    outcome: AttemptOutcome
    failure: Failure | None = None
    usage: Usage = Usage()
    # The structured output of an attempt whose node names an output schema. Above 16 KB it is left out and
    # `output_truncated` is true (the completion check still saw all of it).
    output: dict[str, JsonValue] | None = None
    output_truncated: bool | None = None


class TeamRoundStartedPayload(ContractModel):
    """The leader of a team stage starts its `round`-th turn (07 §5). Team events carry the stage's attempt as their entity:
    {kind: team, id: attempt_id}, with a version of their own."""

    node_id: NodeId
    attempt_id: AttemptId
    round: int = Field(ge=1)
    max_rounds: int = Field(ge=1)
    # The stage's other limits and the messages used so far, so a view needs no defaults.
    max_messages: int | None = Field(default=None, ge=1)
    max_members: int | None = Field(default=None, ge=1)
    max_hops: int | None = Field(default=None, ge=0)
    messages: int | None = Field(default=None, ge=0)


class TeamRoundFinishedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    round: int = Field(ge=1)
    # `assigned`: the leader handed work to members and goes on once they answer; `completed`: the leader's final answer;
    # `stopped`: a limit or a budget ended the stage (`reason` says which).
    outcome: Literal["assigned", "completed", "stopped"]
    # Messages used so far in the stage (assignments, results and notes), against `max_messages`.
    messages: int | None = Field(default=None, ge=0)
    assignments: int = Field(default=0, ge=0)
    reason: str = ""
    usage: Usage = Usage()


class TeamMemberTurnStartedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    round: int = Field(ge=1)
    role: str
    label: str = ""
    executor: VersionedRef
    # The member's own session is derived from the stage's attempt and its role; this is its id in the worker's records.
    member_attempt_id: AttemptId
    # What the leader asked, cut to 300 characters.
    task: str = ""


class TeamMemberTurnFinishedPayload(ContractModel):
    node_id: NodeId
    attempt_id: AttemptId
    round: int = Field(ge=1)
    role: str
    label: str = ""
    executor: VersionedRef
    member_attempt_id: AttemptId
    outcome: Literal["completed", "failed"]
    # What the member answered, cut to 500 characters.
    summary: str = ""
    usage: Usage = Usage()
    # Names of the files the member left in its copy of the workspace.
    artifacts: list[str] = Field(default_factory=list)


class TeamArtifactRef(ContractModel):
    name: str
    blob_ref: Sha256Ref | None = None


TeamMessageKind = Literal["assign", "reply", "note", "review", "user", "system"]


class TeamMessagePayload(ContractModel):
    """One utterance in the team's group conversation, whatever made it (07 §5): the leader assigning work (`assign`), a member
    answering (`reply`), a note posted for the team (`note`), the leader's review or final answer (`review`), the user (`user`) or
    the runtime explaining a refusal (`system`). `to_roles` is who it is addressed to (mentions); empty is the whole group. A stage
    emits them with entity {team, attempt_id}; plan-level ones (a TaskCreate given to a member, its result, a user message to a
    member) have no stage: `attempt_id` is the attempt that made it (None for the user) and `round` is 0.
    `role` and `label` repeat `from_role` and `from_label` (kept for consumers of the first shape of this event)."""

    node_id: NodeId
    attempt_id: AttemptId | None = None
    seq: int | None = Field(default=None, ge=1)
    role: str
    label: str = ""
    from_role: str = ""
    from_label: str = ""
    to_roles: list[str] = Field(default_factory=list)
    text: str
    kind: TeamMessageKind = "note"
    round: int = Field(default=0, ge=0)
    # How many member-to-member wakes deep the utterance is inside its leader round (0: the leader's or the user's own).
    hop: int = Field(default=0, ge=0)
    artifacts: list[TeamArtifactRef] = Field(default_factory=list)


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


class _TeamStamp(ContractModel):
    """Who in a team stage an event of an attempt is about: the stage's agents share one attempt id, so an event made while a member
    (or the leader) worked says which one, and `team_session` (the id of that agent's own session) groups its steps under its turn.
    None outside a team stage."""

    team_role: str | None = None
    team_label: str | None = None
    team_session: str | None = None


class UserMessagePayload(ContractModel):
    message_seq: int = Field(ge=1)
    client_message_id: CommandId
    text: str
    attachments: list[Attachment] = Field(default_factory=list)
    delivery: Delivery = "queue"
    mentions: list[str] = Field(default_factory=list)


class AgentFinalMessagePayload(_TeamStamp):
    attempt_id: AttemptId
    text: str


class BudgetExhaustedPayload(ContractModel):
    scope: Literal["task", "node", "exploration"]
    node_id: NodeId | None = None
    # Which limit and why, for a person to read.
    detail: str = ""


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
    model: str | None = None
    permissions: PermissionSpec | None = None


class ProfileSwitchedPayload(ContractModel):
    node_id: NodeId
    from_profile: VersionedRef
    to_profile: VersionedRef
    reason: str
    approval_id: ApprovalId | None = None


class ToolCallStartedPayload(_TeamStamp):
    attempt_id: AttemptId
    tool_call_id: str
    tool_name: str
    args_preview: str = ""


class ToolCallFinishedPayload(ToolCallStartedPayload):
    state: ToolCallState
    result_preview: str = ""


class UsagePayload(_TeamStamp):
    attempt_id: AttemptId
    usage: Usage


class TextDeltaPayload(_TeamStamp):
    attempt_id: AttemptId
    text: str
    # One model round streams one block: what a client groups the deltas by.
    block_id: str | None = None


class ToolProgressPayload(_TeamStamp):
    attempt_id: AttemptId
    tool_call_id: str
    text: str


class ExecOutputPayload(_TeamStamp):
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


class PlanReviewLimitReachedEvent(_Durable):
    type: Literal["plan.review_limit_reached"] = "plan.review_limit_reached"
    payload: PlanReviewLimitReachedPayload


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


class TeamRoundStartedEvent(_Durable):
    type: Literal["team.round_started"] = "team.round_started"
    payload: TeamRoundStartedPayload


class TeamRoundFinishedEvent(_Durable):
    type: Literal["team.round_finished"] = "team.round_finished"
    payload: TeamRoundFinishedPayload


class TeamMemberTurnStartedEvent(_Durable):
    type: Literal["team.member_turn_started"] = "team.member_turn_started"
    payload: TeamMemberTurnStartedPayload


class TeamMemberTurnFinishedEvent(_Durable):
    type: Literal["team.member_turn_finished"] = "team.member_turn_finished"
    payload: TeamMemberTurnFinishedPayload


class TeamMessageEvent(_Durable):
    type: Literal["team.message"] = "team.message"
    payload: TeamMessagePayload


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
    PlanReviewLimitReachedEvent,
    NodeStatusChangedEvent,
    AttemptStartedEvent,
    AttemptResumedEvent,
    AttemptParkedEvent,
    AttemptFinishedEvent,
    TeamRoundStartedEvent,
    TeamRoundFinishedEvent,
    TeamMemberTurnStartedEvent,
    TeamMemberTurnFinishedEvent,
    TeamMessageEvent,
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
    | PlanReviewLimitReachedEvent
    | NodeStatusChangedEvent
    | AttemptStartedEvent
    | AttemptResumedEvent
    | AttemptParkedEvent
    | AttemptFinishedEvent
    | TeamRoundStartedEvent
    | TeamRoundFinishedEvent
    | TeamMemberTurnStartedEvent
    | TeamMemberTurnFinishedEvent
    | TeamMessageEvent
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

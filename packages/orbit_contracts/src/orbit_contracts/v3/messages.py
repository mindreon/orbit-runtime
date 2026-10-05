"""TaskWorkflow and AttemptWorkflow messages: Updates, Signals and their results (04 §4)."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from orbit_contracts.v3.common import (
    ApprovalId,
    ArtifactUri,
    AttemptId,
    Budget,
    CommandId,
    ConfigMode,
    ConnectorSnapshot,
    ContractModel,
    Failure,
    ManifestId,
    NodeId,
    NonEmptyText,
    PermissionRuleSpec,
    Policy,
    Risk,
    Sha256Ref,
    TaskConfig,
    TaskStatus,
    Team,
    Usage,
    VersionedRef,
    go_union,
)
from orbit_contracts.v3.nodes import NodeType

Delivery = Literal["queue", "interrupt"]
Decision = Literal["approve", "reject"]
ControlAction = Literal["pause", "resume", "stop", "cancel", "takeover", "handback"]
AttemptOutcome = Literal["completed", "failed", "cancelled"]
ParkReason = Literal["approval", "input"]
ApprovalSubjectKind = Literal[
    "tool_call",
    "plan_change",
    "budget_extension",
    "profile_switch",
    "completion",
    "non_idempotent_retry",
    # A person's say on a step of a compiled SOP: before it starts or after it finished (`human_approval`).
    "sop_step",
    # A person's say on an approval node of the plan (an agent or a person added it, or an SOP did).
    "node_approval",
]
# Reasons a validator refuses an Update (04 §4). The message is the code.
UpdateRejectCode = Literal[
    "TASK_CLOSED",
    "STALE_ATTEMPT",
    "UNKNOWN_APPROVAL",
    "APPROVAL_ALREADY_DECIDED",
    "INVALID_TRANSITION",
    "NOT_ALLOWED",
    "CONFIG_VERSION_CONFLICT",
]


class Attachment(ContractModel):
    uri: ArtifactUri
    name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)


class SendMessageInput(ContractModel):
    """Update ``sendMessage``; ``client_message_id`` doubles as the command id."""

    command_id: CommandId
    client_message_id: CommandId
    text: NonEmptyText
    attachments: list[Attachment] = Field(default_factory=list)
    delivery: Delivery = "queue"


class SendMessageResult(ContractModel):
    message_seq: int = Field(ge=1)


class DecideApprovalInput(ContractModel):
    command_id: CommandId
    approval_id: ApprovalId
    decision: Decision
    comment: str = ""
    # With an approval: also allow what the approval offered (`subject.allow_rule`) for the rest of the task.
    always: bool = False


class DecideApprovalResult(ContractModel):
    approval_id: ApprovalId
    status: Literal["APPROVED", "REJECTED"]


class TaskControlInput(ContractModel):
    """Updates ``pause``, ``resume``, ``stop``, ``cancel``, ``takeover`` and ``handback``."""

    command_id: CommandId
    action: ControlAction
    reason: str = ""


class TaskControlResult(ContractModel):
    status: TaskStatus


class GrantBudgetInput(ContractModel):
    command_id: CommandId
    delta: Budget


class GrantBudgetResult(ContractModel):
    budgets: Budget


class RequestProfileSwitchInput(ContractModel):
    command_id: CommandId
    node_id: NodeId
    to_profile: VersionedRef
    reason: str = Field(min_length=1)


class RequestProfileSwitchResult(ContractModel):
    # The switch applies to the next attempt, never the running one (11 §3).
    effective_attempt_no: int = Field(ge=1)
    needs_approval: bool
    # The `profile_switch` approval that has to be decided first, when `needs_approval` is true.
    approval_id: ApprovalId | None = None


class CompleteNodeInput(ContractModel):
    """Update ``completeNode``: a person completes a node by hand (04 §4). Allowed while the task is paused or taken over,
    for a node that is not frozen and has no attempt running. It is not verified: the person is the verdict."""

    command_id: CommandId
    node_id: NodeId
    reason: str = ""


class CompleteNodeResult(ContractModel):
    node_id: NodeId
    status: Literal["COMPLETED"] = "COMPLETED"


class UpdateTaskConfigInput(ContractModel):
    """Replaces the task's configuration. `base_config_version` is the version the caller read; a newer one in
    place means the caller decided on stale data and gets CONFIG_VERSION_CONFLICT."""

    command_id: CommandId
    base_config_version: int = Field(ge=1)
    expert: VersionedRef | None = None
    skills: list[str] | None = None
    connectors: list[ConnectorSnapshot] | None = None
    mode: ConfigMode = "default"
    team: Team | None = None


class UpdateTaskConfigResult(ContractModel):
    config_version: int = Field(ge=2)
    # The attempt that is running keeps what it started with (11 §3).
    effective: Literal["next_attempt"] = "next_attempt"


class CompletionProposal(ContractModel):
    """Update ``proposeCompletion``. Completion is a claim until verified (04 §5)."""

    schema_version: Literal["orbit.completion/1"] = "orbit.completion/1"
    command_id: CommandId
    node_id: NodeId
    attempt_id: AttemptId
    output: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_manifest_id: ManifestId | None = None
    checkpoint_ref: Sha256Ref
    claimed_side_effects: list[str] = Field(default_factory=list)


class CompletionAccepted(ContractModel):
    status: Literal["accepted_for_verification"] = "accepted_for_verification"


class CompletionRejected(ContractModel):
    status: Literal["rejected"] = "rejected"
    code: UpdateRejectCode
    detail: str


CompletionResult = Annotated[
    CompletionAccepted | CompletionRejected,
    go_union("CompletionResult", "status"),
]


class ApprovalSubject(ContractModel):
    kind: ApprovalSubjectKind
    digest: Sha256Ref
    summary: str
    risk: Risk
    # What the call is made with, for a person to read: the command, the path.
    detail: str = ""
    # What "always allow" would allow for the rest of the task, if there is a rule that says it.
    allow_rule: PermissionRuleSpec | None = None


class ParkedToolCall(ContractModel):
    tool_call_id: str = Field(min_length=1)
    subject: ApprovalSubject


class AttemptResult(ContractModel):
    handover_summary: str = ""
    manifest_id: ManifestId | None = None
    manifest_entries: list[dict[str, JsonValue]] = Field(default_factory=list)
    manifest_hash: Sha256Ref | None = None
    checkpoint_ref: Sha256Ref
    usage: Usage = Usage()
    # The exploration node spent its tool budget (05 §4). With no plan committed, the task needs a review.
    budget_exhausted: bool = False
    # What the attempt produced as structured output, when its node's completion contract names an `output_schema_ref`;
    # the completion check validates it against that schema (04 §5). Empty otherwise.
    output: dict[str, JsonValue] = Field(default_factory=dict)


class ExternalEventSignal(ContractModel):
    wait_key: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class InboxMessage(ContractModel):
    message_seq: int = Field(ge=1)
    client_message_id: CommandId
    text: NonEmptyText
    attachments: list[Attachment] = Field(default_factory=list)
    delivery: Delivery = "queue"


class AttemptFinishedSignal(ContractModel):
    """Child AttemptWorkflow to parent, sent to the workflow id without a run id."""

    attempt_workflow_id: str = Field(pattern=r"^attempt/")
    node_id: NodeId
    attempt_no: int = Field(ge=1)
    attempt_id: AttemptId
    outcome: AttemptOutcome
    result: AttemptResult | None = None
    failure: Failure | None = None
    # Messages that were handed to the attempt and not consumed by an activity before it ended. The parent puts them
    # back in the task inbox, so a cancelled or failed attempt does not take them with it.
    unconsumed_messages: list[InboxMessage] = Field(default_factory=list)
    # What the attempt spent, whatever way it ended (`result.usage` is the same for a completed one). The parent settles
    # its reservation against it.
    usage: Usage | None = None


class AttemptParkedSignal(ContractModel):
    node_id: NodeId
    attempt_no: int = Field(ge=1)
    attempt_id: AttemptId
    reason: ParkReason
    approvals: list[ParkedToolCall] = Field(default_factory=list)
    question: str | None = None


class ApprovalDecidedSignal(ContractModel):
    """Parent to child: the decision the parked agent is waiting for."""

    approval_id: ApprovalId
    tool_call_id: str | None = None
    decision: Decision
    comment: str = ""
    # The rule to allow from now on, when the decision was "always".
    rule: PermissionRuleSpec | None = None


class DeliverMessagesSignal(ContractModel):
    messages: list[InboxMessage] = Field(min_length=1)


class AttemptWorkflowInput(ContractModel):
    """Input for one durable child attempt."""

    task_id: str = Field(min_length=1)
    tenant_id: str = Field(default="default", min_length=1)
    node_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_no: int = Field(ge=1)
    node_type: NodeType
    profile: VersionedRef
    goal: str = Field(min_length=1)
    policy: Policy = Field(default_factory=Policy)
    checkpoint_ref: Sha256Ref | None = None
    # Only a node that declares `write` takes the task workspace's write lease (08 §1, serial writes).
    workspace_access: Literal["none", "read", "write"] = "none"
    messages: list[InboxMessage] = Field(default_factory=list)
    config: TaskConfig = Field(default_factory=TaskConfig)
    # What a person allowed for the rest of the task ("always allow"). Not part of the configuration: it grows by
    # approvals, not by editing, and never makes the configuration stale.
    allow_rules: list[PermissionRuleSpec] = Field(default_factory=list)
    # A follow-up carries on the agent session of this attempt, the one that ran before it (the task stays open).
    continue_from: AttemptId | None = None
    # Why the node's previous attempt was rejected or failed, when this attempt is a retry of it (04 §2). The agent is
    # told, so it does not repeat the mistake.
    retry_reason: str = ""
    # What the parent reserved for this attempt from the task's remaining budget (05 §4). None limits nothing. The worker
    # spends only within it.
    budget: Budget | None = None
    # The profile the node ran as before a switch (11 §3). Such an attempt does not carry the old session: it is given
    # `handover` instead.
    switched_from: VersionedRef | None = None
    handover: str = ""
    # The schema (`schema://name/1`) the node's output must satisfy: the worker has the agent end its reply with an object
    # of that schema and reports it as the attempt's output.
    output_schema_ref: str | None = Field(default=None, pattern=r"^schema://\S+/[1-9][0-9]*$")
    # Internal continuation payload. It is written only by the attempt's own Continue-As-New (04 §6).
    carry: dict[str, JsonValue] | None = None

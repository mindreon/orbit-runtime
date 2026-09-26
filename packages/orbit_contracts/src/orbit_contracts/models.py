"""Activity inputs/outputs and the Room workflow surface.

Field names match the platform contract in orbit-infra ARCHITECTURE.md.
Framework types (AgentScope events, AgentState) never appear here.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

PermissionPreset = Literal["workspace-write", "read-only", "danger-full-access"]
TurnStatus = Literal["continue", "needs_approval", "needs_external", "completed", "failed"]
# "mock" means the turn ran on the in-process fake model, not a provider.
ModelMode = Literal["mock", "real"]
# Why a failed turn failed. Clients map these to text and never parse ``error``.
TurnErrorCode = Literal["timeout", "auth", "rate_limited", "provider_error", "config"]
RoomStatus = Literal[
    "idle",
    "running",
    "awaiting_approval",
    "awaiting_external",
    "closed",
]
ToolResultStateName = Literal["success", "error"]
AgentFinishReason = Literal["completed", "aborted", "dissolved", "failed"]
ToolErrorCode = Literal[
    "APPROVAL_REJECTED",
    "PERMISSION_DENIED",
    "DELEGATION_LIMIT_EXCEEDED",
    "QUESTION_CANCELLED",
    "SESSION_ABORTED",
]
# No default: every tool declares its risk explicitly.
ToolRisk = Literal["read", "write", "sensitive", "destructive"]
ToolState = Literal["success", "error", "denied", "interrupted"]

_CAMEL = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AgentRef(BaseModel):
    """Identity of one agent in a room. The main agent is always ``main``."""

    model_config = _CAMEL

    agent_id: str = "main"
    parent_agent_id: str | None = None
    parent_session_id: str | None = None
    depth: int = 0
    # Built by the room at spawn time, e.g. "main/ag-x/ag-y".
    agent_path: str = "main"
    persona: str = ""


class QuestionChoice(BaseModel):
    model_config = _CAMEL

    id: str
    label: str


class QuestionAsk(BaseModel):
    model_config = _CAMEL

    question_id: str
    text: str
    choices: list[QuestionChoice] = Field(default_factory=list)
    allow_custom: bool = True
    agent: AgentRef = Field(default_factory=AgentRef)


class QuestionAnswer(BaseModel):
    model_config = _CAMEL

    question_id: str
    choice_ids: list[str] = Field(default_factory=list)
    text: str = ""


class TodoItem(BaseModel):
    model_config = _CAMEL

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed", "cancelled"] = "pending"


class ApprovalAsk(BaseModel):
    """What a human must decide before a parked tool call continues."""

    approval_request_id: str
    tool_name: str
    call_id: str | None = None
    reason: str | None = None


class ExternalCall(BaseModel):
    """A tool call the workflow must finish outside the agent process."""

    tool_name: str
    call_id: str
    arguments: dict[str, str] = Field(default_factory=dict)


class OpenSessionInput(BaseModel):
    room_id: str
    turn_id: str
    permission_preset: PermissionPreset = "workspace-write"


class OpenSessionOutput(BaseModel):
    session_id: str
    state_version: int


class RunTurnInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str
    message: str
    state_version: int


class TurnResult(BaseModel):
    """One agent step. ``failed`` leaves the saved state at ``state_version``."""

    status: TurnStatus
    session_id: str
    state_version: int
    approval: ApprovalAsk | None = None
    external: ExternalCall | None = None
    text: str = ""
    error: str = ""
    error_code: TurnErrorCode | None = None
    retryable: bool = False
    model_mode: ModelMode = "mock"
    model_name: str = "mock"


class ResolveApprovalInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str
    approval_request_id: str
    outcome: Literal["allowed-once", "rejected"]


class DeliverToolResultInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str
    state_version: int
    tool_name: str
    call_id: str
    output: str
    metadata: dict[str, str] = Field(default_factory=dict)
    result_state: ToolResultStateName = "success"


class SteerInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str
    state_version: int
    hint: str


class AbortSessionInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str


class CloseSessionInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str


class CloseSessionOutput(BaseModel):
    closed: bool
    state_version: int


class GatewayExecuteInput(BaseModel):
    room_id: str
    session_id: str
    tool_name: str
    call_id: str
    arguments: dict[str, str] = Field(default_factory=dict)
    credential_ref: str = ""


class GatewayExecuteOutput(BaseModel):
    output: str
    metadata: dict[str, str] = Field(default_factory=dict)
    result_state: ToolResultStateName = "success"


class CloneRepoInput(BaseModel):
    session_id: str
    repo_url: str
    revision: str = "main"


class CloneRepoOutput(BaseModel):
    workdir: str


class PushBranchInput(BaseModel):
    session_id: str
    workdir: str
    branch: str


class PushBranchOutput(BaseModel):
    branch: str


class OpenPrInput(BaseModel):
    session_id: str
    repo_url: str
    branch: str
    title: str


class OpenPrOutput(BaseModel):
    pr_url: str


class TurnFailure(BaseModel):
    """Payload of a ``turn.failed`` event: ``{turnId, agentId, errorCode, retryable, message}``.

    ``agent_id`` is the contract agent id (``main`` for the room agent), not
    the session id. ``message`` is the fixed text for ``error_code``, never
    built from provider or exception text.
    """

    model_config = _CAMEL

    turn_id: str
    agent_id: str
    error_code: TurnErrorCode
    retryable: bool
    message: str


class OrbitEvent(BaseModel):
    """Normalized event the worker may ingest. Control rejects unknown types.

    The wire form is camelCase (``model_dump(by_alias=True)``).
    """

    model_config = _CAMEL

    type: Literal[
        "session.status",
        "assistant.message",
        "assistant.delta",
        "tool.call",
        "tool.result",
        "approval.asked",
        "approval.resolved",
        "question.asked",
        "question.answered",
        "todo.updated",
        "usage",
        "agent.started",
        "agent.finished",
        "agent.spawn_rejected",
        "turn.failed",
    ]
    event_id: str = ""
    occurred_at: str = ""
    session_id: str
    room_id: str
    job_id: str = ""
    text: str = ""
    runtime: str = "agentscope"
    runtime_version: str = "2.0.8"
    permission_preset: str = ""
    turn_id: str = ""
    agent_id: str = "main"
    parent_agent_id: str | None = None
    parent_session_id: str | None = None
    depth: int = 0
    persona: str = ""
    agent_path: str = "main"
    tool_name: str = ""
    call_id: str = ""
    approval_request_id: str = ""
    risk: ToolRisk | None = None
    error_code: ToolErrorCode | None = None
    # session.status, agent.finished reason, and similar.
    status: str = ""
    question: QuestionAsk | None = None
    answer: QuestionAnswer | None = None
    todos: list[TodoItem] | None = None
    todo_revision: int = 0
    # agent.spawn_rejected
    limit: str = ""
    limit_max: int = 0
    limit_current: int = 0
    # Every worker event carries both; the worker always sets model_mode.
    model_mode: ModelMode = "mock"
    model_name: str = ""
    failure: TurnFailure | None = None
    # tool.result. ``text`` is capped at 4096 UTF-8 bytes; ``truncated`` says it was cut.
    tool_state: ToolState | None = None
    truncated: bool = False
    # tool.call: redacted, then truncated to 256 characters.
    args_preview: str = ""
    # approval.resolved
    outcome: str = ""
    decided_by: str = ""
    # assistant.delta
    block_id: str = ""
    seq: int = 0
    delta: str = ""
    activity_attempt: int = 1
    # usage, one per model call
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_ms: int = 0


class RoomWorkflowInput(BaseModel):
    """Start payload. CamelCase aliases match the Go control client."""

    model_config = ConfigDict(populate_by_name=True)

    room_id: str = Field(alias="roomId")
    permission_preset: PermissionPreset = Field(default="workspace-write", alias="permissionPreset")
    kind: str = "solo"
    max_fanout: int = 4
    max_depth: int = 2
    gateway_task_queue: str = "orbit-gateway"


class RoomCommand(BaseModel):
    """A signal payload. The workflow validates the FSM before any Activity."""

    kind: Literal["open", "message", "approve", "abort", "steer"]
    turn_id: str
    message: str = ""
    outcome: Literal["allowed-once", "rejected"] = "allowed-once"


class RoomSnapshot(BaseModel):
    status: RoomStatus
    room_id: str
    session_id: str | None = None
    state_version: int = 0
    approval: ApprovalAsk | None = None
    last_text: str = ""
    error: str = Field(default="")
    child_workflow_ids: list[str] = Field(default_factory=list)


class AgentRunInput(BaseModel):
    room_id: str
    prompt: str
    permission_preset: PermissionPreset = "workspace-write"
    depth: int = 1
    max_depth: int = 2


class CloudAgentJobInput(BaseModel):
    job_id: str
    repo_url: str
    prompt: str
    permission_preset: PermissionPreset = "workspace-write"
    branch: str = "orbit/cloud-agent"


class CloudAgentSnapshot(BaseModel):
    status: RoomStatus
    job_id: str
    session_id: str | None = None
    workdir: str = ""
    branch: str = ""
    pr_url: str = ""
    last_text: str = ""


def contract_models() -> list[type[BaseModel]]:
    """Every public contract model, in export order."""

    return [
        ApprovalAsk,
        ExternalCall,
        OpenSessionInput,
        OpenSessionOutput,
        RunTurnInput,
        TurnResult,
        ResolveApprovalInput,
        DeliverToolResultInput,
        SteerInput,
        AbortSessionInput,
        CloseSessionInput,
        CloseSessionOutput,
        GatewayExecuteInput,
        GatewayExecuteOutput,
        CloneRepoInput,
        CloneRepoOutput,
        PushBranchInput,
        PushBranchOutput,
        OpenPrInput,
        OpenPrOutput,
        AgentRef,
        QuestionChoice,
        QuestionAsk,
        QuestionAnswer,
        TodoItem,
        TurnFailure,
        OrbitEvent,
        RoomWorkflowInput,
        RoomCommand,
        RoomSnapshot,
        AgentRunInput,
        CloudAgentJobInput,
        CloudAgentSnapshot,
    ]

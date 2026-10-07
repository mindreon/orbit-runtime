"""Activity inputs/outputs and the Room workflow surface.

Field names match the platform contract in orbit-infra ARCHITECTURE.md.
Framework types (AgentScope events, AgentState) never appear here.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from orbit_contracts.v3.common import PermissionSpec, Risk

# C34 §10.2 / §13 item 10. Web imports the same constants from schema/RoomFailure.json.
DECIDED_APPROVALS_LIMIT = "DECIDED_APPROVALS_LIMIT"
DECIDED_APPROVALS_LIMIT_MESSAGE = (
    "此任务的审批次数已达上限，无法继续。你可以查看记录，或新建任务继续工作。"
)

PermissionPreset = Literal["workspace-write", "read-only", "danger-full-access"]
TurnStatus = Literal["continue", "needs_approval", "needs_external", "completed", "failed"]
# "mock" means the turn ran on the in-process fake model, not a provider.
ModelMode = Literal["mock", "real"]
# Why a failed turn failed. Clients map these to text and never parse ``error``.
# ``state_unreadable``: the worker cannot read the session's saved agent state
# (plaintext blob in production, or a blob the current key cannot decrypt).
# ``budget``: the attempt spent what the task reserved for it (05 §4); it stopped between two steps and cannot be retried.
TurnErrorCode = Literal[
    "timeout", "auth", "rate_limited", "provider_error", "config", "state_unreadable", "budget"
]
ToolResultStateName = Literal["success", "error"]
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
# New decide models reject unknown fields. resumeTurnId must not sneak back in.


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
    # What the call is made with, to read; and the rule "always allow" would add (`tool_name`, `rule_content`).
    detail: str = ""
    allow_rule: dict[str, str | None] | None = None
    # How risky the call is, for the person who decides (`orbit_worker.permissions.call_risk`).
    risk: Risk = "medium"


class ExternalCall(BaseModel):
    """A tool call the workflow must finish outside the agent process."""

    tool_name: str
    call_id: str
    arguments: dict[str, str] = Field(default_factory=dict)


class McpHeaderRef(BaseModel):
    """Header name whose value is an environment variable on the worker.

    ``env`` is the variable name. The value is never part of this model.
    """

    model_config = _CAMEL

    name: str
    env: str


class McpConnectorSpec(BaseModel):
    """How the worker reaches one MCP server.

    ``stdio`` starts ``command``. ``streamable_http`` connects to ``url``.
    ``env_refs`` and ``header_refs`` are names only.
    """

    model_config = _CAMEL

    id: str
    name: str
    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env_refs: list[str] = Field(default_factory=list)
    url: str = ""
    header_refs: list[McpHeaderRef] = Field(default_factory=list)


class OpenSessionInput(BaseModel):
    room_id: str
    turn_id: str
    permission_preset: PermissionPreset = "workspace-write"
    # The task's own choice of what the agent may do without asking (`workspace-write` only; absent is preset "default").
    permissions: PermissionSpec | None = None
    # Chosen for this room. Empty for sessions opened before connectors existed.
    mcp_connectors: list[McpConnectorSpec] = Field(default_factory=list)
    # The session this one carries on: a follow-up message to a task that had finished (the task stays open).
    continue_from: str = ""
    # What a person allowed for the rest of the task: each is `tool_name` and `rule_content`.
    allow_rules: list[dict[str, str | None]] = Field(default_factory=list)


class OpenSessionOutput(BaseModel):
    session_id: str
    state_version: int
    # The session carries on the state of the one named by `continue_from`. False when none was asked for, or when it was
    # gone or unreadable and this one starts empty.
    carried: bool = False


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
    # Every call that asked in this step, `approval` being the first. A person may allow some and refuse others.
    approvals: list[ApprovalAsk] = Field(default_factory=list)
    external: ExternalCall | None = None
    # Every external call that is open, `external` being the first. A step of a team's leader may open several (07 §5).
    externals: list[ExternalCall] = Field(default_factory=list)
    # What the turn posted for the team (`team_note`, 07 §6), in order; read from the state it saved.
    notes: list[str] = Field(default_factory=list)
    # The roles each note addressed (`mentions` of `team_note`), parallel to `notes`.
    note_mentions: list[list[str]] = Field(default_factory=list)
    text: str = ""
    error: str = ""
    error_code: TurnErrorCode | None = None
    retryable: bool = False
    # The structured output of a turn that was asked for one (`structured_schema`), else None.
    output: dict[str, Any] | None = None
    model_mode: ModelMode = "mock"
    model_name: str = "mock"


class ResolveApprovalInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str
    approval_request_id: str
    outcome: Literal["allowed-once", "rejected"]
    # Per call id, true to allow. A call not listed follows `outcome`.
    decisions: dict[str, bool] = Field(default_factory=dict)
    # Per call id, a rule to allow from now on (`tool_name`, `rule_content`), for a call that was allowed.
    rules: dict[str, dict[str, str | None]] = Field(default_factory=dict)


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


class ExternalResult(BaseModel):
    """The answer to one external call, for `DeliverToolResultsInput`."""

    call_id: str
    tool_name: str
    output: str
    result_state: ToolResultStateName = "success"


class DeliverToolResultsInput(BaseModel):
    """Answers to every external call a session is parked on, at once: a step of a team's leader may open several, and
    AgentScope resumes the reply when it is given the results of all of them (07 §5)."""

    room_id: str
    session_id: str
    turn_id: str
    state_version: int
    results: list[ExternalResult] = Field(min_length=1)


class RoomFailure(BaseModel):
    """Payload of ``room.failed``. Both fields are constants in the generated schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    code: Literal["DECIDED_APPROVALS_LIMIT"] = DECIDED_APPROVALS_LIMIT
    message: Literal["此任务的审批次数已达上限，无法继续。你可以查看记录，或新建任务继续工作。"] = (
        DECIDED_APPROVALS_LIMIT_MESSAGE
    )


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
        "assistant.thinking",
        "tool.call",
        "tool.call_progress",
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
        "turn.started",
        "room.failed",
    ]
    event_id: str = ""
    occurred_at: str = ""
    session_id: str
    room_id: str
    job_id: str = ""
    text: str = ""
    runtime: str = "agentscope"
    runtime_version: str = "2.0.9"
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
    # turn.failed uses TurnFailure. room.failed uses RoomFailure. Same JSON field.
    failure: TurnFailure | RoomFailure | None = None
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



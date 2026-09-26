"""Activity inputs/outputs and the Room workflow surface.

Field names match the platform contract in orbit-infra ARCHITECTURE.md.
Framework types (AgentScope events, AgentState) never appear here.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class OrbitEvent(BaseModel):
    """Normalized event the worker may ingest. Control rejects unknown types."""

    type: Literal[
        "session.status",
        "assistant.message",
        "tool.call",
        "tool.result",
        "approval.asked",
        "usage",
        "agent.started",
        "agent.finished",
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
    model_mode: ModelMode | None = None
    model_name: str = ""
    error_code: TurnErrorCode | None = None


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
        OrbitEvent,
        RoomWorkflowInput,
        RoomCommand,
        RoomSnapshot,
        AgentRunInput,
        CloudAgentJobInput,
        CloudAgentSnapshot,
    ]

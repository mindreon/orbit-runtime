"""Activity inputs/outputs and the Room workflow surface.

Field names match the platform contract in orbit-infra ARCHITECTURE.md.
Framework types (AgentScope events, AgentState) never appear here.
"""

from typing import Literal

from pydantic import BaseModel, Field

PermissionPreset = Literal["workspace-write", "read-only", "danger-full-access"]
TurnStatus = Literal["continue", "needs_approval", "completed"]
RoomStatus = Literal["idle", "running", "awaiting_approval", "closed"]


class ApprovalAsk(BaseModel):
    """What a human must decide before a parked tool call continues."""

    approval_request_id: str
    tool_name: str
    call_id: str | None = None
    reason: str | None = None


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
    status: TurnStatus
    session_id: str
    state_version: int
    approval: ApprovalAsk | None = None
    text: str = ""


class ResolveApprovalInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str
    approval_request_id: str
    outcome: Literal["allowed-once", "rejected"]


class CloseSessionInput(BaseModel):
    room_id: str
    session_id: str
    turn_id: str


class CloseSessionOutput(BaseModel):
    closed: bool
    state_version: int


class OrbitEvent(BaseModel):
    """Normalized event the worker may ingest. Control rejects unknown types."""

    type: Literal[
        "session.status",
        "assistant.message",
        "tool.call",
        "tool.result",
        "approval.asked",
        "usage",
    ]
    session_id: str
    room_id: str
    text: str = ""
    runtime: str = "agentscope"
    runtime_version: str = ""


class RoomWorkflowInput(BaseModel):
    room_id: str
    permission_preset: PermissionPreset = "workspace-write"


class RoomCommand(BaseModel):
    """A signal payload. The workflow validates the FSM before any Activity."""

    kind: Literal["open", "message", "approve", "abort"]
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

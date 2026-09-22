"""Orbit tools the workflow executes. They never run inside the model call."""

from typing import Any, ClassVar

from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import ToolBase


class _ExternalTool(ToolBase):
    """A tool the agent can name but cannot call. The workflow runs it."""

    is_external_tool: bool = True
    is_concurrency_safe: bool = True
    is_read_only: bool = False
    is_state_injected: bool = False

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message=f"{self.name} is executed by the Orbit workflow.",
        )


class GatewayChargeTool(_ExternalTool):
    """A gateway write. The human approves it before the gateway Activity."""

    name = "gateway_charge"
    description = "Charge an account through the Tool Gateway. Requires approval."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"amount": {"type": "string"}},
        "required": ["amount"],
    }
    metadata_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message="gateway tool requires confirmation",
        )


class GatewayLookupTool(_ExternalTool):
    """A read-only gateway call. It still runs as an external Activity."""

    name = "gateway_lookup"
    description = "Look up a record through the Tool Gateway."
    is_read_only = True
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    metadata_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "value": {"type": "string"},
        },
        "required": ["ok", "value"],
    }


class AgentSpawnTool(_ExternalTool):
    name = "agent_spawn"
    description = "Start a worker agent as its own Temporal child workflow."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "persona": {"type": "string"},
        },
        "required": ["prompt"],
    }
    metadata_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["workflow_id", "status"],
    }


class AgentSendTool(_ExternalTool):
    name = "agent_send"
    description = "Signal a worker child workflow."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "workflow_id": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["workflow_id", "message"],
    }
    metadata_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }


class AgentWaitTool(_ExternalTool):
    name = "agent_wait"
    description = "Wait until the spawned worker workflows finish."
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    metadata_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }


class TeamDissolveTool(_ExternalTool):
    name = "team_dissolve"
    description = "Cancel worker child workflows and close their sessions."
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    metadata_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"dissolved": {"type": "boolean"}},
        "required": ["dissolved"],
    }


def orbit_tools() -> list[ToolBase]:
    return [
        GatewayChargeTool(),
        GatewayLookupTool(),
        AgentSpawnTool(),
        AgentSendTool(),
        AgentWaitTool(),
        TeamDissolveTool(),
    ]

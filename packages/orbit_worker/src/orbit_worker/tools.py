"""Tools of a task attempt that are not planning tools: asking the user, asking for more budget."""

from typing import Any, ClassVar

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk


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


class AskUserTool(_ExternalTool):
    """The agent asks the user a question. The attempt parks until a message answers it (04 §2)."""

    name = "ask_user"
    description = "Ask the user a question and wait for the answer."
    is_read_only = True
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    }
    metadata_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}


class RequestBudgetExtensionTool(ToolBase):
    """Ask a person for more exploration budget. The call itself is the approval: allowed means granted."""

    name = "orbit_request_budget_extension"
    description = "Ask for more exploration budget. A person decides; if allowed, more tool calls become available."
    is_concurrency_safe = False
    is_read_only = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
    }
    EXTRA_TOOL_CALLS: ClassVar[int] = 10

    async def check_permissions(
        self, tool_input: dict[str, Any], context: PermissionContext
    ) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(
            behavior=PermissionBehavior.ASK, message="more exploration budget was requested"
        )

    async def call(self, **kwargs: Any) -> ToolChunk:
        return ToolChunk(
            content=[
                TextBlock(
                    text=f"granted {self.EXTRA_TOOL_CALLS} more tool calls: {kwargs.get('reason', '')}"
                )
            ],
            state=ToolResultState.SUCCESS,
        )


def orbit_tools() -> list[ToolBase]:
    return [AskUserTool(), RequestBudgetExtensionTool()]

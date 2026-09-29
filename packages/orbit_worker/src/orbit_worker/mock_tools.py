"""Tools that only exist with the mock model, so an end-to-end test can hold a tool call open."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk

from orbit_worker.settings import MockSettings


def _append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class SlowEchoTool(ToolBase):
    """A non-idempotent tool that takes ORBIT_MOCK_TOOL_DELAY_MS and logs each run to ORBIT_MOCK_TOOL_LOG.

    The log is how a test counts real executions: one line per run, written when the run starts.
    """

    name = "slow_echo"
    description = "Echo text after a delay. It has a side effect, so it must not be repeated."
    is_read_only = False
    is_concurrency_safe = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def check_permissions(
        self, tool_input: dict[str, Any], context: PermissionContext
    ) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="mock tool")

    async def call(self, **kwargs: Any) -> ToolChunk:
        text = str(kwargs.get("text", ""))
        settings = MockSettings()
        if settings.tool_log:
            await asyncio.to_thread(_append_line, settings.tool_log, f"{self.name}:{text}")
        await asyncio.sleep(settings.tool_delay_ms / 1000)
        return ToolChunk(content=[TextBlock(text=f"slow:{text}")], state=ToolResultState.SUCCESS)


def mock_tools(mock_mode: bool) -> list[ToolBase]:
    return [SlowEchoTool()] if mock_mode else []

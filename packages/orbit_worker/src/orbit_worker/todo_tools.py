"""`TodoWrite`: the agent's own checklist for a task of several steps, shown to the user as it changes.

Each call replaces the whole list, which is how the agent edits it: it writes the list again with an item added, reworded or
ticked off. The list is not kept anywhere by the tool and changes nothing in the workspace or the plan; a person reads it from
the call itself. The call's input does not fit the 256 characters of `tool.call_started`'s `args_preview` (which is also not
durable), so the ledger puts the whole normalized list in the durable `tool.call_finished` event (`todo_list_preview`).

It is not `TaskCreate`/`TaskUpdate` (`planning_tools`), which create nodes of the task's plan.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk

from orbit_worker.secrets import redact_value

TODO_WRITE = "TodoWrite"
STATUSES = ("pending", "in_progress", "completed")
MAX_TODOS = 50
TODO_CHARS = 200


def normalize_todos(raw: Any) -> list[dict[str, str]]:
    """The checklist of a `todos` input: each item's text trimmed and cut to `TODO_CHARS`. Raises ValueError, naming what is
    wrong, for anything that is not a list of at most `MAX_TODOS` items each with a text and a known status."""
    if not isinstance(raw, list):
        raise ValueError("todos must be a list")  # noqa: TRY004 (one error type for every bad input)
    if len(raw) > MAX_TODOS:
        raise ValueError(f"at most {MAX_TODOS} todos")
    todos: list[dict[str, str]] = []
    for index, item in enumerate(raw, 1):
        content = item.get("content") if isinstance(item, dict) else None
        status = item.get("status") if isinstance(item, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"todo {index} needs a non-empty content")
        if status not in STATUSES:
            raise ValueError(f"todo {index}: status must be one of {', '.join(STATUSES)}")
        todos.append({"content": " ".join(content.split())[:TODO_CHARS], "status": status})
    return todos


def todo_list_preview(call_input: str) -> str:
    """The checklist of a `TodoWrite` call as the compact JSON `{"todos": [...]}` a client reads, secrets redacted; empty for
    an input that is not a valid checklist (the call itself ended in an error then)."""
    try:
        todos = normalize_todos(json.loads(call_input or "{}").get("todos"))
    except (ValueError, AttributeError):
        return ""
    return json.dumps(redact_value({"todos": todos}), ensure_ascii=False, separators=(",", ":"))


class TodoWriteTool(ToolBase):
    name = TODO_WRITE
    description = (
        "Write the checklist of a task that takes three or more steps, and keep it up to date as you work: the user sees it. "
        "Every call replaces the whole list, so send all items each time, with their current status. Keep exactly one "
        "item in_progress, mark an item completed as soon as it is done, and rewrite the list when the plan changes. Items "
        "are short imperative phrases. Do not use it for a one-step task."
    )
    is_concurrency_safe = False
    # It changes nothing in the workspace or the plan, and writing the same list twice is the same as once: a read-only task
    # may use it and a replay after a crash is harmless.
    is_read_only = True
    is_state_injected = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "maxItems": MAX_TODOS,
                "description": "The whole checklist, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to do, as a short imperative phrase."},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Writing the checklist is always allowed.")

    async def call(self, **kwargs: Any) -> ToolChunk:
        try:
            todos = normalize_todos(kwargs.get("todos"))
        except ValueError as exc:
            return ToolChunk(content=[TextBlock(text=f"The checklist was not saved: {exc}.")], state=ToolResultState.ERROR)
        done = sum(1 for todo in todos if todo["status"] == "completed")
        return ToolChunk(
            content=[TextBlock(text=f"Todos updated ({done}/{len(todos)} done). Carry on with the list.")],
            state=ToolResultState.SUCCESS,
        )


def todo_tools() -> list[ToolBase]:
    return [TodoWriteTool()]

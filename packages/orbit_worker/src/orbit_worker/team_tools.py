"""The tools of a team stage (07): `team_assign` for the leader, `team_note` for everyone, and what the agents are told.

The protocol is AgentScope's `TeamPipeline`, driven by Temporal instead of by its executor. `team_assign` is an external tool
like `ask_user`: the leader names a member and a task, the turn ends with the call open, and the AttemptWorkflow runs the
member as an activity and hands the answer back as the tool's result. `team_note` is an ordinary tool that posts a line for the
team's mailbox; the notes a turn made are read from its saved state (`notes_of`), so a turn that ran twice (an activity retry)
reports them once.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from agentscope.message import TextBlock, ToolCallBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.state import AgentState
from agentscope.tool import ToolBase, ToolChunk

from orbit_worker.task_stream import TeamTurn
from orbit_worker.tools import _ExternalTool

ASSIGN = "team_assign"
NOTE = "team_note"
NOTE_CHARS = 1000


class TeamAssignTool(_ExternalTool):
    """The leader gives a member a task and gets its answer back as the result. Assignments of one step to different members
    run side by side; to the same member, one after the other."""

    name = ASSIGN
    is_read_only = False

    async def check_read_only(self, tool_input: dict[str, Any]) -> bool:
        # Only the leader has it, and the leader runs read-only: handing a task to a member writes nothing here.
        return True

    def __init__(self, members: tuple[tuple[str, str], ...]) -> None:
        super().__init__()
        listing = "\n".join(f"- {role}: {description}" if description else f"- {role}" for role, description in members)
        self.description = (
            "Assign a task to a team member and get its reply back as the result. Members cannot see your conversation "
            "or each other's: write the whole task. The members are:\n" + listing
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "member": {"type": "string", "enum": [role for role, _ in members], "description": "The member to assign to."},
                "task": {"type": "string", "description": "The complete task for the member."},
            },
            "required": ["member", "task"],
        }


class TeamNoteTool(ToolBase):
    """Post a short note for the whole team. Everyone's next turn is shown it."""

    name = NOTE
    description = (
        "Post a short note for the whole team (what you found, what you decided, what to watch for). Every member and the "
        "leader see it at the start of their next turn. @mention a member (the `mentions` list, or @role in the text) to "
        "wake it: it takes a turn with your note as its input and answers the group."
    )
    is_concurrency_safe = True
    # It only writes a line to the team's mailbox, so a replay after a crash is harmless and a read-only task may use it.
    is_read_only = True
    is_state_injected = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The note. Write @role (or @label) in it to address a member."},
            "mentions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Roles to address. Each mentioned member takes a turn with the note as its input.",
            },
        },
        "required": ["text"],
    }

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="A note for the team is always allowed.")

    async def call(self, **kwargs: Any) -> ToolChunk:
        del kwargs
        return ToolChunk(content=[TextBlock(text="noted")], state=ToolResultState.SUCCESS)


def team_tools(turn: TeamTurn) -> list[ToolBase]:
    """What the agent of a team stage can call beside its workspace: the leader assigns, everybody posts notes."""
    return [*([TeamAssignTool(turn.members)] if turn.leader else []), TeamNoteTool()]


def notes_of(state: AgentState, agent_name: str, start: int) -> list[tuple[str, list[str]]]:
    """The notes the agent made in the messages of its state from `start` on, in order: the text (cut to `NOTE_CHARS`) and the
    roles it listed in `mentions`."""
    notes: list[tuple[str, list[str]]] = []
    for message in state.context[start:]:
        if message.role != "assistant" or message.name != agent_name or isinstance(message.content, str):
            continue
        for block in message.content:
            if isinstance(block, ToolCallBlock) and block.name == NOTE:
                try:
                    arguments = json.loads(block.input or "{}")
                    text = arguments.get("text", "")
                    mentions = arguments.get("mentions") or []
                except (json.JSONDecodeError, AttributeError):
                    continue
                if isinstance(text, str) and text.strip():
                    listed = [str(item) for item in mentions if isinstance(item, str)] if isinstance(mentions, list) else []
                    notes.append((text.strip()[:NOTE_CHARS], listed))
    return notes


def stage_prompt(turn: TeamTurn) -> str:
    """What the agent is told about the stage it is in, for its system prompt."""
    if turn.leader:
        lines = [
            (
                "You lead a team in one stage of a task. Work with your members to reach the goal: give each piece of work to "
                f"the member who should do it with {ASSIGN}. Assignments you make in one step run side by side (two for the "
                "same member run one after the other), and each member's reply comes back as that call's result. A member "
                f"cannot see your conversation: write the whole task. Post a note for the whole team with {NOTE}. When the "
                "goal is reached, answer with the final result: that answer is the stage's."
            ),
            "Your team:",
        ]
        lines += [f"- {role}: {' '.join(description.split())}" if description else f"- {role}" for role, description in turn.members]
        return "\n".join(lines)
    return (
        f"You are the member {turn.role}{f' ({turn.label})' if turn.label else ''} of a team led by {turn.leader_role}{f' ({turn.leader_label})' if turn.leader_label else ''}. The leader gives you tasks: do each and answer "
        f"with what you did and found; your answer goes to the leader only. Post a note for the whole team with {NOTE}. "
        "Your workspace is a copy of the task's: what you change there stays in it, and the files you leave are handed to the "
        "leader."
    )

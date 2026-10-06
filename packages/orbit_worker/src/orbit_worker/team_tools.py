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


# What the leader's brief to a member holds. The member cannot see the conversation, so unlike the leader's own messages (short) a
# brief is complete and detailed. Used by `team_assign`, the stage prompt and `TaskCreate`.
BRIEF_GUIDE = (
    "Write the brief in full, with these parts: (1) Role: the member's own role in the team (its label, what it is for) and what it is "
    "responsible for in this task, never a different persona such as a project lead; and the files it must read first "
    "(absolute paths). (2) Background shared by all members: the user's request in their words, what the user already "
    "confirmed, constraints (stack, scope), environment notes. (3) Inputs: absolute paths of upstream results to read in "
    "full, not a summary, plus a few key points to orient. (4) Existing work: what already exists and where (working "
    "directory, files) that the member must build on and stay consistent with. (5) The task: a numbered list of what must "
    "be done or contained. (6) Output and done criteria: where to write the result (absolute path), how to verify it, and "
    "what to report back (paths, key conclusions, open issues, briefly)."
)


class TeamAssignTool(_ExternalTool):
    """The leader gives a member a task and gets its answer back as the result. Assignments of one step to different members
    run side by side; to the same member, one after the other."""

    name = ASSIGN
    is_read_only = False

    async def check_read_only(self, tool_input: dict[str, Any]) -> bool:
        # Only the leader has it, and it may run read-only: handing a task to a member writes nothing here.
        return True

    def __init__(self, members: tuple[tuple[str, ...], ...]) -> None:
        super().__init__()
        listing = "\n".join(member_line(member) for member in members)
        self.description = (
            "Assign a task to a team member and get its reply back as the result. Members cannot see your conversation "
            "or each other's, so the `task` is a complete, detailed brief, never shortened. " + BRIEF_GUIDE + " The members are:\n"
            + listing
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "member": {"type": "string", "enum": [member[0] for member in members], "description": "The member to assign to."},
                "task": {
                    "type": "string",
                    "description": "The member's whole brief: role, background, inputs, existing work, the numbered task, "
                    "output and done criteria.",
                },
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


def member_line(member: tuple[str, ...]) -> str:
    """A member as the leader reads it: the label a person knows it by first, the id (the `team_assign` value) beside it."""
    role, description = member[0], " ".join(member[1].split()) if len(member) > 1 else ""
    label = " ".join(member[2].split()) if len(member) > 2 else ""
    name = f"{label} (id: {role})" if label else role
    return f"- {name}: {description}" if description else f"- {name}"


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
                "cannot see your conversation, so each assignment is a complete, detailed brief (never shortened): role and files to read "
                "first, background and what the user confirmed, inputs as absolute paths, existing work to build on, the numbered task, "
                f"and the output path with done criteria and what to report back. Post a note for the whole team with {NOTE}. When the "
                "goal is reached, answer with the final result: that answer is the stage's. Keep your own messages short: one short "
                "message per planning step (what you assigned and to whom), not a line per action, and never paste a member's reply "
                "back. In anything you write (to the user, in briefs) call members by their label; the id is only the value of "
                f"{ASSIGN}'s `member`, never shown. The user sees the final result, so write it as result, how to use it and key files."
            ),
            "Your team:",
        ]
        lines += [member_line(member) for member in turn.members]
        return "\n".join(lines)
    return (
        f"You are the member {turn.role}{f' ({turn.label})' if turn.label else ''} of a team led by {turn.leader_role}{f' ({turn.leader_label})' if turn.leader_label else ''}. The leader gives you tasks: do each and answer "
        f"with a short handover summary, a few lines: what you did, the files you left (paths), and any open issue. Do not "
        "paste file contents or long code into it; the files are handed over with it. If the environment or a tool fails, say "
        f"so in a sentence and stop. Your answer goes to the leader only. Post a note for the whole team with {NOTE}. "
        "Your workspace is a copy of the task's: what you change there stays in it, and the files you leave are handed to the "
        "leader."
    )

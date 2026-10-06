"""TodoWrite: the checklist replaces itself on every call, and its whole input reaches the durable event."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import ToolResultState
from agentscope.tool import ToolResponse
from orbit_worker.ledger_middleware import OrbitLedgerMiddleware
from orbit_worker.task_stream import TaskStreamContext, TeamTurn
from orbit_worker.team_tools import team_tools
from orbit_worker.todo_tools import TodoWriteTool, normalize_todos, todo_list_preview, todo_tools


def _todos(*statuses: str) -> dict[str, Any]:
    return {"todos": [{"content": f"step {index}", "status": status} for index, status in enumerate(statuses)]}


async def test_a_call_acknowledges_briefly() -> None:
    chunk = await TodoWriteTool().call(**_todos("completed", "in_progress", "pending"))
    assert chunk.state == ToolResultState.SUCCESS
    assert "1/3" in chunk.content[0].text


@pytest.mark.parametrize(
    "raw",
    [None, "x", [{"content": "", "status": "pending"}], [{"content": "a", "status": "done"}], [{"status": "pending"}], ["a"]],
)
async def test_a_bad_list_is_an_error_not_a_crash(raw: Any) -> None:
    chunk = await TodoWriteTool().call(todos=raw)
    assert chunk.state == ToolResultState.ERROR
    with pytest.raises(ValueError):
        normalize_todos(raw)


def test_it_is_read_only_and_never_asks() -> None:
    tool = TodoWriteTool()
    assert tool.is_read_only
    assert tool.input_schema["properties"]["todos"]["items"]["properties"]["status"]["enum"] == [
        "pending", "in_progress", "completed",
    ]


def test_the_preview_is_the_whole_normalized_list() -> None:
    long = "x" * 1000
    raw = json.dumps({"todos": [{"content": f"  a  b {long}", "status": "pending"}] + _todos("completed")["todos"]})
    preview = json.loads(todo_list_preview(raw))
    assert len(preview["todos"]) == 2 and preview["todos"][0]["content"].startswith("a b x")
    assert len(preview["todos"][0]["content"]) == 200
    assert todo_list_preview("{not json") == "" and todo_list_preview(json.dumps({"todos": "no"})) == ""


def test_leader_and_single_agent_have_it_members_do_not() -> None:
    from orbit_contracts.v3 import Team, TeamMember
    from orbit_worker.agent_config import AgentConfig
    from orbit_worker.runtime import _has_todo_tool

    member = TeamTurn(role="m", leader=False, leader_role="l", leader_label="", label="", stage_attempt_id="s", members=())
    leader = TeamTurn(role="l", leader=True, leader_role="l", leader_label="", label="", stage_attempt_id="s", members=())
    alone = AgentConfig()
    assert _has_todo_tool(None, alone) and _has_todo_tool(leader, alone) and not _has_todo_tool(member, alone)
    # A stage leader keeps it even when its expert is also the leader of the task's plan-level team.
    assert _has_todo_tool(leader, AgentConfig(team=Team(leader="l", members=[TeamMember(role="l", expert="e@1")])))
    assert [tool.name for tool in todo_tools()] == ["TodoWrite"]
    assert "TodoWrite" not in [tool.name for tool in team_tools(leader)]


_PLAN_TEAM = {
    "leader": "lead",
    "members": [{"role": "lead", "expert": "writer@1"}, {"role": "review", "expert": "reviewer@2"}],
}


def test_a_plan_level_team_has_no_checklist_for_its_leader_or_members_but_a_lone_agent_has_one() -> None:
    from orbit_worker.agent_config import AgentConfig, with_task_config
    from orbit_worker.runtime import _has_todo_tool

    def config(profile: str, raw: dict[str, Any] | None) -> AgentConfig:
        return with_task_config(AgentConfig(), raw, profile)

    # The member's attempt has no stage turn and no `team` of its own, yet it is not a single agent.
    member = config("reviewer@2", {"team": _PLAN_TEAM})
    assert member.team is None and member.in_team_member
    assert not _has_todo_tool(None, member)
    # The plan-level leader ends its attempt after creating the members' tasks: the page builds the list from the plan.
    leader = config("writer@1", {"team": _PLAN_TEAM})
    assert leader.team is not None and not _has_todo_tool(None, leader)
    assert _has_todo_tool(None, config("writer@1", None))
    assert _has_todo_tool(None, config("writer@1", {"team": {"leader": "nobody", "members": []}}))


class _Ledger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def publish_events(self, events: list[dict[str, Any]]) -> None:
        self.events.extend(events)


async def test_tool_call_finished_carries_the_whole_checklist() -> None:
    ledger = _Ledger()
    context = TaskStreamContext(tenant_id="t", task_id="task", attempt_id="att", activity_attempt=1)
    arguments = json.dumps(_todos(*["completed"] * 20, "in_progress"))
    assert len(arguments) > 256  # more than `args_preview` of tool.call_started keeps
    call = SimpleNamespace(id="call-1", name="TodoWrite", input=arguments)
    response = ToolResponse(content=[], state=ToolResultState.SUCCESS)
    await OrbitLedgerMiddleware(ledger)._finished(context, call, response)  # type: ignore[arg-type]
    [event] = ledger.events
    assert event["type"] == "tool.call_finished" and event["retention"] == "durable"
    todos = json.loads(event["payload"]["args_preview"])["todos"]
    assert len(todos) == 21 and todos[-1] == {"content": "step 20", "status": "in_progress"}


async def test_other_tools_keep_no_arguments_on_the_finished_event() -> None:
    ledger = _Ledger()
    context = TaskStreamContext(tenant_id="t", task_id="task", attempt_id="att", activity_attempt=1)
    call = SimpleNamespace(id="call-2", name="Write", input='{"file_path": "/workspace/a"}')
    await OrbitLedgerMiddleware(ledger)._finished(context, call, ToolResponse(content=[], state=ToolResultState.SUCCESS))  # type: ignore[arg-type]
    assert "args_preview" not in ledger.events[0]["payload"]

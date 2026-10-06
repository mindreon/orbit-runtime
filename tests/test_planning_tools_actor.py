"""What the planning tools tell the workflow about who is asking (05 §3, §6) and where a node they create belongs (03 §3).

How it can go wrong, written down before the code:
  - TaskList asks for the plan without naming the attempt, so the workflow cannot show it what it may see (and shows it all);
  - work the leader gives to a team member is not nested under the leader's node, so the team's depth is never counted.
"""

from __future__ import annotations

from typing import Any

import pytest
from agentscope.message import ToolResultState
from orbit_contracts.v3 import Actor, Team, TeamMember
from orbit_contracts.v3.plan import AddNodeOp, PlanChangeAccepted, PlanChangeCommand
from orbit_contracts.v3.views import PlanView
from orbit_worker.agent_config import AgentConfig
from orbit_worker.planning_tools import TaskCreateTool, TemporalPlanPort
from orbit_worker.task_stream import TaskStreamContext, executing_tool_call, streaming_for

NODE = "n_01J9Z3K4M5N6P7Q8R9S0T1V2W4"
ATTEMPT = "att_01J9Z3K4M5N6P7Q8R9S0T1V2W5"
TEAM = Team(leader="lead", members=[TeamMember(role="lead", expert="boss@1"), TeamMember(role="dev", expert="dev@1")])


def _context(team: Team | None = None) -> TaskStreamContext:
    return TaskStreamContext(
        tenant_id="t", task_id="task_01J9Z3K4M5N6P7Q8R9S0T1V2W3", attempt_id=ATTEMPT, activity_attempt=1, node_id=NODE,
        profile="boss@1", agent=AgentConfig(team=team),
    )


class _Plan:
    def __init__(self) -> None:
        self.commands: list[PlanChangeCommand] = []

    async def get_plan(self, context: TaskStreamContext) -> PlanView:
        return PlanView(plan_version=4, hash="sha256:" + "a" * 64)

    async def submit(self, context: TaskStreamContext, command: PlanChangeCommand) -> PlanChangeAccepted:
        self.commands.append(command)
        return PlanChangeAccepted(plan_version=5, id_map={"tmp:1": "n_01J9Z3K4M5N6P7Q8R9S0T1V2X4"})


async def _create(plan: _Plan, team: Team | None, **metadata: Any) -> AddNodeOp:
    with streaming_for(_context(team)), executing_tool_call("call-1"):
        await TaskCreateTool(plan).call(subject="write it", description="write the report", metadata=metadata or None)
    op = plan.commands[-1].ops[0]
    assert isinstance(op, AddNodeOp)
    return op


async def test_work_given_to_a_team_member_is_nested_under_the_node_that_gave_it() -> None:
    op = await _create(_Plan(), TEAM, owner="dev")
    assert op.node.owner_profile == "dev@1"
    assert op.node.parent_node_id == NODE
    assert op.node.depends_on == [NODE], "nesting is not ordering: it still waits for the node that planned it"


async def test_work_nobody_owns_is_not_nested() -> None:
    assert (await _create(_Plan(), None)).node.parent_node_id is None


@pytest.mark.parametrize("metadata", [{}, {"owner": "lead"}])
async def test_in_a_team_a_task_without_a_member_owner_is_rejected_and_not_created(metadata: dict[str, Any]) -> None:
    plan = _Plan()
    with streaming_for(_context(TEAM)), executing_tool_call("call-1"):
        chunk = await TaskCreateTool(plan).call(subject="write it", description="d", metadata=metadata or None)
    assert not plan.commands
    assert chunk.state == ToolResultState.ERROR
    assert "in a team every task goes to a member" in chunk.content[0].text and "dev" in chunk.content[0].text


async def test_the_plan_is_asked_for_as_the_attempt_that_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, tuple[Any, ...], Any]] = []

    class Handle:
        async def query(self, name: str, *args: Any, result_type: Any = None) -> PlanView:
            asked.append((name, args, result_type))
            return PlanView(plan_version=1, hash="sha256:" + "a" * 64)

    class Client:
        def get_workflow_handle(self, workflow_id: str) -> Handle:
            assert workflow_id == "task/t/task_01J9Z3K4M5N6P7Q8R9S0T1V2W3"
            return Handle()

    import temporalio.activity

    monkeypatch.setattr(temporalio.activity, "client", lambda: Client())
    port = TemporalPlanPort(lambda context: f"task/{context.tenant_id}/{context.task_id}")
    await port.get_plan(_context())
    name, args, result_type = asked[0]
    assert name == "getPlan" and result_type is PlanView
    assert args == (Actor(kind="agent", id=ATTEMPT, attempt_id=ATTEMPT),)

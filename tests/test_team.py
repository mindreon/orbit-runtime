"""What a team's leader is told, and how a node is given to a member (15 M8, T8.6).

How it can go wrong, written down before the code:
  - a member's attempt, or an attempt of a task without a team, is shown the roster and starts handing work out;
  - the leader names a role the team does not have and the node is created anyway, or goes to some member by guess;
  - `owner` is given on a task that has no team, or is not a string, and the tool crashes instead of refusing;
  - the refusal does not say which roles exist, so the leader cannot correct itself;
  - a member's description with a line break or a heading breaks the structure of the prompt it is put into;
  - no `owner` lets a task be created without saying who does it; the leader's own role is a valid owner (it does the work itself);
  - the leader is forced read-only whatever its expert's mode says, or read-only blocks the leader's planning tools too.
"""

from unittest.mock import patch

import pytest
from orbit_worker.agent_config import (
    AgentConfig,
    owner_profile_for,
    permission_preset_for,
    team_prompt,
    with_task_config,
)

TEAM = {
    "leader": "lead",
    "members": [
        {"role": "lead", "expert": "writer@1", "description": "plans and writes"},
        {"role": "review", "expert": "reviewer@2", "description": "checks the work"},
    ],
}


def _config(**fields) -> dict:
    return {"config_version": 1, "mode": "default", **fields}


def test_the_leader_is_told_its_team() -> None:
    config = with_task_config(AgentConfig(instructions="Be brief."), _config(team=TEAM), "writer@1")
    assert config.instructions.startswith("Be brief.")
    assert "- review: checks the work" in config.instructions and "- lead (you): plans and writes" in config.instructions
    assert "metadata" in config.instructions and "owner" in config.instructions
    assert "You lead a team" in config.instructions
    assert "split by area" in config.instructions and "Do not bundle unrelated work" in config.instructions
    assert "review turn, not a member's task" in config.instructions
    assert [member.role for member in config.team.members] == ["lead", "review"]


@pytest.mark.parametrize("profile", ["reviewer@2", "other@1", ""])
def test_a_member_or_a_stranger_is_not_given_the_roster(profile: str) -> None:
    config = with_task_config(AgentConfig(instructions="Be brief."), _config(team=TEAM), profile)
    assert config.team is None and "You lead a team" not in config.instructions
    assert config.instructions.startswith("Be brief.")


def test_a_member_node_is_told_to_end_with_a_handover_block_but_a_task_without_a_team_is_not() -> None:
    from orbit_worker.agent_config import HANDOVER_MARKER, MEMBER_HANDOVER_PROMPT

    member = with_task_config(AgentConfig(instructions="Be brief."), _config(team=TEAM), "reviewer@2")
    assert member.in_team_member and member.instructions == f"Be brief.\n\n{MEMBER_HANDOVER_PROMPT}"
    for heading in ("Result", "Evidence", "Files", "Verification", "Limits and open issues", "Needs from the leader"):
        assert heading in MEMBER_HANDOVER_PROMPT
    assert HANDOVER_MARKER in MEMBER_HANDOVER_PROMPT
    assert with_task_config(AgentConfig(instructions="Be brief."), _config(), "reviewer@2").instructions == "Be brief."


def test_a_task_without_a_team_has_no_roster() -> None:
    assert with_task_config(AgentConfig(), _config(), "writer@1").team is None
    assert with_task_config(AgentConfig(), _config(team=None), "writer@1").team is None


def test_an_invalid_team_is_ignored_not_fatal() -> None:
    broken = {"leader": "nobody", "members": [{"role": "lead", "expert": "writer@1"}]}
    assert with_task_config(AgentConfig(), _config(team=broken), "writer@1").team is None
    assert with_task_config(AgentConfig(), _config(team="nope"), "writer@1").team is None


def test_a_description_cannot_break_the_prompt() -> None:
    team = {"leader": "lead", "members": [{"role": "lead", "expert": "writer@1", "description": "line one\n# Heading\n\nline two"}]}
    prompt = team_prompt(with_task_config(AgentConfig(), _config(team=team), "writer@1").team)
    assert "\n# Heading" not in prompt and "line one # Heading line two" in prompt


def _leader() -> AgentConfig:
    return with_task_config(AgentConfig(), _config(team=TEAM), "writer@1")


def test_a_role_becomes_that_members_expert() -> None:
    assert owner_profile_for(_leader().team, {"owner": "review"}) == ("reviewer@2", "")


@pytest.mark.parametrize("metadata", [None, {}, {"other": "x"}])
def test_in_a_team_a_task_without_an_owner_is_refused(metadata) -> None:
    ref, error = owner_profile_for(_leader().team, metadata)
    assert ref is None
    assert "every task has an owner" in error and "review" in error and "lead" in error


def test_a_task_may_be_owned_by_the_leader_itself() -> None:
    assert owner_profile_for(_leader().team, {"owner": "lead"}) == ("writer@1", "")


@pytest.mark.parametrize("metadata", [None, {}, {"other": "x"}])
def test_without_a_team_no_owner_is_no_error(metadata) -> None:
    assert owner_profile_for(None, metadata) == (None, "")


def test_the_leader_gets_the_preset_of_its_experts_mode_like_anyone_else() -> None:
    assert permission_preset_for(_config()) == "workspace-write"
    assert permission_preset_for(None) == "workspace-write"
    assert permission_preset_for(_config(mode="ask")) == "read-only"
    assert _leader().team is not None  # a leader's attempt is not read-only for being the leader


def test_the_leader_prompt_lets_it_work_itself_and_lists_it_with_the_members() -> None:
    prompt = _leader().instructions
    assert "Do work yourself when it fits your own role" in prompt and "too small to be worth handing off" in prompt
    assert "If your workspace is read-only you only plan, coordinate and review" in prompt
    assert "only coordinate" not in prompt and "never implement" not in prompt and "never to yourself" not in prompt
    assert "- lead (you): plans and writes" in prompt and "- review: checks the work" in prompt
    assert prompt.index("- lead (you)") < prompt.index("- review:")
    assert "call members by their label; the owner id is only for TaskCreate metadata" in prompt
    assert "one short message at the end (what was split and to whom), not a line per step" in prompt
    assert "Integration, documentation, scripts and verification" in prompt and "review turn" in prompt
    assert "split by area" in prompt and "Do not bundle unrelated work" in prompt


async def test_the_read_only_leader_may_still_plan_and_hand_out() -> None:
    from types import SimpleNamespace

    from orbit_worker.planning_tools import TaskCreateTool, TaskListTool
    from orbit_worker.team_tools import TeamAssignTool

    create = TaskCreateTool(SimpleNamespace())
    assert not await create.check_read_only({}), "no task context: not read-only"
    for team in (_leader().team, None):
        context = SimpleNamespace(agent=SimpleNamespace(team=team))
        with patch("orbit_worker.planning_tools.current_task_context", return_value=context):
            assert await create.check_read_only({}) is (team is not None)
            assert await TaskListTool(SimpleNamespace()).check_read_only({})
    assert await TeamAssignTool((("review", ""),)).check_read_only({})


def test_an_unknown_role_is_refused_with_the_roles_that_exist() -> None:
    ref, error = owner_profile_for(_leader().team, {"owner": "designer"})
    assert ref is None
    assert "designer" in error and "review" in error


@pytest.mark.parametrize("metadata", [{"owner": 7}, {"owner": ["review"]}, {"owner": ""}])
def test_an_owner_that_is_not_a_role_name_is_refused(metadata) -> None:
    ref, error = owner_profile_for(_leader().team, metadata)
    assert ref is None and error != ""


def test_an_owner_on_a_task_without_a_team_is_refused() -> None:
    ref, error = owner_profile_for(None, {"owner": "review"})
    assert ref is None and "team" in error


def test_a_members_label_is_shown_to_the_leader_beside_its_description() -> None:
    team = {"leader": "lead", "members": [
        {"role": "lead", "expert": "writer@1"},
        {"role": "review", "expert": "reviewer@2", "description": "checks the work", "label": "审阅员"},
        {"role": "docs", "expert": "docs@1", "label": "文档"},
    ]}
    prompt = with_task_config(AgentConfig(), _config(team=team), "writer@1").instructions
    assert "- 审阅员 (owner id: review): checks the work" in prompt and "- 文档 (owner id: docs)\n" in prompt + "\n"
    assert "- lead (you)\n" in prompt + "\n", "a member without a label is listed by its id"
    from orbit_contracts.v3 import TeamMember
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TeamMember(role="x", expert="a@1", label="y" * 41)


async def test_read_only_really_blocks_write_edit_and_mutating_bash_but_not_planning() -> None:
    """The preset is AgentScope's EXPLORE mode: the engine denies every call that is not read-only, per invocation for Bash."""
    from types import SimpleNamespace

    from agentscope.permission import (
        PermissionBehavior,
        PermissionContext,
        PermissionEngine,
        PermissionMode,
    )
    from agentscope.tool import Bash, Edit, Write
    from orbit_worker.planning_tools import TaskCreateTool

    engine = PermissionEngine(PermissionContext(mode=PermissionMode.EXPLORE))

    async def verdict(tool, tool_input) -> PermissionBehavior:
        return (await engine.check_permission(tool, tool_input)).behavior

    backend = SimpleNamespace()
    deny = PermissionBehavior.DENY
    assert await verdict(Write(backend=backend), {"file_path": "/workspace/a.py", "content": "x"}) == deny
    assert await verdict(Edit(backend=backend), {"file_path": "/workspace/a.py", "old_string": "a", "new_string": "b"}) == deny
    bash = Bash(cwd="/workspace", backend=backend)
    for command in ("npx create-vite app", "echo x > a.py", "rm -rf app", "ls $(rm -rf /)", "git commit -am x"):
        assert await verdict(bash, {"command": command}) == deny, command
    for command in ("ls -a", "cat a.py", "grep -rn foo ."):
        assert await verdict(bash, {"command": command}) == PermissionBehavior.ALLOW, command
    context = SimpleNamespace(agent=SimpleNamespace(team=_leader().team))
    with patch("orbit_worker.planning_tools.current_task_context", return_value=context):
        assert await verdict(TaskCreateTool(SimpleNamespace()), {"subject": "s", "description": "d"}) == PermissionBehavior.ALLOW


def test_the_leader_is_told_to_write_complete_briefs_and_members_to_reply_short() -> None:
    from orbit_worker.planning_tools import TaskCreateTool
    from orbit_worker.task_stream import TeamTurn
    from orbit_worker.team_tools import TeamAssignTool, stage_prompt

    leader = TeamTurn(role="l", leader=True, leader_role="l", leader_label="", label="", stage_attempt_id="s", members=(("m", "builds"),))
    member = TeamTurn(role="m", leader=False, leader_role="l", leader_label="", label="", stage_attempt_id="s", members=())
    assert "complete, detailed brief" in stage_prompt(leader) and "absolute paths" in stage_prompt(leader)
    assign = TeamAssignTool(leader.members)
    for part in ("Role", "Background", "Inputs", "Existing work", "numbered list", "done criteria"):
        assert part in assign.description or part.lower() in assign.description
    assert "never a different persona" in assign.description
    assert "whole brief" in assign.input_schema["properties"]["task"]["description"]
    assert "never shortened" in TaskCreateTool(None).description  # type: ignore[arg-type]
    assert "## Handover" in stage_prompt(member) and "Needs from the leader" in stage_prompt(member)
    assert "is a claim" in stage_prompt(leader) and "unverified" in stage_prompt(leader)
    assert "call members by their label" in stage_prompt(leader) and "one short message per planning step" in stage_prompt(leader)


def test_a_stage_leader_reads_members_label_first_with_the_id_beside_it() -> None:
    from orbit_worker.task_stream import TeamTurn
    from orbit_worker.team_tools import TeamAssignTool, stage_prompt

    leader = TeamTurn(
        role="l", leader=True, leader_role="l", stage_attempt_id="s",
        members=(("member-2", "builds the API", "后端开发专家"), ("docs", "writes docs")),
    )
    prompt = stage_prompt(leader)
    assert "- 后端开发专家 (id: member-2): builds the API" in prompt and "- docs: writes docs" in prompt
    assert "- 后端开发专家 (id: member-2): builds the API" in TeamAssignTool(leader.members).description
    assert TeamAssignTool(leader.members).input_schema["properties"]["member"]["enum"] == ["member-2", "docs"]


# ---- the leader's own work node executes, it does not plan --------------------------------------------------------


def _node(node_id: str, **fields):
    from orbit_contracts.v3.views import NodeView

    return NodeView(
        node_id=node_id, type="agent_turn", title="t", status="RUNNING", workspace_access="write", owner_profile="writer@1",
        **fields,
    )


class _ViewPlan:
    def __init__(self, *nodes, fail: bool = False) -> None:
        self.nodes, self.fail = nodes, fail

    async def get_plan(self, context):
        from orbit_contracts.v3.views import PlanView

        if self.fail:
            raise RuntimeError("no workflow")
        return PlanView(plan_version=1, hash="sha256:" + "a" * 64, nodes=list(self.nodes))


async def test_only_a_node_nested_under_another_and_not_a_review_is_the_leaders_own_work() -> None:
    from orbit_worker.planning_tools import is_leader_work_node
    from orbit_worker.task_stream import TaskStreamContext

    parent, work, review = "n_01J9Z3K4M5N6P7Q8R9S0T1V2W1", "n_01J9Z3K4M5N6P7Q8R9S0T1V2W2", "n_01J9Z3K4M5N6P7Q8R9S0T1V2W3"
    plan = _ViewPlan(_node(parent), _node(work, parent_node_id=parent), _node(review, review_round=1))

    def context(node: str) -> TaskStreamContext:
        return TaskStreamContext(tenant_id="t", task_id="task", attempt_id="att", activity_attempt=1, node_id=node)

    assert await is_leader_work_node(plan, context(work))
    assert not await is_leader_work_node(plan, context(parent)), "the planning node plans"
    assert not await is_leader_work_node(plan, context(review)), "a review reviews"
    assert not await is_leader_work_node(plan, context("n_01J9Z3K4M5N6P7Q8R9S0T1V2W9")), "a node it cannot see"
    assert not await is_leader_work_node(_ViewPlan(fail=True), context(work)), "an unreadable plan is the leader's own prompt"


def test_the_leaders_own_work_node_is_told_to_execute_not_to_plan() -> None:
    from orbit_worker.agent_config import OWN_TASK_PROMPT

    config = with_task_config(AgentConfig(instructions="Be brief."), _config(team=TEAM), "writer@1", own_task=True)
    assert config.own_task and config.team is not None, "still the leader's expert, so no checklist of its own"
    assert config.instructions == f"Be brief.\n\n{OWN_TASK_PROMPT}"
    assert "Do not create tasks" in config.instructions and "You lead a team" not in config.instructions


async def test_the_leaders_own_work_node_has_no_tool_that_changes_the_plan() -> None:
    from orbit_worker.agent_config import AgentConfig
    from orbit_worker.runtime import AgentRuntime

    runtime = AgentRuntime()
    names = lambda config: {tool.name for tool in runtime._planning_tools_for(config)}
    assert names(AgentConfig(own_task=True)) == {"TaskGet", "TaskList"}
    assert {"TaskCreate", "TaskUpdate", "orbit_declare_unplannable"} <= names(AgentConfig())

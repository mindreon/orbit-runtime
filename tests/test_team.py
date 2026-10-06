"""What a team's leader is told, and how a node is given to a member (15 M8, T8.6).

How it can go wrong, written down before the code:
  - a member's attempt, or an attempt of a task without a team, is shown the roster and starts handing work out;
  - the leader names a role the team does not have and the node is created anyway, or goes to some member by guess;
  - `owner` is given on a task that has no team, or is not a string, and the tool crashes instead of refusing;
  - the refusal does not say which roles exist, so the leader cannot correct itself;
  - a member's description with a line break or a heading breaks the structure of the prompt it is put into;
  - no `owner`, or the leader's own role, lets the leader keep the work instead of giving it to a member;
  - the leader's attempt keeps write permission, or read-only blocks the leader's planning tools too.
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
    assert "review: checks the work" in config.instructions and "lead: plans and writes" in config.instructions
    assert "metadata" in config.instructions and "owner" in config.instructions
    assert "only coordinate" in config.instructions
    assert [member.role for member in config.team.members] == ["lead", "review"]


@pytest.mark.parametrize("profile", ["reviewer@2", "other@1", ""])
def test_a_member_or_a_stranger_is_not_given_the_roster(profile: str) -> None:
    config = with_task_config(AgentConfig(instructions="Be brief."), _config(team=TEAM), profile)
    assert config.team is None and config.instructions == "Be brief."


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


@pytest.mark.parametrize("metadata", [None, {}, {"other": "x"}, {"owner": "lead"}])
def test_in_a_team_a_task_without_a_member_owner_is_refused(metadata) -> None:
    ref, error = owner_profile_for(_leader().team, metadata)
    assert ref is None
    assert "in a team every task goes to a member" in error and "review" in error and "lead" not in error.split("one of")[1]


@pytest.mark.parametrize("metadata", [None, {}, {"other": "x"}])
def test_without_a_team_no_owner_is_no_error(metadata) -> None:
    assert owner_profile_for(None, metadata) == (None, "")


def test_the_leader_runs_read_only_and_nobody_else_does() -> None:
    assert permission_preset_for(_config(), _leader().team) == "read-only"
    assert permission_preset_for(None, _leader().team) == "read-only"
    member = with_task_config(AgentConfig(), _config(team=TEAM), "reviewer@2")
    assert permission_preset_for(_config(), member.team) == "workspace-write"
    assert permission_preset_for(_config()) == "workspace-write"
    assert permission_preset_for(_config(mode="ask")) == "read-only"


def test_the_leader_prompt_says_it_never_implements() -> None:
    prompt = _leader().instructions
    assert "never implement" in prompt and "Every task goes to a member" in prompt and "never to yourself" in prompt
    assert "Integration, documentation, scripts and verification" in prompt and "review turn" in prompt


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
    assert "- review: 审阅员: checks the work" in prompt and "- docs: 文档" in prompt and "- lead\n" in prompt + "\n"
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

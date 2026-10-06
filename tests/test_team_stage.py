"""A team stage (07): a leader and members as activities of one attempt, driven by the workflow.

How it can go wrong, written down before the code:
  - a member runs on the leader's session, or two members share one, or a member's second assignment starts from nothing;
  - the members of one round run one after another (the stage is as slow as its slowest sum), or two assignments for the same
    member run side by side (the member's own session is changed twice at once);
  - the leader is given a member's answer before the others are in, or the answers come back in the order they finished and not
    the order the leader called them, or one of them is lost;
  - a note reaches the member that wrote it and not the others, or reaches nobody, or is shown again and again;
  - a limit (rounds, messages) or the budget does not stop the stage, or stops it without saying which;
  - a member that needs approval parks nobody, or parks the leader's session, or the decision resumes the wrong agent;
  - a worker that died in the middle of a member's turn makes it start again from nothing, or twice;
  - a Continue-As-New between two rounds forgets a session, the mailbox or the answers the leader is owed.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from typing import Any

import pytest
from orbit_contracts import v3
from orbit_contracts.v3 import (
    Actor,
    Budget,
    DecideApprovalInput,
    PlanChangeCommand,
    TaskControlInput,
    TaskWorkflowInput,
)
from orbit_contracts.v3.nodes import TeamStageLimits, TeamStageMember, TeamStageNode, TeamStageSpec
from orbit_contracts.v3.plan import AddNodeOp
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from orbit_orch.team_stage import member_attempt_id
from pydantic import ValidationError
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus, WorkflowUpdateFailedError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker
from waiting import polls

TURNS: list[dict[str, Any]] = []
EVENTS: list[dict[str, Any]] = []
LOG: list[tuple[str, str]] = []
WORKFLOWS: dict[str, str] = {}
# What the agents of the stage do: the leader by the number of its turn, a member by its role.
SCRIPT: dict[str, Any] = {}
GATE = {"started": set(), "release": asyncio.Event()}
FAST = {
    "max_heartbeat_throttle_interval": timedelta(milliseconds=100),
    "default_heartbeat_throttle_interval": timedelta(milliseconds=100),
}
REF = "sha256:" + "1" * 64
TEAM = TeamStageSpec(
    goal="make the release notes",
    members=[
        TeamStageMember(role="lead", executor="writer@1", description="plans and writes"),
        TeamStageMember(role="review", executor="reviewer@1", description="checks the work"),
        TeamStageMember(role="docs", executor="docs@1", description="writes the docs"),
    ],
    leader="lead",
)


def assign(*calls: tuple[str, str], notes: tuple[str, ...] = (), turn: int = 1) -> dict[str, Any]:
    return {
        "status": "team_external", "checkpoint_ref": REF, "session_id": "leader-session", "state_version": turn + 1,
        "notes": list(notes),
        "externals": [
            {"call_id": f"call-{index}", "tool_name": "team_assign", "arguments": {"member": member, "task": task}}
            for index, (member, task) in enumerate(calls)
        ],
    }


def final(text: str) -> dict[str, Any]:
    return {
        "status": "completed", "checkpoint_ref": REF, "session_id": "leader-session", "state_version": 9, "text": text,
        "handover_summary": text, "manifest_id": deterministic_id(f"man:{text}", "man"), "manifest_entries": [],
    }


async def _hold(holding: bool) -> None:
    """Wait for the test to release the gate. Bounded: a test that never releases it (it failed first) must end with its own
    error, not leave the worker waiting for this activity for ever when the test's `async with` closes."""
    for _ in range(1500):  # 30s
        if not holding or GATE["release"].is_set():
            return
        activity.heartbeat()
        await asyncio.sleep(0.02)
    raise AssertionError("the test never released the gate")


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    TURNS.append(payload)
    team = payload.get("team")
    if team is None:
        await _hold(payload["goal"] == "hold")
        return {"status": "completed", "checkpoint_ref": REF, "session_id": "s", "state_version": 2}
    role = str(team["role"])
    if team["leader"]:
        number = 1 + sum(1 for turn in TURNS[:-1] if turn.get("team") and turn["team"]["leader"])
        return SCRIPT["leader"](number, payload)
    LOG.append(("start", role))
    try:
        return await SCRIPT.get(role, _member)(payload)
    finally:
        LOG.append(("end", role))


async def _member(payload: dict[str, Any]) -> dict[str, Any]:
    role = payload["team"]["role"]
    return {
        "status": "completed", "checkpoint_ref": REF, "session_id": payload["attempt_id"],
        "state_version": int(payload["state_version"]) + 1, "text": f"{role} did: {payload['goal'].splitlines()[0]}",
        "usage": {"tokens_in": 10, "tokens_out": 5},
    }


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "failures": [], "workspace_snapshot_ref": REF}


@activity.defn(name="publish_events")
async def _publish_events(payload: list[dict[str, Any]]) -> dict[str, Any]:
    EVENTS.extend(payload)
    return {"ok": True, "count": len(payload)}


@activity.defn(name="checkpoint_commit")
async def _checkpoint_commit(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


COMMITS: list[dict[str, Any]] = []


@activity.defn(name="commit_checkpoints")
async def _commit_checkpoints(payload: dict[str, Any]) -> dict[str, Any]:
    COMMITS.append(payload)
    return {"ok": True, "committed": 1}


def _stack(env: WorkflowEnvironment, runner: Any = None) -> tuple[Worker, Worker, Worker]:
    return (
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=runner or sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn], **FAST),
        Worker(
            env.client, task_queue="orbit.io",
            activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
        ),
    )


def _reset() -> None:
    TURNS.clear()
    EVENTS.clear()
    LOG.clear()
    WORKFLOWS.clear()
    COMMITS.clear()
    SCRIPT.clear()
    GATE["started"] = set()
    GATE["release"] = asyncio.Event()


def _events(kind: str) -> list[dict[str, Any]]:
    return [event["payload"] for event in EVENTS if event["type"] == kind]


def _of(role: str) -> list[dict[str, Any]]:
    return [turn for turn in TURNS if turn.get("team") and turn["team"]["role"] == role]


async def _query(handle, query, *args):
    for _ in range(20):
        try:
            return await handle.query(query, *args, rpc_timeout=timedelta(seconds=3))
        except RPCError:
            continue
    raise AssertionError("the query was never answered")


async def _until(handle, ready, what: str):
    view = plan = None
    async for _ in polls():
        view = await _query(handle, TaskWorkflow.get_task_view)
        plan = await _query(handle, TaskWorkflow.get_plan)
        if ready(view, plan):
            return view, plan
    assert view is not None and plan is not None
    raise AssertionError(f"timed out waiting for {what}: {view.status} {[(n.title, n.status) for n in plan.nodes]} {len(TURNS)} turns")


def _stage(plan):
    return next(node for node in plan.nodes if node.type == "team_stage")


async def _start(
    env: WorkflowEnvironment,
    name: str,
    spec: TeamStageSpec = TEAM,
    *,
    budgets: Budget | None = None,
    goal: str = "explore",
    profile: str = "default@1",
    config: v3.TaskConfig | None = None,
    stage: bool = True,
):
    """A task whose plan is the exploration node and one team stage, added by a person. With a budget the stage is added when
    the exploration is done: it would hold what the stage is to be given."""
    task_id = deterministic_id(f"team:{name}", "task")
    inp = TaskWorkflowInput(
        task_id=task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="user-a"), title="Team", goal=goal,
        profile=profile, node_type_registry_version=1, budgets=budgets or Budget(), **({"config": config} if config else {}),
    )
    handle = await env.client.start_workflow(TaskWorkflow.run, inp, id=f"task/tenant-a/team-{name}", task_queue="orbit.orch")
    WORKFLOWS[task_id] = handle.id
    if not stage:
        return handle
    if budgets is not None:
        await _until(handle, lambda v, p: p.nodes[0].status == "COMPLETED", "the exploration")
    plan = await _query(handle, TaskWorkflow.get_plan)
    result = await handle.execute_update(
        TaskWorkflow.submit_plan_change,
        PlanChangeCommand(
            command_id=hashlib.sha256(f"stage:{name}".encode()).hexdigest(), task_id=task_id, base_plan_version=plan.plan_version,
            actor=Actor(kind="user", id="user-a"),
            ops=[AddNodeOp(node=TeamStageNode(node_id="tmp:1", title="Stage", spec=spec))],
        ),
    )
    assert result.status == "accepted", result
    return handle


async def _done(handle, what: str = "the stage"):
    return await _until(handle, lambda v, p: _stage(p).status == "COMPLETED", what)


async def _finish(handle) -> None:
    await handle.execute_update(
        TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"cancel").hexdigest(), action="cancel")
    )
    async for _ in polls():
        if (await handle.describe()).status != WorkflowExecutionStatus.RUNNING:
            return
    raise AssertionError("the workflow did not end")


# ---- the spec ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"members": [TEAM.members[0]]},  # a leader alone is no team
        {"leader": "nobody"},
        {"members": [TEAM.members[0], TEAM.members[0]]},
        {"limits": TeamStageLimits(max_members=2)},  # three members, room for two
    ],
)
def test_a_team_stage_must_be_a_team(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TeamStageSpec(**{**TEAM.model_dump(), **changes})


@pytest.mark.parametrize("limits", [{"max_members": 9}, {"max_rounds": 31}, {"max_messages": 201}, {"max_rounds": 0}])
def test_the_limits_have_ceilings(limits: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        TeamStageLimits(**limits)
    assert (TeamStageLimits().max_members, TeamStageLimits().max_rounds, TeamStageLimits().max_messages) == (4, 10, 100)


# ---- the protocol -----------------------------------------------------------------------------------------------------


async def test_the_leader_assigns_two_members_and_merges_their_answers() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: (
        assign(("review", "check the notes"), ("docs", "write the docs"))
        if n == 1
        else final("merged: " + " | ".join(item["output"] for item in payload["team_results"]))
    )
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "pair")
            _, plan = await _done(handle)
            await _finish(handle)
    leader, review, docs = _of("lead"), _of("review"), _of("docs")
    assert len(leader) == 2 and len(review) == 1 and len(docs) == 1
    stage_attempt = leader[0]["attempt_id"]
    assert (review[0]["profile"], docs[0]["profile"], leader[0]["profile"]) == ("reviewer@1", "docs@1", "writer@1"), "each runs as its executor"
    assert review[0]["attempt_id"] == member_attempt_id(stage_attempt, "review") != docs[0]["attempt_id"], "each has a session of its own"
    assert review[0]["goal"] == "check the notes" and docs[0]["goal"] == "write the docs"
    assert (leader[0]["workspace_access"], review[0]["workspace_access"]) == ("write", "none"), "only the leader writes the workspace"
    assert review[0]["team"]["workspace"] == "read" and review[0]["team"]["leader"] is False
    assert [item[0] for item in leader[0]["team"]["members"]] == ["review", "docs"], "the leader may assign to the others, not itself"
    assert leader[0]["config"]["team"] is None, "the agents of a stage are not shown the plan's team"
    # The answers come back together, in the order the leader called, to the session it had.
    resumed = leader[1]
    assert [item["call_id"] for item in resumed["team_results"]] == ["call-0", "call-1"]
    assert resumed["team_results"][0]["output"] == "review did: check the notes"
    assert resumed["team_results"][1]["output"] == "docs did: write the docs"
    assert (resumed["session_id"], resumed["state_version"]) == ("leader-session", 2)
    assert plan.nodes[-1].status == "COMPLETED" or _stage(plan).status == "COMPLETED"
    finished = [p for p in _events("attempt.finished") if p["attempt_id"] == stage_attempt]
    assert [p["outcome"] for p in finished] == ["completed"]
    assert finished[0]["usage"]["tokens_in"] == 20, "what the members spent is the attempt's"
    committed = {item["attempt_id"] for item in COMMITS}
    assert {member_attempt_id(stage_attempt, "review"), member_attempt_id(stage_attempt, "docs"), stage_attempt} <= committed, (
        "each session's checkpoints are committed under the id they are stored under"
    )


async def test_assignments_to_one_member_run_one_after_the_other_on_its_own_session() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: (
        assign(("review", "first"), ("review", "second"), ("docs", "other")) if n == 1 else final("ok")
    )

    async def slow_review(payload: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return await _member(payload)

    SCRIPT["review"] = slow_review
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "serial")
            await _done(handle)
            await _finish(handle)
    review = [(kind, role) for kind, role in LOG if role == "review"]
    assert review == [("start", "review"), ("end", "review"), ("start", "review"), ("end", "review")], "never two at once"
    first, second = _of("review")
    assert first["goal"] == "first" and second["goal"] == "second", "in the order the leader called"
    assert second["session_id"] == first["attempt_id"] and second["state_version"] == 1, "the second carries on the first's session"
    resumed = _of("lead")[1]["team_results"]
    assert [item["output"] for item in resumed] == ["review did: first", "review did: second", "docs did: other"]


async def test_members_of_one_round_run_side_by_side() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a"), ("docs", "b")) if n == 1 else final("ok")

    async def together(payload: dict[str, Any]) -> dict[str, Any]:
        # Each waits until the other has started: one after the other, this never ends.
        GATE["started"].add(payload["team"]["role"])
        for _ in range(500):
            if GATE["started"] >= {"review", "docs"}:
                break
            activity.heartbeat()
            await asyncio.sleep(0.02)
        assert GATE["started"] >= {"review", "docs"}, "the other member was never started"
        return await _member(payload)

    SCRIPT["review"] = SCRIPT["docs"] = together
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "side-by-side")
            await _done(handle)
            await _finish(handle)
    starts = [index for index, (kind, _) in enumerate(LOG) if kind == "start"]
    ends = [index for index, (kind, _) in enumerate(LOG) if kind == "end"]
    assert max(starts) < min(ends)


async def test_a_note_reaches_everyone_but_its_author_once() -> None:
    _reset()

    def leader(n: int, payload: dict[str, Any]) -> dict[str, Any]:
        if n == 1:
            return assign(("review", "a"), ("docs", "b"), notes=("kickoff: use the short form",))
        if n == 2:
            return assign(("docs", "c"), turn=2)
        return final("ok")

    async def noting_review(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["the changelog is stale"]}

    SCRIPT["leader"] = leader
    SCRIPT["review"] = noting_review
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "mailbox")
            await _done(handle)
            await _finish(handle)
    review, docs = _of("review")[0], _of("docs")
    assert "团队消息:\n- [领队] kickoff: use the short form" in review["goal"], "the leader's note is in the member's input"
    assert "团队消息:\n- [领队] kickoff: use the short form" in docs[0]["goal"]
    assert "[review]" not in review["goal"] and "[review]" not in docs[0]["goal"] and "[lead]" not in docs[0]["goal"], "a note is posted after the turn that wrote it"
    second = _of("lead")[1]["team_results"][-1]["output"]
    assert "团队消息:\n- [review] the changelog is stale" in second, "the leader is shown what the members posted, with its answers"
    assert "kickoff" not in second, "and not what it wrote itself"
    assert docs[1]["goal"].splitlines()[0] == "c" and "- [review] the changelog is stale" in docs[1]["goal"], "the member's next turn too"
    assert "kickoff" not in docs[1]["goal"] and "[review] the changelog is stale" not in review["goal"]
    notes = [n for n in _events("team.message") if n["kind"] == "note"]
    assert [(n["role"], n["from_role"]) for n in notes] == [("lead", "lead"), ("review", "review")]
    seqs = [n["seq"] for n in _events("team.message")]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "every message of the stage is numbered in order"
    versions = [e["entity"]["version"] for e in EVENTS if e["type"].startswith("team.")]
    assert versions == sorted(versions) and len(set(versions)) == len(versions), "one entity, one increasing version"
    assert all(e["entity"]["kind"] == "team" for e in EVENTS if e["type"].startswith("team."))


async def test_the_events_say_round_by_round_and_member_by_member() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "check the notes")) if n == 1 else final("ok")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "events")
            await _done(handle)
            await _finish(handle)
    rounds = [(e["round"], e.get("outcome")) for e in _events("team.round_started") + _events("team.round_finished")]
    assert sorted(rounds, key=lambda item: (item[0], item[1] or "")) == [(1, None), (1, "assigned"), (2, None), (2, "completed")]
    started, finished = _events("team.member_turn_started")[0], _events("team.member_turn_finished")[0]
    assert (started["role"], started["executor"], started["task"]) == ("review", "reviewer@1", "check the notes")
    assert finished["outcome"] == "completed" and finished["summary"] == "review did: check the notes"
    assert finished["usage"]["tokens_in"] == 10 and started["member_attempt_id"] == finished["member_attempt_id"]
    assert _events("team.round_finished")[0]["assignments"] == 1


# ---- limits and budget ------------------------------------------------------------------------------------------------


async def test_a_leader_that_never_finishes_is_stopped_at_the_round_limit_with_the_reason() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", f"again {n}"), turn=n)
    spec = TEAM.model_copy(update={"limits": TeamStageLimits(max_rounds=2)})
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "rounds", spec)
            _view, plan = await _until(handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the task to ask for a review")
            await _finish(handle)
    assert _stage(plan).status == "BLOCKED"
    assert len(_of("lead")) == 2 and len(_of("review")) == 1, "the work the leader had no round left for was not started"
    failure = next(p for p in _events("attempt.finished") if p["outcome"] == "failed")["failure"]
    assert failure["failure_class"] == "policy" and failure["retryable"] is False
    assert "limit of 2 rounds" in failure["message"]
    reasons = [p["reason"] for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"]
    assert "limit of 2 rounds" in reasons[0]
    assert [p["outcome"] for p in _events("team.round_finished")] == ["assigned", "stopped"]


async def test_too_many_messages_stop_the_stage_before_the_work_that_would_pass_the_limit() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a"), ("docs", "b"))
    spec = TEAM.model_copy(update={"limits": TeamStageLimits(max_messages=3)})
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "messages", spec)
            await _until(handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the task to ask for a review")
            await _finish(handle)
    assert not _of("review") and not _of("docs")
    failure = next(p for p in _events("attempt.finished") if p["outcome"] == "failed")["failure"]
    assert "limit of 3 messages" in failure["message"] and failure["failure_class"] == "policy"


async def test_the_stage_stops_when_the_budget_reserved_for_it_is_spent() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a"), ("docs", "b")) if n == 1 else final("ok")

    async def spends(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "usage": {"tokens_in": 1000, "tokens_out": 500}}

    SCRIPT["review"] = SCRIPT["docs"] = spends
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "budget", budgets=Budget(tokens=3000))
            view, _plan = await _until(handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the task to ask for a review")
            await _finish(handle)
    leader = _of("lead")[0]["budget"]["tokens"]
    assert _of("review")[0]["budget"]["tokens"] == _of("docs")[0]["budget"]["tokens"] == leader // 2, "members side by side share it"
    failure = next(p for p in _events("attempt.finished") if p["outcome"] == "failed")["failure"]
    assert failure["failure_class"] == "budget" and failure["retryable"] is False and "budget is spent" in failure["message"]
    assert view.usage.tokens_in + view.usage.tokens_out == 3000, "what the members spent is the task's"
    assert len(_of("lead")) == 1, "no round starts on a spent reservation"


async def test_a_member_that_runs_out_of_budget_ends_the_stage_as_a_budget_failure() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a")) if n == 1 else final("ok")

    async def broke(payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "failed", "error": "the token budget is spent", "failure_class": "budget", "retryable": False}

    SCRIPT["review"] = broke
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "member-budget", budgets=Budget(tokens=3000))
            await _until(handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the task to ask for a review")
            await _finish(handle)
    failure = next(p for p in _events("attempt.finished") if p["outcome"] == "failed")["failure"]
    assert failure["failure_class"] == "budget" and len(_of("lead")) == 1


async def test_a_member_that_fails_is_reported_to_the_leader_which_decides() -> None:
    _reset()

    async def failing(payload: dict[str, Any]) -> dict[str, Any]:
        return {"status": "failed", "error": "no network", "failure_class": "tool", "retryable": False}

    SCRIPT["leader"] = lambda n, payload: assign(("review", "a")) if n == 1 else final("worked around it")
    SCRIPT["review"] = failing
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "member-fails")
            await _done(handle)
            await _finish(handle)
    result = _of("lead")[1]["team_results"][0]
    assert result["result_state"] == "error" and "no network" in result["output"]


async def test_an_assignment_to_a_member_that_does_not_exist_is_an_error_result_not_a_stop() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("ghost", "a"), ("docs", "b")) if n == 1 else final("ok")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "ghost")
            await _done(handle)
            await _finish(handle)
    results = _of("lead")[1]["team_results"]
    assert results[0]["result_state"] == "error" and "ghost" in results[0]["output"] and "review" in results[0]["output"]
    assert results[1]["result_state"] == "success"


# ---- people -----------------------------------------------------------------------------------------------------------


async def test_a_member_that_needs_approval_parks_the_attempt_and_the_decision_resumes_that_member_only() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "needs a command"), ("docs", "b")) if n == 1 else final("ok")

    async def asking(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("approval") is None:
            return {
                "status": "parked_approval", "checkpoint_ref": REF, "session_id": payload["attempt_id"], "state_version": 2,
                "approval_request_id": "apr-call-1",
                "approvals": [{"tool_call_id": "call-1", "subject": {
                    "kind": "tool_call", "digest": REF, "summary": "Bash", "risk": "medium", "detail": "make build",
                }}],
            }
        return {**await _member(payload), "text": "built it", "state_version": 3}

    SCRIPT["review"] = asking
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "approval")
            view, plan = await _until(handle, lambda v, p: bool(v.pending_approvals), "the approval")
            await _until(handle, lambda v, p: bool(_of("docs")) and ("end", "docs") in LOG, "the other member to finish meanwhile")
            assert len(_of("lead")) == 1, "the leader waits for the member that is parked"
            assert _stage(plan).status == "AWAITING_APPROVAL"
            requested = _events("approval.requested")[0]
            assert requested["tool_call_id"] == "review:call-1"
            assert requested["subject"]["role"] == "review" and requested["subject"]["summary"] == "review: Bash"
            assert requested["subject"]["detail"] == "make build"
            parked = [p for p in _events("attempt.parked")]
            assert [p["reason"] for p in parked] == ["approval"]
            await handle.execute_update(
                TaskWorkflow.decide_approval,
                DecideApprovalInput(command_id="01J00000000000000000000050", approval_id=view.pending_approvals[0], decision="approve"),
            )
            await _done(handle)
            await _finish(handle)
    first, resumed = _of("review")
    assert resumed["approval"]["decision"] == "approve" and resumed["approval"]["decisions"] == {"call-1": "approve"}
    assert (resumed["session_id"], resumed["state_version"]) == (first["attempt_id"], 2), "the same member, on its own session"
    assert len(_of("docs")) == 1, "the other member was not run again"
    assert [item["output"] for item in _of("lead")[1]["team_results"]] == ["built it", "docs did: b"]


async def test_a_member_that_asks_the_user_parks_the_attempt_and_the_answer_resumes_it() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "ask")) if n == 1 else final("ok")

    async def asking(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("team_results"):
            return {
                "status": "team_external", "checkpoint_ref": REF, "session_id": payload["attempt_id"], "state_version": 2,
                "externals": [{"call_id": "call-ask", "tool_name": "ask_user", "arguments": {"question": "Which branch?", "questions": [{"header": "Branch", "question": "Which one?", "options": [{"label": "main"}, {"label": "dev"}]}]}}],
            }
        return {**await _member(payload), "text": "used " + payload["team_results"][0]["output"], "state_version": 3}

    SCRIPT["review"] = asking
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "ask")
            await _until(handle, lambda v, p: _stage(p).status == "AWAITING_INPUT", "the question")
            assert [p["question"] for p in _events("attempt.parked")] == ["[review] Which branch?\n1. Which one?（main / dev）"]
            assert [p["questions"][0]["header"] for p in _events("attempt.parked")] == ["Branch"]
            await handle.execute_update(
                TaskWorkflow.send_message,
                v3.SendMessageInput(command_id="01J00000000000000000000051", client_message_id="01J00000000000000000000052", text="main"),
            )
            await _done(handle)
            await _finish(handle)
    assert _of("lead")[1]["team_results"][0]["output"] == "used main"


# ---- crashes and Continue-As-New --------------------------------------------------------------------------------------


async def test_a_member_activity_that_crashes_is_retried_on_the_same_attempt_and_session() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a")) if n == 1 else final("ok")

    async def crashing(payload: dict[str, Any]) -> dict[str, Any]:
        if activity.info().attempt == 1:
            raise RuntimeError("the worker died")
        return await _member(payload)

    SCRIPT["review"] = crashing
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "crash")
            await _done(handle)
            await _finish(handle)
    first, second = _of("review")
    assert first == second, "the retry is given exactly what the first try was: the member's attempt, session and state"
    assert len(_events("team.member_turn_started")) == 1 and len(_events("team.member_turn_finished")) == 1
    assert _of("lead")[1]["team_results"][0]["output"] == "review did: a"


async def test_a_continue_as_new_between_rounds_carries_the_sessions_the_mailbox_and_the_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset()

    def leader(n: int, payload: dict[str, Any]) -> dict[str, Any]:
        return assign(("review", "a"), ("docs", "b"), notes=("kickoff",)) if n == 1 else final("ok")

    async def noting(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["a finding"]}

    SCRIPT["leader"] = leader
    SCRIPT["review"] = noting
    # `is_continue_as_new_suggested` turns true after thousands of events; here it is true whenever it is asked.
    monkeypatch.setattr(AttemptWorkflow, "_should_continue_as_new", lambda self: True)
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env, UnsandboxedWorkflowRunner())
        async with orch, agent, io:
            handle = await _start(env, "can")
            view, plan = await _done(handle)
            stage_attempt = _of("lead")[0]["attempt_id"]
            history = await env.client.get_workflow_handle(f"attempt/{view.task_id}/{_stage(plan).node_id}/1").fetch_history()
            await _finish(handle)
    assert history.events[0].workflow_execution_started_event_attributes.continued_execution_run_id, "it continued as new"
    resumed = _of("lead")[1]
    assert (resumed["session_id"], resumed["state_version"]) == ("leader-session", 2), "the leader's session was carried"
    assert [item["call_id"] for item in resumed["team_results"]] == ["call-0", "call-1"], "and the answers it was owed"
    assert "团队消息:\n- [review] a finding" in resumed["team_results"][-1]["output"], "and the mailbox (the leader's note is its own)"
    assert [p["round"] for p in _events("team.round_started")] == [1, 2], "the round counter went on, no round was started twice"
    assert len(_events("team.member_turn_started")) == 2, "no member was run again"
    assert resumed["attempt_id"] == stage_attempt
    versions = [e["entity"]["version"] for e in EVENTS if e["type"].startswith("team.")]
    assert versions == sorted(versions) and len(set(versions)) == len(versions), "the events' version went on across the run"


async def test_a_retried_stage_carries_on_the_leaders_and_the_members_sessions() -> None:
    _reset()
    state = {"failed": False}

    def leader(n: int, payload: dict[str, Any]) -> dict[str, Any]:
        if n == 1 and not state["failed"]:
            state["failed"] = True
            return {"status": "failed", "error": "model overloaded", "failure_class": "model", "retryable": True}
        return assign(("review", "a")) if n <= 2 else final("ok")

    SCRIPT["leader"] = leader
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "retry")
            await _until(handle, lambda v, p: _stage(p).status in {"RETRY_PENDING", "RUNNING"} and len(_of("lead")) >= 1, "the failure")
            await env.sleep(timedelta(seconds=60))
            await _done(handle)
            await _finish(handle)
    first_leader, second_leader = _of("lead")[0], _of("lead")[1]
    assert second_leader["attempt_no"] == 2 and second_leader["continue_from"] == first_leader["attempt_id"]
    review = _of("review")[0]
    assert review["attempt_no"] == 2
    assert review["continue_from"] == member_attempt_id(first_leader["attempt_id"], "review"), "a member carries on its session of the attempt before"


# ---- who may create one -----------------------------------------------------------------------------------------------


def _plan_command(plan, task_id: str, actor: Actor, spec: TeamStageSpec, salt: str) -> PlanChangeCommand:
    return PlanChangeCommand(
        command_id=hashlib.sha256(f"{task_id}:{salt}".encode()).hexdigest(), task_id=task_id, base_plan_version=getattr(plan, "plan_version", None) or plan.version,
        actor=actor, ops=[AddNodeOp(node=TeamStageNode(node_id="tmp:1", title="Stage", spec=spec))],
    )


def test_the_registry_defaults_of_a_team_stage_and_the_rule_for_agents() -> None:
    from orbit_orch.plan_engine import (
        NODE_TYPES,
        NODE_TYPES_WITH_TEAM,
        PlanPolicy,
        apply,
        initial_plan,
    )

    task_id = deterministic_id("team:rules", "task")
    plan = initial_plan(task_id, "goal", "writer@1")
    person = Actor(kind="user", id="u")
    agent = Actor(kind="agent", id="att", attempt_id=deterministic_id("team:rules:att", "att"))
    active = frozenset({agent.attempt_id})

    def judged(actor: Actor, policy: PlanPolicy):
        return apply(plan, _plan_command(plan, task_id, actor, TEAM, actor.kind), policy).result

    assert judged(person, PlanPolicy(allowed_node_types=NODE_TYPES)).code == "TYPE_NOT_ALLOWED", "off until the workflow enables it"
    enabled = PlanPolicy(allowed_node_types=NODE_TYPES_WITH_TEAM, active_attempt_ids=active)
    outcome = apply(plan, _plan_command(plan, task_id, person, TEAM, "p"), enabled)
    assert outcome.result.status == "accepted"
    node = outcome.plan.nodes[outcome.result.id_map["tmp:1"]].draft
    assert (node.workspace_access, node.owner_profile) == ("write", "writer@1"), "the leader writes the workspace, as its expert"
    assert judged(agent, enabled).code == "TYPE_NOT_ALLOWED", "an agent without a team has no stage to make"
    teams = PlanPolicy(allowed_node_types=NODE_TYPES_WITH_TEAM, active_attempt_ids=active, agent_team_experts=frozenset({"writer@1", "reviewer@1", "docs@1"}))
    assert judged(agent, teams).status == "accepted"
    strangers = PlanPolicy(allowed_node_types=NODE_TYPES_WITH_TEAM, active_attempt_ids=active, agent_team_experts=frozenset({"writer@1"}))
    refused = judged(agent, strangers)
    assert refused.code == "POLICY_DENIED" and "reviewer@1" in refused.detail and "docs@1" in refused.detail


async def test_an_agent_creates_a_team_stage_only_in_a_task_with_a_team_and_only_with_that_teams_experts() -> None:
    _reset()
    team = v3.Team(
        leader="lead",
        members=[v3.TeamMember(role="lead", expert="writer@1"), v3.TeamMember(role="review", expert="reviewer@1")],
    )
    config = v3.TaskConfig(expert="writer@1", team=team)
    spec = TeamStageSpec(
        goal="g", leader="lead",
        members=[TeamStageMember(role="lead", executor="writer@1"), TeamStageMember(role="review", executor="reviewer@1")],
    )
    stranger = spec.model_copy(update={"members": [spec.members[0], TeamStageMember(role="review", executor="stranger@1")]})
    SCRIPT["leader"] = lambda n, payload: final("the stage's answer")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            with_team = await _start(env, "agent-team", goal="hold", profile="writer@1", config=config, stage=False)
            without = await _start(env, "agent-no-team", goal="hold", stage=False)
            answers = {}
            for name, handle, candidate in (("team", with_team, spec), ("stranger", with_team, stranger), ("none", without, spec)):
                _, plan = await _until(handle, lambda v, p: p.nodes[0].status == "RUNNING", "the exploration")
                task_id = (await _query(handle, TaskWorkflow.get_task_view)).task_id
                attempt = plan.nodes[0].current_attempt_id
                actor = Actor(kind="agent", id=attempt, attempt_id=attempt)
                answers[name] = await handle.execute_update(
                    TaskWorkflow.submit_plan_change, _plan_command(plan, task_id, actor, candidate, name)
                )
            GATE["release"].set()
            # The stage that was accepted really runs: let it end before the task is cancelled, or the workers are closed while a
            # cancelled stage attempt still waits for its activity to confirm the cancel.
            await _until(with_team, lambda v, p: all(n.status == "COMPLETED" for n in p.nodes), "the accepted stage to finish")
            await _until(without, lambda v, p: all(n.status == "COMPLETED" for n in p.nodes), "the exploration to finish")
            await _finish(with_team)
            await _finish(without)
    assert answers["team"].status == "accepted"
    assert answers["stranger"].code == "POLICY_DENIED"
    assert answers["none"].code == "TYPE_NOT_ALLOWED"


async def test_nodes_say_their_stage_limits_and_events_carry_labels_and_counters() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a"), notes=("kickoff",)) if n == 1 else final("ok")
    spec = TEAM.model_copy(update={
        "members": [TEAM.members[0], TEAM.members[1].model_copy(update={"label": "审阅员"}), TEAM.members[2]],
        "limits": TeamStageLimits(max_members=3, max_rounds=6, max_messages=50, max_hops=2),
    })
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "facts", spec)
            _, plan = await _done(handle)
            await _finish(handle)
    stage = _stage(plan)
    assert stage.team is not None and (stage.team.max_members, stage.team.max_rounds, stage.team.max_messages, stage.team.max_hops) == (3, 6, 50, 2)
    assert stage.review_round is None and all(n.team is None for n in plan.nodes if n.type != "team_stage")
    changed = [p for p in _events("node.status_changed") if p["node_id"] == stage.node_id]
    assert changed and all(p["team"] == {"max_members": 3, "max_rounds": 6, "max_messages": 50, "max_hops": 2} for p in changed)
    first = _events("team.round_started")[0]
    assert (first["max_rounds"], first["max_messages"], first["max_members"], first["max_hops"], first["messages"]) == (6, 50, 3, 2, 0)
    assert [p["messages"] for p in _events("team.round_finished")] == [3, 3]  # a note, an assignment, an answer
    assert _events("team.member_turn_started")[0]["label"] == "审阅员" and _events("team.member_turn_finished")[0]["label"] == "审阅员"
    assert "label" not in next(n for n in _events("team.message") if n["from_role"] == "lead"), "the leader has no label"
    assert "审阅员" in _of("lead")[0]["team"]["members"][0][1] or _of("lead")[0]["team"]["members"][0][2] == "审阅员"


# ---- @mentions (07 §6a) -----------------------------------------------------------------------------------------------

MENTION_TEAM = TEAM.model_copy(update={
    "members": [TEAM.members[0], TEAM.members[1].model_copy(update={"label": "审阅员"}), TEAM.members[2]],
    "limits": TeamStageLimits(max_hops=2),
})


def _messages(kind: str | None = None) -> list[dict[str, Any]]:
    return [p for p in _events("team.message") if kind is None or p["kind"] == kind]


TASK_TEAM = v3.Team(
    leader="lead",
    members=[
        v3.TeamMember(role="lead", expert="writer@1", label="组长"),
        v3.TeamMember(role="review", expert="reviewer@1", label="审阅员"),
        v3.TeamMember(role="docs", expert="docs@1"),
    ],
)


async def _mentions_stack(env, name: str, spec: TeamStageSpec = MENTION_TEAM):
    return await _start(env, name, spec, config=v3.TaskConfig(expert="writer@1", team=TASK_TEAM), profile="writer@1")


async def test_a_member_that_mentions_another_wakes_it_and_the_reply_goes_to_the_group() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "check it")) if n == 1 else final("ok")

    async def mentioning(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["@docs please update the changelog"], "note_mentions": [[]]}

    SCRIPT["review"] = mentioning
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _mentions_stack(env, "wake")
            await _done(handle)
            await _finish(handle)
    woken = _of("docs")
    assert len(woken) == 1 and woken[0]["goal"].startswith("团队成员 审阅员 在群里 @ 了你: @docs please update the changelog")
    assert woken[0]["team"]["role"] == "docs" and woken[0]["attempt_id"] == member_attempt_id(_of("lead")[0]["attempt_id"], "docs")
    note = next(m for m in _messages("note") if m["from_role"] == "review")
    assert note["to_roles"] == ["docs"] and note["hop"] == 0 and note["from_label"] == "审阅员"
    reply = next(m for m in _messages("reply") if m["from_role"] == "docs")
    assert reply["to_roles"] == ["review"] and reply["hop"] == 1, "the woken member answers whoever woke it"
    assert [m["kind"] for m in _messages() if m["from_role"] == "lead"] == ["assign", "review"]
    leader_input = _of("lead")[1]["team_results"][0]["output"]
    assert "[docs] docs did:" in leader_input, "the leader sees the woken member's answer at its next turn"


async def test_mentions_by_list_label_and_text_wake_each_named_member_once_and_never_the_speaker() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a")) if n == 1 else final("ok")

    async def noting(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["@review @docs @docs hello", "no one here"], "note_mentions": [["docs"], []]}

    SCRIPT["review"] = noting
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _mentions_stack(env, "mention-forms")
            await _done(handle)
            await _finish(handle)
    assert len(_of("docs")) == 1, "the explicit list wins, and a role is woken once per note"
    assert len(_of("review")) == 1, "nobody wakes itself"
    first, second = _messages("note")[:2]
    assert first["to_roles"] == ["docs"] and second["to_roles"] == [], "no mention is the whole group"


async def test_woken_members_run_side_by_side_and_one_member_woken_twice_runs_serially() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a")) if n == 1 else final("ok")

    async def noting(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["one", "two"], "note_mentions": [["docs"], ["docs"]]}

    async def slow_docs(payload: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return await _member(payload)

    SCRIPT["review"], SCRIPT["docs"] = noting, slow_docs
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _mentions_stack(env, "serial-wakes")
            await _done(handle)
            await _finish(handle)
    docs = [(kind, role) for kind, role in LOG if role == "docs"]
    assert docs == [("start", "docs"), ("end", "docs"), ("start", "docs"), ("end", "docs")], "the same member never runs twice at once"
    first, second = _of("docs")
    assert second["session_id"] == first["attempt_id"], "the second wake carries on the member's session"


async def test_a_chain_of_wakes_stops_at_max_hops_and_the_group_is_told_why() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "ping-pong start")) if n == 1 else final("ok")

    async def ping(payload: dict[str, Any]) -> dict[str, Any]:
        role = payload["team"]["role"]
        other = "docs" if role == "review" else "review"
        return {**await _member(payload), "notes": [f"@{other} ping"], "note_mentions": [[other]]}

    SCRIPT["review"] = SCRIPT["docs"] = ping
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _mentions_stack(env, "hops")
            await _done(handle)
            await _finish(handle)
    assert len(_of("review")) + len(_of("docs")) == 3, "the assignment and two wakes (max_hops 2); the third is refused"
    refusal = _messages("system")
    assert len(refusal) == 1 and "limit is 2" in refusal[0]["text"] and refusal[0]["from_role"] == "system"
    assert max(m["hop"] for m in _messages("reply")) == 2
    assert _events("team.round_started")[0]["max_hops"] == 2


async def test_a_wake_that_would_pass_the_message_limit_is_refused_in_the_group() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a")) if n == 1 else final("ok")

    async def noting(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["@docs now"], "note_mentions": [["docs"]]}

    SCRIPT["review"] = noting
    spec = MENTION_TEAM.model_copy(update={"limits": TeamStageLimits(max_messages=3, max_hops=2)})
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _mentions_stack(env, "wake-limit", spec)
            await _until(handle, lambda v, p: v.status in {"PAUSED_NEEDS_REVIEW", "COMPLETED"}, "the stage to end")
            await _finish(handle)
    assert not _of("docs"), "nobody was woken past the limit"
    assert any("limit of 3 messages" in m["text"] for m in _messages("system"))


async def test_the_user_mentioning_a_member_in_a_running_stage_wakes_it_and_the_leader_sees_it() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "hold")) if n == 1 else final("ok")

    async def holding(payload: dict[str, Any]) -> dict[str, Any]:
        await _hold(payload["goal"].startswith("hold"))
        return await _member(payload)

    SCRIPT["review"] = holding
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _mentions_stack(env, "user-mention")
            await _until(handle, lambda v, p: ("start", "review") in LOG, "the member to be working")
            await handle.execute_update(
                TaskWorkflow.send_message,
                v3.SendMessageInput(
                    command_id="01J00000000000000000000070", client_message_id="01J00000000000000000000071",
                    text="@docs please also write the intro", mentions=["docs"],
                ),
            )
            await _until(handle, lambda v, p: ("end", "docs") in LOG, "docs to answer the user")
            GATE["release"].set()
            await _done(handle)
            await _finish(handle)
    woken = _of("docs")
    assert len(woken) == 1 and "@docs please also write the intro" in woken[0]["goal"]
    user = _messages("user")[0]
    assert user["from_role"] == "user" and user["to_roles"] == ["docs"] and user["hop"] == 0
    assert _messages("reply")[0]["from_role"] == "docs" and _messages("reply")[0]["to_roles"] == ["user"] or any(
        m["from_role"] == "docs" and m["to_roles"] == ["user"] for m in _messages("reply")
    )
    seen = _of("lead")[1]["team_results"][-1]["output"]
    assert "[用户] @docs please also write the intro" in seen, "the leader reads the user's words from the mailbox"
    assert len(_of("lead")) == 2, "the message was the member's: the leader was not given it as its own"


async def test_events_of_concurrent_members_say_who_made_them() -> None:
    from orbit_contracts.models import OrbitEvent
    from orbit_worker.task_stream import TaskStreamContext, TeamTurn, to_v3

    def event(kind: str, **fields: Any) -> OrbitEvent:
        return OrbitEvent(type=kind, session_id="s", room_id="task-1", turn_id="t", **fields)  # type: ignore[arg-type]

    stage = "att_STAGE"
    contexts = [
        TaskStreamContext(
            tenant_id="t", task_id="task-1", attempt_id=member_attempt_id(stage, role), activity_attempt=1, node_id="n",
            team=TeamTurn(role=role, leader=False, leader_role="lead", label=label, stage_attempt_id=stage),
        )
        for role, label in (("review", "审阅员"), ("docs", ""))
    ]
    outs = [
        to_v3(event("assistant.delta", delta="x", block_id="b", seq=1), contexts[0]),
        to_v3(event("assistant.delta", delta="y", block_id="b", seq=1), contexts[1]),
        to_v3(event("tool.call", call_id="c1", tool_name="Bash", args_preview="ls"), contexts[0]),
    ]
    assert [(o["payload"]["attempt_id"], o["entity"]["id"]) for o in outs] == [(stage, stage)] * 3, "what a consumer knows"
    assert [o["payload"]["team_role"] for o in outs] == ["review", "docs", "review"]
    assert outs[0]["payload"]["team_label"] == "审阅员" and "team_label" not in outs[1]["payload"]
    assert outs[0]["payload"]["team_session"] == member_attempt_id(stage, "review") != outs[1]["payload"]["team_session"]
    assert outs[0]["event_id"] != outs[1]["event_id"], "two members' deltas of the same block are two events"
    plain = TaskStreamContext(tenant_id="t", task_id="task-1", attempt_id="att_PLAIN", activity_attempt=1)
    assert "team_role" not in to_v3(event("assistant.delta", delta="z", block_id="b", seq=1), plain)["payload"]


async def test_a_continue_as_new_carries_the_wakes_and_the_message_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a")) if n == 1 else final("ok")

    async def noting(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["@docs go"], "note_mentions": [["docs"]]}

    SCRIPT["review"] = noting
    monkeypatch.setattr(AttemptWorkflow, "_should_continue_as_new", lambda self: True)
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env, UnsandboxedWorkflowRunner())
        async with orch, agent, io:
            handle = await _mentions_stack(env, "can-wake")
            await _done(handle)
            await _finish(handle)
    assert len(_of("docs")) == 1, "woken once, before the round ended"
    seqs = [m["seq"] for m in _messages()]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "the message numbers went on across the continue-as-new"
    assert _of("lead")[1]["team_results"][0]["output"].count("[docs]") >= 1


# ---- the group at plan level (07 §5, §6b) -----------------------------------------------------------------------------


async def _team_task(env, name: str, goal: str = "explore"):
    """A task with a team whose exploration node is the leader's; nodes are added as the test needs them."""
    return await _start(
        env, name, goal=goal, profile="writer@1", config=v3.TaskConfig(expert="writer@1", team=TASK_TEAM), stage=False
    )


async def test_a_message_that_mentions_a_member_becomes_a_node_the_member_owns_and_carries_on_its_session() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _team_task(env, "plan-mention")
            await _until(handle, lambda v, p: v.status == "COMPLETED", "the first round")
            say = lambda n, text, mentions: handle.execute_update(
                TaskWorkflow.send_message,
                v3.SendMessageInput(
                    command_id=f"01J0000000000000000000008{n}", client_message_id=f"01J1000000000000000000008{n}", text=text, mentions=mentions
                ),
            )
            await say(1, "@review check the numbers", ["review"])
            _, plan = await _until(handle, lambda v, p: len(p.nodes) == 2 and p.nodes[1].status == "COMPLETED", "the member's node")
            await say(2, "@review and again", ["review"])
            _, plan = await _until(handle, lambda v, p: len(p.nodes) == 3 and p.nodes[2].status == "COMPLETED", "its second node")
            await say(3, "just the leader", [])
            await _until(handle, lambda v, p: len(p.nodes) == 4 and p.nodes[3].status == "COMPLETED", "the leader's node")
            await _finish(handle)
    first, second = plan.nodes[1], plan.nodes[2]
    assert (first.owner_role, first.owner_label) == ("review", "审阅员") and plan.nodes[0].owner_role == "lead"
    turns = {t["goal"]: t for t in TURNS}
    assert turns["@review check the numbers"]["profile"] == "reviewer@1" and turns["just the leader"]["profile"] == "writer@1"
    assert turns["@review check the numbers"].get("continue_from") is None, "no session of its own yet"
    assert turns["@review and again"]["continue_from"] == turns["@review check the numbers"]["attempt_id"], "its last session"
    user = [m for m in _messages("user")]
    assert [m["to_roles"] for m in user] == [["review"], ["review"]] and user[0]["from_role"] == "user"
    replies = _messages("reply")
    assert [(m["from_role"], m["to_roles"]) for m in replies] == [("review", ["user"])] * 2
    assert second.node_id != first.node_id
    owners = [p for p in _events("node.status_changed") if p["node_id"] == first.node_id]
    assert owners and all(p["owner_role"] == "review" and p["owner_label"] == "审阅员" for p in owners)


async def test_a_mention_of_a_role_the_team_does_not_have_is_refused_and_a_task_without_a_team_has_none() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            with_team = await _team_task(env, "unknown-mention")
            without = await _start(env, "no-team-mention", goal="explore", stage=False)
            for handle, mention in ((with_team, "ghost"), (without, "review")):
                with pytest.raises(WorkflowUpdateFailedError) as caught:
                    await handle.execute_update(
                        TaskWorkflow.send_message,
                        v3.SendMessageInput(
                            command_id="01J00000000000000000000090", client_message_id="01J00000000000000000000091",
                            text=f"@{mention} hi", mentions=[mention],
                        ),
                    )
                assert "UNKNOWN_MENTION" in str(caught.value.cause), caught.value.cause
            await _finish(with_team)
            await _finish(without)


async def test_the_leaders_task_create_for_a_member_is_said_in_the_group_and_its_result_comes_back() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _team_task(env, "plan-assign", goal="hold")
            _, plan = await _until(handle, lambda v, p: p.nodes[0].status == "RUNNING", "the leader's attempt")
            task_id = (await _query(handle, TaskWorkflow.get_task_view)).task_id
            attempt = plan.nodes[0].current_attempt_id
            actor = Actor(kind="agent", id=attempt, attempt_id=attempt)
            from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec

            result = await handle.execute_update(
                TaskWorkflow.submit_plan_change,
                PlanChangeCommand(
                    command_id=hashlib.sha256(b"assign").hexdigest(), task_id=task_id, base_plan_version=plan.plan_version, actor=actor,
                    ops=[
                        AddNodeOp(node=AgentTurnNode(
                            node_id="tmp:1", title="Check the numbers", owner_profile="reviewer@1",
                            depends_on=[plan.nodes[0].node_id], spec=AgentTurnSpec(goal="verify every figure"),
                        ))
                    ],
                ),
            )
            assert result.status == "accepted"
            GATE["release"].set()
            _, plan = await _until(
                handle, lambda v, p: any(n.title == "领队复盘" and n.status == "COMPLETED" for n in p.nodes), "the leader's review"
            )
            await _finish(handle)
    member_node = next(n for n in plan.nodes if n.title == "Check the numbers")
    assign = _messages("assign")[0]
    assert (assign["from_role"], assign["to_roles"], assign["node_id"]) == ("lead", ["review"], member_node.node_id)
    assert "Check the numbers" in assign["text"] and "verify every figure" in assign["text"] and assign["attempt_id"] == attempt
    reply = _messages("reply")[0]
    assert (reply["from_role"], reply["from_label"], reply["to_roles"]) == ("review", "审阅员", ["lead"])
    assert member_node.owner_role == "review" and member_node.owner_label == "审阅员"
    review = _messages("review")[0]
    assert review["from_role"] == "lead" and review["to_roles"] == [], "the leader's review is said to the whole group"


async def test_agents_are_shown_labels_not_role_ids_with_the_leader_called_领队_when_it_has_none() -> None:
    _reset()
    SCRIPT["leader"] = lambda n, payload: assign(("review", "a"), notes=("kickoff",)) if n == 1 else final("ok")

    async def noting(payload: dict[str, Any]) -> dict[str, Any]:
        return {**await _member(payload), "notes": ["@docs 看一下"], "note_mentions": [["docs"]]}

    SCRIPT["review"] = noting
    spec = MENTION_TEAM.model_copy(update={"members": [TEAM.members[0], TEAM.members[1].model_copy(update={"label": "审阅员"}), TEAM.members[2]]})
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _mentions_stack(env, "names", spec)
            await _done(handle)
            await _finish(handle)
    docs = _of("docs")[0]["goal"]
    assert docs.startswith("团队成员 审阅员 在群里 @ 了你"), "the one who woke it, by label"
    assert "- [领队] kickoff" in docs, "the leader has no label: 领队, never lead"
    assert "lead]" not in docs and "[review]" not in docs
    leader_input = _of("lead")[1]["team_results"][-1]["output"]
    assert "[审阅员]" in leader_input and "[docs]" in leader_input, "a member without a label is its role id"
    assert _of("review")[0]["team"]["leader_label"] == "领队" and _of("lead")[0]["team"]["label"] == "领队"


# ---- several members @-mentioned at plan level run in parallel (07 §6b, 08 §1) ---------------------------------------


async def _two_mentions(env, name: str):
    handle = await _team_task(env, name)
    await _until(handle, lambda v, p: v.status == "COMPLETED", "the first round")
    await handle.execute_update(
        TaskWorkflow.send_message,
        v3.SendMessageInput(
            command_id="01J00000000000000000000095", client_message_id="01J00000000000000000000096",
            text="@review @docs both of you, now", mentions=["review", "docs"],
        ),
    )
    return handle


def _overlap(plan) -> bool:
    return LOG.index(("start", "review")) < LOG.index(("end", "docs")) and LOG.index(("start", "docs")) < LOG.index(("end", "review"))


async def test_mentioned_members_follow_ups_run_in_parallel_and_leave_their_files_as_artifacts() -> None:
    _reset()

    async def member(payload: dict[str, Any]) -> dict[str, Any]:
        role = payload["profile"].split("@")[0].replace("reviewer", "review")
        GATE["started"].add(role)
        LOG.append(("start", role))
        for _ in range(500):  # each waits for the other: one after the other, this never ends
            if len(GATE["started"]) >= 2:
                break
            activity.heartbeat()
            await asyncio.sleep(0.02)
        # Bounded, and it fails clearly: a scheduler that starts the two one at a time is a product bug, not a hang.
        assert len(GATE["started"]) >= 2, f"{role} was never joined by the other mentioned member: they were serialized"
        LOG.append(("end", role))
        return {
            "status": "completed", "checkpoint_ref": REF, "session_id": payload["attempt_id"], "state_version": 2,
            "handover_summary": f"{role} done", "manifest_id": deterministic_id(f"man:{payload['attempt_id']}", "man"),
            "manifest_entries": [{"name": f"{role}.md", "media_type": "text/markdown", "size_bytes": 3, "blob_ref": REF}],
        }

    async def plain(payload: dict[str, Any]) -> dict[str, Any]:
        if payload["goal"].startswith("@"):
            return await member(payload)
        return {"status": "completed", "checkpoint_ref": REF, "session_id": "s", "state_version": 2}

    @activity.defn(name="agent_turn")
    async def turn(payload: dict[str, Any]) -> dict[str, Any]:
        TURNS.append(payload)
        return await plain(payload)

    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch = Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner())
        agent = Worker(env.client, task_queue="orbit.agent", activities=[turn], **FAST)
        io = Worker(env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints])
        async with orch, agent, io:
            handle = await _two_mentions(env, "parallel")
            _, plan = await _until(handle, lambda v, p: len(p.nodes) == 3 and all(n.status == "COMPLETED" for n in p.nodes), "both nodes")
            await _finish(handle)
    nodes = {n.owner_role: n for n in plan.nodes[1:]}
    assert set(nodes) == {"review", "docs"} and all(n.workspace_access == "read" for n in nodes.values()), "mention follow-ups are read-only"
    assert plan.nodes[0].workspace_access != "read", "other nodes keep their access: write-serialization is unaffected"
    assert _overlap(plan), "both attempts were RUNNING at once"
    artifacts = [e["payload"] for e in EVENTS if e["type"] == "artifact.manifest_created"]
    assert sorted(entry["name"] for a in artifacts for entry in a["entries"]) == ["docs.md", "review.md"]
    turns = [t for t in TURNS if t["goal"].startswith("@")]
    assert [t["workspace_access"] for t in turns] == ["read", "read"]


async def test_a_message_without_a_mention_is_not_made_read_only() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _team_task(env, "writers")
            _, plan = await _until(handle, lambda v, p: v.status == "COMPLETED", "the first round")
            for number in (1, 2):
                await handle.execute_update(
                    TaskWorkflow.send_message,
                    v3.SendMessageInput(
                        command_id=f"01J0000000000000000000010{number}", client_message_id=f"01J1000000000000000000010{number}",
                        text=f"leader work {number}",
                    ),
                )
                await _until(handle, lambda v, p, n=number: len(p.nodes) == 1 + n and p.nodes[-1].status == "COMPLETED", "a leader follow-up")
            await _finish(handle)
    assert all(n.workspace_access != "read" for n in plan.nodes), "only mention follow-ups are read-only; other nodes are as before"


# ---- a cancel, a stop or an interrupt while the stage's agents work (04 §6) ------------------------------------------------

ACTIVE: set[str] = set()


async def _hold_until_cancelled(payload: dict[str, Any]) -> dict[str, Any]:
    """A turn that heartbeats until it is cancelled: the activity is `ACTIVE` while it runs, and what it was cancelled by is kept."""
    key = payload["team"]["role"] if payload.get("team") else "plain"
    ACTIVE.add(key)
    try:
        for _ in range(3000):
            activity.heartbeat()
            await asyncio.sleep(0.02)
    finally:
        ACTIVE.discard(key)
    raise AssertionError("never cancelled")


async def _until_active(wanted: set[str]) -> None:
    async for _ in polls():
        if ACTIVE >= wanted:
            return
    raise AssertionError(f"{wanted} never started, only {ACTIVE}")


async def _closes(env, handle_id: str, what: str) -> None:
    """The workflow is closed within a bound: the heartbeat timeout (30s) and a margin, in the time-skipping server's clock."""
    h = env.client.get_workflow_handle(handle_id)
    async for _ in polls():
        if (await h.describe()).status != WorkflowExecutionStatus.RUNNING:
            return
        await env.sleep(timedelta(seconds=5))
    raise AssertionError(f"{what} did not close: {handle_id}")


@pytest.mark.parametrize("who", ["leader", "member", "two"])
@pytest.mark.parametrize("action", ["cancel", "stop", "interrupt"])
async def test_a_task_is_cancelled_stopped_or_interrupted_while_the_stage_works_and_nothing_is_left_running(who: str, action: str) -> None:
    _reset()
    ACTIVE.clear()
    expected = {"leader": {"lead"}, "member": {"review"}, "two": {"review", "docs"}}[who]

    def leader(n: int, payload: dict[str, Any]) -> dict[str, Any]:
        if who == "leader":
            raise _Hold
        return assign(("review", "work"), ("docs", "work")) if who == "two" else assign(("review", "work"))

    class _Hold(Exception):
        pass

    @activity.defn(name="agent_turn")
    async def turn(payload: dict[str, Any]) -> dict[str, Any]:
        TURNS.append(payload)
        team = payload.get("team")
        if team is None:
            return {"status": "completed", "checkpoint_ref": REF, "session_id": "s", "state_version": 2}
        if team["leader"]:
            if who == "leader" or payload.get("team_results") or len([t for t in TURNS if t.get("team") and t["team"]["leader"]]) > 1:
                return final("done") if who != "leader" else await _hold_until_cancelled(payload)
            return assign(("review", "work"), ("docs", "work")) if who == "two" else assign(("review", "work"))
        return await _hold_until_cancelled(payload)

    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch = Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner())
        agent = Worker(env.client, task_queue="orbit.agent", activities=[turn], **FAST)
        io = Worker(env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints])
        async with orch, agent, io:
            handle = await _start(env, f"cancel-{who}-{action}")
            await _until_active(expected)
            _, plan = await _until(handle, lambda v, p: any(n.type == "team_stage" and n.status == "RUNNING" for n in p.nodes), "the stage")
            stage = _stage(plan)
            task_id = (await _query(handle, TaskWorkflow.get_task_view)).task_id
            child = f"attempt/{task_id}/{stage.node_id}/1"
            if action == "interrupt":
                await handle.execute_update(
                    TaskWorkflow.send_message,
                    v3.SendMessageInput(command_id="01J00000000000000000000120", client_message_id="01J00000000000000000000121", text="stop that", delivery="interrupt"),
                )
            else:
                await handle.execute_update(
                    TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(action.encode()).hexdigest(), action=action)
                )
            await _closes(env, child, "the stage's attempt")
            if action != "interrupt":  # an interrupt starts the replacement attempt, whose own turns run
                assert not ACTIVE, f"activities still running after the attempt closed: {ACTIVE}"
            if action == "cancel":
                await _closes(env, handle.id, "the task")
                assert (await handle.describe()).status != WorkflowExecutionStatus.RUNNING
            else:
                view, plan = await _until(handle, lambda v, p: not ACTIVE and _stage(p).status != "RUNNING" or action == "interrupt", "the stage to leave RUNNING")
                if action == "stop":
                    assert view.status == "PAUSED"
                await _finish(handle)
    assert any(p["outcome"] == "cancelled" for p in _events("attempt.finished")), "the attempt reported its end as cancelled"

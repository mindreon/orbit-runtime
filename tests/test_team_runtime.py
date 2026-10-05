"""The agents of a team stage on the real worker: the tools, the mock leader's script, the member's copy of the workspace.

How it can go wrong, written down before the code:
  - a leader step that assigns to two members is told about one of them, or answered for one, and AgentScope refuses to go on;
  - a member is given the plan's tools (and changes the plan), or the leader is given no way to assign;
  - a member writes into the task's own workspace, so the leader's head snapshot changes under it;
  - what a member left in its copy is lost, or is written over the leader's own files instead of being offered under `.team/`;
  - a note is counted twice because the turn that wrote it ran again;
  - two members that run side by side take the task's write lease from each other.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from orbit_contracts.models import (
    DeliverToolResultsInput,
    ExternalResult,
    OpenSessionInput,
    RunTurnInput,
    TurnResult,
)
from orbit_contracts.v3 import (
    Actor,
    Budget,
    DecideApprovalInput,
    PlanChangeCommand,
    TaskControlInput,
    TaskWorkflowInput,
)
from orbit_contracts.v3.nodes import TeamStageMember, TeamStageNode, TeamStageSpec
from orbit_contracts.v3.plan import AddNodeOp
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from orbit_worker import task_activities
from orbit_worker.runtime import AgentRuntime
from orbit_worker.runtime_holder import set_runtime
from orbit_worker.sandbox import SandboxSession
from orbit_worker.store import MemoryStateStore
from orbit_worker.task_stream import TaskStreamContext, TeamTurn, streaming_for
from orbit_worker.team_tools import team_tools
from orbit_worker.workspace import LocalWorkspaceAdapter, PersistentWorkspaceAdapter
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from test_interrupted_session import _Store

MEMBERS = (("review", "checks the work"), ("docs", "writes the docs"))
LEADER = TeamTurn(role="lead", leader=True, leader_role="lead", members=MEMBERS)
MEMBER = TeamTurn(role="review", leader=False, leader_role="lead")
FAST = {
    "max_heartbeat_throttle_interval": timedelta(milliseconds=100),
    "default_heartbeat_throttle_interval": timedelta(milliseconds=100),
}


def _context(attempt_id: str, team: TeamTurn) -> TaskStreamContext:
    return TaskStreamContext(tenant_id="t", task_id="task-1", attempt_id=attempt_id, activity_attempt=1, team=team)


async def _open(runtime: AgentRuntime, attempt_id: str, team: TeamTurn) -> str:
    with streaming_for(_context(attempt_id, team)):
        return (await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id=f"{attempt_id}:open"))).session_id


async def _say(runtime: AgentRuntime, attempt_id: str, team: TeamTurn, text: str) -> TurnResult:
    with streaming_for(_context(attempt_id, team)):
        blob = await runtime._store.get(attempt_id)
        assert blob is not None
        return await runtime.run_turn(
            RunTurnInput(
                room_id="task-1", session_id=attempt_id, turn_id=f"{attempt_id}:{blob.state_version}",
                message=text, state_version=blob.state_version,
            )
        )


async def _deliver(runtime: AgentRuntime, attempt_id: str, team: TeamTurn, results: list[ExternalResult], turn: str) -> TurnResult:
    with streaming_for(_context(attempt_id, team)):
        blob = await runtime._store.get(attempt_id)
        assert blob is not None
        return await runtime.deliver_tool_results(
            DeliverToolResultsInput(
                room_id="task-1", session_id=attempt_id, turn_id=f"{attempt_id}:{turn}",
                state_version=blob.state_version, results=results,
            )
        )


def test_the_leader_can_assign_and_a_member_can_only_post_notes() -> None:
    assert [tool.name for tool in team_tools(LEADER)] == ["team_assign", "team_note"]
    assert [tool.name for tool in team_tools(MEMBER)] == ["team_note"]
    assign = team_tools(LEADER)[0]
    assert assign.is_external_tool and assign.input_schema["properties"]["member"]["enum"] == ["review", "docs"]
    assert "review: checks the work" in assign.description and "docs: writes the docs" in assign.description


async def test_a_step_that_assigns_to_two_members_is_answered_with_both_results_at_once() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await _open(runtime, "att-lead", LEADER)
    parked = await _say(runtime, "att-lead", LEADER, "team: check it;;@docs write it")
    assert parked.status == "needs_external"
    assert [(call.tool_name, call.arguments["member"], call.arguments["task"]) for call in parked.externals] == [
        ("team_assign", "review", "check it"), ("team_assign", "docs", "write it"),
    ], "both calls of the step, in the order the leader made them"
    assert parked.notes == ["kickoff: 2 task(s)"], "the note the leader made in the same step"
    ids = [call.call_id for call in parked.externals]

    # One result is not enough: the reply stays parked until every open call has its answer.
    with pytest.raises(ValueError, match="parked on external calls"):
        await _deliver(runtime, "att-lead", LEADER, [ExternalResult(call_id=ids[0], tool_name="team_assign", output="x")], "partial")
    done = await _deliver(
        runtime, "att-lead", LEADER,
        [ExternalResult(call_id=ids[1], tool_name="team_assign", output="docs: done"),
         ExternalResult(call_id=ids[0], tool_name="team_assign", output="review: ok")],
        "all",
    )
    assert done.status == "completed" and done.text == "team-final=review: ok | docs: done", "in the order of the calls"
    assert done.notes == [], "the notes of an earlier step are not reported again"


async def test_the_state_of_a_turn_that_ran_again_reports_its_notes_once() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await _open(runtime, "att-m", MEMBER)
    first = await _say(runtime, "att-m", MEMBER, "note:found a bug")
    assert first.status == "completed" and first.notes == ["found a bug"]
    blob = await runtime._store.get("att-m")
    assert blob is not None
    # The activity ran again with the same turn id (a retry): the saved result comes back, with the notes it had.
    with streaming_for(_context("att-m", MEMBER)):
        again = await runtime.run_turn(
            RunTurnInput(room_id="task-1", session_id="att-m", turn_id="att-m:1", message="note:found a bug", state_version=1)
        )
    assert again.notes == ["found a bug"] and blob.state_version == 2


async def test_a_member_is_not_given_the_plans_tools() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await _open(runtime, "att-m", MEMBER)
    listing = (await _say(runtime, "att-m", MEMBER, "tools:")).text
    assert "team_note" in listing and "TaskCreate" not in listing and "team_assign" not in listing
    await _open(runtime, "att-lead", LEADER)
    listing = (await _say(runtime, "att-lead", LEADER, "tools:")).text
    assert "team_assign" in listing and "team_note" in listing and "TaskCreate" not in listing
    prompt = (await _say(runtime, "att-lead", LEADER, "prompt:")).text
    assert "You lead a team in one stage" in prompt and "- docs: writes the docs" in prompt


# ---- the member's copy of the workspace -------------------------------------------------------------------------------


class _Leases:
    async def acquire_workspace_lease(self, **kwargs: Any) -> None: ...
    async def renew_workspace_lease(self, **kwargs: Any) -> None: ...
    async def release_workspace_lease(self, **kwargs: Any) -> None: ...


async def test_a_replica_is_a_copy_whose_changes_come_back_as_files_and_never_as_the_head(tmp_path: Path) -> None:
    adapter = PersistentWorkspaceAdapter(LocalWorkspaceAdapter(tmp_path / "ws"), _Leases())
    store = _Store(tmp_path / "store")
    head = await adapter.acquire("tenant-a", "task-1")
    (tmp_path / "ws" / "tenant-a" / head.workspace_id / "base.md").write_text("the head", encoding="utf-8")
    snapshot = await adapter.snapshot(head)
    await adapter.release(head)

    writer = await adapter.acquire("tenant-a", "task-1", holder="att-lead")  # the leader holds the write lease meanwhile
    replicas = []
    for name in ("review", "docs"):
        session = SandboxSession(adapter, store, tenant_id="tenant-a", task_id="task-1", holder=f"att-{name}", restore_from=snapshot, replica=True)
        replicas.append(session)
        assert (await session.backend.read_file("/workspace/base.md")) == b"the head", "seeded from the head snapshot"
        await session.backend.write_file("/workspace/base.md", b"changed by " + name.encode())
        await session.backend.write_file(f"/workspace/{name}.md", b"mine")
    assert [(f.name, f.payload) for f in await replicas[0].files()] == [("base.md", b"changed by review"), ("review.md", b"mine")], (
        "what differs from the head is what it left"
    )
    assert [f.name for f in await replicas[1].files()] == ["base.md", "docs.md"], "each replica is its own copy"
    for session in replicas:
        assert await session.close() is None, "a replica is released, never snapshotted into the head"
    assert len(list((tmp_path / "ws" / "snapshots" / "tenant-a").iterdir())) == 1, "no snapshot of a replica was taken"
    await adapter.release(writer)


async def test_what_the_members_left_is_in_the_leaders_workspace_under_team_and_not_in_its_snapshot(tmp_path: Path) -> None:
    adapter = PersistentWorkspaceAdapter(LocalWorkspaceAdapter(tmp_path / "ws"), _Leases())
    store = _Store(tmp_path / "store")
    session = SandboxSession(adapter, store, tenant_id="tenant-a", task_id="task-1", holder="att-lead")
    session.offer_team_files("review", [("notes.md", b"found two issues")])
    assert await session.backend.read_file("/workspace/.team/review/notes.md") == b"found two issues"
    await session.backend.write_file("/workspace/final.md", b"merged")
    snapshot = await session.close()
    assert snapshot is not None
    archive = await adapter.load_snapshot("tenant-a", snapshot)
    from orbit_worker.sandbox import files_in_archive

    assert [f.name for f in files_in_archive(archive)] == ["final.md"], ".team is data of the attempt, not of the task"


# ---- the whole stage on the real activities and the mock model -------------------------------------------------------


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "failures": [], "workspace_snapshot_ref": None}


EVENTS: list[dict[str, Any]] = []


@activity.defn(name="publish_events")
async def _publish_events(payload: list[dict[str, Any]]) -> dict[str, Any]:
    EVENTS.extend(payload)
    return {"ok": True, "count": len(payload)}


@activity.defn(name="checkpoint_commit")
async def _checkpoint_commit(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


@activity.defn(name="commit_checkpoints")
async def _commit_checkpoints(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "committed": 1}


async def _query(handle, query):
    for _ in range(20):
        try:
            return await handle.query(query, rpc_timeout=timedelta(seconds=3))
        except RPCError:
            continue
    raise AssertionError("the query was never answered")


async def test_the_mock_leader_assigns_two_members_who_work_in_copies_and_the_stage_merges_it(tmp_path: Path) -> None:
    EVENTS.clear()
    store = _Store(tmp_path / "store")
    task_activities.set_task_store(store)
    adapter = LocalWorkspaceAdapter(tmp_path / "ws")
    task_activities.set_workspace_adapter(PersistentWorkspaceAdapter(adapter, _Leases()))
    set_runtime(AgentRuntime(MemoryStateStore()))
    spec = TeamStageSpec(
        goal="team: @review file:review.md|checked it;;@docs note:the docs need a changelog",
        leader="lead",
        members=[TeamStageMember(role="lead", executor="writer@1"), TeamStageMember(role="review", executor="reviewer@1"), TeamStageMember(role="docs", executor="docs@1")],
    )
    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[task_activities.agent_turn], **FAST),
        Worker(env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]),
    ):
        task_id = deterministic_id("team-runtime", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run,
            TaskWorkflowInput(
                task_id=task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="u"), title="Team", goal="explore",
                profile="default@1", node_type_registry_version=1, budgets=Budget(),
            ),
            id="task/tenant-a/team-runtime", task_queue="orbit.orch",
        )
        plan = await _query(handle, TaskWorkflow.get_plan)
        await handle.execute_update(
            TaskWorkflow.submit_plan_change,
            PlanChangeCommand(
                command_id=hashlib.sha256(b"team-runtime").hexdigest(), task_id=task_id, base_plan_version=plan.plan_version,
                actor=Actor(kind="user", id="u"), ops=[AddNodeOp(node=TeamStageNode(node_id="tmp:1", title="Stage", spec=spec))],
            ),
        )
        for _ in range(3000):
            plan = await _query(handle, TaskWorkflow.get_plan)
            if any(n.type == "team_stage" and n.status == "COMPLETED" for n in plan.nodes):
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError(f"the stage did not complete: {[(n.title, n.status) for n in plan.nodes]} {[e['payload'] for e in EVENTS if e['type'] == 'attempt.finished']}")
        await handle.execute_update(TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"cancel").hexdigest(), action="cancel"))
        for _ in range(3000):
            if (await handle.describe()).status != WorkflowExecutionStatus.RUNNING:
                break
            await asyncio.sleep(0.02)
    finals = [e["payload"]["text"] for e in store.events if e["type"] == "message.agent_final"]
    notes = [e["payload"] for e in EVENTS if e["type"] == "team.message" and e["payload"]["kind"] == "note"]
    assert [(n["role"], n["text"]) for n in notes] == [("lead", "kickoff: 2 task(s)"), ("docs", "the docs need a changelog")]
    finished = [e["payload"] for e in EVENTS if e["type"] == "team.member_turn_finished"]
    assert sorted((p["role"], p["outcome"]) for p in finished) == [("docs", "completed"), ("review", "completed")]
    review = next(p for p in finished if p["role"] == "review")
    assert review["artifacts"] == ["review.md"], "the file the member wrote in its copy is reported by name"
    assert review["summary"].startswith("file-result="), "what the member answered is what the leader is given"
    rounds = [e["payload"]["outcome"] for e in EVENTS if e["type"] == "team.round_finished"]
    assert rounds == ["assigned", "completed"]
    merged = next(text for text in finals if text.startswith("team-final="))
    assert "file-result=" in merged and "noted: the docs need a changelog" in merged
    assert "团队消息:\n- [docs] the docs need a changelog" in merged, "the leader is shown the note a member posted"
    assert "Files review left, readable in your workspace under .team/review/: review.md" in merged, "the leader is told where they are"
    # The task's own workspace holds nothing of the member's copy: the leader's snapshot has no review.md.
    snapshots = list((tmp_path / "ws" / "snapshots" / "tenant-a").glob("*")) if (tmp_path / "ws" / "snapshots" / "tenant-a").exists() else []
    from orbit_worker.sandbox import files_in_archive

    for item in snapshots:
        assert "review.md" not in [f.name for f in files_in_archive(item.read_bytes())], "the member's file is not in the head"


async def test_the_mock_member_that_needs_approval_parks_the_stage_and_the_decision_resumes_it(tmp_path: Path) -> None:
    EVENTS.clear()
    store = _Store(tmp_path / "store")
    task_activities.set_task_store(store)
    task_activities.set_workspace_adapter(None)  # type: ignore[arg-type]
    set_runtime(AgentRuntime(MemoryStateStore()))
    spec = TeamStageSpec(
        goal="team: @review echo:once",
        leader="lead",
        members=[TeamStageMember(role="lead", executor="writer@1"), TeamStageMember(role="review", executor="reviewer@1")],
    )
    async with (
        await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env,
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[task_activities.agent_turn], **FAST),
        Worker(env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]),
    ):
        task_id = deterministic_id("team-runtime-approval", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run,
            TaskWorkflowInput(
                task_id=task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="u"), title="Team", goal="explore",
                profile="default@1", node_type_registry_version=1, budgets=Budget(),
            ),
            id="task/tenant-a/team-runtime-approval", task_queue="orbit.orch",
        )
        plan = await _query(handle, TaskWorkflow.get_plan)
        await handle.execute_update(
            TaskWorkflow.submit_plan_change,
            PlanChangeCommand(
                command_id=hashlib.sha256(b"team-runtime-approval").hexdigest(), task_id=task_id,
                base_plan_version=plan.plan_version, actor=Actor(kind="user", id="u"),
                ops=[AddNodeOp(node=TeamStageNode(node_id="tmp:1", title="Stage", spec=spec))],
            ),
        )
        for _ in range(3000):
            view = await _query(handle, TaskWorkflow.get_task_view)
            if view.pending_approvals:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the member never asked for approval")
        requested = next(e["payload"] for e in EVENTS if e["type"] == "approval.requested")
        assert requested["subject"]["role"] == "review" and requested["subject"]["summary"].startswith("review: gated_echo")
        assert requested["tool_call_id"].startswith("review:")
        await handle.execute_update(
            TaskWorkflow.decide_approval,
            DecideApprovalInput(command_id="01J00000000000000000000060", approval_id=view.pending_approvals[0], decision="approve"),
        )
        for _ in range(3000):
            plan = await _query(handle, TaskWorkflow.get_plan)
            if any(n.type == "team_stage" and n.status == "COMPLETED" for n in plan.nodes):
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the stage did not complete after the decision")
        await handle.execute_update(TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"cancel").hexdigest(), action="cancel"))
    finals = [e["payload"]["text"] for e in store.events if e["type"] == "message.agent_final"]
    assert "team-final=done" in finals, "the member went on after the decision and answered the leader"

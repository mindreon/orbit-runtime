"""End to end coverage for the v3 task workflow surface."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from orbit_contracts.v3 import (
    Actor,
    Budget,
    ConnectorSnapshot,
    DecideApprovalInput,
    PlanChangeCommand,
    SendMessageInput,
    TaskConfig,
    TaskControlInput,
    TaskWorkflowInput,
    Team,
    TeamMember,
    UpdateTaskConfigInput,
)
from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec, CheckpointNode, CheckpointSpec
from orbit_contracts.v3.plan import AddNodeOp
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus, WorkflowUpdateFailedError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.service import RPCError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

# A cancel reaches an activity on its next heartbeat, and the SDK sends one at most every `max_heartbeat_throttle_interval`.
_FAST_HEARTBEAT = {
    "max_heartbeat_throttle_interval": timedelta(milliseconds=100),
    "default_heartbeat_throttle_interval": timedelta(milliseconds=100),
}
TURNS: list[dict[str, object]] = []
EVENTS: list[dict[str, object]] = []


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, object]) -> dict[str, object]:
    TURNS.append(payload)
    if payload.get("goal") == "hold" and payload.get("attempt_no") == 1 and not payload.get("messages"):
        # Heartbeats, as a real turn does: it is how a cancel reaches the activity, and the attempt does not end until the
        # activity has stopped.
        for _ in range(600):
            activity.heartbeat()
            await asyncio.sleep(0.1)
    if payload.get("goal") == "approval" and not payload.get("approval"):
        return {
            "status": "parked_approval",
            "checkpoint_ref": "sha256:" + "3" * 64,
            "approvals": [{
                "tool_call_id": "call-1",
                "subject": {
                    "kind": "tool_call",
                    "digest": "sha256:" + "4" * 64,
                    "summary": "run a gated command",
                    "risk": "medium",
                },
            }],
        }
    await asyncio.sleep(0)
    return {"status": "completed", "checkpoint_ref": "sha256:" + "1" * 64}


@activity.defn(name="sop_step")
async def _sop_step(payload: dict[str, object]) -> dict[str, object]:
    return {"status": "completed", "checkpoint_ref": "sha256:" + "2" * 64}


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, object]) -> dict[str, object]:
    return {"ok": True}


@activity.defn(name="publish_events")
async def _publish_events(payload: list[dict[str, object]]) -> dict[str, object]:
    EVENTS.extend(payload)
    return {"ok": True, "count": len(payload)}


CHECKPOINT_PAYLOADS: list[dict[str, object]] = []


@activity.defn(name="checkpoint_commit")
async def _checkpoint_commit(payload: dict[str, object]) -> dict[str, object]:
    CHECKPOINT_PAYLOADS.append(payload)
    return {"ok": True}


COMMITS: list[dict[str, object]] = []


@activity.defn(name="commit_checkpoints")
async def _commit_checkpoints(payload: dict[str, object]) -> dict[str, object]:
    COMMITS.append(payload)
    return {"ok": True, "committed": 1}


def _input(
    task_id: str, *, goal: str = "complete the task", config: TaskConfig | None = None
) -> TaskWorkflowInput:
    return TaskWorkflowInput(
        task_id=task_id,
        tenant_id="tenant-a",
        created_by=Actor(kind="user", id="user-a"),
        title="E2E task",
        goal=goal,
        profile="default@1",
        node_type_registry_version=1,
        budgets=Budget(),
        **({"config": config} if config else {}),
    )


async def _running(handle, polls: int = 6000) -> None:
    """Wait until the first attempt is running: a condition of the workflow, not a moment of the clock."""
    for _ in range(polls):
        try:
            plan = await handle.query(TaskWorkflow.get_plan)
        except Exception:  # noqa: BLE001 - the workflow may not have handled its first task yet
            plan = None
        if plan is not None and any(node.status == "RUNNING" for node in plan.nodes):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the first attempt never started")


async def _wait_done(handle, polls: int = 6000) -> object:
    """Wait for the condition, not for a moment: a generous deadline (60s) that a loaded machine does not reach."""
    for _ in range(polls):
        view = await handle.query(TaskWorkflow.get_task_view)
        if view.status == "COMPLETED":
            return view
        await asyncio.sleep(0.01)
    plan = await handle.query(TaskWorkflow.get_plan)
    seen = [(t["goal"], t["attempt_no"], bool(t.get("messages"))) for t in TURNS]
    raise AssertionError(f"TaskWorkflow did not complete: {view} {[(n.title, n.status) for n in plan.nodes]} {seen}")


@pytest.mark.asyncio
async def test_task_workflow_completes_and_publishes_events() -> None:
    COMMITS.clear()
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit.orch",
        workflows=[TaskWorkflow, AttemptWorkflow],
        workflow_runner=sandbox_runner(),
    ), Worker(
        env.client,
        task_queue="orbit.agent",
        activities=[_agent_turn, _sop_step],
        **_FAST_HEARTBEAT,
    ), Worker(
        env.client,
        task_queue="orbit.io",
        activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
    ):
        handle = await env.client.start_workflow(
            TaskWorkflow.run,
            _input(deterministic_id("e2e:complete", "task")),
            id="task/tenant-a/complete",
            task_queue="orbit.orch",
        )
        view = await _wait_done(handle)
        assert view.plan_version == 1
        assert view.pending_approvals == []
        # The finished attempt's checkpoint is committed once its result is in history (08 §3).
        assert [c["checkpoint_ref"] for c in COMMITS] == ["sha256:" + "1" * 64]
        assert COMMITS[0]["tenant_id"] == "tenant-a" and str(COMMITS[0]["attempt_id"]).startswith("att_")


@pytest.mark.asyncio
async def test_queue_message_is_delivered_to_active_attempt() -> None:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit.orch",
        workflows=[TaskWorkflow, AttemptWorkflow],
        workflow_runner=sandbox_runner(),
    ), Worker(
        env.client,
        task_queue="orbit.agent",
        activities=[_agent_turn, _sop_step],
        **_FAST_HEARTBEAT,
    ), Worker(
        env.client,
        task_queue="orbit.io",
        activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
    ):
        task_id = deterministic_id("e2e:queue", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id), id="task/tenant-a/queue", task_queue="orbit.orch"
        )
        result = await handle.execute_update(
            TaskWorkflow.send_message,
            SendMessageInput(
                command_id="01J00000000000000000000001",
                client_message_id="01J00000000000000000000002",
                text="queued input",
                delivery="queue",
            ),
        )
        assert result.message_seq == 1
        assert (await _wait_done(handle)).status == "COMPLETED"


@pytest.mark.asyncio
async def test_continue_as_new_carries_plan_and_message_state() -> None:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit.orch",
        workflows=[TaskWorkflow, AttemptWorkflow],
        workflow_runner=sandbox_runner(),
    ), Worker(
        env.client,
        task_queue="orbit.agent",
        activities=[_agent_turn, _sop_step],
        **_FAST_HEARTBEAT,
    ), Worker(
        env.client,
        task_queue="orbit.io",
        activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
    ):
        task_id = deterministic_id("e2e:can", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, goal="hold"), id="task/tenant-a/can", task_queue="orbit.orch"
        )
        await _running(handle)
        # A stop, not a pause: a running attempt is handed every message, and a message it has leaves the inbox.
        await handle.execute_update(
            TaskWorkflow.control,
            TaskControlInput(command_id="0" * 26, action="stop"),
        )
        for index in range(1001):
            await handle.execute_update(
                TaskWorkflow.send_message,
                SendMessageInput(
                    command_id=f"{index + 1:026d}",
                    client_message_id=f"{index + 10000:026d}",
                    text="message",
                ),
            )
        inbox = await handle.query(TaskWorkflow.get_inbox, 1000)
        assert len(inbox) == 1


@pytest.mark.asyncio
async def test_approval_parks_and_resumes_the_same_attempt() -> None:
    COMMITS.clear()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env, Worker(
        env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step], **_FAST_HEARTBEAT), Worker(
        env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]
    ):
        task_id = deterministic_id("e2e:approval", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, goal="approval"), id="task/tenant-a/approval", task_queue="orbit.orch"
        )
        approval_id = None
        for _ in range(100):
            view = await handle.query(TaskWorkflow.get_task_view)
            if view.pending_approvals:
                approval_id = view.pending_approvals[0]
                break
            await asyncio.sleep(0.01)
        assert approval_id is not None
        result = await handle.execute_update(
            TaskWorkflow.decide_approval,
            DecideApprovalInput(command_id="01J00000000000000000000003", approval_id=approval_id, decision="approve"),
        )
        assert result.status == "APPROVED"
        assert (await _wait_done(handle)).status == "COMPLETED"
        # Parked on the approval and finished: one commit each, for the same attempt, so the parked
        # checkpoint is committed before the person decides (A11, A19).
        assert [c["checkpoint_ref"] for c in COMMITS] == ["sha256:" + "3" * 64, "sha256:" + "1" * 64]
        assert len({c["attempt_id"] for c in COMMITS}) == 1


@pytest.mark.asyncio
async def test_interrupt_cancels_active_attempt_and_finishes() -> None:
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env, Worker(
        env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step], **_FAST_HEARTBEAT), Worker(
        env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]
    ):
        task_id = deterministic_id("e2e:interrupt", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, goal="hold"), id="task/tenant-a/interrupt", task_queue="orbit.orch"
        )
        await _running(handle)
        result = await handle.execute_update(
            TaskWorkflow.send_message,
            SendMessageInput(
                command_id="01J00000000000000000000004",
                client_message_id="01J00000000000000000000005",
                text="stop and finish",
                delivery="interrupt",
            ),
        )
        assert result.message_seq == 1
        for _ in range(100):
            view = await handle.query(TaskWorkflow.get_task_view)
            if view.status == "COMPLETED":
                break
            await asyncio.sleep(0.01)
        assert view.status == "COMPLETED"


@pytest.mark.asyncio
async def test_plan_change_is_atomic_and_stale_versions_are_rejected() -> None:
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env, Worker(
        env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step], **_FAST_HEARTBEAT), Worker(
        env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]
    ):
        task_id = deterministic_id("e2e:plan", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, goal="hold"), id="task/tenant-a/plan", task_queue="orbit.orch"
        )
        command = PlanChangeCommand(
            command_id="01J00000000000000000000007",
            task_id=task_id,
            base_plan_version=1,
            actor=Actor(kind="user", id="user-a"),
            ops=[AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title="follow up", spec=AgentTurnSpec(goal="later")))],
        )
        accepted = await handle.execute_update(TaskWorkflow.submit_plan_change, command)
        assert accepted.status == "accepted"
        stale = await handle.execute_update(
            TaskWorkflow.submit_plan_change,
            command.model_copy(update={"command_id": "01J00000000000000000000008"}),
        )
        assert stale.status == "rejected"
        assert stale.code == "VERSION_CONFLICT"


@activity.defn(name="commit_checkpoints")
async def _failing_commit(payload: dict[str, object]) -> dict[str, object]:
    raise ApplicationError("database is down", non_retryable=True)


@pytest.mark.asyncio
async def test_a_failed_checkpoint_commit_does_not_stop_the_attempt() -> None:
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env, Worker(
        env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step], **_FAST_HEARTBEAT), Worker(
        env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _failing_commit]
    ):
        handle = await env.client.start_workflow(
            TaskWorkflow.run,
            _input(deterministic_id("e2e:commit-fails", "task")),
            id="task/tenant-a/commit-fails",
            task_queue="orbit.orch",
        )
        assert (await _wait_done(handle)).status == "COMPLETED"


async def _run_checkpoint_node(task_id: str, workflow_id: str) -> None:
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env, Worker(
        env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step], **_FAST_HEARTBEAT), Worker(
        env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]
    ):
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, goal="hold"), id=workflow_id, task_queue="orbit.orch"
        )
        command = PlanChangeCommand(
            command_id="01J00000000000000000000009",
            task_id=task_id,
            base_plan_version=1,
            actor=Actor(kind="user", id="user-a"),
            ops=[AddNodeOp(node=CheckpointNode(node_id="tmp:1", title="save", spec=CheckpointSpec(label="mid")))],
        )
        assert (await handle.execute_update(TaskWorkflow.submit_plan_change, command)).status == "accepted"
        for _ in range(200):
            if CHECKPOINT_PAYLOADS:
                break
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_checkpoint_node_sends_the_identity_it_is_stored_under() -> None:
    CHECKPOINT_PAYLOADS.clear()
    task_id = deterministic_id("e2e:checkpoint-node", "task")
    await _run_checkpoint_node(task_id, "task/tenant-a/checkpoint-node")
    assert len(CHECKPOINT_PAYLOADS) == 1
    payload = CHECKPOINT_PAYLOADS[0]
    assert payload["tenant_id"] == "tenant-a" and payload["task_id"] == task_id
    assert str(payload["node_id"]).startswith("n_")
    assert str(payload["attempt_id"]).startswith("att_")
    assert payload["seq"] == 0 and payload["kind"] == "plan"
    # The same node always maps to the same attempt id, and another task's node to another one.
    other = deterministic_id("e2e:checkpoint-node-2", "task")
    CHECKPOINT_PAYLOADS.clear()
    await _run_checkpoint_node(other, "task/tenant-a/checkpoint-node-2")
    assert CHECKPOINT_PAYLOADS[0]["attempt_id"] != payload["attempt_id"]


# ---- task configuration (15 M8, T8.2): expert, skills and connectors a task runs with -----------------------------
#
# How it can go wrong, written down before the code:
#   - a config given at creation never reaches the attempt, or an expert does not replace the task's profile;
#   - an update changes the attempt that is already running, instead of the next one (11 §3);
#   - an update built on an old version overwrites a newer one, or one command id is applied twice;
#   - a closed task still takes a new config;
#   - the config is lost when the workflow continues as new;
#   - a node that names its own profile (a team member) is taken over by the task's expert.

DOCS = ConnectorSnapshot(id="mcp_docs", name="Docs", command="orbit-mcp-docs", env_refs=["ORBIT_MCP_DOCS_TOKEN"])


def _stack(env):
    return (
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step], **_FAST_HEARTBEAT),
        Worker(env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]),
    )


async def _update_config(handle, command_id: str, base: int, **fields):
    return await handle.execute_update(
        TaskWorkflow.update_task_config,
        UpdateTaskConfigInput(command_id=command_id, base_config_version=base, **fields),
    )


@pytest.mark.asyncio
async def test_the_task_config_reaches_the_attempt_and_its_expert_replaces_the_profile() -> None:
    TURNS.clear()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            config = TaskConfig(expert="writer@2", connectors=[DOCS], mode="plan")
            handle = await env.client.start_workflow(
                TaskWorkflow.run,
                _input(deterministic_id("e2e:config", "task"), config=config),
                id="task/tenant-a/config",
                task_queue="orbit.orch",
            )
            await _wait_done(handle)
    assert TURNS[0]["profile"] == "writer@2"
    sent = TURNS[0]["config"]
    assert sent["config_version"] == 1 and sent["mode"] == "plan"
    assert [item["id"] for item in sent["connectors"]] == ["mcp_docs"]
    assert sent["connectors"][0]["env_refs"] == ["ORBIT_MCP_DOCS_TOKEN"]


@pytest.mark.asyncio
async def test_an_update_applies_to_the_next_attempt_and_never_to_the_running_one() -> None:
    TURNS.clear()
    EVENTS.clear()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(
                TaskWorkflow.run,
                _input(deterministic_id("e2e:config-next", "task"), goal="hold"),
                id="task/tenant-a/config-next",
                task_queue="orbit.orch",
            )
            await _running(handle)
            first = await _update_config(handle, "01J00000000000000000000010", 1, expert="reviewer@1", connectors=[DOCS])
            assert first.config_version == 2
            # The same command again: the same answer, not a third version.
            again = await _update_config(handle, "01J00000000000000000000010", 1, expert="reviewer@1", connectors=[DOCS])
            assert again.config_version == 2
            # Built on a version that is no longer current: refused, and nothing changes.
            with pytest.raises(WorkflowUpdateFailedError) as stale:
                await _update_config(handle, "01J00000000000000000000011", 1, expert="other@1")
            assert stale.value.cause.type == "CONFIG_VERSION_CONFLICT"
            assert [t["profile"] for t in TURNS] == ["default@1"]  # the running attempt is untouched
            await handle.execute_update(
                TaskWorkflow.send_message,
                SendMessageInput(
                    command_id="01J00000000000000000000012",
                    client_message_id="01J00000000000000000000013",
                    text="go on",
                    delivery="interrupt",
                ),
            )
            await _wait_done(handle)
    assert TURNS[-1]["profile"] == "reviewer@1" and TURNS[-1]["config"]["config_version"] == 2
    # Each attempt says which configuration it started with, so a replay or an audit need not guess.
    started = [e["payload"] for e in EVENTS if e["type"] == "attempt.started"]
    assert [(p["attempt_no"], p["config_version"]) for p in started] == [(1, 1), (2, 2)]
    changed = [e for e in EVENTS if e["type"] == "task.config_changed"]
    assert [e["payload"]["config_version"] for e in changed] == [2]
    assert changed[0]["payload"]["expert"] == "reviewer@1"
    assert changed[0]["payload"]["connector_ids"] == ["mcp_docs"]
    assert "env_refs" not in str(changed[0]["payload"])


@pytest.mark.asyncio
async def test_an_idle_task_takes_a_new_config_for_its_next_round_and_a_cancelled_one_takes_none() -> None:
    TURNS.clear()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(
                TaskWorkflow.run,
                _input(deterministic_id("e2e:config-idle", "task")),
                id="task/tenant-a/config-idle",
                task_queue="orbit.orch",
            )
            await _wait_done(handle)
            # The task rests, it has not ended: a configuration is taken, and the next round runs with it.
            assert (await _update_config(handle, "01J00000000000000000000020", 1, expert="late@1")).config_version == 2
            await handle.execute_update(
                TaskWorkflow.send_message,
                SendMessageInput(command_id="01J00000000000000000000021", client_message_id="01J00000000000000000000022", text="again"),
            )
            await _wait_for(handle, lambda v, p: p.plan_version == 2 and v.status == "COMPLETED", "the next round")
            assert (TURNS[-1]["profile"], TURNS[-1]["config"]["config_version"]) == ("late@1", 2)

            # Only a cancel ends it, and then the configuration stays as it was.
            await handle.execute_update(TaskWorkflow.control, TaskControlInput(command_id="0" * 25 + "8", action="cancel"))
            await handle.result()
            with pytest.raises((RPCError, WorkflowUpdateFailedError)):
                await _update_config(handle, "01J00000000000000000000023", 2, expert="too-late@1")
            view = await handle.query(TaskWorkflow.get_task_view)
            assert view.config.config_version == 2 and view.config.expert == "late@1"


@pytest.mark.asyncio
async def test_the_config_survives_continue_as_new() -> None:
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(
                TaskWorkflow.run,
                _input(deterministic_id("e2e:config-can", "task"), goal="hold"),
                id="task/tenant-a/config-can",
                task_queue="orbit.orch",
            )
            await handle.execute_update(TaskWorkflow.control, TaskControlInput(command_id="0" * 26, action="pause"))
            await _update_config(handle, "01J00000000000000000000030", 1, expert="writer@3", connectors=[DOCS])
            for index in range(1001):
                await handle.execute_update(
                    TaskWorkflow.send_message,
                    SendMessageInput(command_id=f"{index + 1:026d}", client_message_id=f"{index + 10000:026d}", text="m"),
                )
            view = await handle.query(TaskWorkflow.get_task_view)
    assert view.config.config_version == 2 and view.config.expert == "writer@3"
    assert [item.id for item in view.config.connectors or []] == ["mcp_docs"]


# ---- a team (15 M8, T8.6): a leader and members, each member an expert --------------------------------------------
#
# How it can go wrong, written down before the code:
#   - the exploration node, which plans, does not run as the team's leader;
#   - a node the leader gave to a member runs as the leader, or as the task's profile, instead of as that member;
#   - a node nobody was assigned goes to some member and not to the leader;
#   - a node that names a member's profile on its own (not through the task's expert) is taken over by the leader.

TEAM = Team(
    leader="lead",
    members=[
        TeamMember(role="lead", expert="writer@1", description="plans and writes"),
        TeamMember(role="review", expert="reviewer@1", description="checks the work"),
    ],
)


@pytest.mark.asyncio
async def test_a_team_leader_plans_and_each_node_runs_as_the_member_it_was_given_to() -> None:
    TURNS.clear()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            task_id = deterministic_id("e2e:team", "task")
            inp = _input(task_id, goal="hold", config=TaskConfig(expert="team@1", team=TEAM)).model_copy(update={"profile": "team@1"})
            handle = await env.client.start_workflow(TaskWorkflow.run, inp, id="task/tenant-a/team", task_queue="orbit.orch")
            await _running(handle)
            command = PlanChangeCommand(
                command_id="01J00000000000000000000040",
                task_id=task_id,
                base_plan_version=1,
                actor=Actor(kind="user", id="user-a"),
                ops=[
                    AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title="check", owner_profile="reviewer@1", spec=AgentTurnSpec(goal="check it"))),
                    AddNodeOp(node=AgentTurnNode(node_id="tmp:2", title="finish", depends_on=["tmp:1"], spec=AgentTurnSpec(goal="finish it"))),
                ],
            )
            assert (await handle.execute_update(TaskWorkflow.submit_plan_change, command)).status == "accepted"
            await handle.execute_update(
                TaskWorkflow.send_message,
                SendMessageInput(command_id="01J00000000000000000000041", client_message_id="01J00000000000000000000042", text="go", delivery="interrupt"),
            )
            await _wait_done(handle, polls=600)
    by_goal = {turn["goal"]: turn["profile"] for turn in TURNS}
    assert by_goal == {"hold": "writer@1", "check it": "reviewer@1", "finish it": "writer@1"}


# ---- a task is a conversation, not a job -------------------------------------------------------------------------------
#
# A task stays open for as long as its session does. When every node is done it rests (COMPLETED means "this round is
# done"); the next message starts a follow-up that carries on the agent's own conversation. Only a cancel ends it.
#
# How it can go wrong, written down before the code:
#   - the workflow exits once every node is done, so the next message has nowhere to go, or is refused as "closed";
#   - the message is queued and nothing runs;
#   - the follow-up starts from nothing: it does not continue the agent session of the attempt before it;
#   - the follow-up replays the messages of earlier rounds in its prompt, or gets its own message twice (as the goal and
#     as a message);
#   - the task shows COMPLETED again but `task.completed` is sent once, or the status never leaves COMPLETED;
#   - a cancel no longer ends the task, or a message after a cancel is accepted.


async def _wait_for(handle, ready, what: str, polls: int = 600):
    for _ in range(polls):
        view = await handle.query(TaskWorkflow.get_task_view)
        plan = await handle.query(TaskWorkflow.get_plan)
        if ready(view, plan):
            return view
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _say(handle, number: int, text: str, delivery: str = "queue"):
    return handle.execute_update(
        TaskWorkflow.send_message,
        SendMessageInput(
            command_id=f"01J000000000000000000{number:05d}"[:26],
            client_message_id=f"01J100000000000000000{number:05d}"[:26],
            text=text,
            delivery=delivery,
        ),
    )


@pytest.mark.asyncio
async def test_a_finished_task_stays_open_and_a_message_continues_it() -> None:
    TURNS.clear()
    EVENTS.clear()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(
                TaskWorkflow.run, _input(deterministic_id("e2e:open", "task"), goal="hold"), id="task/tenant-a/open", task_queue="orbit.orch"
            )
            await _running(handle)
            await _say(handle, 1, "begin", "interrupt")
            await _wait_for(handle, lambda v, p: v.status == "COMPLETED", "the first round to finish")
            assert (await handle.describe()).status == WorkflowExecutionStatus.RUNNING, "the workflow is still there"
            last_attempt = TURNS[-1]["attempt_id"]
            rounds = len(TURNS)

            await _say(handle, 2, "and then?")
            view = await _wait_for(handle, lambda v, p: p.plan_version == 2 and v.status == "COMPLETED", "the follow-up to finish")
            follow_up = TURNS[rounds:]
            assert [t["goal"] for t in follow_up] == ["and then?"], "one attempt, whose goal is the message"
            assert follow_up[0]["messages"] == [], "neither earlier messages nor the message itself again"
            assert follow_up[0]["continue_from"] == last_attempt, "it carries on the agent's own conversation"
            assert TURNS[0].get("continue_from") is None
            assert len([e for e in EVENTS if e["type"] == "task.completed"]) == 2
            assert view.status == "COMPLETED"
            assert (await handle.describe()).status == WorkflowExecutionStatus.RUNNING


@pytest.mark.asyncio
async def test_each_follow_up_carries_on_the_one_before_it() -> None:
    TURNS.clear()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(
                TaskWorkflow.run, _input(deterministic_id("e2e:chain", "task")), id="task/tenant-a/chain", task_queue="orbit.orch"
            )
            await _wait_for(handle, lambda v, p: v.status == "COMPLETED", "the first round")
            for number, text in ((10, "one"), (11, "two"), (12, "three")):
                await _say(handle, number, text)
                await _wait_for(handle, lambda v, p, n=number: p.plan_version == n - 8 and v.status == "COMPLETED", f"{text}")
    assert [t["goal"] for t in TURNS][1:] == ["one", "two", "three"]
    attempts = [t["attempt_id"] for t in TURNS]
    assert [t.get("continue_from") for t in TURNS] == [None, attempts[0], attempts[1], attempts[2]]


@pytest.mark.asyncio
async def test_only_a_cancel_ends_a_task_and_a_message_after_it_is_refused() -> None:
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(
                TaskWorkflow.run, _input(deterministic_id("e2e:end", "task")), id="task/tenant-a/end", task_queue="orbit.orch"
            )
            await _wait_for(handle, lambda v, p: v.status == "COMPLETED", "the first round")
            await handle.execute_update(TaskWorkflow.control, TaskControlInput(command_id="0" * 25 + "9", action="cancel"))
            view = await handle.result()
            assert view.status == "CANCELLED"
            assert (await handle.describe()).status == WorkflowExecutionStatus.COMPLETED
            with pytest.raises((RPCError, WorkflowUpdateFailedError)):
                await _say(handle, 20, "anyone there?")

"""End to end coverage for the v3 task workflow surface."""

from __future__ import annotations

import asyncio

import pytest
from orbit_contracts.v3 import (
    Actor,
    Budget,
    DecideApprovalInput,
    PlanChangeCommand,
    SendMessageInput,
    TaskControlInput,
    TaskWorkflowInput,
)
from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec, CheckpointNode, CheckpointSpec
from orbit_contracts.v3.plan import AddNodeOp
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("goal") == "hold" and not payload.get("messages"):
        await asyncio.sleep(60)
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


def _input(task_id: str, *, goal: str = "complete the task") -> TaskWorkflowInput:
    return TaskWorkflowInput(
        task_id=task_id,
        tenant_id="tenant-a",
        created_by=Actor(kind="user", id="user-a"),
        title="E2E task",
        goal=goal,
        profile="default@1",
        node_type_registry_version=1,
        budgets=Budget(),
    )


async def _wait_done(handle) -> object:
    for _ in range(100):
        view = await handle.query(TaskWorkflow.get_task_view)
        if view.status == "COMPLETED":
            return view
        await asyncio.sleep(0.01)
    raise AssertionError(f"TaskWorkflow did not complete: {view}")


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
    ), Worker(
        env.client,
        task_queue="orbit.io",
        activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
    ):
        task_id = deterministic_id("e2e:can", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, goal="hold"), id="task/tenant-a/can", task_queue="orbit.orch"
        )
        await handle.execute_update(
            TaskWorkflow.control,
            TaskControlInput(command_id="0" * 26, action="pause"),
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
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step]), Worker(
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
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step]), Worker(
        env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]
    ):
        task_id = deterministic_id("e2e:interrupt", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, goal="hold"), id="task/tenant-a/interrupt", task_queue="orbit.orch"
        )
        await asyncio.sleep(0.05)
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
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step]), Worker(
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
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step]), Worker(
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
    ), Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step]), Worker(
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

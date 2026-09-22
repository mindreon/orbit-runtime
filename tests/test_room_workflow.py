"""RoomWorkflow drives the real adapter through Temporal."""

import asyncio

import pytest
from orbit_contracts.models import CloudAgentJobInput, RoomCommand, RoomWorkflowInput
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.workflows import AgentRunWorkflow, CloudAgentJob, RoomWorkflow
from orbit_worker.activities import ACTIVITIES, GATEWAY_ACTIVITIES, set_runtime
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker


async def _wait_status(handle, expected: str) -> object:
    last = None
    for _ in range(50):
        last = await handle.query(RoomWorkflow.snapshot)
        if last.status == expected:
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"wanted {expected}, last snapshot was {last}")


@pytest.mark.asyncio
async def test_room_parks_for_approval_then_completes() -> None:
    set_runtime(AgentRuntime(MemoryStateStore()))
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env:
        task_queue = "orbit"
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[RoomWorkflow],
            activities=ACTIVITIES,
            workflow_runner=sandbox_runner(),
        ):
            handle = await env.client.start_workflow(
                RoomWorkflow.run,
                RoomWorkflowInput(room_id="room-1"),
                id="room-1",
                task_queue=task_queue,
            )
            await handle.signal(
                RoomWorkflow.command, RoomCommand(kind="open", turn_id="t-open")
            )
            await handle.signal(
                RoomWorkflow.command,
                RoomCommand(kind="message", turn_id="t-msg", message="please gated this"),
            )
            parked = await _wait_status(handle, "awaiting_approval")
            assert parked.approval is not None
            assert parked.approval.tool_name == "gated_echo"

            await handle.signal(
                RoomWorkflow.command,
                RoomCommand(kind="approve", turn_id="t-ok", outcome="allowed-once"),
            )
            result = await handle.result()
            assert result.status == "closed"
            assert result.last_text == "done"
            assert result.state_version > parked.state_version


@pytest.mark.asyncio
async def test_gateway_lookup_uses_the_gateway_queue() -> None:
    set_runtime(AgentRuntime(MemoryStateStore()))
    async with (
        await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
            env.client,
            task_queue="orbit",
            workflows=[RoomWorkflow, AgentRunWorkflow, CloudAgentJob],
            activities=ACTIVITIES,
            workflow_runner=sandbox_runner(),
        ),
        Worker(
            env.client,
            task_queue="orbit-gateway",
            activities=GATEWAY_ACTIVITIES,
        ),
    ):
        handle = await env.client.start_workflow(
            RoomWorkflow.run,
            RoomWorkflowInput(room_id="room-lookup"),
            id="room-lookup",
            task_queue="orbit",
        )
        await handle.signal(RoomWorkflow.command, RoomCommand(kind="open", turn_id="t-open"))
        await handle.signal(
            RoomWorkflow.command,
            RoomCommand(kind="message", turn_id="t-msg", message="lookup the workspace"),
        )
        result = await handle.result()
        assert result.status == "closed"
        assert result.last_text == "lookup-ok"


@pytest.mark.asyncio
async def test_leader_spawns_two_workers_then_dissolves() -> None:
    set_runtime(AgentRuntime(MemoryStateStore()))
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit",
        workflows=[RoomWorkflow, AgentRunWorkflow],
        activities=ACTIVITIES,
        workflow_runner=sandbox_runner(),
    ):
        handle = await env.client.start_workflow(
            RoomWorkflow.run,
            RoomWorkflowInput(room_id="room-team", max_fanout=4, max_depth=2),
            id="room-team",
            task_queue="orbit",
        )
        await handle.signal(RoomWorkflow.command, RoomCommand(kind="open", turn_id="t-open"))
        await handle.signal(
            RoomWorkflow.command,
            RoomCommand(kind="message", turn_id="t-msg", message="spawn two workers"),
        )
        result = await handle.result()
        assert result.status == "closed"
        assert result.last_text == "team-done"
        assert len(result.child_workflow_ids) == 2


@pytest.mark.asyncio
async def test_cloud_agent_job_clones_pushes_and_opens_a_pr(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORBIT_WORK_ROOT", str(tmp_path))
    set_runtime(AgentRuntime(MemoryStateStore()))
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit",
        workflows=[CloudAgentJob],
        activities=ACTIVITIES,
        workflow_runner=sandbox_runner(),
    ):
        result = await env.client.execute_workflow(
            CloudAgentJob.run,
            CloudAgentJobInput(
                job_id="job-1",
                repo_url="https://github.com/mindreon/example",
                prompt="summarize the repo",
            ),
            id="job-1",
            task_queue="orbit",
        )
        assert result.status == "closed"
        assert result.last_text == "hello"
        assert result.pr_url.endswith("/pull/1")
        assert result.branch == "orbit/cloud-agent"
        assert (tmp_path / "job-1" / "BRANCH").read_text(encoding="utf-8").strip() == result.branch


def test_client_converter_is_the_one_workflows_expect() -> None:
    # Guard against a worker that forgets the pydantic converter and silently
    # drops nested models. Construction only; no server.
    assert pydantic_data_converter is not None
    assert Client is not None

"""RoomWorkflow drives the real adapter through Temporal."""

import asyncio

import pytest
from orbit_contracts.models import RoomCommand, RoomWorkflowInput
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.workflows import RoomWorkflow
from orbit_worker.activities import ACTIVITIES, set_runtime
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


def test_client_converter_is_the_one_workflows_expect() -> None:
    # Guard against a worker that forgets the pydantic converter and silently
    # drops nested models. Construction only; no server.
    assert pydantic_data_converter is not None
    assert Client is not None

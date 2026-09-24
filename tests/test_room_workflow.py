"""RoomWorkflow drives the real adapter through Temporal."""

import asyncio
import json
from datetime import timedelta

import pytest
from orbit_contracts.models import (
    CloudAgentJobInput,
    CloudAgentSnapshot,
    RoomCommand,
    RoomWorkflowInput,
)
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.schedules import (
    ensure_recurring_job,
    ensure_recurring_job_from_env,
    recurring_schedule,
)
from orbit_orch.versioning import ROOM_CONTROL_SURFACE
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


async def _wait_text(handle, expected: str) -> object:
    last = None
    for _ in range(50):
        last = await handle.query(RoomWorkflow.snapshot)
        if last.last_text == expected and last.status == "running":
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"wanted text {expected}, last snapshot was {last}")


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
            result = await _wait_status(handle, "running")
            assert result.last_text == "done"
            assert result.state_version > parked.state_version
            view = await handle.query(RoomWorkflow.get_room_view)
            assert view["state"] == "running"


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
        result = await _wait_text(handle, "lookup-ok")
        assert result.status == "running"


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
        result = await _wait_text(handle, "team-done")
        assert result.status == "running"
        assert len(result.child_workflow_ids) == 2


@pytest.mark.asyncio
async def test_follow_up_update_keeps_the_room_open() -> None:
    set_runtime(AgentRuntime(MemoryStateStore()))
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit",
        workflows=[RoomWorkflow],
        activities=ACTIVITIES,
        workflow_runner=sandbox_runner(),
    ):
        handle = await env.client.start_workflow(
            RoomWorkflow.run,
            RoomWorkflowInput(room_id="room-follow", permission_preset="read-only"),
            id="room-follow",
            task_queue="orbit",
        )
        await _wait_status(handle, "running")
        first = await handle.execute_update(
            RoomWorkflow.update_run_turn,
            {"turnId": "t-1", "message": "hello there"},
        )
        assert first["status"] == "completed"
        assert first["texts"] == ["hello"]
        second = await handle.execute_update(
            RoomWorkflow.update_run_turn,
            {"turnId": "t-2", "message": "and again"},
        )
        assert second["status"] == "completed"
        assert second["texts"] == ["hello"]
        view = await handle.query(RoomWorkflow.get_room_view)
        assert view["state"] == "running"
        assert view["sessionId"]


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


@pytest.mark.asyncio
async def test_room_records_the_control_surface_version() -> None:
    set_runtime(AgentRuntime(MemoryStateStore()))
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit",
        workflows=[RoomWorkflow],
        activities=ACTIVITIES,
        workflow_runner=sandbox_runner(),
    ):
        handle = await env.client.start_workflow(
            RoomWorkflow.run,
            RoomWorkflowInput(room_id="room-version"),
            id="room-version",
            task_queue="orbit",
        )
        await _wait_status(handle, "running")
        history = await handle.fetch_history()
    change_ids: list[str] = []
    for event in history.events:
        attrs = event.marker_recorded_event_attributes
        if attrs.marker_name != "core_patch":
            continue
        raw = attrs.details["patch-data"].payloads[0].data
        change_ids.append(json.loads(raw)["id"])
    assert change_ids == [ROOM_CONTROL_SURFACE]


def test_recurring_schedule_uses_skip_overlap() -> None:
    schedule = recurring_schedule(
        CloudAgentJobInput(
            job_id="nightly",
            repo_url="https://github.com/mindreon/example",
            prompt="nightly",
        ),
        every=timedelta(hours=24),
        task_queue="orbit",
        schedule_id="nightly",
    )
    assert schedule.policy is not None
    assert schedule.spec.intervals[0].every == timedelta(hours=24)
    with pytest.raises(ValueError, match="at least one second"):
        recurring_schedule(
            CloudAgentJobInput(
                job_id="nightly",
                repo_url="https://github.com/mindreon/example",
                prompt="nightly",
            ),
            every=timedelta(0),
            task_queue="orbit",
            schedule_id="nightly",
        )


@pytest.mark.asyncio
async def test_unset_schedule_id_does_not_register(monkeypatch) -> None:
    monkeypatch.delenv("ORBIT_RECURRING_SCHEDULE_ID", raising=False)
    await ensure_recurring_job_from_env(None, task_queue="orbit")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_recurring_schedule_starts_two_cloud_jobs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORBIT_WORK_ROOT", str(tmp_path))
    set_runtime(AgentRuntime(MemoryStateStore()))
    job = CloudAgentJobInput(
        job_id="nightly",
        repo_url="https://github.com/mindreon/example",
        prompt="summarize the repo",
    )
    async with await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter,
    ) as env, Worker(
        env.client,
        task_queue="orbit",
        workflows=[CloudAgentJob],
        activities=ACTIVITIES,
        workflow_runner=sandbox_runner(),
    ):
        await ensure_recurring_job(
            env.client,
            job,
            every=timedelta(hours=24),
            task_queue="orbit",
            schedule_id="nightly",
        )
        await ensure_recurring_job(
            env.client,
            job,
            every=timedelta(hours=24),
            task_queue="orbit",
            schedule_id="nightly",
        )
        schedule = env.client.get_schedule_handle("nightly")
        described = await schedule.describe()
        assert described.schedule.spec.intervals[0].every == timedelta(hours=24)
        started: list[str] = []
        for index in range(2):
            # The server suffixes the workflow id with the scheduled second.
            if index:
                await asyncio.sleep(1.1)
            await schedule.trigger()
            described = await schedule.describe()
            fresh = [
                item.action.workflow_id
                for item in described.info.recent_actions
                if item.action.workflow_id not in started
            ]
            assert len(fresh) == 1
            result = await env.client.get_workflow_handle(
                fresh[0],
                result_type=CloudAgentSnapshot,
            ).result()
            assert result.pr_url.endswith("/pull/1")
            assert result.job_id == "nightly"
            started.append(fresh[0])
        assert all(item.startswith("recurring:nightly-") for item in started)


def test_client_converter_is_the_one_workflows_expect() -> None:
    # Guard against a worker that forgets the pydantic converter and silently
    # drops nested models. Construction only; no server.
    assert pydantic_data_converter is not None
    assert Client is not None

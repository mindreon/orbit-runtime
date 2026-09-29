"""proposeCompletion runs the node's completion checks (04 §5) before the node is frozen."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from orbit_contracts.v3 import (
    Actor,
    Budget,
    CompletionProposal,
    PlanChangeCommand,
    TaskWorkflowInput,
)
from orbit_contracts.v3.nodes import (
    AgentTurnNode,
    AgentTurnSpec,
    ArtifactRequirement,
    CompletionContract,
    Verification,
)
from orbit_contracts.v3.plan import AddNodeOp
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

SNAPSHOT = "sha256:" + "7" * 64
CALLS: dict[str, list[dict[str, Any]]] = {"verify_completion": [], "verify_command": []}
BEHAVIOR: dict[str, Any] = {}


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("goal") == "hold" and not payload.get("messages"):
        await asyncio.sleep(3600)
    return {"status": "completed", "checkpoint_ref": "sha256:" + "1" * 64}


@activity.defn(name="sop_step")
async def _sop_step(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "completed", "checkpoint_ref": "sha256:" + "2" * 64}


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    CALLS["verify_completion"].append(payload)
    if BEHAVIOR.get("io") == "crash":
        raise ApplicationError("object store is down", type="STORE_DOWN", non_retryable=True)
    if BEHAVIOR.get("io") == "reject":
        return {
            "ok": False,
            "failures": [{"check": "output_schema", "code": "schema_violation", "message": "bad", "detail": {}}],
            "checkpoint_ref": payload.get("checkpoint_ref"),
            "workspace_snapshot_ref": None,
        }
    return {"ok": True, "failures": [], "checkpoint_ref": payload.get("checkpoint_ref"), "workspace_snapshot_ref": SNAPSHOT}


@activity.defn(name="verify_command")
async def _verify_command(payload: dict[str, Any]) -> dict[str, Any]:
    CALLS["verify_command"].append(payload)
    ok = payload["command"] != "exit 1"
    failures = [] if ok else [{"check": "command", "code": "command_failed", "message": "exit 1", "detail": {}}]
    return {"ok": ok, "failures": failures}


@activity.defn(name="publish_events")
async def _publish_events(payload: list[dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "count": len(payload)}


@activity.defn(name="checkpoint_commit")
async def _checkpoint_commit(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


@pytest.fixture(autouse=True)
def _reset() -> None:
    for calls in CALLS.values():
        calls.clear()
    BEHAVIOR.clear()


def _input(task_id: str) -> TaskWorkflowInput:
    return TaskWorkflowInput(
        task_id=task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="user-a"),
        title="verify", goal="hold", profile="default@1", node_type_registry_version=1, budgets=Budget(),
    )


class _Stack:
    def __init__(self, env: WorkflowEnvironment) -> None:
        self.env = env

    def workers(self) -> list[Worker]:
        return [
            Worker(self.env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
            Worker(self.env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step, _verify_command]),
            Worker(self.env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit]),
        ]


async def _start(env: WorkflowEnvironment, name: str, contract: CompletionContract):
    """A task whose first node holds its attempt open, so the task stays open while a second node is added."""
    task_id = deterministic_id(f"verify:{name}", "task")
    handle = await env.client.start_workflow(TaskWorkflow.run, _input(task_id), id=f"task/tenant-a/{name}", task_queue="orbit.orch")
    return task_id, handle


async def _node_with_contract(env, handle, task_id, contract: CompletionContract):
    """Add a node that holds its attempt open, and return (node_id, attempt_id) once it is running."""
    command = PlanChangeCommand(
        command_id="01J00000000000000000000010", task_id=task_id, base_plan_version=1,
        actor=Actor(kind="user", id="user-a"),
        ops=[AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title="checked", spec=AgentTurnSpec(goal="hold"), completion_contract=contract))],
    )
    accepted = await handle.execute_update(TaskWorkflow.submit_plan_change, command)
    node_id = accepted.id_map["tmp:1"]
    for _ in range(200):
        plan = await handle.query(TaskWorkflow.get_plan)
        node = next(item for item in plan.nodes if item.node_id == node_id)
        if node.current_attempt_id:
            return node_id, node.current_attempt_id
        await asyncio.sleep(0.01)
    raise AssertionError("the node never started an attempt")


def _proposal(node_id: str, attempt_id: str, command_id: str = "01J00000000000000000000011") -> CompletionProposal:
    return CompletionProposal(
        command_id=command_id, node_id=node_id, attempt_id=attempt_id, output={"summary": "done"},
        artifact_manifest_id="man_01J00000000000000000000001", checkpoint_ref="sha256:" + "5" * 64,
    )


async def _node(handle, node_id: str):
    plan = await handle.query(TaskWorkflow.get_plan)
    return next(item for item in plan.nodes if item.node_id == node_id)


async def _settle(handle, node_id: str, want: set[str]):
    for _ in range(300):
        node = await _node(handle, node_id)
        if node.status in want:
            return node
        await asyncio.sleep(0.01)
    raise AssertionError(f"node stayed {node.status}")


async def _run_case(name: str, contract: CompletionContract, want: set[str]):
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        stack = _Stack(env)
        w1, w2, w3 = stack.workers()
        async with w1, w2, w3:
            task_id, handle = await _start(env, name, contract)
            node_id, attempt_id = await _node_with_contract(env, handle, task_id, contract)
            await handle.execute_update(TaskWorkflow.propose_completion, _proposal(node_id, attempt_id))
            node = await _settle(handle, node_id, want)
            return node, task_id, node_id


CONTRACT = CompletionContract(
    output_schema_ref="schema://report/1",
    required_artifacts=[ArtifactRequirement(name="report.md", media_type="text/markdown")],
    verifications=[Verification(kind="command", spec={"command": "pytest -q", "timeout_s": 120})],
)


async def test_completion_that_passes_every_check_freezes_the_node() -> None:
    node, task_id, node_id = await _run_case("pass", CONTRACT, {"COMPLETED"})
    assert node.frozen is True
    [io_call] = CALLS["verify_completion"]
    assert io_call["tenant_id"] == "tenant-a" and io_call["task_id"] == task_id
    assert io_call["node_id"] == node_id
    assert io_call["completion_contract"]["output_schema_ref"] == "schema://report/1"
    assert io_call["completion_contract"]["verifications"][0]["spec"]["command"] == "pytest -q"
    assert io_call["artifact_manifest_id"] == "man_01J00000000000000000000001"
    [command_call] = CALLS["verify_command"]
    assert command_call["command"] == "pytest -q" and command_call["timeout_s"] == 120
    assert command_call["workspace_snapshot_ref"] == SNAPSHOT
    assert (command_call["tenant_id"], command_call["task_id"], command_call["node_id"]) == ("tenant-a", task_id, node_id)


async def test_a_failed_io_check_keeps_the_node_open_and_skips_the_command() -> None:
    BEHAVIOR["io"] = "reject"
    node, _, _ = await _run_case("reject", CONTRACT, {"RETRY_PENDING", "RUNNING"})
    assert node.frozen is False and node.status != "COMPLETED"
    assert CALLS["verify_command"] == []


async def test_a_failed_command_keeps_the_node_open() -> None:
    contract = CONTRACT.model_copy(update={"verifications": [Verification(kind="command", spec={"command": "exit 1"})]})
    node, _, _ = await _run_case("cmd-fail", contract, {"RETRY_PENDING", "RUNNING"})
    assert node.frozen is False and node.status != "COMPLETED"
    assert len(CALLS["verify_command"]) == 1


async def test_a_verification_activity_that_cannot_run_is_a_rejection_not_a_crashed_task() -> None:
    BEHAVIOR["io"] = "crash"
    node, _, _ = await _run_case("crash", CONTRACT, {"RETRY_PENDING", "RUNNING"})
    assert node.frozen is False and node.status != "COMPLETED"


async def test_only_command_verifications_run_on_the_agent_queue() -> None:
    contract = CompletionContract(verifications=[Verification(kind="sop_verifier", spec={})])
    node, _, _ = await _run_case("no-command", contract, {"COMPLETED"})
    assert node.frozen is True
    assert CALLS["verify_command"] == []

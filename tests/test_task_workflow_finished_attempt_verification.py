"""An attempt that ends `completed` is verified like a completion proposal before its node is frozen (17 G15, 04 §5).

The worker reports an attempt's end with `attemptFinished`; that is the production path to COMPLETED, so the node's
completion contract is checked there. A rejected attempt is spent: a report of its end that arrives late, or one for
an attempt that was replaced, never completes the node.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest
from orbit_contracts.v3 import (
    Actor,
    Budget,
    CompletionProposal,
    PlanChangeCommand,
    TaskControlInput,
    TaskWorkflowInput,
)
from orbit_contracts.v3.messages import AttemptFinishedSignal, AttemptResult
from orbit_contracts.v3.nodes import (
    AgentTurnNode,
    AgentTurnSpec,
    ArtifactRequirement,
    CompletionContract,
    Verification,
)
from orbit_contracts.v3.plan import AddNodeOp
from orbit_orch.plan_engine import attempt_workflow_id, deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

CHECKPOINT = "sha256:" + "1" * 64
SNAPSHOT = "sha256:" + "7" * 64
MANIFEST = "man_01J00000000000000000000001"

CALLS: dict[str, list[dict[str, Any]]] = {"verify_completion": [], "verify_command": [], "agent_turn": []}
EVENTS: list[dict[str, Any]] = []
# (goal, attempt_no) -> the attempt's activity waits for the event before it reports; no entry means it does not wait.
GATES: dict[tuple[str, int], asyncio.Event] = {}
# The answers of the verification activities, one per call in order; when they run out the completion is accepted.
VERDICTS: list[str] = []


def _artifact_missing() -> dict[str, Any]:
    return {
        "check": "required_artifact", "code": "artifact_missing", "message": "0 of 1 required artifacts named report.md",
        "detail": {"name": "report.md"},
    }


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    CALLS["agent_turn"].append(payload)
    goal, number = str(payload["goal"]), int(payload["attempt_no"])
    if goal == "hold":
        await asyncio.sleep(3600)
    gate = GATES.get((goal, number))
    if gate is not None:
        await gate.wait()
    return {"status": "completed", "checkpoint_ref": CHECKPOINT, "manifest_id": MANIFEST, "manifest_entries": []}


@activity.defn(name="sop_step")
async def _sop_step(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "completed", "checkpoint_ref": "sha256:" + "2" * 64}


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    CALLS["verify_completion"].append(payload)
    verdict = VERDICTS.pop(0) if VERDICTS else "accept"
    if verdict == "crash":
        raise ApplicationError("object store is down", type="STORE_DOWN", non_retryable=True)
    failures = [_artifact_missing()] if verdict == "reject" else []
    return {"ok": not failures, "failures": failures, "checkpoint_ref": payload.get("checkpoint_ref"), "workspace_snapshot_ref": SNAPSHOT}


@activity.defn(name="verify_command")
async def _verify_command(payload: dict[str, Any]) -> dict[str, Any]:
    CALLS["verify_command"].append(payload)
    failed = payload["command"] == "exit 1"
    failures = [{"check": "command", "code": "command_failed", "message": "exit 1", "detail": {}}] if failed else []
    return {"ok": not failed, "failures": failures}


# How long publishing takes: the main loop is inside it while a slower report is being verified.
PUBLISH_DELAY_S: list[float] = [0.0]


@activity.defn(name="publish_events")
async def _publish_events(payload: list[dict[str, Any]]) -> dict[str, Any]:
    await asyncio.sleep(PUBLISH_DELAY_S[0])
    EVENTS.extend(payload)
    return {"ok": True, "count": len(payload)}


@activity.defn(name="checkpoint_commit")
async def _checkpoint_commit(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


@activity.defn(name="commit_checkpoints")
async def _commit_checkpoints(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "committed": 1}


@pytest.fixture(autouse=True)
def _reset() -> None:
    for calls in CALLS.values():
        calls.clear()
    EVENTS.clear()
    GATES.clear()
    VERDICTS.clear()
    PUBLISH_DELAY_S[0] = 0.0


CONTRACT = CompletionContract(
    required_artifacts=[ArtifactRequirement(name="report.md", media_type="text/markdown")],
    verifications=[Verification(kind="command", spec={"command": "pytest -q", "timeout_s": 120})],
)


class _Run:
    """A task whose exploration node holds forever, plus one node `work` with the contract under test."""

    def __init__(self, env: WorkflowEnvironment, name: str, contract: CompletionContract) -> None:
        self.env, self.name, self.contract = env, name, contract
        self.task_id = deterministic_id(f"finished:{name}", "task")
        self.node_id = ""
        self.handle: Any = None

    async def start(self) -> None:
        self.handle = await self.env.client.start_workflow(
            TaskWorkflow.run,
            TaskWorkflowInput(
                task_id=self.task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="user-a"), title="finished",
                goal="hold", profile="default@1", node_type_registry_version=1, budgets=Budget(),
            ),
            id=f"task/tenant-a/{self.name}", task_queue="orbit.orch",
        )
        command = PlanChangeCommand(
            command_id="01J00000000000000000000010", task_id=self.task_id, base_plan_version=1,
            actor=Actor(kind="user", id="user-a"),
            ops=[AddNodeOp(node=AgentTurnNode(
                node_id="tmp:1", title="checked", spec=AgentTurnSpec(goal="work"), completion_contract=self.contract
            ))],
        )
        self.node_id = (await self.handle.execute_update(TaskWorkflow.submit_plan_change, command)).id_map["tmp:1"]

    async def node(self) -> Any:
        plan = await self.handle.query(TaskWorkflow.get_plan)
        return next(item for item in plan.nodes if item.node_id == self.node_id)

    async def until(self, predicate: Any, what: str) -> Any:
        node = None
        deadline = asyncio.get_running_loop().time() + 60  # wall clock: a wait for a retry sits out its 5s backoff in real time
        while asyncio.get_running_loop().time() < deadline:
            node = await self.node()
            if predicate(node):
                return node
            await asyncio.sleep(0.05)
        raise AssertionError(f"{what}; the node is {node}")

    def attempt_id(self, number: int) -> str:
        return deterministic_id(f"{self.task_id}:{self.node_id}:{number}", "att")

    def finished(self, number: int, outcome: str = "completed", attempt_id: str | None = None) -> AttemptFinishedSignal:
        return AttemptFinishedSignal(
            attempt_workflow_id=attempt_workflow_id(self.task_id, self.node_id, number),
            node_id=self.node_id, attempt_no=number, attempt_id=attempt_id or self.attempt_id(number), outcome=outcome,  # type: ignore[arg-type]
            result=AttemptResult(checkpoint_ref=CHECKPOINT, manifest_id=MANIFEST),
        )


@contextlib.asynccontextmanager
async def _run(name: str, contract: CompletionContract = CONTRACT) -> AsyncIterator[_Run]:
    env_ctx = await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)
    async with env_ctx as env, contextlib.AsyncExitStack() as stack:
        for worker in (
            Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
            Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step, _verify_command]),
            Worker(
                env.client, task_queue="orbit.io",
                activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
            ),
        ):
            await stack.enter_async_context(worker)
        run = _Run(env, name, contract)
        await run.start()
        yield run


async def _events(kind: str, node_id: str | None = None, at_least: int = 0) -> list[dict[str, Any]]:
    """The published payloads of one kind; publishing is asynchronous, so wait for `at_least` of them."""
    found: list[dict[str, Any]] = []
    for _ in range(400):
        found = [
            event["payload"] for event in EVENTS
            if event["type"] == kind and (node_id is None or event["payload"].get("node_id") == node_id)
        ]
        if len(found) >= at_least:
            break
        await asyncio.sleep(0.01)
    return found


async def test_an_attempt_that_passes_the_contract_completes_and_freezes_its_node() -> None:
    async with _run("pass") as run:
        node = await run.until(lambda n: n.status == "COMPLETED", "the node never completed")
        assert node.frozen is True and node.attempt_count == 1
        [io_call] = CALLS["verify_completion"]
        assert io_call["tenant_id"] == "tenant-a" and io_call["task_id"] == run.task_id and io_call["node_id"] == run.node_id
        assert io_call["attempt_id"] == run.attempt_id(1)
        assert io_call["artifact_manifest_id"] == MANIFEST and io_call["checkpoint_ref"] == CHECKPOINT
        assert io_call["completion_contract"]["required_artifacts"][0]["name"] == "report.md"
        [command_call] = CALLS["verify_command"]
        assert command_call["command"] == "pytest -q" and command_call["workspace_snapshot_ref"] == SNAPSHOT
        assert command_call["attempt_id"] == run.attempt_id(1)


async def test_a_node_with_an_empty_contract_is_still_checked_and_completes() -> None:
    async with _run("empty", CompletionContract()) as run:
        node = await run.until(lambda n: n.status == "COMPLETED", "the node never completed")
        assert node.frozen is True
        assert len(CALLS["verify_completion"]) == 1 and CALLS["verify_command"] == []


async def test_a_rejected_attempt_leaves_the_node_open_and_a_new_attempt_takes_over() -> None:
    VERDICTS.extend(["reject", "accept"])
    async with _run("reject") as run:
        node = await run.until(lambda n: n.status == "COMPLETED", "the retry never completed the node")
        assert node.attempt_count == 2 and node.frozen is True
        # The rejection stopped the command checks: only the accepted attempt reached them.
        assert [call["attempt_id"] for call in CALLS["verify_command"]] == [run.attempt_id(2)]
        [first, second] = await _events("attempt.finished", run.node_id, at_least=2)
        assert first["attempt_id"] == run.attempt_id(1) and first["outcome"] == "failed"
        assert first["failure"]["failure_class"] == "verification" and first["failure"]["retryable"] is True
        assert "artifact_missing" in first["failure"]["message"] and "report.md" in first["failure"]["message"]
        assert second["outcome"] == "completed" and second["failure"] is None
        changes = await _events("node.status_changed", run.node_id, at_least=4)
        retry = [item for item in changes if item["to_status"] == "RETRY_PENDING"]
        assert len(retry) == 1 and "artifact_missing" in retry[0]["reason"]
        assert not [item for item in changes if item["to_status"] == "COMPLETED" and item["from_status"] == "RUNNING"]


async def test_a_failed_command_check_rejects_the_attempt() -> None:
    contract = CONTRACT.model_copy(update={"verifications": [Verification(kind="command", spec={"command": "exit 1"})]})
    async with _run("cmd", contract) as run:
        await run.until(lambda n: n.attempt_count >= 2, "the rejected node was never retried")
        node = await run.node()
        assert node.status != "COMPLETED" and node.frozen is False
        [first, *_] = await _events("attempt.finished", run.node_id, at_least=1)
        assert first["outcome"] == "failed" and "command_failed" in first["failure"]["message"]


async def test_verification_that_cannot_run_rejects_the_attempt_instead_of_stopping_the_task() -> None:
    VERDICTS.extend(["crash", "accept"])
    async with _run("crash") as run:
        node = await run.until(lambda n: n.status == "COMPLETED", "the retry never completed the node")
        assert node.attempt_count == 2
        [first, _] = await _events("attempt.finished", run.node_id, at_least=2)
        assert "verification_unavailable" in first["failure"]["message"]


async def test_a_report_of_a_replaced_attempt_does_not_complete_the_node() -> None:
    VERDICTS.extend(["reject"])
    GATES[("work", 2)] = asyncio.Event()
    async with _run("replaced") as run:
        await run.until(lambda n: n.attempt_count == 2 and n.status == "RUNNING", "attempt 2 never started")
        checks = len(CALLS["verify_completion"])
        await run.handle.signal(TaskWorkflow.attempt_finished, run.finished(1))
        await asyncio.sleep(0.3)
        node = await run.node()
        assert node.status == "RUNNING" and node.frozen is False and node.current_attempt_id == run.attempt_id(2)
        assert len(CALLS["verify_completion"]) == checks


async def test_a_report_naming_another_attempt_id_is_ignored() -> None:
    GATES[("work", 1)] = asyncio.Event()
    async with _run("wrong-id") as run:
        await run.until(lambda n: n.status == "RUNNING", "attempt 1 never started")
        await run.handle.signal(TaskWorkflow.attempt_finished, run.finished(1, attempt_id=run.attempt_id(9)))
        await asyncio.sleep(0.3)
        node = await run.node()
        assert node.status == "RUNNING" and CALLS["verify_completion"] == []


async def test_the_same_report_delivered_twice_is_verified_once() -> None:
    GATES[("work", 1)] = asyncio.Event()
    async with _run("twice") as run:
        await run.until(lambda n: n.status == "RUNNING", "attempt 1 never started")
        await run.handle.signal(TaskWorkflow.attempt_finished, run.finished(1))
        await run.handle.signal(TaskWorkflow.attempt_finished, run.finished(1))
        await run.until(lambda n: n.status == "COMPLETED", "the node never completed")
        assert len(CALLS["verify_completion"]) == 1


def _proposal(run: _Run, number: int, command_id: str) -> CompletionProposal:
    return CompletionProposal(
        command_id=command_id, node_id=run.node_id, attempt_id=run.attempt_id(number),
        artifact_manifest_id=MANIFEST, checkpoint_ref=CHECKPOINT,
    )


async def test_a_proposal_that_was_rejected_makes_the_attempts_own_end_report_harmless() -> None:
    VERDICTS.extend(["reject", "accept"])
    GATES[("work", 1)] = asyncio.Event()
    GATES[("work", 2)] = asyncio.Event()
    async with _run("proposal-then-finished") as run:
        await run.until(lambda n: n.status == "RUNNING", "attempt 1 never started")
        await run.handle.execute_update(TaskWorkflow.propose_completion, _proposal(run, 1, "01J00000000000000000000011"))
        await run.until(lambda n: n.status == "RETRY_PENDING", "the proposal was not rejected")
        # The attempt was told nothing and finishes as if it had succeeded.
        GATES[("work", 1)].set()
        node = await run.until(lambda n: n.attempt_count == 2 and n.status == "RUNNING", "attempt 2 never started")
        assert node.frozen is False
        # Only the proposal was verified: the spent attempt's own report is not a second chance.
        assert len(CALLS["verify_completion"]) == 1
        [finished] = await _events("attempt.finished", run.node_id, at_least=1)
        assert finished["attempt_id"] == run.attempt_id(1) and finished["outcome"] == "failed"
        assert finished["failure"]["failure_class"] == "verification"
        GATES[("work", 2)].set()
        await run.until(lambda n: n.status == "COMPLETED", "attempt 2 never completed the node")


async def test_an_accepted_proposal_is_not_verified_again_when_its_attempt_ends() -> None:
    GATES[("work", 1)] = asyncio.Event()
    async with _run("proposal-accepted") as run:
        await run.until(lambda n: n.status == "RUNNING", "attempt 1 never started")
        await run.handle.execute_update(TaskWorkflow.propose_completion, _proposal(run, 1, "01J00000000000000000000012"))
        await run.until(lambda n: n.status == "COMPLETED", "the proposal was not accepted")
        GATES[("work", 1)].set()
        await run.until(lambda n: n.current_attempt_id is None, "the finished attempt was never released")
        node = await run.node()
        assert node.status == "COMPLETED" and node.frozen is True and node.attempt_count == 1
        assert len(CALLS["verify_completion"]) == 1


async def test_a_second_proposal_from_a_rejected_attempt_is_stale() -> None:
    VERDICTS.extend(["reject"])
    GATES[("work", 1)] = asyncio.Event()
    async with _run("proposal-twice") as run:
        await run.until(lambda n: n.status == "RUNNING", "attempt 1 never started")
        await run.handle.execute_update(TaskWorkflow.propose_completion, _proposal(run, 1, "01J00000000000000000000013"))
        await run.until(lambda n: n.status == "RETRY_PENDING", "the proposal was not rejected")
        with pytest.raises(Exception) as caught:
            await run.handle.execute_update(TaskWorkflow.propose_completion, _proposal(run, 1, "01J00000000000000000000014"))
        assert "STALE_ATTEMPT" in repr(caught.value.__cause__ or caught.value)


async def test_what_a_verified_report_emits_is_published_while_the_task_is_paused() -> None:
    """The report settles after the main loop last looked (the check is asynchronous), and a paused task has nothing
    else to wake it: its events must not wait for the next command."""
    GATES[("work", 1)] = asyncio.Event()
    PUBLISH_DELAY_S[0] = 0.5
    async with _run("paused") as run:
        await run.until(lambda n: n.status == "RUNNING", "attempt 1 never started")
        await run.handle.execute_update(
            TaskWorkflow.control, TaskControlInput(command_id="01J00000000000000000000030", action="pause")
        )
        GATES[("work", 1)].set()
        await run.until(lambda n: n.status == "COMPLETED", "the node never completed")
        [finished] = await _events("attempt.finished", run.node_id, at_least=1)
        assert finished["outcome"] == "completed"
        assert (await run.handle.query(TaskWorkflow.get_task_view)).status == "PAUSED"

"""Approval nodes that no SOP made, and structured output for a node whose contract names a schema (04 §5).

How it can go wrong, written down before the code:
  - an approval node a person or an agent planned is never asked about, so the node waits for ever;
  - a refused approval completes the node, or leaves the task running as if nothing happened;
  - a node with `output_schema_ref` reports an empty object, so it can never complete;
  - an output that breaks the schema completes the node, or is retried without telling the agent why;
  - a schema that cannot be loaded is retried for ever instead of asking a person;
  - the agent is never asked for structured output, or is asked again on resume and loses what it was doing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from orbit_contracts.v3.nodes import (
    AgentTurnNode,
    AgentTurnSpec,
    ApprovalNode,
    ApprovalSpec,
    CompletionContract,
)
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from orbit_worker import task_activities, verify, verify_activities
from orbit_worker.runtime import AgentRuntime
from orbit_worker.runtime_holder import set_runtime
from orbit_worker.store import MemoryStateStore
from orbit_worker.verify import DirectorySchemaRegistry
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker
from test_interrupted_session import _payload, _Store
from test_sop_expansion import (
    FAST,
    TURNS,
    _add,
    _checkpoint_commit,
    _commit_checkpoints,
    _decide,
    _events,
    _finish,
    _publish_events,
    _ref,
    _reset,
    _start,
    _until,
)

SCHEMA = {"type": "object", "required": ["summary", "count"], "properties": {"summary": {"type": "string"}, "count": {"type": "integer"}}}
REGISTRY: list[DirectorySchemaRegistry] = []


def _schemas(root: Path) -> DirectorySchemaRegistry:
    (root / "report").mkdir(parents=True)
    (root / "report" / "1.json").write_text(json.dumps(SCHEMA))
    return DirectorySchemaRegistry(root)


# ---- approval nodes ---------------------------------------------------------------------------------------------------


@activity.defn(name="agent_turn")
async def _turn(payload: dict[str, Any]) -> dict[str, Any]:
    """Says what the goal says: `output:{json}` is the structured output; the second attempt of `fix:` is valid."""
    TURNS.append(payload)
    goal = str(payload["goal"])
    ref = payload.get("output_schema_ref")
    result: dict[str, Any] = {"status": "completed", "checkpoint_ref": _ref("1"), "session_id": "s", "state_version": 2}
    if ref and REGISTRY[0].get(ref) is None:  # as the worker does when the schema cannot be loaded
        return {"status": "failed", "error": f"the output schema {ref} is not registered", "failure_class": "policy", "retryable": False}
    if ref:
        result["output"] = (
            {"summary": "fixed", "count": 2} if goal == "fix" and int(payload["attempt_no"]) > 1
            else json.loads(goal[len("output:"):]) if goal.startswith("output:") else {}
        )
    return result


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    ref = payload["completion_contract"].get("output_schema_ref")
    if not ref:
        return {"ok": True, "failures": [], "workspace_snapshot_ref": None}
    failures = await verify.verify_schema(REGISTRY[0], ref, payload.get("output") or {})
    return {"ok": not failures, "failures": failures, "workspace_snapshot_ref": None}


def _stack(env: WorkflowEnvironment) -> tuple[Worker, Worker, Worker]:
    return (
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[_turn], **FAST),
        Worker(env.client, task_queue="orbit.io", activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints]),
    )


def _gate(summary: str = "Ship it?") -> ApprovalNode:
    return ApprovalNode(node_id="tmp:1", title="Gate", spec=ApprovalSpec(summary=summary, risk="high"))


@pytest.mark.asyncio
async def test_an_approval_node_a_person_planned_is_asked_and_completes_when_approved() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "gate-ok")
            await _add(handle, _gate(), AgentTurnNode(node_id="tmp:2", title="After", depends_on=["tmp:1"], spec=AgentTurnSpec(goal="after")))
            view, plan = await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the approval")
            gate = next(n for n in plan.nodes if n.type == "approval")
            assert gate.status == "AWAITING_APPROVAL"
            requested = _events("approval.requested")[-1]
            assert requested["node_id"] == gate.node_id
            assert (requested["subject"]["kind"], requested["subject"]["summary"], requested["subject"]["risk"]) == ("node_approval", "Ship it?", "high")
            assert not any(t["goal"] == "after" for t in TURNS), "what depends on the gate waits"
            await _decide(handle, view.pending_approvals[0], "approve")
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the end")
            await _finish(handle)
    assert next(n for n in plan.nodes if n.type == "approval").status == "COMPLETED"
    assert next(n for n in plan.nodes if n.title == "After").status == "COMPLETED"


@pytest.mark.asyncio
async def test_a_refused_approval_node_blocks_and_asks_for_a_review() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "gate-no")
            await _add(handle, _gate())
            view, _ = await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the approval")
            await _decide(handle, view.pending_approvals[0], "reject")
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
            await _finish(handle)
    gate = next(n for n in plan.nodes if n.type == "approval")
    assert gate.status == "BLOCKED" and gate.frozen is False
    assert "was refused" in next(p["reason"] for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW")


# ---- structured output through the workflow ---------------------------------------------------------------------------


def _node(goal: str, ref: str = "schema://report/1") -> AgentTurnNode:
    return AgentTurnNode(
        node_id="tmp:1", title="Report", spec=AgentTurnSpec(goal=goal),
        completion_contract=CompletionContract(output_schema_ref=ref),
    )


@pytest.mark.asyncio
async def test_a_valid_output_completes_the_node_and_the_schema_goes_to_the_worker(tmp_path: Path) -> None:
    _reset()
    REGISTRY[:] = [_schemas(tmp_path)]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "out-ok")
            await _add(handle, _node('output:{"summary":"hi","count":3}'))
            await _until(env, handle, lambda v, p: v.status == "COMPLETED" and any(n.title == "Report" and n.status == "COMPLETED" for n in p.nodes), "the node")
            await _finish(handle)
    assert next(t for t in TURNS if t["goal"].startswith("output:"))["output_schema_ref"] == "schema://report/1"
    assert next(t for t in TURNS if not t["goal"].startswith("output:"))["output_schema_ref"] is None, "a node without a schema asks for none"


@pytest.mark.asyncio
async def test_an_output_that_breaks_the_schema_is_retried_with_the_reason(tmp_path: Path) -> None:
    _reset()
    REGISTRY[:] = [_schemas(tmp_path)]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "out-bad")
            await _add(handle, _node("fix"))
            _, plan = await _until(env, handle, lambda v, p: any(n.title == "Report" and n.status == "COMPLETED" for n in p.nodes), "the node", skip_s=10)
            await _finish(handle)
    report = next(n for n in plan.nodes if n.title == "Report")
    assert report.attempt_count == 2
    retry = [t for t in TURNS if t["goal"] == "fix"][1]
    assert "schema" in retry["retry_reason"] and "summary" in retry["retry_reason"]


@pytest.mark.asyncio
async def test_a_schema_that_is_not_registered_blocks_the_node_at_once(tmp_path: Path) -> None:
    _reset()
    REGISTRY[:] = [_schemas(tmp_path)]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "out-unknown")
            await _add(handle, _node("anything", "schema://nothing/1"))
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
            await _finish(handle)
    assert next(n for n in plan.nodes if n.title == "Report").status == "BLOCKED"
    assert len([t for t in TURNS if t["goal"] == "anything"]) == 1, "not retried"
    reason = next(p["reason"] for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW")
    assert "schema://nothing/1 is not registered" in reason


@pytest.mark.asyncio
async def test_attempt_finished_carries_the_output_and_leaves_a_big_one_out(tmp_path: Path) -> None:
    _reset()
    REGISTRY[:] = [_schemas(tmp_path)]
    big = json.dumps({"summary": "y" * 20000, "count": 1})
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "out-event")
            await _add(handle, _node('output:{"summary":"hi","count":3}'))
            await _until(env, handle, lambda v, p: any(n.title == "Report" and n.status == "COMPLETED" for n in p.nodes), "small")
            await _add(handle, AgentTurnNode(
                node_id="tmp:1", title="Big", spec=AgentTurnSpec(goal="output:" + big),
                completion_contract=CompletionContract(output_schema_ref="schema://report/1")), salt="b")
            await _until(env, handle, lambda v, p: any(n.title == "Big" and n.status == "COMPLETED" for n in p.nodes), "big")
            await _finish(handle)
    finished = [p for p in _events("attempt.finished") if "output" in p or "output_truncated" in p]
    assert finished[0]["output"] == {"summary": "hi", "count": 3} and "output_truncated" not in finished[0]
    assert finished[1]["output_truncated"] is True and "output" not in finished[1]
    assert all("output" not in p for p in _events("attempt.finished") if p["node_id"] not in {f["node_id"] for f in finished})


# ---- structured output in the worker ---------------------------------------------------------------------------------


@pytest.fixture
def worker(tmp_path: Path):
    task_activities.set_task_store(_Store(tmp_path / "store"))
    set_runtime(AgentRuntime(MemoryStateStore()))
    verify_activities.set_schema_registry(_schemas(tmp_path / "schemas"))
    yield
    verify_activities.set_schema_registry(None)


async def _turn_with(**fields: Any) -> dict[str, Any]:
    return await ActivityEnvironment().run(task_activities.agent_turn, _payload(output_schema_ref="schema://report/1", **fields))


async def test_the_agent_ends_in_an_object_of_the_schema_and_the_attempt_reports_it(worker: None) -> None:
    outcome = await _turn_with(goal='output:{"summary":"hi","count":3}')
    assert outcome["status"] == "completed" and outcome["output"] == {"summary": "hi", "count": 3}
    assert json.loads(outcome["handover_summary"]) == {"summary": "hi", "count": 3}, "the steps after are handed the object"
    assert (await _turn_with(goal="just do it"))["output"] == {"summary": "", "count": 0}, "the mock's smallest valid object"
    plain = await ActivityEnvironment().run(task_activities.agent_turn, _payload(goal="hello"))
    assert "output" not in plain, "no schema, no structured output"


async def test_a_schema_that_cannot_be_loaded_fails_the_attempt_without_retries(worker: None) -> None:
    missing = await ActivityEnvironment().run(task_activities.agent_turn, _payload(output_schema_ref="schema://nope/1"))
    assert (missing["status"], missing["failure_class"], missing["retryable"]) == ("failed", "policy", False)
    assert "schema://nope/1 is not registered" in missing["error"]
    verify_activities.set_schema_registry(None)
    unconfigured = await _turn_with(goal="x")
    assert unconfigured["retryable"] is False and "ORBIT_OUTPUT_SCHEMA_DIR" in unconfigured["error"]

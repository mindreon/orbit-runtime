"""A `sop_stage` node compiled into the plan (`orbit_orch.task_sop`) and the verifier that judges its steps.

How it can go wrong, written down before the code:
  - a step runs before the steps it depends on, or two independent steps are chained;
  - what depended on the SOP node starts before the SOP ends, or never starts;
  - a step is not told what the steps before it handed over, or is told a step that has not finished;
  - a refused step is not tried again with the verifier's reason, or is tried for ever, or its retries do not stop at the
    step's own `max_attempts`;
  - a blocked step leaves the SOP node running, so the task looks alive while it waits for a person;
  - an approval node waits for a person who has no way to answer it, or a refusal goes unnoticed;
  - the SOP node does not complete because compaction took one of its finished steps out of the plan;
  - a definition that cannot be read, is unknown or does not fit in the plan fails the whole task.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from typing import Any

import pytest
from orbit_contracts.v3 import (
    Actor,
    Budget,
    DecideApprovalInput,
    PlanChangeCommand,
    TaskControlInput,
    TaskWorkflowInput,
)
from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec, SopStageNode, SopStageSpec
from orbit_contracts.v3.plan import AddNodeOp
from orbit_contracts.v3.sop import SopDefinition, SopStep, resolve_steps
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from pydantic import ValidationError
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from waiting import polls

TURNS: list[dict[str, Any]] = []
EVENTS: list[dict[str, Any]] = []
VERIFIES: list[dict[str, Any]] = []
SOPS: dict[str, dict[str, Any]] = {}
TASK_IDS: dict[str, str] = {}
RELEASE = asyncio.Event()
FAST = {
    "max_heartbeat_throttle_interval": timedelta(milliseconds=100),
    "default_heartbeat_throttle_interval": timedelta(milliseconds=100),
}


def _ref(char: str) -> str:
    return "sha256:" + char * 64


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """A turn that says what it did: `did: <goal's step subject>`, and leaves a file. A step whose subject is `hold` waits
    until the test releases it."""
    TURNS.append(payload)
    goal = str(payload["goal"])
    if ": hold." in goal:
        while not RELEASE.is_set():
            activity.heartbeat()
            await asyncio.sleep(0.02)
    subject = goal.split(": ", 1)[1].split(".\n", 1)[0] if goal.startswith("You are running step") else goal[:40]
    return {
        "status": "completed",
        "checkpoint_ref": _ref("1"),
        "session_id": "s",
        "state_version": 2,
        "handover_summary": f"did: {subject}",
        "manifest_id": deterministic_id(f"man:{payload['attempt_id']}", "man"),
        "manifest_entries": [{"name": f"{subject}.md", "media_type": "text/markdown", "size_bytes": 3, "blob_ref": _ref("5")}],
    }


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "failures": [], "workspace_snapshot_ref": _ref("7")}


@activity.defn(name="verify_sop_step")
async def _verify_sop_step(payload: dict[str, Any]) -> dict[str, Any]:
    """The worker's verifier, scripted as the mock model does: `flaky-<n>` is refused on its first n attempts."""
    VERIFIES.append(payload)
    subject, attempt = str(payload["subject"]), int(payload["attempt_no"])
    if subject.startswith("flaky-") and attempt <= int(subject.removeprefix("flaky-") or 1):
        failure = {
            "check": "sop_verifier", "code": "sop_step_failed",
            "message": f"step {subject!r} was refused: not good enough on attempt {attempt}", "detail": {},
        }
        return {"ok": False, "failures": [failure]}
    return {"ok": True, "failures": []}


@activity.defn(name="load_sop")
async def _load_sop(payload: dict[str, Any]) -> dict[str, Any]:
    found = SOPS.get(str(payload["sop_ref"]))
    return {"found": False, "steps": []} if found is None else {"found": True, **found}


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


def _stack(env: WorkflowEnvironment) -> tuple[Worker, Worker, Worker]:
    return (
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _verify_sop_step], **FAST),
        Worker(
            env.client, task_queue="orbit.io",
            activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints, _load_sop],
        ),
    )


def _reset() -> None:
    TURNS.clear()
    EVENTS.clear()
    TASK_IDS.clear()
    VERIFIES.clear()
    SOPS.clear()
    RELEASE.clear()


def _events(kind: str) -> list[dict[str, Any]]:
    return [event["payload"] for event in EVENTS if event["type"] == kind]


def _sop(ref: str, steps: list[Any], name: str = "release") -> None:
    SOPS[ref] = {"name": name, "description": "", "steps": steps}


async def _query(handle, query, *args):
    for _ in range(20):
        try:
            return await handle.query(query, *args, rpc_timeout=timedelta(seconds=3))
        except RPCError:
            continue
    raise AssertionError("the query was never answered")


async def _until(env: WorkflowEnvironment, handle, ready, what: str, *, skip_s: int = 0):
    view = plan = None
    async for _ in polls():
        view = await _query(handle, TaskWorkflow.get_task_view)
        plan = await _query(handle, TaskWorkflow.get_plan)
        if ready(view, plan):
            return view, plan
        if skip_s:
            await env.sleep(timedelta(seconds=skip_s))
    assert view is not None and plan is not None
    raise AssertionError(f"timed out waiting for {what}: {view.status} {[(n.title, n.status) for n in plan.nodes]} {len(TURNS)} turns")


async def _start(env: WorkflowEnvironment, name: str, *, carry: dict[str, Any] | None = None):
    task_id = deterministic_id(f"sop:{name}", "task")
    inp = TaskWorkflowInput(
        task_id=task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="user-a"),
        title="SOP", goal="explore", profile="default@1", node_type_registry_version=1, budgets=Budget(),
    )
    if carry is not None:
        inp = inp.model_copy(update={"carry": carry})
    handle = await env.client.start_workflow(TaskWorkflow.run, inp, id=f"task/tenant-a/sop-{name}", task_queue="orbit.orch")
    TASK_IDS[handle.id] = task_id
    return handle


async def _add(handle, *nodes: Any, salt: str = "a"):
    """Add nodes as a person; the first is `tmp:1`. Returns the answer: `id_map` of what was accepted."""
    plan = await _query(handle, TaskWorkflow.get_plan)
    command = PlanChangeCommand(
        command_id=hashlib.sha256(f"{plan.hash}:{salt}".encode()).hexdigest(), task_id=TASK_IDS[handle.id],
        base_plan_version=plan.plan_version, actor=Actor(kind="user", id="user-a"),
        ops=[AddNodeOp(node=node) for node in nodes],
    )
    return await handle.execute_update(TaskWorkflow.submit_plan_change, command)


def _sop_node(ref: str, title: str = "Release", *, depends_on: list[str] | None = None) -> SopStageNode:
    return SopStageNode(node_id="tmp:1", title=title, spec=SopStageSpec(sop=ref), depends_on=depends_on or [])


def _after(title: str) -> AgentTurnNode:
    return AgentTurnNode(node_id="tmp:2", title=title, depends_on=["tmp:1"], spec=AgentTurnSpec(goal=f"after: {title}"))


def _sop_of(plan):
    return next(n for n in plan.nodes if n.type == "sop_stage")


def _status(plan, fragment: str) -> str | None:
    """The status of the node with `fragment` in its title, or None while the plan does not hold one yet."""
    return next((n.status for n in plan.nodes if fragment in n.title and n.type != "approval"), None)


def _by_title(plan, fragment: str):
    return next(node for node in plan.nodes if fragment in node.title and node.type != "approval")


async def _finish(handle) -> None:
    await handle.execute_update(
        TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"cancel").hexdigest(), action="cancel")
    )
    async for _ in polls():
        if (await handle.describe()).status != WorkflowExecutionStatus.RUNNING:
            return
    raise AssertionError("the workflow did not end")




# ---- the definition ---------------------------------------------------------------------------------------------------


def test_a_v1_definition_is_a_linear_v2_one() -> None:
    steps = resolve_steps([SopStep.model_validate(item) for item in ["draft", {"subject": "review", "description": "check it", "max_attempts": 2}]])
    assert [(s.id, s.depends_on, s.description, s.max_attempts) for s in steps] == [
        ("s1", (), "draft", 3), ("s2", ("s1",), "check it", 2),
    ]


def test_depends_on_makes_a_graph_and_an_empty_list_a_root() -> None:
    steps = resolve_steps([
        SopStep(id="a", subject="a"), SopStep(id="b", subject="b", depends_on=[]), SopStep(id="c", subject="c", depends_on=["a", "b"]),
    ])
    assert [s.id for s in steps] == ["a", "b", "c"] and steps[1].depends_on == () and steps[2].depends_on == ("a", "b")


@pytest.mark.parametrize(
    "steps",
    [
        [SopStep(id="a", subject="a"), SopStep(id="a", subject="b")],
        [SopStep(id="a", subject="a", depends_on=["zzz"])],
        [SopStep(id="a", subject="a", depends_on=["a"])],
        [SopStep(id="a", subject="a", depends_on=["b"]), SopStep(id="b", subject="b", depends_on=["a"])],
        [SopStep(id="s2", subject="a"), SopStep(subject="b"), SopStep(subject="c")],
    ],
)
def test_a_definition_that_is_not_a_dag_is_refused(steps: list[SopStep]) -> None:
    with pytest.raises(ValueError):
        resolve_steps(steps)
    with pytest.raises(ValidationError):
        SopDefinition(name="x", steps=steps)


# ---- linear, handover, events -----------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_linear_sop_runs_its_steps_in_order_each_told_what_the_one_before_handed_over() -> None:
    _reset()
    _sop("release@1", ["draft", {"subject": "review", "description": "check the draft"}, "publish"])
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "linear")
            accepted = await _add(handle, _sop_node("release@1"), _after("Announce"))
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED" and len(TURNS) == 5, "the SOP and what follows it")
            await _finish(handle)
    sop = _sop_of(plan)
    steps = [n for n in plan.nodes if n.parent_node_id == sop.node_id]
    assert [n.title for n in steps] == ["Release 1/3: draft", "Release 2/3: review", "Release 3/3: publish"]
    assert all(n.type == "agent_turn" and n.status == "COMPLETED" and n.frozen for n in steps)
    assert sop.status == "COMPLETED" and sop.frozen
    # `getPlan` says which step of which SOP each node is (the authoritative source of what the task page shows).
    assert sop.sop_step is not None and sop.sop_step.model_dump(exclude_none=True) == {"sop": "release@1", "role": "sop", "total": 3}
    assert [n.sop_step.model_dump(exclude_none=True) for n in steps] == [
        {"sop": "release@1", "role": "step", "total": 3, "step_id": f"s{i}", "index": i, "subject": subject}
        for i, subject in enumerate(["draft", "review", "publish"], start=1)
    ]
    assert _by_title(plan, "Announce").sop_step is None, "a node outside the SOP has none"
    # One after the other: each step depends on the one before; the SOP node's dependent now depends on the last step.
    assert [n.depends_on for n in steps] == [[], [steps[0].node_id], [steps[1].node_id]]
    after = _by_title(plan, "Announce")
    assert after.depends_on == [steps[2].node_id] and after.node_id == accepted.id_map["tmp:2"]
    # The turns ran in order, and every step was told its place and what the steps before handed over.
    goals = [t["goal"] for t in TURNS[1:]]  # the first turn is the exploration node
    assert [g.split("\n")[0] for g in goals[:3]] == [
        'You are running step 1 of 3 of the procedure "release": draft.',
        'You are running step 2 of 3 of the procedure "release": review.',
        'You are running step 3 of 3 of the procedure "release": publish.',
    ]
    assert "check the draft" in goals[1] and "handed over" not in goals[0]
    assert '- Step 1 "draft": did: draft' in goals[1] and "draft.md" in goals[1]
    assert '- Step 2 "review": did: review' in goals[2] and "draft" not in goals[2].split("handed over")[1]
    assert goals[3] == "after: Announce", "a node outside the SOP is not given a handover"
    # Every step was judged by a verifier that was given what the step must achieve and what the executor said.
    assert [v["subject"] for v in VERIFIES] == ["draft", "review", "publish"]
    assert VERIFIES[1]["description"] == "check the draft" and VERIFIES[1]["text"] == "did: review"
    assert VERIFIES[1]["workspace_snapshot_ref"] == _ref("7") and VERIFIES[1]["manifest_entries"][0]["name"] == "review.md"
    # The events say which step of which SOP a node is.
    commit = next(p for p in _events("plan.version_committed") if p.get("reason") == "sop expansion")
    assert commit["sop_node_id"] == sop.node_id and commit["actor"] == {"kind": "system", "id": "task-workflow"}
    changes = [p for p in _events("node.status_changed") if p["node_id"] == steps[1].node_id]
    assert changes[0]["parent_node_id"] == sop.node_id
    assert changes[0]["sop_step"] == {"sop": "release@1", "role": "step", "total": 3, "step_id": "s2", "index": 2, "subject": "review"}
    group = [p for p in _events("node.status_changed") if p["node_id"] == sop.node_id]
    assert [(p["from_status"], p["to_status"]) for p in group][-2:] == [("READY", "RUNNING"), ("RUNNING", "COMPLETED")]
    assert group[-1]["sop_step"] == {"sop": "release@1", "role": "sop", "total": 3}


# ---- parallel steps, rewiring, executors ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_independent_steps_do_not_wait_for_each_other_and_a_join_waits_for_both() -> None:
    _reset()
    _sop("fan@1", [
        {"id": "a", "subject": "left", "depends_on": []},
        {"id": "b", "subject": "right", "depends_on": [], "executor": "coder@2"},
        {"id": "c", "subject": "join", "depends_on": ["a", "b"]},
    ], name="fan")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "parallel")
            await _add(handle, _sop_node("fan@1", "Fan"), _after("Next"))
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED" and len(TURNS) == 5, "the SOP")
            await _finish(handle)
    left, right, join = (_by_title(plan, name) for name in ("left", "right", "join"))
    assert left.depends_on == [] and right.depends_on == []
    assert sorted(join.depends_on) == sorted([left.node_id, right.node_id])
    assert right.owner_profile == "coder@2" and left.owner_profile == "default@1", "a step runs as its executor"
    assert _by_title(plan, "Next").depends_on == [join.node_id], "the only sink"
    joined = next(t for t in TURNS if "step 3 of 3" in t["goal"])
    assert 'Step 1 "left": did: left' in joined["goal"] and 'Step 2 "right": did: right' in joined["goal"]
    assert next(t for t in TURNS if "right" in t["goal"] and "step 2" in t["goal"])["profile"] == "coder@2"
    assert _sop_of(plan).status == "COMPLETED"


@pytest.mark.asyncio
async def test_whatever_waited_for_the_sop_waits_for_every_sink_of_it() -> None:
    _reset()
    _sop("two@1", [{"id": "a", "subject": "one", "depends_on": []}, {"id": "b", "subject": "hold", "depends_on": []}])
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "sinks")
            await _add(handle, _sop_node("two@1", "Two"), _after("Next"))
            _, plan = await _until(env, handle, lambda v, p: _status(p, "one") == "COMPLETED", "the first sink")
            held = _by_title(plan, "hold")
            assert held.status in {"READY", "RUNNING"} and _by_title(plan, "Next").status == "PENDING", "one sink is not enough"
            assert not any(t["goal"] == "after: Next" for t in TURNS)
            RELEASE.set()
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED" and _status(p, "Next") == "COMPLETED", "the end")
            await _finish(handle)
    assert sorted(_by_title(plan, "Next").depends_on) == sorted([_by_title(plan, "one").node_id, _by_title(plan, "hold").node_id])


# ---- the verifier ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_step_is_tried_again_with_the_verifiers_reason_and_then_passes() -> None:
    _reset()
    _sop("flaky@1", ["draft", "flaky-1", "publish"], name="flaky")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "flaky")
            await _add(handle, _sop_node("flaky@1", "Flaky"))
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the SOP", skip_s=10)
            await _finish(handle)
    flaky = _by_title(plan, "flaky-1")
    assert flaky.status == "COMPLETED" and flaky.attempt_count == 2
    assert [(v["subject"], v["attempt_no"]) for v in VERIFIES] == [("draft", 1), ("flaky-1", 1), ("flaky-1", 2), ("publish", 1)]
    runs = [t for t in TURNS if "step 2 of 3" in t["goal"]]
    assert len(runs) == 2 and not runs[0]["retry_reason"]
    assert "[sop_verifier/sop_step_failed]" in runs[1]["retry_reason"] and "not good enough on attempt 1" in runs[1]["retry_reason"]
    assert runs[1]["continue_from"], "the retry carries on the session of the attempt that was refused"
    failed = [p for p in _events("attempt.finished") if p["node_id"] == flaky.node_id]
    assert [p["outcome"] for p in failed] == ["failed", "completed"]
    assert failed[0]["failure"]["failure_class"] == "verification" and failed[0]["failure"]["retryable"]
    assert _sop_of(plan).status == "COMPLETED"


@pytest.mark.asyncio
async def test_a_step_that_keeps_being_refused_stops_at_its_own_attempts_and_blocks_the_sop_and_what_follows() -> None:
    _reset()
    _sop("strict@1", ["draft", {"subject": "flaky-5", "max_attempts": 2}, "publish"], name="strict")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "strict")
            await _add(handle, _sop_node("strict@1", "Strict"), _after("Next"))
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
            await _finish(handle)
    assert _by_title(plan, "flaky-5").status == "BLOCKED" and _by_title(plan, "flaky-5").attempt_count == 2
    assert _sop_of(plan).status == "BLOCKED", "the SOP is blocked by its step"
    assert _by_title(plan, "publish").status == "PENDING" and _by_title(plan, "Next").status == "PENDING"
    assert [v["attempt_no"] for v in VERIFIES if v["subject"] == "flaky-5"] == [1, 2]
    assert not any("step 3 of 3" in t["goal"] or t["goal"] == "after: Next" for t in TURNS)
    reasons = [p["reason"] for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"]
    assert reasons and "2 of 2 attempts failed" in reasons[0]
    blocked = [p for p in _events("node.status_changed") if p["node_id"] == _sop_of(plan).node_id and p["to_status"] == "BLOCKED"]
    assert blocked and "step 2 (flaky-5) is blocked" in blocked[0]["reason"]


@pytest.mark.asyncio
async def test_resuming_a_blocked_step_puts_the_sop_back_in_play_and_it_completes() -> None:
    _reset()
    _sop("again@1", [{"subject": "flaky-2", "max_attempts": 2}], name="again")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "resume")
            await _add(handle, _sop_node("again@1", "Again"))
            await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
            await handle.execute_update(
                TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"resume").hexdigest(), action="resume")
            )
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the end", skip_s=10)
            await _finish(handle)
    assert _sop_of(plan).status == "COMPLETED"
    states = [(p["from_status"], p["to_status"]) for p in _events("node.status_changed") if p["node_id"] == _sop_of(plan).node_id]
    assert ("BLOCKED", "RUNNING") in states


# ---- approvals --------------------------------------------------------------------------------------------------------


async def _decide(handle, approval_id: str, decision: str) -> None:
    await handle.execute_update(
        TaskWorkflow.decide_approval,
        DecideApprovalInput(command_id=hashlib.sha256(f"{approval_id}{decision}".encode()).hexdigest(), approval_id=approval_id, decision=decision),
    )


@pytest.mark.asyncio
async def test_a_step_with_human_approval_waits_for_a_person_before_and_after() -> None:
    _reset()
    _sop("gated@1", [
        {"id": "a", "subject": "prepare", "human_approval": "before"},
        {"id": "b", "subject": "ship", "human_approval": "after"},
        {"id": "c", "subject": "tell"},
    ], name="gated")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "gated")
            await _add(handle, _sop_node("gated@1", "Gated"), _after("Next"))
            view, plan = await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the first approval")
            gate = next(n for n in plan.nodes if n.type == "approval")
            assert gate.title.startswith("Approve the start of Gated 1/3: prepare") and gate.status == "AWAITING_APPROVAL"
            assert gate.parent_node_id == _sop_of(plan).node_id
            assert _by_title(plan, "prepare").depends_on == [gate.node_id] and _by_title(plan, "prepare").status == "PENDING"
            assert view.status == "WAITING" and not any("step 1 of 3" in t["goal"] for t in TURNS)
            requested = _events("approval.requested")[-1]
            assert requested["node_id"] == gate.node_id and requested["subject"]["kind"] == "sop_step"
            await _decide(handle, view.pending_approvals[0], "approve")
            # The step runs, then its result waits for approval; the step after it waits for that.
            view, plan = await _until(env, handle, lambda v, p: bool(v.pending_approvals) and _status(p, "ship") == "COMPLETED", "the second approval")
            after_gate = next(n for n in plan.nodes if n.title.startswith("Approve the result of"))
            assert after_gate.depends_on == [_by_title(plan, "ship").node_id] and _by_title(plan, "tell").depends_on == [after_gate.node_id]
            assert not any("step 3 of 3" in t["goal"] for t in TURNS)
            await _decide(handle, view.pending_approvals[0], "approve")
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the end")
            await _finish(handle)
    assert _sop_of(plan).status == "COMPLETED" and all(n.status == "COMPLETED" for n in plan.nodes)
    assert [p["status"] for p in _events("approval.decided")] == ["APPROVED", "APPROVED"]
    assert 'Step 2 "ship": did: ship' in next(t for t in TURNS if "step 3 of 3" in t["goal"])["goal"], "the handover skips the approval node"


@pytest.mark.asyncio
async def test_a_refused_approval_blocks_the_sop_and_a_resume_asks_again() -> None:
    _reset()
    _sop("ask@1", [{"subject": "prepare", "human_approval": "before"}], name="ask")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "refused")
            await _add(handle, _sop_node("ask@1", "Ask"))
            view, _ = await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the approval")
            first = view.pending_approvals[0]
            await _decide(handle, first, "reject")
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
            assert next(n for n in plan.nodes if n.type == "approval").status == "BLOCKED" and _sop_of(plan).status == "BLOCKED"
            assert not TURNS[1:], "nothing ran behind a refused approval"
            await handle.execute_update(
                TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"resume").hexdigest(), action="resume")
            )
            view, _ = await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the approval again")
            assert view.pending_approvals != [first], "a new question, not the old answer"
            await _decide(handle, view.pending_approvals[0], "approve")
            _, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the end")
            await _finish(handle)
    assert _sop_of(plan).status == "COMPLETED"


# ---- what cannot be compiled ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_sop_blocks_its_node_and_asks_for_a_review_without_failing_the_task() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "unknown")
            await _add(handle, _sop_node("nothing@1", "Nothing"))
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
            await _finish(handle)
    assert _sop_of(plan).status == "BLOCKED"
    assert "unknown SOP nothing@1" in next(p["reason"] for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW")


@pytest.mark.asyncio
async def test_a_definition_that_is_not_valid_blocks_its_node() -> None:
    _reset()
    _sop("cycle@1", [{"id": "a", "subject": "a", "depends_on": ["b"]}, {"id": "b", "subject": "b", "depends_on": ["a"]}])
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "cycle")
            await _add(handle, _sop_node("cycle@1", "Cycle"))
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
            await _finish(handle)
    assert _sop_of(plan).status == "BLOCKED" and not [n for n in plan.nodes if n.parent_node_id]
    assert "not a valid definition" in next(p["reason"] for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW")


# ---- compaction -------------------------------------------------------------------------------------------------------


def _filler(count: int) -> list[dict[str, Any]]:
    """Finished, frozen nodes that make the plan big enough to be compacted (half of what a plan change may reach)."""
    return [
        {
            "node_id": deterministic_id(f"filler:{index}", "n"),
            "draft": AgentTurnNode(
                node_id=deterministic_id(f"filler:{index}", "n"), title=f"done {index}", spec=AgentTurnSpec(goal="g" * 9000)
            ).model_dump(mode="json", by_alias=True, exclude_none=True),
            "status": "COMPLETED", "frozen": True, "current_attempt_id": None, "attempt_count": 1, "created_by": "user-a",
        }
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_a_sop_compiles_into_a_big_plan_and_completes_though_compaction_took_a_finished_step_out() -> None:
    _reset()
    _sop("big@1", [{"id": "a", "subject": "left", "depends_on": []}, {"id": "b", "subject": "hold", "depends_on": []}], name="big")
    carry = {"status": "COMPLETED", "plan_version": 7, "nodes": _filler(20), "edges": [], "next_message_seq": 1}
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "big", carry=carry)
            await _until(env, handle, lambda v, p: v.status == "COMPLETED" and p.plan_version == 7, "the carried plan")
            accepted = await _add(handle, _sop_node("big@1", "Big"))
            assert accepted.status == "accepted", accepted
            # `left` finishes and nothing depends on it. Seventeen nodes finish after it, so it is no longer among the newest
            # sixteen that compaction keeps, and the plan change after them (the plan is over the size that is compacted) takes
            # it out of the plan while `hold` still runs.
            await _until(env, handle, lambda v, p: _status(p, "left") == "COMPLETED", "the first step")
            more = [AgentTurnNode(node_id=f"tmp:{i}", title=f"small {i}", spec=AgentTurnSpec(goal=f"small {i}")) for i in range(1, 18)]
            await _add(handle, *more, salt="b")
            await _until(env, handle, lambda v, p: all(_status(p, f"small {i}") == "COMPLETED" for i in range(1, 18)), "the small nodes")
            await _add(handle, AgentTurnNode(node_id="tmp:1", title="one more", spec=AgentTurnSpec(goal="one more")), salt="c")
            _, plan = await _until(env, handle, lambda v, p: _status(p, "left") is None, "the compaction")
            sop = _sop_of(plan)
            assert sop.status == "RUNNING" and _status(plan, "hold") in {"READY", "RUNNING"}, "the SOP is still going"
            RELEASE.set()
            _, plan = await _until(env, handle, lambda v, p: _sop_of(p).status == "COMPLETED", "the SOP")
            await _finish(handle)
    assert plan.archived is not None and plan.archived.count >= 1
    assert not _events("plan.change_rejected")
    committed = _events("plan.version_committed")
    assert any(p.get("reason") == "sop expansion" for p in committed)
    assert any(p.get("reason") == "compaction" for p in committed)


# ---- Continue-As-New --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_sop_carried_across_a_continue_as_new_goes_on_with_what_its_finished_steps_handed_over() -> None:
    """The record of the expansion is part of what a Continue-As-New carries: a run that starts from it finishes the SOP,
    tells the next step what the finished one handed over and completes the SOP node."""
    _reset()
    sop_id, one, two = (deterministic_id(f"carry:{name}", "n") for name in ("sop", "one", "two"))

    def node(draft: Any, status: str, frozen: bool = False) -> dict[str, Any]:
        return {
            "node_id": draft.node_id, "draft": draft.model_dump(mode="json", by_alias=True, exclude_none=True), "status": status,
            "frozen": frozen, "current_attempt_id": None, "attempt_count": 1 if frozen else 0, "created_by": "task-workflow",
        }

    nodes = [
        node(SopStageNode(node_id=sop_id, title="Carry", spec=SopStageSpec(sop="carry@1")), "RUNNING"),
        node(AgentTurnNode(node_id=one, title="Carry 1/2: one", parent_node_id=sop_id, spec=AgentTurnSpec(goal="one")), "COMPLETED", True),
        node(
            AgentTurnNode(
                node_id=two, title="Carry 2/2: two", parent_node_id=sop_id, depends_on=[one], workspace_access="write",
                spec=AgentTurnSpec(goal='You are running step 2 of 2 of the procedure "carry": two.\n\nWhat this step must achieve:\ntwo'),
            ),
            "READY",
        ),
    ]
    parts = {
        one: {"role": "step", "step_id": "s1", "index": 1, "subject": "one"},
        two: {"role": "step", "step_id": "s2", "index": 2, "subject": "two"},
    }
    carry = {
        "status": "RUNNING", "plan_version": 4, "nodes": nodes, "edges": [[one, two]], "next_message_seq": 1,
        "sops": {sop_id: {
            "ref": "carry@1", "name": "carry", "title": "Carry", "total": 2, "nodes": parts, "open": [two],
            "agent": {"s1": one, "s2": two}, "deps": {one: [], two: [one]},
            "outputs": {one: {"summary": "carried summary", "artifacts": ["one.md"]}},
        }},
    }
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "carry", carry=carry)
            _, plan = await _until(env, handle, lambda v, p: _status(p, "Carry") == "COMPLETED", "the SOP")
            await _finish(handle)
    assert 'Step 1 "one": carried summary' in TURNS[0]["goal"] and "one.md" in TURNS[0]["goal"]
    assert _status(plan, "2/2") == "COMPLETED"
    group = [p for p in _events("node.status_changed") if p["node_id"] == sop_id]
    assert group[-1]["to_status"] == "COMPLETED" and group[-1]["sop_step"]["role"] == "sop"

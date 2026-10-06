"""How a task copes when an attempt does not simply complete: failures, retries, interrupts, cancels and long conversations.

How it can go wrong, written down before the code:
  - a node that can never succeed is tried again for ever; or a failure that retrying cannot fix is retried anyway;
  - the retry starts from nothing, and does not know why the attempt before it was rejected;
  - an interrupt or a stop throws away what the agent knew; the replacement is not given the message that interrupted it;
  - a cancelled attempt is counted as a failure, so a person who interrupts a few times blocks the node;
  - the replacement starts while the cancelled attempt still holds the workspace;
  - an approval of an attempt that is gone stays pending, and a person can still answer it;
  - an agent is refused when it changes a node it created, once the node is READY;
  - a wait node's timer is lost when the workflow continues as new;
  - the state a Continue-As-New carries grows with every message and every follow-up of the conversation;
  - a follow-up that does not fit in the plan is dropped without a word;
  - an AttemptWorkflow with a long history never continues as new.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

import pytest
from orbit_contracts.v3 import (
    Actor,
    Budget,
    DecideApprovalInput,
    ExternalEventSignal,
    GrantBudgetInput,
    PlanChangeCommand,
    SendMessageInput,
    TaskControlInput,
    TaskWorkflowInput,
)
from orbit_contracts.v3.nodes import (
    AgentTurnNode,
    AgentTurnSpec,
    WaitNode,
    WaitSpec,
)
from orbit_contracts.v3.plan import AddNodeOp, NodePatch, RemoveNodeOp, UpdateNodeOp
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio import activity
from temporalio.client import WorkflowUpdateFailedError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from waiting import polls

TURNS: list[dict[str, Any]] = []
EVENTS: list[dict[str, Any]] = []
ORDER: list[tuple[str, int]] = []
CLEANUP_S = 0.05

# A cancel reaches an activity on its next heartbeat, and the SDK sends one at most every `max_heartbeat_throttle_interval`.
FAST = {
    "max_heartbeat_throttle_interval": timedelta(milliseconds=100),
    "default_heartbeat_throttle_interval": timedelta(milliseconds=100),
}


def _ref(char: str) -> str:
    return "sha256:" + char * 64


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """A turn whose goal says what it does: `fail:<class>:<retryable>` always fails, `fail-until:<n>` fails until attempt n,
    `hold-always` holds until it is cancelled (and takes a moment to let go), `approval` parks on an approval."""
    TURNS.append(payload)
    goal = str(payload["goal"])
    attempt_no = int(payload["attempt_no"])
    if goal.startswith("fail:"):
        _, failure_class, retryable = goal.split(":")
        return {"status": "failed", "error": f"boom {attempt_no}", "failure_class": failure_class, "retryable": retryable == "true"}
    if goal.startswith("fail-until:") and attempt_no < int(goal.split(":")[1]):
        return {"status": "failed", "error": f"boom {attempt_no}", "failure_class": "model", "retryable": True}
    if goal == "hold-always":
        ORDER.append(("start", attempt_no))
        try:
            for _ in range(6000):
                activity.heartbeat()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            await asyncio.sleep(CLEANUP_S)
            ORDER.append(("end", attempt_no))
            raise
    if goal == "approval" and not payload.get("approval"):
        return {
            "status": "parked_approval",
            "checkpoint_ref": _ref("3"),
            "approvals": [{
                "tool_call_id": "call-1",
                "subject": {"kind": "tool_call", "digest": _ref("4"), "summary": "run a gated command", "risk": "medium"},
            }],
            "session_id": "s",
            "state_version": 2,
        }
    await asyncio.sleep(0)
    return {"status": "completed", "checkpoint_ref": _ref("1"), "session_id": "s", "state_version": 2}


@activity.defn(name="sop_step")
async def _sop_step(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "completed", "checkpoint_ref": _ref("2")}


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


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


def _stack(env: WorkflowEnvironment, runner: Any = None, turn: Any = _agent_turn) -> tuple[Worker, Worker, Worker]:
    return (
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=runner or sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[turn, _sop_step], **FAST),
        Worker(
            env.client, task_queue="orbit.io",
            activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
        ),
    )


def _reset() -> None:
    TURNS.clear()
    EVENTS.clear()
    ORDER.clear()


def _input(task_id: str, goal: str) -> TaskWorkflowInput:
    return TaskWorkflowInput(
        task_id=task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="user-a"), title="Resilience",
        goal=goal, profile="default@1", node_type_registry_version=1, budgets=Budget(),
    )


async def _start(env: WorkflowEnvironment, name: str, goal: str):
    from orbit_orch.plan_engine import deterministic_id

    return await env.client.start_workflow(
        TaskWorkflow.run, _input(deterministic_id(f"resilience:{name}", "task"), goal),
        id=f"task/tenant-a/{name}", task_queue="orbit.orch",
    )


async def _until(env: WorkflowEnvironment, handle, ready, what: str, *, skip_s: int = 0):
    """Poll the task until `ready(view, plan)`, for up to a minute of wall time. With `skip_s` the test server's clock is moved on
    between polls, which is how a backoff or a timer of the workflow runs out."""
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


async def _closed(handle) -> None:
    """Wait until the workflow has ended, by asking where it stands. `handle.result()` follows the chain of runs of a
    workflow that continued as new, and under a loaded machine its long poll on the time-skipping test server was seen not
    to return for a workflow that had completed (its history ends with the completion); describing the latest run does."""
    from temporalio.client import WorkflowExecutionStatus

    async for _ in polls():
        if (await handle.describe()).status != WorkflowExecutionStatus.RUNNING:
            return
    raise AssertionError("the workflow did not end")


async def _query(handle, query, *args):
    """A query sent at the moment the workflow continues as new is lost with the run it was sent to, and times out; a client
    asks again."""
    for _ in range(20):
        try:
            return await handle.query(query, *args, rpc_timeout=timedelta(seconds=3))
        except RPCError:
            continue
    raise AssertionError("the query was never answered")


def _say(handle, number: int, text: str, delivery: str = "queue"):
    return handle.execute_update(
        TaskWorkflow.send_message,
        SendMessageInput(
            command_id=f"01J000000000000000000{number:05d}"[:26],
            client_message_id=f"01J100000000000000000{number:05d}"[:26],
            text=text, delivery=delivery,
        ),
    )


def _control(handle, number: int, action: str):
    return handle.execute_update(TaskWorkflow.control, TaskControlInput(command_id=f"01JC{number:022d}", action=action))


def _events(kind: str) -> list[dict[str, Any]]:
    return [e["payload"] for e in EVENTS if e["type"] == kind]


def _when(kind: str, index: int) -> datetime:
    return datetime.fromisoformat([e for e in EVENTS if e["type"] == kind][index]["occurred_at"])


def _node(plan, node_id: str):
    return next(n for n in plan.nodes if n.node_id == node_id)


# ---- 1: retries have a limit, a backoff and a way out ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failing_node_retries_with_a_backoff_and_is_then_blocked_for_a_review() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "retry-limit", "fail:model:true")
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
            assert plan.nodes[0].status == "BLOCKED"
            assert [t["attempt_no"] for t in TURNS] == [1, 2, 3], "three tries by default, and no more"
            await env.sleep(timedelta(seconds=600))
            assert len(TURNS) == 3, "a blocked node is not tried again"
    # The failure class the worker named is what the attempt reports.
    failures = [p["failure"] for p in _events("attempt.finished")]
    assert [(f["failure_class"], f["retryable"]) for f in failures] == [("model", True)] * 3
    # A retry waits: 5 seconds after the first failure, 30 after the second.
    assert (_when("attempt.started", 1) - _when("attempt.finished", 0)).total_seconds() >= 5
    assert (_when("attempt.started", 2) - _when("attempt.finished", 1)).total_seconds() >= 30
    blocked = [p for p in _events("node.status_changed") if p["to_status"] == "BLOCKED"]
    assert len(blocked) == 1 and "3 of 3 attempts failed" in blocked[0]["reason"] and "boom 3" in blocked[0]["reason"]
    review = [p for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"]
    assert len(review) == 1 and "is blocked" in review[0]["reason"]


@pytest.mark.asyncio
async def test_a_failure_that_retrying_cannot_fix_blocks_the_node_at_once() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "not-retryable", "fail:policy:false")
            await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
            assert len(TURNS) == 1
    blocked = [p for p in _events("node.status_changed") if p["to_status"] == "BLOCKED"]
    assert "policy failure that cannot be retried" in blocked[0]["reason"]


@pytest.mark.asyncio
async def test_a_retry_carries_on_the_session_and_knows_why_and_a_resume_gives_the_tries_back() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "resume-resets", "fail-until:5")
            await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
            assert [t["attempt_no"] for t in TURNS] == [1, 2, 3]
            await _control(handle, 1, "resume")
            await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the node to complete", skip_s=10)
            # Attempt 4 fails again and is retried: the count started again at zero, or it would have blocked at once.
            assert [t["attempt_no"] for t in TURNS] == [1, 2, 3, 4, 5]
    first = TURNS[0]
    assert first["continue_from"] is None and first["retry_reason"] == "", "a fresh node starts empty"
    for before, after in pairwise(TURNS):
        assert after["continue_from"] == before["attempt_id"], "a retry carries on the attempt before it"
        assert f"boom {before['attempt_no']}" in after["retry_reason"], "and is told why that one failed"


@pytest.mark.asyncio
async def test_a_plan_change_on_a_blocked_node_puts_it_back_in_play() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "plan-unblocks", "fail:model:true")
            view, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
            node_id = plan.nodes[0].node_id
            result = await handle.execute_update(
                TaskWorkflow.submit_plan_change,
                PlanChangeCommand(
                    command_id="01J00000000000000000000050", task_id=view.task_id, base_plan_version=plan.plan_version,
                    actor=Actor(kind="user", id="user-a"),
                    ops=[UpdateNodeOp(node_id=node_id, patch=NodePatch(title="Explore and plan, differently"))],
                ),
            )
            assert result.status == "accepted"
            plan = await _query(handle, TaskWorkflow.get_plan)
            assert _node(plan, node_id).status == "READY", "the person looked at it and changed it"


# ---- 3 and 1: an interrupt or a stop keeps the agent's session and is not a failure -----------------------------------


@pytest.mark.asyncio
async def test_an_interrupted_attempt_is_replaced_by_one_that_carries_on_its_session_and_is_not_held_against_the_node() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "interrupts", "hold-always")
            await _until(env, handle, lambda v, p: len(TURNS) == 1, "the first attempt")
            # More interrupts than the node has tries: they are not failures, so the node is never blocked.
            for number in range(1, 5):
                await _say(handle, number, f"change of plan {number}", "interrupt")
                await _until(env, handle, lambda v, p, n=number: len(TURNS) == n + 1, f"the replacement {number}")
            await _until(env, handle, lambda v, p: v.status == "RUNNING" and p.nodes[0].status == "RUNNING", "running")
            for number, (before, after) in enumerate(pairwise(TURNS), start=1):
                assert after["continue_from"] == before["attempt_id"], "the replacement carries on the interrupted session"
                assert [m["text"] for m in after["messages"]] == [f"change of plan {number}"], "with the new message as input"
                assert after["retry_reason"] == ""

            # A stop is the same: the session is carried on when the person resumes.
            await _control(handle, 1, "stop")
            await _until(env, handle, lambda v, p: v.status == "PAUSED", "the pause")
            assert len(TURNS) == 5
            await _say(handle, 9, "while it was stopped")
            await _control(handle, 2, "resume")
            await _until(env, handle, lambda v, p: len(TURNS) == 6, "the attempt after the stop")
            assert TURNS[5]["continue_from"] == TURNS[4]["attempt_id"]
            assert [m["text"] for m in TURNS[5]["messages"]] == ["while it was stopped"], "a message that came meanwhile waits for it"
            inbox = await _query(handle, TaskWorkflow.get_inbox, 0)
            assert inbox == [], "and has left the inbox now that an attempt has it"
            await _control(handle, 3, "cancel")
            await _closed(handle)


@pytest.mark.asyncio
async def test_the_replacement_starts_only_after_the_cancelled_attempt_has_let_go(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset()
    # The cancelled activity takes a while to wind down (it saves its state and releases the workspace lease). The
    # replacement must not start before that: it would race the old attempt for the single writer lease.
    global CLEANUP_S
    previous, CLEANUP_S = CLEANUP_S, 2.5
    try:
        async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
            orch, agent, io = _stack(env)
            async with orch, agent, io:
                handle = await _start(env, "release", "hold-always")
                await _until(env, handle, lambda v, p: len(TURNS) == 1, "the first attempt")
                await _say(handle, 1, "go", "interrupt")
                await _until(env, handle, lambda v, p: len(TURNS) == 2, "the replacement")
                await asyncio.sleep(0.3)
                await _control(handle, 1, "cancel")
                await _closed(handle)
    finally:
        CLEANUP_S = previous
    assert ORDER[:3] == [("start", 1), ("end", 1), ("start", 2)], ORDER


# ---- 7: the approvals of an attempt that is gone ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_approvals_of_an_interrupted_attempt_are_cancelled_and_cannot_be_decided() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "approvals", "approval")
            view, _ = await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the approval")
            old = view.pending_approvals[0]
            await _say(handle, 1, "never mind that", "interrupt")
            # The replacement parks on an approval of its own; the old one is no longer pending.
            await _until(env, handle, lambda v, p: len(TURNS) == 2 and v.pending_approvals and old not in v.pending_approvals, "the new approval")
            with pytest.raises(WorkflowUpdateFailedError) as refused:
                await handle.execute_update(
                    TaskWorkflow.decide_approval,
                    DecideApprovalInput(command_id="01J00000000000000000000060", approval_id=old, decision="approve"),
                )
            assert refused.value.cause.type == "UNKNOWN_APPROVAL"
            await _control(handle, 1, "cancel")
            await _closed(handle)
    decided = [p for p in _events("approval.decided") if p["approval_id"] == old]
    assert [p["status"] for p in decided] == ["CANCELLED"]
    # Cancelling the task cancels the approval the replacement was waiting on too.
    assert sorted(p["status"] for p in _events("approval.decided")) == ["CANCELLED", "CANCELLED"]


# ---- 4: an agent keeps its own node ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_agent_can_still_change_and_remove_its_own_node_once_it_is_ready() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "own-node", "hold-always")
            view, plan = await _until(env, handle, lambda v, p: len(TURNS) == 1, "the attempt")
            # Paused, so the node the agent adds is READY and not started.
            await _control(handle, 1, "pause")
            me = Actor(kind="agent", id="agent-x", attempt_id=TURNS[0]["attempt_id"])

            async def change(number: int, version: int, *ops: Any):
                return await handle.execute_update(
                    TaskWorkflow.submit_plan_change,
                    PlanChangeCommand(
                        command_id=f"01JA{number:022d}", task_id=view.task_id, base_plan_version=version,
                        actor=me, ops=list(ops),
                    ),
                )

            added = await change(1, 1, AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title="mine", spec=AgentTurnSpec(goal="later"))))
            assert added.status == "accepted"
            node_id = added.id_map["tmp:1"]
            plan = await _query(handle, TaskWorkflow.get_plan)
            assert _node(plan, node_id).status == "READY"
            updated = await change(2, plan.plan_version, UpdateNodeOp(node_id=node_id, patch=NodePatch(title="mine, renamed")))
            assert updated.status == "accepted", updated
            plan = await _query(handle, TaskWorkflow.get_plan)
            removed = await change(3, plan.plan_version, RemoveNodeOp(node_id=node_id))
            assert removed.status == "accepted", removed
            await _control(handle, 2, "cancel")
            await _closed(handle)


# ---- 5: wait nodes ----------------------------------------------------------------------------------------------------


async def _cause_continue_as_new(handle, first: int, count: int = 1010) -> None:
    first_run = (await handle.describe()).run_id
    for index in range(count):
        await handle.execute_update(
            TaskWorkflow.grant_budget, GrantBudgetInput(command_id=f"01JB{first + index:022d}", delta=Budget(tokens=1))
        )
    # The run ends once the loop has drained its handlers and flushed, a moment after the last update was answered.
    async for _ in polls():
        if (await handle.describe()).run_id != first_run:
            return


def _wait_node(title: str, **spec: Any) -> AddNodeOp:
    return AddNodeOp(node=WaitNode(node_id="tmp:1", title=title, spec=WaitSpec(**spec)))


@pytest.mark.asyncio
async def test_a_wait_nodes_timer_survives_continue_as_new() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "wait-timer", "complete the task")
            view, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the first round")
            first_run = (await handle.describe()).run_id
            added = await handle.execute_update(
                TaskWorkflow.submit_plan_change,
                PlanChangeCommand(
                    command_id="01J00000000000000000000080", task_id=view.task_id, base_plan_version=plan.plan_version,
                    actor=Actor(kind="user", id="user-a"), ops=[_wait_node("hold on", timer_s=100)],
                ),
            )
            node_id = added.id_map["tmp:1"]
            plan = await _query(handle, TaskWorkflow.get_plan)
            assert _node(plan, node_id).status == "AWAITING_INPUT"
            await _cause_continue_as_new(handle, 1)
            assert (await handle.describe()).run_id != first_run, "the workflow continued as new"
            plan = await _query(handle, TaskWorkflow.get_plan)
            assert _node(plan, node_id).status == "AWAITING_INPUT", "and the node is still waiting"
            await env.sleep(timedelta(seconds=150))
            _, plan = await _until(env, handle, lambda v, p: _node(p, node_id).status == "COMPLETED", "the timer")
            assert _node(plan, node_id).frozen, "a node the timer completes is frozen, like any completed node"


@pytest.mark.asyncio
async def test_a_wait_for_an_event_that_already_came_completes_frozen() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "wait-key", "complete the task")
            view, plan = await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the first round")
            await handle.signal(TaskWorkflow.external_event, ExternalEventSignal(wait_key="go"))
            added = await handle.execute_update(
                TaskWorkflow.submit_plan_change,
                PlanChangeCommand(
                    command_id="01J00000000000000000000081", task_id=view.task_id, base_plan_version=plan.plan_version,
                    actor=Actor(kind="user", id="user-a"), ops=[_wait_node("the event", wait_key="go")],
                ),
            )
            node_id = added.id_map["tmp:1"]
            _, plan = await _until(env, handle, lambda v, p: _node(p, node_id).status == "COMPLETED", "the wait")
            assert _node(plan, node_id).frozen


# ---- 8: what a Continue-As-New carries is bounded --------------------------------------------------------------------


def _continued_as_new_inputs(history) -> list[dict[str, Any]]:
    """What the runs of a history carried over: the input of a run that started by continuing as new (the first run of a
    workflow has none), and the input a run continued as new with."""
    out = []
    for event in history.events:
        attrs = event.workflow_execution_continued_as_new_event_attributes
        if attrs.input.payloads:
            out.append(json.loads(attrs.input.payloads[0].data))
        started = event.workflow_execution_started_event_attributes
        if started.continued_execution_run_id and started.input.payloads:
            out.append(json.loads(started.input.payloads[0].data))
    return out


@pytest.mark.asyncio
async def test_two_thousand_messages_do_not_make_the_carried_state_grow() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "bounded", "hold-always")
            first_run = handle.result_run_id
            await _until(env, handle, lambda v, p: len(TURNS) == 1, "the attempt")
            for number in range(1, 2001):
                await _say(handle, number, f"message {number} " + "x" * 40)
            async for _ in polls():  # the main loop hands them over one by one, behind the updates that took them
                if not await _query(handle, TaskWorkflow.get_inbox, 0):
                    break
            assert await _query(handle, TaskWorkflow.get_inbox, 0) == [], "every message was handed to the attempt"
            async for _ in polls():  # the run ends a moment after its last update is answered: wait for the next one
                if (await handle.describe()).run_id != first_run:
                    break
            assert (await handle.describe()).run_id != first_run
            # What the first run carried over is the input the next run started with. It is read from there: under load the
            # test server was seen to answer a request for the 20,000-event history of the first run with the events it had
            # before the run ended, none of them its continuation.
            history = await env.client.get_workflow_handle(handle.id, run_id=(await handle.describe()).run_id).fetch_history()
            await _control(handle, 1, "cancel")
            await _closed(handle)
    carried = _continued_as_new_inputs(history)
    assert carried, "the first run continued as new"
    sizes = [len(json.dumps(item)) for item in carried]
    assert max(sizes) < 256 * 1024, sizes
    carry = carried[0]["carry"]
    assert carry["inbox"] == []
    assert len(carry["dedup"]) <= 512, "the dedup window keeps the 512 most recent"
    # The run ends at 1000 updates, or earlier when the server suggests it for the size of the history (which depends on how
    # much each message adds): either way the sequence is carried and goes on, it does not start again.
    assert carry["next_message_seq"] > 100, "the sequence goes on"


@pytest.mark.asyncio
async def test_a_message_waits_in_the_inbox_until_an_attempt_has_it() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "inbox", "hold-always")
            await _until(env, handle, lambda v, p: len(TURNS) == 1, "the attempt")
            await _control(handle, 1, "stop")
            await _until(env, handle, lambda v, p: v.status == "PAUSED", "the stop")
            await _say(handle, 1, "one")
            await _say(handle, 2, "two")
            assert [m.text for m in await _query(handle, TaskWorkflow.get_inbox, 0)] == ["one", "two"]
            assert [m.text for m in await _query(handle, TaskWorkflow.get_inbox, 1)] == ["two"]
            await _control(handle, 2, "resume")
            await _until(env, handle, lambda v, p: len(TURNS) == 2, "the next attempt")
            assert [m["text"] for m in TURNS[1]["messages"]] == ["one", "two"]
            assert await _query(handle, TaskWorkflow.get_inbox, 0) == []
            await _control(handle, 3, "cancel")
            await _closed(handle)


# ---- 9: a long conversation does not fill the plan --------------------------------------------------------------------


async def _follow_up(env, handle, number: int, text: str) -> None:
    await _say(handle, number, text)
    async for _ in polls(timeout=90, interval=0.01):  # a deadline of 90s of wall time, which a loaded machine does not reach
        view = await _query(handle, TaskWorkflow.get_task_view)
        # Answered and settled: this follow-up's turn ran, and no node of the plan is still open (a COMPLETED left over from the
        # round before says nothing about the message just sent).
        if view.status == "COMPLETED" and len(TURNS) == number + 1:
            plan = await _query(handle, TaskWorkflow.get_plan)
            if all(node.status == "COMPLETED" for node in plan.nodes) and not await _query(handle, TaskWorkflow.get_inbox, 0):
                return
    raise AssertionError(f"follow-up {number} was not answered: {view.status} {len(TURNS)} turns")


@pytest.mark.asyncio
async def test_a_thousand_long_follow_ups_are_all_answered_and_the_plan_stays_small() -> None:
    _reset()
    count = 1000
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "follow-ups", "complete the task")
            await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the first round")
            for number in range(1, count + 1):
                await _follow_up(env, handle, number, f"{number:04d} " + "word " * 400)  # about 2KB each
            plan = await _query(handle, TaskWorkflow.get_plan)
    assert len(TURNS) == count + 1, "every follow-up was answered"
    # The attempt is given the whole message, and the node keeps a bounded goal.
    assert all(len(turn["goal"]) <= 500 for turn in TURNS[1:])
    assert TURNS[-1]["goal"].startswith(f"{count:04d} ") and len(TURNS[-1]["goal"]) == 500
    assert [m["text"] for m in TURNS[-1]["messages"]] == [f"{count:04d} " + "word " * 400]
    assert not _events("plan.change_rejected")
    assert plan.archived is not None and plan.archived.count > 900 and len(plan.nodes) < 100, (len(plan.nodes), plan.archived)
    assert len(plan.archived.recent_titles) <= 10
    versions = [p["plan_version"] for p in _events("plan.version_committed")]
    assert versions == sorted(versions) and len(set(versions)) == len(versions), "every version is committed once, in order"
    compactions = [p for p in _events("plan.version_committed") if p.get("reason") == "compaction"]
    assert compactions and all(p["actor"]["kind"] == "system" for p in compactions)


@pytest.mark.asyncio
async def test_a_follow_up_that_cannot_be_added_is_reported_and_the_task_asks_for_a_review() -> None:
    _reset()
    from orbit_orch.plan_engine import deterministic_id

    task_id = deterministic_id("resilience:reject", "task")
    # A plan that is full of finished nodes nothing may remove (they are not frozen), carried in as it would be by a
    # Continue-As-New, so that the next message does not fit.
    nodes = [
        {
            "node_id": deterministic_id(f"reject:{index}", "n"),
            "draft": AgentTurnNode(
                node_id=deterministic_id(f"reject:{index}", "n"), title=f"done {index}", spec=AgentTurnSpec(goal="g" * 14000)
            ).model_dump(mode="json", by_alias=True, exclude_none=True),
            "status": "COMPLETED", "frozen": False, "current_attempt_id": None, "attempt_count": 1, "created_by": "user-a",
        }
        for index in range(19)
    ]
    inp = _input(task_id, "complete the task").model_copy(
        update={"carry": {"status": "COMPLETED", "plan_version": 7, "nodes": nodes, "edges": [], "next_message_seq": 1}}
    )
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(TaskWorkflow.run, inp, id="task/tenant-a/reject", task_queue="orbit.orch")
            await _until(env, handle, lambda v, p: v.status == "COMPLETED" and p.plan_version == 7, "the carried plan")
            await _say(handle, 1, "does this fit?")
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
            await _control(handle, 1, "cancel")
            await _closed(handle)
    assert plan.plan_version == 7 and TURNS == []
    rejected = _events("plan.change_rejected")
    assert len(rejected) == 1 and rejected[0]["code"] == "TOO_MANY_OPS"
    review = [p for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"]
    assert len(review) == 1 and "message 1 could not be added to the plan" in review[0]["reason"]


@pytest.mark.asyncio
async def test_a_rejected_follow_up_is_reported_once_across_many_loop_passes_and_a_continue_as_new() -> None:
    _reset()
    from orbit_orch.plan_engine import deterministic_id

    task_id = deterministic_id("resilience:reject-once", "task")
    # A plan that is full of finished nodes nothing may remove (they are not frozen), carried in as it would be by a
    # Continue-As-New, so that the next message does not fit.
    nodes = [
        {
            "node_id": deterministic_id(f"reject:{index}", "n"),
            "draft": AgentTurnNode(
                node_id=deterministic_id(f"reject:{index}", "n"), title=f"done {index}", spec=AgentTurnSpec(goal="g" * 14000)
            ).model_dump(mode="json", by_alias=True, exclude_none=True),
            "status": "COMPLETED", "frozen": False, "current_attempt_id": None, "attempt_count": 1, "created_by": "user-a",
        }
        for index in range(19)
    ]
    inp = _input(task_id, "complete the task").model_copy(
        update={"carry": {"status": "COMPLETED", "plan_version": 7, "nodes": nodes, "edges": [], "next_message_seq": 1}}
    )
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(TaskWorkflow.run, inp, id="task/tenant-a/reject-once", task_queue="orbit.orch")
            await _until(env, handle, lambda v, p: v.status == "COMPLETED" and p.plan_version == 7, "the carried plan")
            await _say(handle, 1, "does this fit?")
            _, plan = await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
            # Many more passes of the main loop, and a Continue-As-New: the message that was refused is not tried again.
            first_run = (await handle.describe()).run_id
            await _cause_continue_as_new(handle, 2000)
            assert (await handle.describe()).run_id != first_run
            assert await _query(handle, TaskWorkflow.get_inbox, 0) == [], "the refused message left the inbox"
            # A grant of budget is not what this review waits for: the task stays held through all of those passes.
            assert (await _query(handle, TaskWorkflow.get_task_view)).status == "PAUSED_NEEDS_REVIEW"
            await _control(handle, 1, "cancel")
            await _closed(handle)
    assert plan.plan_version == 7 and TURNS == []
    await asyncio.sleep(0.2)
    rejected = _events("plan.change_rejected")
    assert len(rejected) == 1 and rejected[0]["code"] == "TOO_MANY_OPS"
    review = [p for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"]
    assert len(review) == 1 and "message 1 could not be added to the plan" in review[0]["reason"]


# ---- 12: an AttemptWorkflow with a long history continues as new ------------------------------------------------------


@pytest.mark.asyncio
async def test_an_attempt_continues_as_new_in_its_approval_loop_and_its_parent_notices_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from temporalio.worker import UnsandboxedWorkflowRunner

    _reset()
    # `is_continue_as_new_suggested` turns true after thousands of events; here it is true from the first reply on, and the
    # workflow runs unsandboxed so that the patch reaches it.
    monkeypatch.setattr(AttemptWorkflow, "_should_continue_as_new", lambda self: True)
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env, UnsandboxedWorkflowRunner())
        async with orch, agent, io:
            handle = await _start(env, "attempt-can", "approval")
            view, plan = await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the approval")
            child_id = f"attempt/{view.task_id}/{plan.nodes[0].node_id}/1"
            await _say(handle, 1, "and this too")  # reaches the attempt while it is parked
            await handle.execute_update(
                TaskWorkflow.decide_approval,
                DecideApprovalInput(command_id="01J00000000000000000000090", approval_id=view.pending_approvals[0], decision="approve"),
            )
            await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the task")
            history = await env.client.get_workflow_handle(child_id).fetch_history()
            started = history.events[0].workflow_execution_started_event_attributes
    assert started.continued_execution_run_id, "the run that finished the attempt began as a continuation of an earlier one"
    # What the second run was given: the session the first run's activity returned, and where its state stood.
    assert len(TURNS) == 2
    assert TURNS[1]["session_id"] == "s" and TURNS[1]["state_version"] == 2
    assert TURNS[1]["approval"]["decision"] == "approve", "the decision taken is carried over"
    assert [m["text"] for m in TURNS[1]["messages"]] == ["and this too"], "so are the messages the attempt had not consumed"
    # The parent saw one attempt, parked once and finished once.
    assert [p["outcome"] for p in _events("attempt.finished")] == ["completed"]
    assert len(_events("attempt.parked")) == 1 and len(_events("attempt.started")) == 1


# ---- what an attempt reports ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_attempt_that_gives_up_with_an_activity_failure_is_reported_as_lost_and_retried() -> None:
    from temporalio.exceptions import ApplicationError

    @activity.defn(name="agent_turn")
    async def broken(payload: dict[str, Any]) -> dict[str, Any]:
        TURNS.append(payload)
        raise ApplicationError("worker crashed", non_retryable=True)

    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env, turn=broken)
        async with orch, agent, io:
            handle = await _start(env, "lost", "anything")
            await _until(env, handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review", skip_s=10)
    assert [t["attempt_no"] for t in TURNS] == [1, 2, 3], "retried, and then given to a person"
    failures = [p["failure"] for p in _events("attempt.finished")]
    assert [(f["failure_class"], f["retryable"]) for f in failures] == [("lost", True)] * 3
    assert "worker crashed" in failures[0]["message"]


@pytest.mark.asyncio
async def test_a_node_that_depends_on_an_archived_node_is_ready_at_once() -> None:
    _reset()
    from orbit_orch.plan_engine import deterministic_id

    task_id = deterministic_id("resilience:archived", "task")
    old = deterministic_id("archived:old", "n")
    nodes = [{
        "node_id": deterministic_id("archived:live", "n"),
        "draft": AgentTurnNode(
            node_id=deterministic_id("archived:live", "n"), title="live", spec=AgentTurnSpec(goal="complete the task")
        ).model_dump(mode="json", by_alias=True, exclude_none=True),
        "status": "COMPLETED", "frozen": True, "current_attempt_id": None, "attempt_count": 1, "created_by": "user-a",
    }]
    inp = _input(task_id, "complete the task").model_copy(update={"carry": {
        "status": "COMPLETED", "plan_version": 7, "nodes": nodes, "edges": [], "next_message_seq": 1,
        "archive": {"count": 1, "hash": "sha256:" + "0" * 64, "recent": ["old"], "ids": [old]},
    }})
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await env.client.start_workflow(TaskWorkflow.run, inp, id="task/tenant-a/archived", task_queue="orbit.orch")
            await _until(env, handle, lambda v, p: v.status == "COMPLETED" and p.plan_version == 7, "the carried plan")
            result = await handle.execute_update(
                TaskWorkflow.submit_plan_change,
                PlanChangeCommand(
                    command_id="01J00000000000000000000099", task_id=task_id, base_plan_version=7, actor=Actor(kind="user", id="u"),
                    ops=[AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title="after old", depends_on=[old], spec=AgentTurnSpec(goal="after")))],
                ),
            )
            assert result.status == "accepted", result
            await _until(env, handle, lambda v, p: len(TURNS) == 1 and v.status == "COMPLETED" and p.plan_version == 8, "the node to run")
            plan = await _query(handle, TaskWorkflow.get_plan)
            assert plan.archived is not None and plan.archived.count == 1
            await _control(handle, 1, "cancel")
            await _closed(handle)
    assert TURNS[0]["goal"] == "after"


@pytest.mark.asyncio
async def test_a_message_that_meets_an_attempt_as_it_closes_does_not_fail_the_task() -> None:
    """The parent signals the attempt that is running with a queued message. An attempt that has just finished has closed its
    workflow before the parent has handled its report, and signalling a closed workflow used to raise "Unable to signal
    external workflow because it was not found" out of the main loop: the whole task ended FAILED, and its queries kept
    answering with the last state (RUNNING). The race is a few milliseconds wide, so the message is sent at many different
    moments after the start; the task must complete every time and never be failed."""
    import random

    from temporalio.client import WorkflowExecutionStatus

    _reset()
    delays = random.Random(11)
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            for number in range(150):
                handle = await _start(env, f"closing-{number}", "complete the task")
                await asyncio.sleep(delays.uniform(0, 0.12))
                await _say(handle, number + 1, "while it closes")
                async for _ in polls():
                    view = await _query(handle, TaskWorkflow.get_task_view)
                    if view.status == "COMPLETED" or (await handle.describe()).status != WorkflowExecutionStatus.RUNNING:
                        break
                described = await handle.describe()
                assert described.status == WorkflowExecutionStatus.RUNNING, f"task {number} ended: {described.status}"
                assert view.status == "COMPLETED", f"task {number} is {view.status}"


@pytest.mark.asyncio
async def test_a_held_task_is_not_completed_by_its_nodes_being_done_until_it_is_resumed() -> None:
    """A task a person paused or took over, or that waits for a review, stays as it is when every node is done (a loop pass
    used to set COMPLETED and drop what the person had to act on); resume or handback then completes it."""
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "held-done", "approval")
            await _until(env, handle, lambda v, p: bool(v.pending_approvals), "the approval")
            await _control(handle, 1, "pause")
            await _until(env, handle, lambda v, p: v.status == "PAUSED", "the pause")
            await handle.execute_update(
                TaskWorkflow.decide_approval,
                DecideApprovalInput(
                    command_id="01JD00000000000000000000A1",
                    approval_id=(await _query(handle, TaskWorkflow.get_task_view)).pending_approvals[0],
                    decision="approve",
                ),
            )
            await _until(env, handle, lambda v, p: p.nodes[0].status == "COMPLETED", "the node to finish")
            async for _ in polls():  # many passes of the loop: none of them completes the task
                await _query(handle, TaskWorkflow.get_task_view)
            assert (await _query(handle, TaskWorkflow.get_task_view)).status == "PAUSED"
            await _control(handle, 2, "resume")
            await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the task to complete after the resume")

            # Taken over, then handed back: the same.
            await _say(handle, 1, "one more thing")
            await _until(env, handle, lambda v, p: len(TURNS) >= 2 and v.status == "COMPLETED", "the follow-up")
            await _control(handle, 3, "takeover")
            assert (await _query(handle, TaskWorkflow.get_task_view)).status == "TAKEN_OVER"
            await asyncio.sleep(0.3)
            assert (await _query(handle, TaskWorkflow.get_task_view)).status == "TAKEN_OVER"
            await _control(handle, 4, "handback")
            await _until(env, handle, lambda v, p: v.status == "COMPLETED", "the task to complete after the handback")


@pytest.mark.asyncio
async def test_a_message_signalled_to_an_attempt_that_never_reads_it_is_answered_as_a_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent signals `deliverMessages` to a child that is about to report its end; the server accepts the signal, but the
    child has left its loop and never gives the message to a turn (and does not list it as unconsumed). The parent must not
    take the signal for consumption: the message is answered after the attempt. The child is made to behave so
    deterministically: it takes the message in and drops it before it reports."""
    from orbit_orch.attempt_workflow import AttemptWorkflow as Attempt
    from temporalio import workflow
    from temporalio.worker import UnsandboxedWorkflowRunner

    _reset()
    original = Attempt._notify_parent_finished

    async def drop_then_finish(self, inp, outcome, result):
        if outcome == "completed" and inp.attempt_no == 1 and inp.continue_from is None and inp.goal != "a late word":
            await workflow.wait_condition(lambda: bool(self._messages), timeout=timedelta(seconds=300))
            self._messages = []  # signalled, and never read by a turn
        await original(self, inp, outcome, result)

    monkeypatch.setattr(Attempt, "_notify_parent_finished", drop_then_finish)
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env, UnsandboxedWorkflowRunner())
        async with orch, agent, io:
            handle = await _start(env, "late-message", "complete the task")
            await _until(env, handle, lambda v, p: any(n.status == "RUNNING" for n in p.nodes), "the attempt")
            await _say(handle, 1, "a late word")
            await _until(
                env, handle,
                lambda v, p: len(TURNS) == 2 and v.status == "COMPLETED" and all(n.status == "COMPLETED" for n in p.nodes),
                "the follow-up",
            )
            await _control(handle, 9, "cancel")
            await _closed(handle)
    assert TURNS[1]["goal"] == "a late word", "answered as a follow-up of its own"


@pytest.mark.asyncio
async def test_a_plan_change_that_reaches_the_workflow_in_the_activation_of_its_start_is_judged_against_the_real_plan() -> None:
    """The workflow is started and a plan change is sent before any worker has polled, so both arrive in the first activation, the
    update ahead of `run`. The update must wait for the plan that `run` builds from its input: judged against one built from
    nothing (an empty goal, a guessed profile) the workflow task fails, and fails again each time it is tried."""
    from orbit_orch.plan_engine import deterministic_id

    _reset()
    task_id = deterministic_id("resilience:early-update", "task")
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        handle = await env.client.start_workflow(
            TaskWorkflow.run, _input(task_id, "hold-always"), id="task/tenant-a/early-update", task_queue="orbit.orch"
        )
        command = PlanChangeCommand(
            command_id="01J00000000000000000000150", task_id=task_id, base_plan_version=1, actor=Actor(kind="user", id="u"),
            ops=[AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title="early", spec=AgentTurnSpec(goal="early work")))],
        )
        update = asyncio.ensure_future(handle.execute_update(TaskWorkflow.submit_plan_change, command))
        await asyncio.sleep(0.5)  # the update is queued at the server with the start: no worker has seen either
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            result = await asyncio.wait_for(update, timeout=60)
            assert result.status == "accepted", result
            plan = await _query(handle, TaskWorkflow.get_plan)
            assert [(n.title, n.owner_profile) for n in plan.nodes] == [("Explore and plan", "default@1"), ("early", "default@1")]
            await _control(handle, 1, "cancel")
            await _closed(handle)

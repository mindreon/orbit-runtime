"""What the task owes its people and its agents: a budget that is reserved and settled, a limit on what runs at once, a view
of the plan that fits the agent, a profile that can be switched and a task that a person can take over (05 §4 and §6,
04 §3, 11 §3).

How it can go wrong, written down before the code:
  - attempts that start together each spend "what was left when they started", and the task overspends;
  - a node the task cannot pay for is started anyway, or the task is left running with nothing it can start;
  - what an attempt did not use is never returned, so a task runs out of budget with most of it unspent;
  - a grant of budget resumes a task that a person paused or took over, or one that is waiting for another reason;
  - every ready node starts at once, in the order of their ids, and a plan with fifty nodes starts fifty attempts;
  - an agent sees, and changes, nodes that are not its own;
  - a switch of profile interrupts the attempt that is running, or the next attempt carries a session of another model;
  - a takeover leaves the agents working, and the person's own completion of a node needs the agents to be verified;
  - a handback from anything but a takeover un-pauses or un-cancels the task.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from orbit_contracts.v3 import (
    Actor,
    Budget,
    CompleteNodeInput,
    DecideApprovalInput,
    GrantBudgetInput,
    PlanChangeCommand,
    Policy,
    RequestProfileSwitchInput,
    TaskConfig,
    TaskControlInput,
    TaskWorkflowInput,
    Team,
    TeamMember,
)
from orbit_contracts.v3.common import Usage
from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec, ApprovalNode, ApprovalSpec
from orbit_contracts.v3.plan import AddEdgeOp, AddNodeOp
from orbit_orch import budgets
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio import activity
from temporalio.client import WorkflowUpdateFailedError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from waiting import polls

TURNS: list[dict[str, Any]] = []
EVENTS: list[dict[str, Any]] = []
STATE: dict[str, Any] = {"running": 0, "max_running": 0, "release": None}
FAST = {
    "max_heartbeat_throttle_interval": timedelta(milliseconds=100),
    "default_heartbeat_throttle_interval": timedelta(milliseconds=100),
}
USER = Actor(kind="user", id="user-a")


def _ref(char: str) -> str:
    return "sha256:" + char * 64


def _done(**extra: Any) -> dict[str, Any]:
    return {"status": "completed", "checkpoint_ref": _ref("1"), "session_id": "s", "state_version": 2, **extra}


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """A turn whose goal says what it does. `spend:<n>` completes having used n tokens; `over:<n>` fails its first attempt
    having spent its budget (n tokens); `gate` (or `gate:<n>`, having used n tokens) waits to be released and then completes; `flaky` waits to be released and fails
    its first attempt; `hold` and `hold-once` (the first attempt only) wait until they are cancelled; `work` takes a moment and
    is counted while it does; `fail-policy` fails for a reason that retrying cannot fix."""
    TURNS.append(payload)
    goal, attempt_no = str(payload["goal"]), int(payload["attempt_no"])
    if goal.startswith("spend:"):
        return _done(usage={"tokens_in": int(goal.split(":")[1]), "tool_calls": 1, "wall_s": 1})
    if goal.startswith("over:") and attempt_no == 1:
        return {
            "status": "failed", "error": "the attempt's token budget is spent", "failure_class": "budget", "retryable": False,
            "usage": {"tokens_in": int(goal.split(":")[1]), "tool_calls": 2, "wall_s": 3},
        }
    if goal == "fail-policy":
        return {"status": "failed", "error": "not allowed", "failure_class": "policy", "retryable": False}
    if goal.startswith("gate") or goal == "flaky":
        await STATE["release"].wait()
        if goal == "flaky" and attempt_no == 1:
            return {"status": "failed", "error": "boom 1", "failure_class": "model", "retryable": True}
        return _done(**({"usage": {"tokens_in": int(goal.split(":")[1])}} if ":" in goal else {}))
    if goal == "hold" or (goal == "hold-once" and attempt_no == 1):
        for _ in range(6000):
            activity.heartbeat()
            await asyncio.sleep(0.05)
    if goal.startswith("work"):
        STATE["running"] += 1
        STATE["max_running"] = max(STATE["max_running"], STATE["running"])
        try:
            await asyncio.sleep(0.15)
        finally:
            STATE["running"] -= 1
    return _done()


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


def _reset() -> None:
    TURNS.clear()
    EVENTS.clear()
    STATE.update(running=0, max_running=0, release=asyncio.Event())


def _stack(env: WorkflowEnvironment) -> tuple[Worker, Worker, Worker]:
    return (
        Worker(env.client, task_queue="orbit.orch", workflows=[TaskWorkflow, AttemptWorkflow], workflow_runner=sandbox_runner()),
        Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn, _sop_step], **FAST),
        Worker(
            env.client, task_queue="orbit.io",
            activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
        ),
    )


class Task:
    """A running TaskWorkflow and the little the tests do to it."""

    def __init__(self, env: WorkflowEnvironment, handle: Any, name: str) -> None:
        self.env, self.handle = env, handle
        self.task_id = deterministic_id(f"budget-scheduling:{name}", "task")
        self.exploration = deterministic_id(f"{self.task_id}:exploration", "n")
        self._numbers = 0

    @classmethod
    async def start(
        cls, env: WorkflowEnvironment, name: str, goal: str, *, budgets: Budget | None = None,
        policy: Policy | None = None, config: TaskConfig | None = None,
    ) -> Task:
        task_id = deterministic_id(f"budget-scheduling:{name}", "task")
        handle = await env.client.start_workflow(
            TaskWorkflow.run,
            TaskWorkflowInput(
                task_id=task_id, tenant_id="tenant-a", created_by=USER, title="Budget and scheduling", goal=goal,
                profile="default@1", node_type_registry_version=1, budgets=budgets or Budget(),
                policy=policy or Policy(), config=config or TaskConfig(),
            ),
            id=f"task/tenant-a/{name}", task_queue="orbit.orch",
        )
        return cls(env, handle, name)

    def command_id(self) -> str:
        self._numbers += 1
        # Stable across processes (`hash()` of a string is salted per process): the ids a test gets are the same on every run.
        digits = int(hashlib.sha256(self.task_id.encode()).hexdigest(), 16) % 10**16
        return f"01JD{self._numbers:06d}{digits:016d}"[:26]

    async def view(self):
        return await self.handle.query(TaskWorkflow.get_task_view)

    async def plan(self, actor: Actor | None = None):
        return await self.handle.query(TaskWorkflow.get_plan, actor)

    async def until(self, ready, what: str, *, skip_s: int = 0):
        view = plan = None
        async for _ in polls():
            view, plan = await self.view(), await self.plan()
            if ready(view, plan):
                return view, plan
            if skip_s:
                await self.env.sleep(timedelta(seconds=skip_s))
        assert view is not None and plan is not None
        raise AssertionError(f"timed out waiting for {what}: {view.status} {[(n.title, n.status) for n in plan.nodes]} {len(TURNS)} turns")

    async def change(self, *ops: Any, actor: Actor = USER, command_id: str | None = None):
        plan = await self.plan()
        return await self.handle.execute_update(
            TaskWorkflow.submit_plan_change,
            PlanChangeCommand(
                command_id=command_id or self.command_id(), task_id=self.task_id, base_plan_version=plan.plan_version,
                actor=actor, ops=list(ops),
            ),
        )

    def control(self, action: str):
        return self.handle.execute_update(TaskWorkflow.control, TaskControlInput(command_id=self.command_id(), action=action))

    def grant(self, **delta: int):
        return self.handle.execute_update(TaskWorkflow.grant_budget, GrantBudgetInput(command_id=self.command_id(), delta=Budget(**delta)))

    def attempt_of(self, node_id: str, number: int = 1) -> str:
        return deterministic_id(f"{self.task_id}:{node_id}:{number}", "att")


def _agent(ref: str, goal: str, *, after: list[str] | None = None, budget: Budget | None = None, parent: str | None = None) -> AddNodeOp:
    return AddNodeOp(node=AgentTurnNode(
        node_id=ref, title=ref, spec=AgentTurnSpec(goal=goal), depends_on=after or [],
        budget=budget or Budget(), parent_node_id=parent,
    ))


def _events(kind: str) -> list[dict[str, Any]]:
    return [e["payload"] for e in EVENTS if e["type"] == kind]


async def _seen(kind: str, count: int = 1) -> list[dict[str, Any]]:
    """Events reach the sink a moment after the update that caused them: wait for `count` of a kind."""
    async for _ in polls():
        if len(_events(kind)) >= count:
            break
    return _events(kind)


def _node(plan, node_id: str):
    return next(n for n in plan.nodes if n.node_id == node_id)


async def _rejected(update) -> str:
    with pytest.raises(WorkflowUpdateFailedError) as refused:
        await update
    assert isinstance(refused.value.cause, ApplicationError)
    return str(refused.value.cause.type)


@contextlib.asynccontextmanager
async def environment() -> AsyncIterator[WorkflowEnvironment]:
    """The time-skipping server, the three workers, and a clean slate."""
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            yield env


# ---- the arithmetic ---------------------------------------------------------------------------------------------------


def test_what_is_left_is_the_limit_less_what_was_spent_and_what_running_attempts_hold() -> None:
    limits = Budget(tokens=1000, tool_calls=10)
    spent = Usage(tokens_in=200, tokens_out=100, tool_calls=3)
    left = budgets.remaining(limits, spent, [Budget(tokens=400), Budget(tokens=50, wall_s=60)])
    assert left == {"tokens": 250, "tool_calls": 7, "wall_s": None, "cost_usd_micros": None}
    assert budgets.remaining(Budget(tokens=10), Usage(tokens_in=50), [])["tokens"] == 0, "overspent is nothing left, not less"


def test_an_attempt_reserves_what_its_node_names_and_a_share_of_what_it_leaves_open() -> None:
    left = {"tokens": 900, "tool_calls": 30, "wall_s": None, "cost_usd_micros": None}
    reserved, why = budgets.reserve(Budget(tokens=200), left, share=3)
    assert reserved == Budget(tokens=200, tool_calls=10, wall_s=None, cost_usd_micros=None) and why == ""
    assert budgets.reserve(Budget(), left, share=1)[0] == Budget(tokens=900, tool_calls=30)
    none, why = budgets.reserve(Budget(tokens=901), left, share=1)
    assert none is None and "needs 901 and 900 is left" in why
    none, why = budgets.reserve(Budget(), {**left, "tokens": 0}, share=1)
    assert none is None and "tokens: nothing is left" in why


def test_an_unknown_cost_is_not_zero_when_usage_is_added_up() -> None:
    unknown = Usage(tokens_in=1)
    assert budgets.usage_add(unknown, unknown).cost_usd_micros is None
    assert budgets.usage_add(unknown, Usage(cost_usd_micros=5)).cost_usd_micros == 5
    assert budgets.spent(unknown, "cost_usd_micros") == 0 and budgets.spent(Usage(tokens_in=2, tokens_out=3), "tokens") == 5
    assert budgets.after(Budget(tokens=10, wall_s=None), Usage(tokens_in=4)) == Budget(tokens=6)


# ---- 1: a budget is reserved, spent within and settled ------------------------------------------------------------------


async def test_a_reservation_is_held_while_the_attempt_runs_and_settled_by_what_it_spent() -> None:
    async with environment() as env:
        task = await Task.start(env, "reserve-settle", "gate", budgets=Budget(tokens=1000, tool_calls=20))
        view, _ = await task.until(lambda v, p: len(TURNS) == 1, "the attempt")
        # The node named no budget: it holds all the task has left, and is told so.
        assert TURNS[0]["budget"] == {"tokens": 1000, "tool_calls": 20}
        view = await task.view()
        assert view.budget_reserved.tokens == 1000 and view.usage.tokens_in == 0
        started = _events("attempt.started")[0]
        assert started["budget_reserved"] == {"tokens": 1000, "tool_calls": 20}

        STATE["release"].set()
        await task.until(lambda v, p: v.status == "COMPLETED", "the attempt to end")
        view = await task.view()
        assert view.budget_reserved.tokens is None, "what was held is released"


async def test_what_an_attempt_spent_joins_the_usage_and_is_recorded_and_the_rest_is_the_tasks_again() -> None:
    async with environment() as env:
        task = await Task.start(env, "settle", "spend:300", budgets=Budget(tokens=1000))
        view, _ = await task.until(lambda v, p: v.status == "COMPLETED", "the attempt")
        assert (view.usage.tokens_in, view.usage.tool_calls, view.usage.wall_s) == (300, 1, 1)
        assert view.usage.cost_usd_micros is None, "an unknown cost stays unknown"
        assert view.budget_reserved.tokens is None
        recorded = _events("usage.recorded")
        assert len(recorded) == 1 and recorded[0]["usage"]["tokens_in"] == 300
        assert recorded[0]["attempt_id"] == task.attempt_of(task.exploration)
        # The next node can use what is left: 700.
        await task.change(_agent("tmp:1", "gate"))
        await task.until(lambda v, p: len(TURNS) == 2, "the next attempt")
        assert TURNS[1]["budget"] == {"tokens": 700}


async def test_attempts_that_start_together_share_what_is_left_instead_of_each_taking_it_all() -> None:
    async with environment() as env:
        task = await Task.start(env, "share", "gate", budgets=Budget(tokens=900))
        await task.until(lambda v, p: len(TURNS) == 1, "the exploration attempt")
        STATE["release"].set()
        await task.until(lambda v, p: v.status == "COMPLETED", "the exploration to end")
        STATE["release"] = asyncio.Event()
        await task.change(_agent("tmp:1", "gate"), _agent("tmp:2", "gate"), _agent("tmp:3", "gate"))
        await task.until(lambda v, p: len(TURNS) == 4, "three attempts")
        assert sorted(t["budget"]["tokens"] for t in TURNS[1:]) == [300, 300, 300]
        view = await task.view()
        assert view.budget_reserved.tokens == 900, "together they hold what the task has, not three times it"


async def test_a_node_the_task_cannot_pay_for_is_not_started_and_a_grant_starts_it() -> None:
    async with environment() as env:
        task = await Task.start(env, "cannot-cover", "gate:600", budgets=Budget(tokens=1000))
        await task.until(lambda v, p: len(TURNS) == 1, "the exploration attempt")
        # 800 fits the plan (the budget the task was given), but not what is left once the exploration has spent 600.
        added = await task.change(_agent("tmp:1", "work", after=[task.exploration], budget=Budget(tokens=800)))
        node = added.id_map["tmp:1"]
        STATE["release"].set()
        _, plan = await task.until(lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the hold")
        assert _node(plan, node).status == "READY" and len(TURNS) == 1, "the node did not start"
        exhausted = await _seen("budget.exhausted")
        assert len(exhausted) == 1 and exhausted[0]["scope"] == "task" and exhausted[0]["node_id"] == node
        assert "tokens: the node needs 800 and 400 is left" in exhausted[0]["detail"]
        reason = [p for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"][-1]["reason"]
        assert "budget exhausted" in reason

        await task.grant(tokens=1000)
        await task.until(lambda v, p: v.status == "COMPLETED", "the node to run and finish")
        assert TURNS[1]["budget"] == {"tokens": 800}
        assert (await _seen("budget.granted"))[0]["delta"] == {"tokens": 1000}
        assert (await task.view()).budgets.tokens == 2000


async def test_an_attempt_that_spent_its_budget_blocks_its_node_for_a_review_and_a_grant_frees_it() -> None:
    async with environment() as env:
        task = await Task.start(env, "spent", "gate", budgets=Budget(tokens=10_000))
        await task.until(lambda v, p: len(TURNS) == 1, "the exploration attempt")
        added = await task.change(_agent("tmp:1", "over:100", after=[task.exploration], budget=Budget(tokens=100)))
        node = added.id_map["tmp:1"]
        STATE["release"].set()
        view, plan = await task.until(lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
        assert _node(plan, node).status == "BLOCKED" and len(TURNS) == 2, "no retry: it would spend the same again"
        finished = next(p for p in _events("attempt.finished") if p["node_id"] == node)
        assert (finished["failure"]["failure_class"], finished["failure"]["retryable"]) == ("budget", False)
        assert finished["usage"]["tokens_in"] == 100
        assert view.usage.tokens_in == 100 and view.usage.tool_calls == 2, "what it spent is settled even though it failed"
        blocked = next(p for p in _events("node.status_changed") if p["to_status"] == "BLOCKED")
        assert "budget failure that cannot be retried" in blocked["reason"]

        await task.grant(tokens=5000)
        view, plan = await task.until(lambda v, p: v.status == "COMPLETED", "the node to run again")
        assert [t["attempt_no"] for t in TURNS if t["goal"] == "over:100"] == [1, 2]
        assert TURNS[2]["continue_from"] == TURNS[1]["attempt_id"], "it goes on from the session that stopped"


async def test_a_grant_resumes_only_a_task_that_waited_for_budget() -> None:
    async with environment() as env:
        # A task a person paused stays paused, and its running attempt is left alone.
        task = await Task.start(env, "grant-paused", "hold", budgets=Budget(tokens=1000))
        await task.until(lambda v, p: len(TURNS) == 1, "the attempt")
        await task.control("pause")
        await task.grant(tokens=500)
        assert (await task.view()).status == "PAUSED"
        # Taken over: the same.
        await task.control("takeover")
        await task.grant(tokens=500)
        assert (await task.view()).status == "TAKEN_OVER"
        assert (await task.view()).budgets.tokens == 2000


async def test_a_grant_does_not_resume_a_task_that_waits_for_another_reason() -> None:
    async with environment() as env:
        task = await Task.start(env, "grant-other", "fail-policy", budgets=Budget(tokens=1000))
        view, plan = await task.until(lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the review")
        await task.grant(tokens=500)
        view, plan = await task.view(), await task.plan()
        assert view.status == "PAUSED_NEEDS_REVIEW", "it did not wait for budget"
        assert plan.nodes[0].status == "BLOCKED", "a node blocked for another reason is not freed by budget"


# ---- 2: what runs at once, and in what order ----------------------------------------------------------------------------


async def test_ready_nodes_start_in_plan_order_and_at_most_the_policys_concurrency_at_a_time() -> None:
    async with environment() as env:
        task = await Task.start(env, "concurrency", "gate", policy=Policy(max_concurrency=2))
        await task.until(lambda v, p: len(TURNS) == 1, "the exploration attempt")
        STATE["release"].set()
        await task.until(lambda v, p: v.status == "COMPLETED", "the exploration to end")
        added = await task.change(*[_agent(f"tmp:{n}", f"work:{n}") for n in range(1, 6)])
        await task.until(lambda v, p: v.status == "COMPLETED" and len(TURNS) == 6, "all five nodes")
        created = [added.id_map[f"tmp:{n}"] for n in range(1, 6)]
        started = [p["node_id"] for p in _events("attempt.started")][1:]
        assert started == created, "in the order the nodes were created in, not the order of their ids"
        assert created != sorted(created), "(and that order is not the order of the ids)"
        assert STATE["max_running"] == 2


async def test_without_a_policy_four_attempts_run_at_once() -> None:
    async with environment() as env:
        task = await Task.start(env, "default-concurrency", "complete")
        await task.until(lambda v, p: v.status == "COMPLETED", "the exploration")
        await task.change(*[_agent(f"tmp:{n}", f"work:{n}") for n in range(1, 8)])
        await task.until(lambda v, p: v.status == "COMPLETED" and len(TURNS) == 8, "all seven nodes")
        assert STATE["max_running"] == 4


# ---- 3: what an agent sees ------------------------------------------------------------------------------------------------


async def test_an_agent_sees_its_own_node_what_it_created_and_their_dependencies_and_a_planner_sees_all() -> None:
    async with environment() as env:
        task = await Task.start(env, "visibility", "hold")
        await task.until(lambda v, p: len(TURNS) == 1, "the exploration attempt")
        added = await task.change(_agent("tmp:1", "hold"), _agent("tmp:2", "hold"))
        a, b = added.id_map["tmp:1"], added.id_map["tmp:2"]
        await task.until(lambda v, p: len(TURNS) == 3, "both attempts")
        planner = Actor(kind="agent", id="x", attempt_id=task.attempt_of(task.exploration))
        actor_a = Actor(kind="agent", id=task.attempt_of(a), attempt_id=task.attempt_of(a))
        actor_b = Actor(kind="agent", id=task.attempt_of(b), attempt_id=task.attempt_of(b))

        def names(plan) -> set[str]:
            return {n.node_id for n in plan.nodes}

        assert names(await task.plan(planner)) == {task.exploration, a, b}, "the node that plans sees the whole plan"
        assert names(await task.plan(actor_a)) == {a}
        assert names(await task.plan(None)) == {task.exploration, a, b}, "a caller that names nobody sees it all"
        assert names(await task.plan(USER)) == {task.exploration, a, b}
        shown = await task.plan(actor_a)
        full = await task.plan()
        assert (shown.plan_version, shown.hash) == (full.plan_version, full.hash), "the version is the whole plan's"
        stranger = Actor(kind="agent", id="s", attempt_id=deterministic_id("no-such-attempt", "att"))
        assert names(await task.plan(stranger)) == set(), "an attempt that is not running sees nothing"

        # A's plan changes stay within what it sees: it cannot reach for B ...
        reach = await task.change(
            AddEdgeOp.model_validate({"op": "add_edge", "from": b, "to": a}), actor=Actor(kind="agent", id=actor_a.id, attempt_id=actor_a.attempt_id)
        )
        assert reach.status == "rejected" and reach.code == "VISIBILITY"
        # ... but it can add to the plan below its own node, and then sees what it added, and B does not.
        child = await task.change(_agent("tmp:1", "work", after=[a]), actor=actor_a)
        assert child.status == "accepted"
        made = child.id_map["tmp:1"]
        assert names(await task.plan(actor_a)) == {a, made}
        assert names(await task.plan(actor_b)) == {b}
        assert names(await task.plan(planner)) == {task.exploration, a, b, made}
        # A person and the planner are not held to any of it.
        link = await task.change(AddEdgeOp.model_validate({"op": "add_edge", "from": b, "to": made}))
        assert link.status == "accepted"
        assert names(await task.plan(actor_a)) == {a, made, b}, "B is now a direct dependency of a node A created"


# ---- 4: nesting is the depth ----------------------------------------------------------------------------------------------


async def test_depth_is_nesting_and_a_long_pipeline_is_only_a_pipeline() -> None:
    async with environment() as env:
        task = await Task.start(env, "depth", "hold")
        await task.until(lambda v, p: len(TURNS) == 1, "the exploration attempt")
        chain = [_agent("tmp:1", "work", after=[task.exploration])] + [_agent(f"tmp:{n}", "work", after=[f"tmp:{n - 1}"]) for n in range(2, 13)]
        assert (await task.change(*chain)).status == "accepted", "twelve steps in a row, past the old limit of eight"

        first = await task.change(_agent("tmp:1", "work", parent=task.exploration, after=[task.exploration]))
        assert first.status == "accepted"
        second = await task.change(_agent("tmp:1", "work", parent=first.id_map["tmp:1"], after=[task.exploration]))
        assert (second.status, second.code) == ("rejected", "DEPTH_EXCEEDED"), "one level of nesting by default"


async def test_a_team_sets_how_deep_it_may_nest() -> None:
    async with environment() as env:
        team = Team(
            leader="lead", max_depth=2,
            members=[TeamMember(role="lead", expert="default@1"), TeamMember(role="dev", expert="dev@1")],
        )
        task = await Task.start(env, "team-depth", "hold", config=TaskConfig(expert="default@1", team=team))
        await task.until(lambda v, p: len(TURNS) == 1, "the exploration attempt")
        first = await task.change(_agent("tmp:1", "work", parent=task.exploration, after=[task.exploration]))
        second = await task.change(_agent("tmp:1", "work", parent=first.id_map["tmp:1"], after=[task.exploration]))
        third = await task.change(_agent("tmp:1", "work", parent=second.id_map["tmp:1"], after=[task.exploration]))
        assert (first.status, second.status) == ("accepted", "accepted")
        assert (third.status, third.code) == ("rejected", "DEPTH_EXCEEDED")


# ---- 5: a profile switch applies to the next attempt ----------------------------------------------------------------------

TEAM = Team(
    leader="lead",
    members=[TeamMember(role="lead", expert="boss@1"), TeamMember(role="dev", expert="helper@1")],
)


def _switch(task: Task, node_id: str, to: str, reason: str = "it needs a stronger model"):
    return task.handle.execute_update(
        TaskWorkflow.request_profile_switch,
        RequestProfileSwitchInput(command_id=task.command_id(), node_id=node_id, to_profile=to, reason=reason),
    )


async def test_a_switch_to_a_team_member_applies_to_the_next_attempt_without_touching_the_running_one() -> None:
    async with environment() as env:
        task = await Task.start(env, "switch-direct", "flaky", config=TaskConfig(expert="boss@1", team=TEAM))
        await task.until(lambda v, p: len(TURNS) == 1, "the first attempt")
        assert TURNS[0]["profile"] == "boss@1"
        result = await _switch(task, task.exploration, "helper@1")
        assert (result.needs_approval, result.approval_id, result.effective_attempt_no) == (False, None, 2)
        assert len(TURNS) == 1 and (await task.plan()).nodes[0].status == "RUNNING", "the running attempt was not interrupted"
        switched = await _seen("profile.switched")
        assert switched == [{"node_id": task.exploration, "from_profile": "boss@1", "to_profile": "helper@1", "reason": "it needs a stronger model"}]

        STATE["release"].set()
        await task.until(lambda v, p: v.status == "COMPLETED", "the next attempt", skip_s=10)
        second = TURNS[1]
        assert second["profile"] == "helper@1" and second["switched_from"] == "boss@1"
        assert second["continue_from"] is None, "another model does not carry the old session (11 §3)"
        assert "boom 1" in second["handover"], "it is told where the last attempt left off instead"
        started = _events("attempt.started")
        assert [p.get("switched_from") for p in started] == [None, "boss@1"] and started[1]["profile"] == "helper@1"


async def test_a_profile_outside_what_the_task_allows_needs_an_approval_and_applies_after_it() -> None:
    async with environment() as env:
        task = await Task.start(env, "switch-approved", "flaky", config=TaskConfig(expert="boss@1", team=TEAM))
        await task.until(lambda v, p: len(TURNS) == 1, "the first attempt")
        result = await _switch(task, task.exploration, "stranger@9")
        assert result.needs_approval and result.approval_id
        requested = await _seen("approval.requested")
        assert requested[0]["subject"]["kind"] == "profile_switch" and requested[0]["approval_id"] == result.approval_id
        assert result.approval_id in (await task.view()).pending_approvals
        assert _events("profile.switched") == [], "nothing has changed yet"

        decided = await task.handle.execute_update(
            TaskWorkflow.decide_approval, DecideApprovalInput(command_id=task.command_id(), approval_id=result.approval_id, decision="approve")
        )
        assert decided.status == "APPROVED"
        switched = await _seen("profile.switched")
        assert len(switched) == 1 and switched[0]["to_profile"] == "stranger@9" and switched[0]["approval_id"] == result.approval_id
        STATE["release"].set()
        await task.until(lambda v, p: v.status == "COMPLETED", "the next attempt", skip_s=10)
        assert TURNS[1]["profile"] == "stranger@9" and TURNS[1]["switched_from"] == "boss@1"


async def test_a_refused_switch_changes_nothing() -> None:
    async with environment() as env:
        task = await Task.start(env, "switch-refused", "flaky", config=TaskConfig(expert="boss@1", team=TEAM))
        await task.until(lambda v, p: len(TURNS) == 1, "the first attempt")
        result = await _switch(task, task.exploration, "stranger@9")
        await task.handle.execute_update(
            TaskWorkflow.decide_approval, DecideApprovalInput(command_id=task.command_id(), approval_id=result.approval_id, decision="reject")
        )
        assert _events("profile.switched") == []
        STATE["release"].set()
        await task.until(lambda v, p: v.status == "COMPLETED", "the next attempt", skip_s=10)
        assert TURNS[1]["profile"] == "boss@1" and TURNS[1]["switched_from"] is None
        assert TURNS[1]["continue_from"] == TURNS[0]["attempt_id"], "the session is carried on as before"


async def test_only_a_node_an_agent_runs_that_is_not_done_can_be_switched() -> None:
    async with environment() as env:
        task = await Task.start(env, "switch-invalid", "complete")
        await task.until(lambda v, p: v.status == "COMPLETED", "the exploration")
        assert await _rejected(_switch(task, task.exploration, "default@2")) == "FROZEN_NODE"
        approval = await task.change(AddNodeOp(node=ApprovalNode(node_id="tmp:1", title="ok?", spec=ApprovalSpec(summary="ship it?"))))
        assert await _rejected(_switch(task, approval.id_map["tmp:1"], "default@2")) == "NOT_ALLOWED"
        assert await _rejected(_switch(task, deterministic_id("ghost", "n"), "default@2")) == "SCHEMA_INVALID"


# ---- 6: a takeover stops the agents and a person can finish a node ---------------------------------------------------------


async def test_a_takeover_stops_the_running_attempt_without_holding_it_against_the_node_and_a_handback_goes_on() -> None:
    async with environment() as env:
        task = await Task.start(env, "takeover", "hold-once")
        await task.until(lambda v, p: len(TURNS) == 1, "the attempt")
        await task.control("takeover")
        view, plan = await task.until(lambda v, p: p.nodes[0].status == "RETRY_PENDING", "the attempt to be stopped")
        assert view.status == "TAKEN_OVER"
        assert [p["outcome"] for p in _events("attempt.finished")] == ["cancelled"]

        # Nothing is started for the agents while a person has the task, even a node that is ready.
        added = await task.change(_agent("tmp:1", "work"))
        await task.env.sleep(timedelta(seconds=60))
        await asyncio.sleep(0.3)
        plan = await task.plan()
        assert len(TURNS) == 1 and _node(plan, added.id_map["tmp:1"]).status == "READY"
        assert (await task.view()).status == "TAKEN_OVER"

        await task.control("handback")
        await task.until(lambda v, p: v.status == "COMPLETED", "the agents to go on")
        # After the handback the stopped node and the node added meanwhile start together: their turns reach TURNS in either order,
        # so the node's own turns are picked out by their goal, not by position.
        mine = [t for t in TURNS if t["goal"] == "hold-once"]
        assert [t["attempt_no"] for t in mine] == [1, 2]
        assert mine[1]["continue_from"] == mine[0]["attempt_id"], "it goes on from the session the takeover stopped"
        assert [p for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"] == [], "not a failure"


async def test_a_person_completes_a_node_by_hand_and_what_depends_on_it_goes_on() -> None:
    async with environment() as env:
        task = await Task.start(env, "complete-by-hand", "hold-once")
        await task.until(lambda v, p: len(TURNS) == 1, "the attempt")
        await task.control("takeover")
        await task.until(lambda v, p: p.nodes[0].status == "RETRY_PENDING", "the attempt to be stopped")
        added = await task.change(_agent("tmp:1", "work"), _agent("tmp:2", "work"))
        one, two = added.id_map["tmp:1"], added.id_map["tmp:2"]
        # tmp:2 waits for tmp:1.
        await task.change(AddEdgeOp.model_validate({"op": "add_edge", "from": one, "to": two}))
        plan_before = await task.plan()
        command = CompleteNodeInput(command_id=task.command_id(), node_id=one, reason="I wrote it myself")
        result = await task.handle.execute_update(TaskWorkflow.complete_node, command)
        assert (result.node_id, result.status) == (one, "COMPLETED")
        plan = await task.plan()
        assert (_node(plan, one).status, _node(plan, one).frozen) == ("COMPLETED", True)
        assert plan.plan_version == plan_before.plan_version, "a node's completion is not a change of the plan"
        assert len(TURNS) == 1, "no verification and no attempt: the person is the verdict"
        # The same command is answered again, not applied twice.
        again = await task.handle.execute_update(TaskWorkflow.complete_node, command)
        assert again == result
        # What waited for it is ready, and starts when the agents have the task back.
        assert _node(plan, two).status == "READY"
        await task.control("handback")
        await task.until(lambda v, p: v.status == "COMPLETED" and _node(p, two).status == "COMPLETED", "the next node")
        assert [t["goal"] for t in TURNS] == ["hold-once", "hold-once", "work"], "the node done by hand was never run"
        changed = [p for p in _events("node.status_changed") if p["node_id"] == one and p["to_status"] == "COMPLETED"]
        assert len(changed) == 1 and changed[0]["reason"] == "I wrote it myself"


async def test_completing_a_node_by_hand_has_its_conditions() -> None:
    async with environment() as env:
        task = await Task.start(env, "complete-rules", "hold")
        await task.until(lambda v, p: len(TURNS) == 1, "the attempt")

        def complete(node_id: str):
            return task.handle.execute_update(
                TaskWorkflow.complete_node, CompleteNodeInput(command_id=task.command_id(), node_id=node_id)
            )

        assert await _rejected(complete(task.exploration)) == "INVALID_TRANSITION", "the agents are working: pause or take over first"
        await task.control("pause")  # a pause leaves the running attempt alone
        assert await _rejected(complete(task.exploration)) == "INVALID_TRANSITION", "the node is running"
        assert await _rejected(complete(deterministic_id("ghost", "n"))) == "SCHEMA_INVALID"
        await task.control("takeover")
        await task.until(lambda v, p: p.nodes[0].status == "RETRY_PENDING", "the attempt to be stopped")
        result = await complete(task.exploration)
        assert result.status == "COMPLETED"
        assert await _rejected(complete(task.exploration)) == "FROZEN_NODE", "it is complete now"


async def test_a_handback_is_only_for_a_task_that_was_taken_over() -> None:
    async with environment() as env:
        task = await Task.start(env, "handback", "hold")
        await task.until(lambda v, p: len(TURNS) == 1, "the attempt")
        await task.control("pause")
        assert await _rejected(task.control("handback")) == "INVALID_TRANSITION"
        assert (await task.view()).status == "PAUSED", "a handback does not un-pause what a person paused"


# ---- events carry what a projection needs --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_carry_the_facts_a_projection_builds_its_rows_from() -> None:
    async with environment() as env:
        task = await Task.start(env, "event-facts", "complete")
        await task.until(lambda v, p: v.status == "COMPLETED", "the exploration")
        added = await task.change(_agent("tmp:1", "work", after=[task.exploration]))
        node = added.id_map["tmp:1"]
        await task.until(lambda v, p: _node(p, node).status == "COMPLETED", "the node")
        await _seen("attempt.finished", 2)

        running = next(p for p in _events("node.status_changed") if p["node_id"] == node and p["to_status"] == "RUNNING")
        assert (running["node_type"], running["title"], running["workspace_access"]) == ("agent_turn", "tmp:1", "none")
        assert running["depends_on"] == [task.exploration] and running["frozen"] is False
        assert running["attempt_count"] == 1 and running["current_attempt_id"] == task.attempt_of(node)
        assert running["owner_profile"] == "default@1"
        done = [p for p in _events("node.status_changed") if p["node_id"] == node][-1]
        assert done["frozen"] is True

        committed = next(p for p in _events("plan.version_committed") if p["plan_version"] == 2)
        assert committed["change_command_id"] == committed["command_id"], "both names, for old consumers"
        assert committed["parent_version"] == 1 and committed["actor"] == {"kind": "user", "id": "user-a"}

        finished = next(p for p in _events("attempt.finished") if p["node_id"] == node)
        assert (finished["attempt_no"], finished["profile"], finished["config_version"]) == (1, "default@1", 1)


@pytest.mark.asyncio
async def test_node_events_carry_the_nodes_own_entity_and_version() -> None:
    async with environment() as env:
        task = await Task.start(env, "node-entity", "complete")
        await task.until(lambda v, p: v.status == "COMPLETED", "the exploration")
        added = await task.change(_agent("tmp:1", "work", after=[task.exploration]))
        node = added.id_map["tmp:1"]
        await task.until(lambda v, p: _node(p, node).status == "COMPLETED", "the node")
        await _seen("node.status_changed", 6)
        mine = [e for e in EVENTS if e["type"] == "node.status_changed" and e["payload"]["node_id"] == node]
        assert len(mine) >= 3
        assert {(e["entity"]["kind"], e["entity"]["id"]) for e in mine} == {("node", node)}
        versions = [e["entity"]["version"] for e in mine]
        assert versions == sorted(versions) and len(set(versions)) == len(versions), "its own counter, going up by one"
        assert versions == list(range(versions[0], versions[0] + len(versions)))
        other = [e for e in EVENTS if e["type"] == "node.status_changed" and e["payload"]["node_id"] == task.exploration]
        assert {e["entity"]["id"] for e in other} == {task.exploration}, "another node has a counter of its own"
        # The task's counter is not moved by nodes: its events are numbered one after the other.
        task_versions = [e["entity"]["version"] for e in EVENTS if e["entity"]["kind"] == "task"]
        assert task_versions == list(range(1, len(task_versions) + 1))

"""The leader's review of what the tasks it created produced (05 §7, `orbit_orch.task_review`).

How it can go wrong, written down before the code:
  - a leader that created tasks never hears what they produced, so a plan can only be one round long;
  - the review is added while a task is still going, or never when they are all done, or for a leader that created nothing;
  - a task that is blocked is left out and the review judges the rest: it must wait until a person resolves the task;
  - the review starts from nothing instead of carrying on the leader's own session, or runs as another agent;
  - its prompt leaves out a task's result, or grows with the number of tasks and the length of what they said;
  - the reviews of a plan that keeps growing never stop, or stop without a word;
  - a Continue-As-New in the middle forgets a review that was owed, or what a finished task had said.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from typing import Any

from orbit_contracts.v3 import (
    Actor,
    Budget,
    CompleteNodeInput,
    GrantBudgetInput,
    PlanChangeCommand,
    Policy,
    TaskControlInput,
    TaskWorkflowInput,
)
from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec
from orbit_contracts.v3.plan import AddNodeOp
from orbit_orch.plan_engine import deterministic_id
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.service import RPCError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from waiting import polls

TURNS: list[dict[str, Any]] = []
EVENTS: list[dict[str, Any]] = []
WORKFLOWS: dict[str, str] = {}
# What a leader's turn creates: the tasks, by the start of its goal. `explore` is the task's goal.
PLANS: dict[str, list[str]] = {}
RELEASE = asyncio.Event()
FAST = {
    "max_heartbeat_throttle_interval": timedelta(milliseconds=100),
    "default_heartbeat_throttle_interval": timedelta(milliseconds=100),
}
REVIEW = "领队复盘"


def _ref(char: str) -> str:
    return "sha256:" + char * 64


@activity.defn(name="agent_turn")
async def _agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """A turn that creates the tasks `PLANS` says for its goal (as an agent, through the plan), then completes saying what it
    did and leaving a file named after its goal. A goal starting with `hold` waits for the test; `fail` fails for good."""
    TURNS.append(payload)
    goal = str(payload["goal"])
    if goal.startswith("hold"):
        while not RELEASE.is_set():
            activity.heartbeat()
            await asyncio.sleep(0.02)
    if goal.startswith("fail"):
        return {"status": "failed", "error": "it cannot be done", "failure_class": "policy", "retryable": False}
    titles = next((items for prefix, items in PLANS.items() if goal.startswith(prefix)), [])
    if titles:
        handle = activity.client().get_workflow_handle(WORKFLOWS[payload["task_id"]])
        actor = Actor(kind="agent", id=payload["attempt_id"], attempt_id=payload["attempt_id"])
        plan = await handle.query(TaskWorkflow.get_plan, actor)
        result = await handle.execute_update(
            TaskWorkflow.submit_plan_change,
            PlanChangeCommand(
                command_id=hashlib.sha256(f"{payload['attempt_id']}|create".encode()).hexdigest(),
                task_id=payload["task_id"], base_plan_version=plan.plan_version, actor=actor,
                ops=[
                    AddNodeOp(node=AgentTurnNode(
                        node_id=f"tmp:{index}", title=title, depends_on=[payload["node_id"]], spec=AgentTurnSpec(goal=title)
                    ))
                    for index, title in enumerate(titles, 1)
                ],
            ),
        )
        assert result.status == "accepted", result
    return {
        "status": "completed", "checkpoint_ref": _ref("1"), "session_id": "s", "state_version": 2,
        "handover_summary": f"did: {goal[:60]}",
        "manifest_id": deterministic_id(f"man:{payload['attempt_id']}", "man"),
        "manifest_entries": [{"name": f"{goal[:20]}.md", "media_type": "text/markdown", "size_bytes": 3, "blob_ref": _ref("5")}],
    }


@activity.defn(name="verify_completion")
async def _verify_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "failures": [], "workspace_snapshot_ref": _ref("7")}


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
        Worker(env.client, task_queue="orbit.agent", activities=[_agent_turn], **FAST),
        Worker(
            env.client, task_queue="orbit.io",
            activities=[_verify_completion, _publish_events, _checkpoint_commit, _commit_checkpoints],
        ),
    )


def _reset() -> None:
    TURNS.clear()
    EVENTS.clear()
    WORKFLOWS.clear()
    PLANS.clear()
    RELEASE.clear()


def _events(kind: str) -> list[dict[str, Any]]:
    return [event["payload"] for event in EVENTS if event["type"] == kind]


async def _query(handle, query, *args):
    for _ in range(20):
        try:
            return await handle.query(query, *args, rpc_timeout=timedelta(seconds=3))
        except RPCError:
            continue
    raise AssertionError("the query was never answered")


async def _until(handle, ready, what: str):
    view = plan = None
    async for _ in polls():
        view = await _query(handle, TaskWorkflow.get_task_view)
        plan = await _query(handle, TaskWorkflow.get_plan)
        if ready(view, plan):
            return view, plan
    assert view is not None and plan is not None
    raise AssertionError(f"timed out waiting for {what}: {view.status} {[(n.title, n.status) for n in plan.nodes]} {len(TURNS)} turns")


async def _start(env: WorkflowEnvironment, name: str, *, policy: Policy | None = None, goal: str = "explore", profile: str = "default@1"):
    task_id = deterministic_id(f"review:{name}", "task")
    inp = TaskWorkflowInput(
        task_id=task_id, tenant_id="tenant-a", created_by=Actor(kind="user", id="user-a"), title="Review", goal=goal,
        profile=profile, node_type_registry_version=1, budgets=Budget(), **({"policy": policy} if policy else {}),
    )
    handle = await env.client.start_workflow(TaskWorkflow.run, inp, id=f"task/tenant-a/review-{name}", task_queue="orbit.orch")
    WORKFLOWS[task_id] = handle.id
    return handle


async def _finish(handle) -> None:
    await handle.execute_update(
        TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"cancel").hexdigest(), action="cancel")
    )
    async for _ in polls():
        if (await handle.describe()).status != WorkflowExecutionStatus.RUNNING:
            return
    raise AssertionError("the workflow did not end")


def _reviews(plan):
    return [node for node in plan.nodes if node.title == REVIEW]


def _turn_of(goal_part: str) -> dict[str, Any]:
    return next(turn for turn in TURNS if goal_part in str(turn["goal"]))


async def test_a_leader_that_created_tasks_reviews_what_they_produced_and_then_finishes() -> None:
    _reset()
    PLANS["explore"] = ["alpha", "beta"]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "one-round")
            _, plan = await _until(handle, lambda v, p: v.status == "COMPLETED" and len(_reviews(p)) == 1, "the review to finish")
            await _finish(handle)
    exploration = TURNS[0]
    review = _turn_of("第 1 轮")
    node = _reviews(plan)[0]
    assert [t["goal"] for t in TURNS[:3]] == ["explore", "alpha", "beta"], "the tasks run before the review"
    assert node.status == "COMPLETED" and len(node.depends_on) == 2, "it depends on every task the leader created"
    assert review["goal"].count("\n") > 3
    assert "alpha [completed]" in review["goal"] and "did: alpha" in review["goal"] and "alpha.md" in review["goal"]
    assert "beta [completed]" in review["goal"] and "did: beta" in review["goal"], "what each task reported is in the prompt"
    assert review["continue_from"] == exploration["attempt_id"], "it carries on the leader's own session"
    assert review["profile"] == exploration["profile"] == "default@1", "and runs as the leader's expert"
    assert len(TURNS) == 4, "a review that creates nothing ends the plan"
    assert node.review_round == 1 and plan.nodes[0].review_round is None, "a review node says its round, no other does"
    assert [p["review_round"] for p in _events("node.status_changed") if p["node_id"] == node.node_id] == [1] * 4, "every event of the node, from its first"
    commits = [p for p in _events("plan.version_committed") if p.get("reason") == "leader review"]
    assert len(commits) == 1 and commits[0]["actor"] == {"kind": "system", "id": "task-workflow"}
    assert len(_events("task.completed")) == 1, "and the task completed once, after the review"


async def test_a_leader_that_created_nothing_gets_no_review() -> None:
    _reset()
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "none")
            _, plan = await _until(handle, lambda v, p: v.status == "COMPLETED", "the task")
            await _finish(handle)
    assert [t["goal"] for t in TURNS] == ["explore"] and not _reviews(plan)


async def test_a_review_that_plans_more_is_followed_by_a_review_of_that_round() -> None:
    _reset()
    PLANS["explore"] = ["alpha"]
    PLANS[f"{REVIEW}（第 1 轮）"] = ["gamma"]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "two-rounds")
            _, _plan = await _until(handle, lambda v, p: v.status == "COMPLETED" and len(_reviews(p)) == 2, "the second review")
            await _finish(handle)
    first, second = _turn_of("第 1 轮"), _turn_of("第 2 轮")
    exploration = TURNS[0]
    assert "gamma [completed]" in second["goal"] and "alpha" not in second["goal"], "the second review is about the second round"
    assert first["continue_from"] == exploration["attempt_id"]
    assert second["continue_from"] == first["attempt_id"], "each review carries on the one before it"
    assert [t["goal"].split("（")[0] for t in TURNS] == ["explore", "alpha", REVIEW, "gamma", REVIEW]


async def test_the_chain_of_reviews_stops_at_the_limit_and_says_so() -> None:
    _reset()
    PLANS["explore"] = ["alpha"]
    PLANS[REVIEW] = ["more"]  # every review plans another task: left alone it would never end
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "limit", policy=Policy(max_review_rounds=2))
            _, _plan = await _until(handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the task to ask for a review")
            plan = _plan
            turns_at_limit = len(TURNS)
            await handle.execute_update(
                TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"resume").hexdigest(), action="resume")
            )
            await _until(handle, lambda v, p: v.status == "COMPLETED", "the task to complete once a person looked")
            await _finish(handle)
    assert len(_reviews(plan)) == 2, "two reviews, no third"
    assert turns_at_limit == len(TURNS) == 6, "explore, alpha, review 1, more, review 2, more"
    limit = _events("plan.review_limit_reached")
    assert len(limit) == 1 and (limit[0]["round"], limit[0]["max_rounds"], limit[0]["children"]) == (3, 2, 1)
    reasons = [p["reason"] for p in _events("task.status_changed") if p["to_status"] == "PAUSED_NEEDS_REVIEW"]
    assert len(reasons) == 1 and "limit of 2 rounds" in reasons[0]


async def test_the_default_limit_is_five_rounds_and_zero_turns_the_reviews_off() -> None:
    _reset()
    PLANS["explore"] = ["alpha"]
    PLANS[REVIEW] = ["more"]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "default-limit")
            await _until(handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the default limit")
            assert len(_reviews(await _query(handle, TaskWorkflow.get_plan))) == 5
            await _finish(handle)
            TURNS.clear()
            off = await _start(env, "off", policy=Policy(max_review_rounds=0))
            _, plan = await _until(off, lambda v, p: v.status == "COMPLETED", "the task without reviews")
            await _finish(off)
    assert not _reviews(plan) and [t["goal"] for t in TURNS] == ["explore", "alpha"]
    assert not [p for p in _events("plan.review_limit_reached") if p["max_rounds"] == 0], "turned off is not a limit that was reached"


async def test_a_blocked_task_keeps_the_review_waiting_until_a_person_resolves_it() -> None:
    _reset()
    PLANS["explore"] = ["alpha", "fail-beta"]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "blocked")
            _, plan = await _until(handle, lambda v, p: v.status == "PAUSED_NEEDS_REVIEW", "the blocked task")
            assert not _reviews(plan), "one task is blocked: the review does not judge the rest"
            blocked = next(node for node in plan.nodes if node.title == "fail-beta")
            assert blocked.status == "BLOCKED"
            await handle.execute_update(
                TaskWorkflow.complete_node, CompleteNodeInput(command_id=hashlib.sha256(b"done").hexdigest(), node_id=blocked.node_id)
            )
            await handle.execute_update(
                TaskWorkflow.control, TaskControlInput(command_id=hashlib.sha256(b"resume").hexdigest(), action="resume")
            )
            _, plan = await _until(handle, lambda v, p: v.status == "COMPLETED" and len(_reviews(p)) == 1, "the review")
            await _finish(handle)
    review = _turn_of("第 1 轮")
    assert "alpha [completed]" in review["goal"] and "did: alpha" in review["goal"]
    assert "fail-beta [completed]" in review["goal"] and "(it reported nothing)" in review["goal"], "the person's word, no report"


async def test_the_prompt_is_bounded_whatever_the_tasks_said() -> None:
    _reset()
    titles = [f"task-{index:02d}" for index in range(30)]
    PLANS["explore"] = titles
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "bounded")
            await _until(handle, lambda v, p: v.status == "COMPLETED" and len(_reviews(p)) == 1, "the review")
            await _finish(handle)
    goal = _turn_of("第 1 轮")["goal"]
    assert len(goal) < 12000 + 30 * 400, "the prompt does not grow with what the tasks said"
    assert all(f"{title} [completed]" in goal for title in titles)


async def test_a_continue_as_new_in_the_middle_keeps_the_review_that_is_owed() -> None:
    _reset()
    PLANS["explore"] = ["alpha", "hold-beta"]
    async with await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter) as env:
        orch, agent, io = _stack(env)
        async with orch, agent, io:
            handle = await _start(env, "can")
            await _until(
                handle,
                lambda v, p: any(n.title == "alpha" and n.status == "COMPLETED" for n in p.nodes)
                and any(n.title == "hold-beta" and n.status == "RUNNING" for n in p.nodes),
                "alpha done and beta running",
            )
            first_run = (await handle.describe()).run_id
            for index in range(1010):
                await handle.execute_update(
                    TaskWorkflow.grant_budget, GrantBudgetInput(command_id=f"01JB{index:022d}", delta=Budget(tokens=1))
                )
            async for _ in polls():
                if (await handle.describe()).run_id != first_run:
                    break
            assert (await handle.describe()).run_id != first_run, "the workflow continued as new"
            RELEASE.set()
            _, _plan = await _until(handle, lambda v, p: v.status == "COMPLETED" and any(n.title == REVIEW and n.status == "COMPLETED" for n in p.nodes), "the review after the continue-as-new")
            await _finish(handle)
    review = _turn_of("第 1 轮")
    assert "alpha [completed]" in review["goal"] and "did: alpha" in review["goal"], "what a finished task said survives the continue-as-new"
    assert "hold-beta [completed]" in review["goal"]
    assert review["continue_from"] == TURNS[0]["attempt_id"]


def test_a_review_flags_a_finished_task_that_handed_over_nothing_usable_and_says_to_ask_again() -> None:
    from orbit_orch.task_review import review_goal

    def child(title: str, summary: str, missing: str = "") -> dict[str, Any]:
        return {"title": title, "status": "completed", "summary": summary, "artifacts": [], "missing": missing}

    goal = review_goal(
        1, 5, [child("api", "## Handover\n### Result\nbuilt"), child("docs", "completed", "writer"), child("ui", "x" * 200)], 0
    )
    assert goal.count("HANDOVER MISSING") == 2, "the flag on the task, and the instruction that names it"
    assert "HANDOVER MISSING (owner id: writer)" in goal and "same owner" in goal and "counts toward the same rounds" in goal
    assert "is a claim" in goal and "unverified" in goal
    quiet = review_goal(1, 5, [child("api", "## Handover\n### Result\nbuilt")], 0)
    assert "HANDOVER MISSING" not in quiet and "is a claim" in quiet


def test_only_a_reply_with_no_block_and_little_else_is_a_missing_handover() -> None:
    from orbit_orch.task_review import MIN_REPORT_CHARS, handover_missing

    assert handover_missing("") and handover_missing("completed") and handover_missing("  ok  ")
    assert not handover_missing("## Handover\n### Result\nok"), "a block, however short"
    assert not handover_missing("Done.\n\n## Handover\n### Result\nok"), "the block may be anywhere in the summary"
    assert not handover_missing("x" * MIN_REPORT_CHARS), "a free-form reply with something in it is the fallback"


def test_a_reply_that_fits_is_kept_as_it_is_and_one_that_does_not_keeps_its_start_and_its_block() -> None:
    from orbit_orch.handover import ELLIPSIS, fit_handover

    reply = "Answer.\n\n## Handover\n### Result\nbuilt the API\n### Files\n/work/api.py"
    assert fit_handover(reply, 2000) == reply, "block included, nothing rewritten"
    assert fit_handover("# handover\nx", 2000) == "# handover\nx"
    # Too long: the answer's opening, a marker where it was cut, then the block.
    long = "A" * 3000 + "\n\n" + reply
    fitted = fit_handover(long, 2000)
    assert len(fitted) <= 2000 and fitted.startswith("AAAA") and fitted.endswith(reply.split("\n\n", 1)[1])
    assert f"A{ELLIPSIS}\n\n## Handover\n### Result" in fitted
    # No block: the first characters, as before. An empty block is no block.
    assert fit_handover("x" * 3000, 2000) == "x" * 2000
    assert fit_handover("y" * 3000 + "\n## Handover\n \n", 2000) == ("y" * 3000 + "\n## Handover\n \n")[:2000]


def test_a_long_handover_block_keeps_every_section_and_gives_up_at_most_a_quarter_of_the_room_for_the_answer() -> None:
    from orbit_orch.handover import fit_handover

    block = (
        "## Handover\n### Result\nbuilt\n### Evidence\n" + "log line\n" * 600
        + "### Limits and open issues\nthe migration is not run\n### Needs from the leader\na decision on the schema"
    )
    fitted = fit_handover("Short answer.\n\n" + block, 2000)
    assert len(fitted) <= 2000 and fitted.startswith("Short answer.")
    assert "the migration is not run" in fitted and "a decision on the schema" in fitted, "a long Evidence pushes nothing out"
    only_block = fit_handover(block, 2000)
    assert only_block.startswith("## Handover\n### Result\nbuilt") and len(only_block) <= 2000
    assert "a decision on the schema" in only_block
    huge = fit_handover("B" * 5000 + "\n\n" + block, 2000)
    assert len(huge) <= 2000 and huge.startswith("B" * 400), "the answer keeps a quarter of the room at least"
    assert "a decision on the schema" in huge


def test_a_person_is_shown_the_reply_above_the_block_or_the_blocks_content_when_that_is_all() -> None:
    from orbit_orch.handover import without_handover_block

    assert without_handover_block("The totals add up.\n\n## Handover\n### Result\nchecked") == "The totals add up."
    assert without_handover_block("## Handover\n### Result\nchecked") == "### Result\nchecked"
    assert without_handover_block("Just an answer.") == "Just an answer."
    assert without_handover_block("Answer.\n## Handover\n  ") == "Answer.\n## Handover\n  ", "an empty block is not one"


def test_the_review_and_the_relay_keep_the_block_when_a_summary_is_cut_to_its_share() -> None:
    from orbit_orch.task_review import relay_goal, review_goal

    summary = "C" * 5000 + "\n\n## Handover\n### Result\nbuilt\n### Limits and open issues\nnot deployed"
    children = [{"title": f"t{i}", "status": "completed", "summary": summary, "artifacts": []} for i in range(10)]
    for goal in (review_goal(1, 5, children, 0), relay_goal("how?", children)):
        assert goal.count("not deployed") == 10 and goal.count("## Handover") == 10, "every task keeps its block"
        assert "C" * 100 in goal and len(goal) < 12000 + 10 * 600


def test_the_relay_treats_what_a_member_said_as_a_claim() -> None:
    from orbit_orch.task_review import relay_goal

    goal = relay_goal("how did it go?", [{"title": "api", "status": "completed", "summary": "all done", "artifacts": []}])
    assert "only a claim" in goal and "unverified" in goal and "HANDOVER MISSING" not in goal

"""The budget an attempt may spend is enforced where the agent works (05 §4).

How it can go wrong, written down before the code:
  - the attempt spends what the whole task has left, not what was reserved for it, so attempts that run side by side
    overspend;
  - a turn is stopped in the middle of a batch of tool calls: a call has no result, the saved state is not whole (08 §3, A27)
    and the model API refuses what is formatted from it;
  - a call that is refused is recorded in the ledger as started, or checkpointed, as if it had run;
  - a spent budget fails the attempt as retryable, so the node is tried again and spends the same again;
  - a cost is enforced without a price (zero looks like "free"), or recorded as 0 when it is unknown;
  - what the activity spent is not reported, so the parent cannot settle the reservation by it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from agentscope.message import ToolResultBlock, ToolResultState
from agentscope.state import AgentState
from orbit_contracts.models import OpenSessionInput, RunTurnInput, TurnResult
from orbit_contracts.v3 import Budget
from orbit_worker import task_activities
from orbit_worker.agent_config import agent_config_from_spec
from orbit_worker.budget_middleware import BudgetMeter, ModelPrice, model_price
from orbit_worker.chat_model import resolve_model_config
from orbit_worker.runtime import AgentRuntime
from orbit_worker.runtime_holder import set_runtime
from orbit_worker.store import MemoryStateStore
from orbit_worker.task_stream import TaskStreamContext, streaming_for
from temporalio.testing import ActivityEnvironment
from test_interrupted_session import _payload, _Store

FOUR_STEPS = "chain:slow:a;;slow:b;;slow:c;;slow:d"


def _context(meter: BudgetMeter, attempt_id: str = "att-1") -> TaskStreamContext:
    return TaskStreamContext(tenant_id="t", task_id="task-1", attempt_id=attempt_id, activity_attempt=1, meter=meter)


async def _run(runtime: AgentRuntime, meter: BudgetMeter, text: str) -> TurnResult:
    with streaming_for(_context(meter)):
        opened = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="att-1:open"))
        return await runtime.run_turn(
            RunTurnInput(
                room_id="task-1", session_id=opened.session_id, turn_id="att-1:turn",
                message=text, state_version=opened.state_version,
            )
        )


def _results(state: dict[str, Any]) -> list[ToolResultBlock]:
    agent_state = AgentState.model_validate(state)
    return [b for m in agent_state.context if not isinstance(m.content, str) for b in m.content if isinstance(b, ToolResultBlock)]


def _calls(state: dict[str, Any]) -> list[str]:
    agent_state = AgentState.model_validate(state)
    return [b.id for m in agent_state.context if not isinstance(m.content, str) for b in m.content if b.type == "tool_call"]


async def test_spent_tokens_stop_the_turn_between_steps_and_leave_the_state_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "10")  # a model call reports 10 tokens in and 10 out
    runtime = AgentRuntime(MemoryStateStore())
    meter = BudgetMeter(limit=Budget(tokens=50))
    result = await _run(runtime, meter, FOUR_STEPS)

    assert (result.status, result.error_code, result.retryable) == ("failed", "budget", False)
    assert "token budget is spent" in result.error
    usage = meter.usage()
    # The third model call took it to 60 of 50: the call it asked for was refused and the fourth model call never happened.
    assert (usage.tokens_in, usage.tokens_out, usage.tool_calls) == (30, 30, 2)
    blob = await runtime._store.get("att-1")
    assert blob is not None
    results = _results(blob.agent_state)
    assert sorted(_calls(blob.agent_state)) == sorted(r.id for r in results), "every call has a result: the state is whole"
    assert [r.state for r in results] == [ToolResultState.SUCCESS, ToolResultState.SUCCESS, ToolResultState.ERROR]
    assert blob.state_version == 2, "the stopped turn's state is kept, for the attempt that goes on after a grant"


async def test_a_spent_tool_call_budget_refuses_the_call_with_a_result_and_ends_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "1")
    runtime = AgentRuntime(MemoryStateStore())
    meter = BudgetMeter(limit=Budget(tool_calls=1))
    result = await _run(runtime, meter, FOUR_STEPS)

    assert (result.status, result.error_code) == ("failed", "budget")
    assert "tool call budget is spent" in result.error
    assert meter.usage().tool_calls == 1, "only the call inside the budget ran"
    blob = await runtime._store.get("att-1")
    assert blob is not None
    results = _results(blob.agent_state)
    assert [r.state for r in results] == [ToolResultState.SUCCESS, ToolResultState.ERROR]
    assert "was not run" in str(results[1].output)


async def test_calls_of_one_batch_that_start_together_cannot_all_take_the_last_slot() -> None:
    """Two calls enter `on_acting` at once with room for one: the second is refused with a result, and only one runs."""
    from types import SimpleNamespace

    from agentscope.tool import ToolResponse
    from orbit_worker.budget_middleware import OrbitBudgetMiddleware

    ran: list[str] = []
    meter = BudgetMeter(limit=Budget(tool_calls=1))

    async def run_call(name: str) -> list[Any]:
        async def next_handler(**kwargs: Any):
            ran.append(kwargs["tool_call"].id)
            await asyncio.sleep(0)
            yield ToolResponse(content=[], state=ToolResultState.SUCCESS)

        call = SimpleNamespace(id=name, name="Bash")
        return [
            item
            async for item in OrbitBudgetMiddleware().on_acting(
                None, {"tool_call": call}, next_handler
            )
        ]

    with streaming_for(_context(meter)):
        first, second = await asyncio.gather(run_call("c1"), run_call("c2"))
    assert ran == ["c1"], "the call past the budget did not run"
    assert [r.state for r in first] == [ToolResultState.SUCCESS]
    assert [r.state for r in second] == [ToolResultState.ERROR], "it has a result, so the batch is whole"
    assert meter.refused == "tool_calls"


async def test_a_turn_inside_its_budget_completes_and_reports_what_it_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "10")
    runtime = AgentRuntime(MemoryStateStore())
    meter = BudgetMeter(limit=Budget(tokens=10_000, tool_calls=10, wall_s=600))
    result = await _run(runtime, meter, "slow:a")
    assert result.status == "completed"
    usage = meter.usage()
    assert (usage.tokens_in, usage.tokens_out, usage.tool_calls) == (20, 20, 1)
    assert usage.cost_usd_micros is None, "no price: the cost is unknown, not zero"


async def test_no_budget_is_no_limit() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    meter = BudgetMeter()
    assert (await _run(runtime, meter, FOUR_STEPS)).status == "completed"
    assert meter.usage().tool_calls == 4


async def test_a_spent_wall_budget_stops_the_turn_before_the_first_model_call() -> None:
    now = [100.0]
    meter = BudgetMeter(limit=Budget(wall_s=5), clock=lambda: now[0])
    now[0] = 106.0  # the activity has been running for 6 seconds
    runtime = AgentRuntime(MemoryStateStore())
    result = await _run(runtime, meter, "hello")
    assert (result.status, result.error_code) == ("failed", "budget")
    assert "time budget is spent" in result.error
    assert meter.usage().wall_s == 6
    assert meter.usage().tokens_in == 0, "no model call was made"


async def test_a_known_price_makes_the_cost_known_and_a_cost_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "1000")
    runtime = AgentRuntime(MemoryStateStore())
    # 3 USD per million input tokens and 15 per million output: 3,000,000 and 15,000,000 micro-dollars.
    price = ModelPrice(input_per_mtok=3_000_000, output_per_mtok=15_000_000)
    meter = BudgetMeter(limit=Budget(cost_usd_micros=40_000), price=price)
    result = await _run(runtime, meter, FOUR_STEPS)
    # One call costs 1000 * 3 + 1000 * 15 = 18,000 micro-dollars: the third call takes it to 54,000 of 40,000.
    assert (result.status, result.error_code) == ("failed", "budget")
    assert "cost budget is spent" in result.error
    assert meter.usage().cost_usd_micros == 54_000


async def test_an_unknown_price_does_not_enforce_a_cost_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "1000")
    runtime = AgentRuntime(MemoryStateStore())
    meter = BudgetMeter(limit=Budget(cost_usd_micros=1))
    assert (await _run(runtime, meter, FOUR_STEPS)).status == "completed"
    assert meter.usage().cost_usd_micros is None


def test_cost_is_rounded_up_so_a_cheap_call_is_never_free() -> None:
    assert ModelPrice(1, 1).cost(1, 0) == 1
    assert ModelPrice(3_000_000, 15_000_000).cost(1000, 1000) == 18_000
    assert ModelPrice(0, 0).cost(1000, 1000) == 0


def test_a_price_is_known_only_when_input_and_output_are_both_set() -> None:
    assert model_price(1, 2) == ModelPrice(1, 2)
    assert model_price(1, None) is None and model_price(None, 2) is None and model_price(None, None) is None


def test_the_price_comes_from_the_environment_or_the_profile() -> None:
    config = resolve_model_config({"ORBIT_MODEL_PRICE_INPUT_PER_MTOK": "3000000", "ORBIT_MODEL_PRICE_OUTPUT_PER_MTOK": "15000000"})
    assert (config.price_input_per_mtok, config.price_output_per_mtok) == (3_000_000, 15_000_000)
    assert resolve_model_config({}).price_input_per_mtok is None
    profile = agent_config_from_spec({"model_params": {"price_input_per_mtok": 1, "price_output_per_mtok": 2}})
    assert (profile.price_input_per_mtok, profile.price_output_per_mtok) == (1, 2)
    bad = agent_config_from_spec({"model_params": {"price_input_per_mtok": -1, "price_output_per_mtok": "x"}})
    assert bad.price_input_per_mtok is None and bad.price_output_per_mtok is None


# ---- the activity: what it reads from the attempt, and what it reports -------------------------------------------------


async def test_the_activity_stops_at_its_reserved_budget_and_fails_the_attempt_as_budget_not_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "10")
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(AgentRuntime(MemoryStateStore()))
    outcome = await ActivityEnvironment().run(
        task_activities.agent_turn, _payload(goal=FOUR_STEPS, budget={"tokens": 50})
    )
    assert outcome["status"] == "failed"
    assert (outcome["failure_class"], outcome["retryable"]) == ("budget", False)
    assert outcome["usage"]["tokens_in"] == 30 and outcome["usage"]["tool_calls"] == 2
    assert "cost_usd_micros" not in outcome["usage"], "an unknown cost is left out, not reported as 0"


async def test_the_activity_reports_what_a_finished_turn_spent_and_a_summary_for_a_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "10")
    monkeypatch.setenv("ORBIT_MODEL_PRICE_INPUT_PER_MTOK", "1000000")
    monkeypatch.setenv("ORBIT_MODEL_PRICE_OUTPUT_PER_MTOK", "2000000")
    task_activities.set_task_store(_Store(tmp_path))
    runtime = AgentRuntime(MemoryStateStore(), model_config=resolve_model_config())
    set_runtime(runtime)
    outcome = await ActivityEnvironment().run(task_activities.agent_turn, _payload(goal="slow:a", budget={"tokens": 1000}))
    assert outcome["status"] == "completed"
    assert outcome["usage"]["tokens_in"] == 20 and outcome["usage"]["tool_calls"] == 1
    assert outcome["usage"]["cost_usd_micros"] == 60, "20 tokens in at 1 and 20 out at 2 micro-dollars per token"
    assert outcome["handover_summary"] == outcome["text"]


async def test_an_activity_without_a_budget_has_no_limit(tmp_path: Path) -> None:
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(AgentRuntime(MemoryStateStore()))
    outcome = await ActivityEnvironment().run(task_activities.agent_turn, _payload(goal=FOUR_STEPS))
    assert outcome["status"] == "completed" and outcome["usage"]["tool_calls"] == 4


# ---- a node switched to another profile is told where the work stands ---------------------------------------------------


def test_a_switched_attempt_is_given_the_handover_with_its_goal() -> None:
    prompt = task_activities.turn_prompt(_payload(handover="the report is half written"), "")
    assert prompt.startswith("do it") and "the report is half written" in prompt
    assert "handed over" in prompt


def test_an_attempt_without_a_handover_is_given_only_its_goal() -> None:
    assert task_activities.turn_prompt(_payload(), "") == "do it"


# ---- a retry of the activity carries on from what the run before it spent -------------------------------------------------


async def test_the_heartbeat_carries_what_was_spent_so_far(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "10")
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(AgentRuntime(MemoryStateStore()))
    beats: list[tuple[Any, ...]] = []
    env = ActivityEnvironment()
    env.on_heartbeat = lambda *details: beats.append(details)
    await env.run(task_activities.agent_turn, _payload(goal="chain:slow:a;;slow:b"))
    spent = [details[0] for details in beats if details and details[0]["tokens_in"]]
    assert spent, "what a model call used is in the heartbeat"
    assert max(item["tool_calls"] for item in spent) == 2 and max(item["tokens_in"] for item in spent) == 30


async def test_a_retried_activity_is_stopped_by_what_the_crashed_run_had_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker crashed after spending 80 of the attempt's 100 tokens. Temporal runs the activity again; counting from zero
    it would spend the reservation a second time."""
    import dataclasses

    monkeypatch.setenv("ORBIT_MOCK_TOKENS_PER_CALL", "10")
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(AgentRuntime(MemoryStateStore()))
    crashed = {"tokens_in": 40, "tokens_out": 40, "tool_calls": 3, "wall_s": 0, "cost_usd_micros": None, "refused": ""}
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, attempt=2, heartbeat_details=[crashed])
    outcome = await env.run(task_activities.agent_turn, _payload(goal=FOUR_STEPS, budget={"tokens": 100}))
    assert (outcome["status"], outcome["failure_class"], outcome["retryable"]) == ("failed", "budget", False)
    # One more model call (20 tokens) took it to 100 of 100; the call it asked for was refused.
    assert outcome["usage"]["tokens_in"] + outcome["usage"]["tokens_out"] == 100
    assert outcome["usage"]["tool_calls"] == 3, "the calls of the first run count too"

    # The same activity with no heartbeat to read (a first run) has the whole budget.
    fresh = await ActivityEnvironment().run(
        task_activities.agent_turn, _payload(attempt_id="att-2", goal=FOUR_STEPS, budget={"tokens": 100})
    )
    assert fresh["usage"]["tool_calls"] > 0 and fresh["usage"]["tokens_in"] + fresh["usage"]["tokens_out"] > 20


# ---- the first attempt after a switch is told what was already done ------------------------------------------------------


def test_a_switched_attempt_is_told_what_was_already_carried_out() -> None:
    prompt = task_activities.turn_prompt(
        _payload(handover="half done", side_effects=["Write [ab12cd34]: wrote /workspace/a.md", "Bash [ff00ff00]: ok"]), ""
    )
    assert "do not repeat them" in prompt
    assert "- Write [ab12cd34]: wrote /workspace/a.md" in prompt and "- Bash [ff00ff00]: ok" in prompt
    many = task_activities.turn_prompt(_payload(side_effects=[f"Bash [{n}]" for n in range(80)]), "")
    assert many.count("\n- ") == 50, "bounded"
    assert "do not repeat" not in task_activities.turn_prompt(_payload(), "")


async def test_a_resumed_event_says_which_activity_run_and_state_version_it_is(tmp_path: Path) -> None:
    import dataclasses

    store = _Store(tmp_path)
    task_activities.set_task_store(store)
    set_runtime(AgentRuntime(MemoryStateStore()))
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, attempt=2)
    await env.run(task_activities.agent_turn, _payload(goal="hello", state_version=0))
    resumed = [e for e in store.events if e["type"] == "attempt.resumed"]
    assert resumed and resumed[0]["payload"]["activity_attempt"] == 2 and resumed[0]["payload"]["state_version"] == 0

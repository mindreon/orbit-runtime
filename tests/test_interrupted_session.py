"""An attempt that is replaced carries on the session of the one before it, and what the worker tells the workflow.

How it can go wrong, written down before the code:
  - the replacement carries a session that still has a call waiting for an answer: under the new attempt id the call would
    run again, and the side-effect ledger (keyed by attempt and call id) would not know it had already run;
  - a cancelled turn saves nothing, so the replacement starts from the state before the turn and the user's message is gone;
  - a failed turn does not say whether trying again can help, so the workflow retries what cannot work or gives up on what can;
  - a retry is not told why the attempt before it was rejected;
  - the checkpoint a result names is a digest of the attempt id and not the checkpoint that was written (08 §3);
  - the context window is the library's default, so the agent compresses its context at the wrong size.
"""

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest
from agentscope.message import ToolCallBlock, ToolCallState, ToolResultBlock, ToolResultState
from agentscope.model import OpenAIChatModel
from agentscope.state import AgentState
from orbit_contracts.models import OpenSessionInput, OpenSessionOutput, RunTurnInput, TurnResult
from orbit_worker import task_activities
from orbit_worker.agent_config import AgentConfig, agent_config_from_spec
from orbit_worker.chat_model import (
    ModelConfig,
    ModelConfigError,
    RealChatModel,
    build_chat_model,
    resolve_model_config,
)
from orbit_worker.runtime import AgentRuntime, close_unfinished_tool_calls
from orbit_worker.runtime_holder import set_runtime
from orbit_worker.store import MemoryStateStore
from orbit_worker.task_store import TaskStore
from orbit_worker.task_stream import TaskStreamContext, streaming_for
from temporalio.activity import ActivityCancellationDetails
from temporalio.testing import ActivityEnvironment


def context(attempt_id: str) -> TaskStreamContext:
    return TaskStreamContext(tenant_id="t", task_id="task-1", attempt_id=attempt_id, activity_attempt=1)


async def open_session(runtime: AgentRuntime, attempt_id: str, *, continue_from: str = "") -> str:
    with streaming_for(context(attempt_id)):
        opened = await runtime.open_session(
            OpenSessionInput(room_id="task-1", turn_id=f"{attempt_id}:open", continue_from=continue_from)
        )
    return opened.session_id


async def turn(runtime: AgentRuntime, attempt_id: str, session_id: str, text: str) -> TurnResult:
    with streaming_for(context(attempt_id)):
        blob = await runtime._store.get(session_id)
        assert blob is not None
        return await runtime.run_turn(
            RunTurnInput(
                room_id="task-1", session_id=session_id, turn_id=f"{attempt_id}:{blob.state_version}",
                message=text, state_version=blob.state_version,
            )
        )


def open_calls(state: dict) -> list[str]:
    """Ids of the calls in a saved agent state that have no result."""
    agent_state = AgentState.model_validate(state)
    ids: list[str] = []
    for message in agent_state.context:
        if isinstance(message.content, str):
            continue
        answered = {block.id for block in message.content if isinstance(block, ToolResultBlock)}
        ids += [block.id for block in message.content if block.type == "tool_call" and block.id not in answered]
    return ids


async def test_a_call_parked_on_approval_is_closed_as_interrupted_for_the_replacement() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    first = await open_session(runtime, "att-1")
    parked = await turn(runtime, "att-1", first, "please gated this")
    assert parked.status == "needs_approval"
    old = await runtime._store.get("att-1")
    assert old is not None and open_calls(old.agent_state), "the session that was interrupted still waits on the call"

    second = await open_session(runtime, "att-2", continue_from="att-1")
    carried = await runtime._store.get(second)
    assert carried is not None
    assert open_calls(carried.agent_state) == []
    state = AgentState.model_validate(carried.agent_state)
    assert state.get_awaiting_tool_calls("orbit") == []
    results = [b for m in state.context if not isinstance(m.content, str) for b in m.content if isinstance(b, ToolResultBlock)]
    assert [r.state for r in results] == [ToolResultState.INTERRUPTED]
    # What the replacement is given goes on from there: no "agent is waiting for a confirmation" error.
    answer = await turn(runtime, "att-2", second, "history:")
    assert answer.status == "completed" and "gated" in (answer.text or "")
    # And the session it carried on is left as it was.
    again = await runtime._store.get("att-1")
    assert again is not None and open_calls(again.agent_state) == open_calls(old.agent_state)


def test_closing_unfinished_calls_ends_each_state_of_call_and_leaves_answered_ones_alone() -> None:
    from agentscope.message import AssistantMsg

    def call(call_id: str, state: ToolCallState) -> ToolCallBlock:
        return ToolCallBlock(id=call_id, name="Bash", input="{}", state=state)

    answered = ToolResultBlock(id="done", name="Bash", output="ok", state=ToolResultState.SUCCESS)
    state = AgentState(
        context=[
            AssistantMsg(
                name="orbit",
                content=[
                    call("done", ToolCallState.FINISHED), answered,
                    call("pending", ToolCallState.PENDING), call("asking", ToolCallState.ASKING),
                    call("allowed", ToolCallState.ALLOWED), call("submitted", ToolCallState.SUBMITTED),
                ],
            )
        ]
    )
    assert close_unfinished_tool_calls(state, "orbit") == 4
    blocks = state.context[0].content
    results = {b.id: b for b in blocks if isinstance(b, ToolResultBlock)}
    assert results["done"].state == ToolResultState.SUCCESS
    assert {i for i, r in results.items() if r.state == ToolResultState.INTERRUPTED} == {"pending", "asking", "allowed", "submitted"}
    assert all(b.state == ToolCallState.FINISHED for b in blocks if isinstance(b, ToolCallBlock))
    assert close_unfinished_tool_calls(state, "orbit") == 0, "closing again changes nothing"


async def test_a_cancelled_turn_is_saved_so_the_replacement_carries_on_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "tools.log"
    monkeypatch.setenv("ORBIT_MOCK_TOOL_LOG", str(log))
    monkeypatch.setenv("ORBIT_MOCK_TOOL_DELAY_MS", "30000")
    runtime = AgentRuntime(MemoryStateStore())
    first = await open_session(runtime, "att-1")
    before = await runtime._store.get(first)
    assert before is not None

    async def run() -> TurnResult:
        return await turn(runtime, "att-1", first, "slow:a")

    task = asyncio.create_task(run())
    for _ in range(500):
        if log.exists():
            break
        await asyncio.sleep(0.01)
    assert log.exists(), "the tool started"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    saved = await runtime._store.get(first)
    assert saved is not None and saved.state_version > before.state_version
    assert open_calls(saved.agent_state) == [], "the call that was running is closed, not left open"

    monkeypatch.setenv("ORBIT_MOCK_TOOL_DELAY_MS", "0")
    second = await open_session(runtime, "att-2", continue_from="att-1")
    heard = await turn(runtime, "att-2", second, "history:")
    assert "slow:a" in (heard.text or ""), "the message of the interrupted turn is in the carried session"


def test_a_call_whose_result_is_in_another_message_is_not_closed_again() -> None:
    """A reply that resumes after a confirmation writes its results into a new assistant message, not the one with the call."""
    from agentscope.message import AssistantMsg

    call = ToolCallBlock(id="c1", name="Bash", input="{}", state=ToolCallState.FINISHED)
    result = ToolResultBlock(id="c1", name="Bash", output="ok", state=ToolResultState.SUCCESS)
    state = AgentState(context=[AssistantMsg(name="orbit", content=[call]), AssistantMsg(name="orbit", content=[result])])
    assert close_unfinished_tool_calls(state, "orbit") == 0
    assert state.context[0].content == [call] and state.context[1].content == [result]


def test_a_truly_unfinished_call_is_closed_once_even_with_answered_calls_in_other_messages() -> None:
    from agentscope.message import AssistantMsg

    done = ToolCallBlock(id="c1", name="Bash", input="{}", state=ToolCallState.FINISHED)
    open_call = ToolCallBlock(id="c2", name="Bash", input="{}", state=ToolCallState.ASKING)
    state = AgentState(
        context=[
            AssistantMsg(name="orbit", content=[done]),
            AssistantMsg(name="orbit", content=[ToolResultBlock(id="c1", name="Bash", output="ok", state=ToolResultState.SUCCESS), open_call]),
        ]
    )
    assert close_unfinished_tool_calls(state, "orbit") == 1
    results = [b for m in state.context for b in m.content if isinstance(b, ToolResultBlock)]
    assert sorted((r.id, r.state) for r in results) == [("c1", ToolResultState.SUCCESS), ("c2", ToolResultState.INTERRUPTED)]
    assert open_call.state == ToolCallState.FINISHED


async def _cancel_a_slow_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, details) -> tuple[AgentRuntime, int, int]:
    """Run a turn inside an activity context, cancel it with `details`, and return the runtime and the version before/after."""
    monkeypatch.setenv("ORBIT_MOCK_TOOL_LOG", str(tmp_path / "tools.log"))
    monkeypatch.setenv("ORBIT_MOCK_TOOL_DELAY_MS", "30000")
    runtime = AgentRuntime(MemoryStateStore())
    session = await open_session(runtime, "att-1")
    before = await runtime._store.get(session)
    assert before is not None
    env = ActivityEnvironment()

    async def run() -> None:
        await turn(runtime, "att-1", session, "slow:a")

    task = asyncio.create_task(env.run(run))
    for _ in range(500):
        if (tmp_path / "tools.log").exists():
            break
        await asyncio.sleep(0.01)
    env.cancel(details)
    with pytest.raises(asyncio.CancelledError):
        await task
    after = await runtime._store.get(session)
    assert after is not None
    return runtime, before.state_version, after.state_version


async def test_a_cancel_the_workflow_asked_for_keeps_the_turn_and_bumps_the_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    details = ActivityCancellationDetails(cancel_requested=True)
    _, before, after = await _cancel_a_slow_turn(tmp_path, monkeypatch, details)
    assert after > before


@pytest.mark.parametrize("why", ["timed_out", "worker_shutdown", "paused", "reset", "not_found"])
async def test_a_cancel_from_a_timeout_or_shutdown_leaves_the_version_for_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    """Temporal retries the same activity with the version it was given; a bumped version would fail that retry."""
    details = ActivityCancellationDetails(**{"cancel_requested": why == "timed_out", why: True})
    runtime, before, after = await _cancel_a_slow_turn(tmp_path, monkeypatch, details)
    assert after == before
    monkeypatch.setenv("ORBIT_MOCK_TOOL_DELAY_MS", "0")
    again = await turn(runtime, "att-1", "att-1", "hello")  # the retry: same session, same version, runs
    assert again.status == "completed"


# ---- what an attempt tells the workflow ----------------------------------------------------------------------------


class _Store(TaskStore):
    """A TaskStore on a directory: events and checkpoint lookups are recorded."""

    def __init__(self, root: Path, ref: str | None = None) -> None:
        super().__init__(url="", root=str(root))
        self.events: list[dict[str, Any]] = []
        self.ref = ref
        self.manifests: list[dict[str, Any]] = []

    async def publish_events(self, events: list[dict[str, Any]]) -> None:
        self.events.extend(events)

    async def latest_checkpoint_ref(self, **kwargs: Any) -> str | None:
        assert kwargs["kind"] == "agent_state"
        return self.ref

    async def put_manifest(self, **kwargs: Any) -> None:
        self.manifests.append(kwargs)


class _FailingRuntime:
    """Stands in for AgentRuntime: its turn fails the way a real model's does."""

    def __init__(self, **failure: Any) -> None:
        self.failure = failure

    def model_config_for(self, config: Any) -> ModelConfig:
        return ModelConfig()

    async def open_session(self, inp: OpenSessionInput) -> OpenSessionOutput:
        return OpenSessionOutput(session_id="s", state_version=1)

    async def run_turn(self, inp: RunTurnInput) -> TurnResult:
        return TurnResult(status="failed", session_id="s", state_version=1, model_mode="mock", model_name="mock", **self.failure)


def _payload(**fields: Any) -> dict[str, Any]:
    return {"task_id": "task-1", "tenant_id": "t", "node_id": "n_1", "attempt_id": "att-1", "attempt_no": 1,
            "goal": "do it", "profile": "default@1", **fields}


@pytest.mark.parametrize(
    ("failure", "failure_class", "retryable"),
    [
        ({"error_code": "rate_limited", "retryable": True, "error": "slow down"}, "model", True),
        ({"error_code": "timeout", "retryable": True, "error": "too slow"}, "model", True),
        # The attempt spent what the task reserved for it: a retry would spend the same again (05 §4).
        ({"error_code": "budget", "retryable": False, "error": "the attempt's token budget is spent"}, "budget", False),
        ({"error_code": "auth", "retryable": False, "error": "bad key"}, "model", False),
        ({"error_code": "state_unreadable", "retryable": False, "error": "gone"}, "lost", False),
    ],
)
async def test_a_failed_turn_says_what_kind_of_failure_it_was_and_whether_to_retry(
    tmp_path: Path, failure: dict[str, Any], failure_class: str, retryable: bool
) -> None:
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(_FailingRuntime(**failure))  # type: ignore[arg-type]
    outcome = await ActivityEnvironment().run(task_activities.agent_turn, _payload())
    assert outcome["status"] == "failed"
    assert (outcome["failure_class"], outcome["retryable"]) == (failure_class, retryable)
    assert outcome["error"] == failure["error"]


async def test_a_retry_is_told_why_the_attempt_before_it_was_rejected(tmp_path: Path) -> None:
    store = _Store(tmp_path)
    task_activities.set_task_store(store)
    set_runtime(AgentRuntime(MemoryStateStore()))
    payload = _payload(goal="history:", retry_reason="completion verification failed: [command/command_failed] pytest exited 1")
    outcome = await ActivityEnvironment().run(task_activities.agent_turn, payload)
    assert outcome["status"] == "completed"
    assert "Your previous attempt was rejected: completion verification failed" in outcome["text"]
    assert "pytest exited 1" in outcome["text"]
    first = await ActivityEnvironment().run(task_activities.agent_turn, _payload(attempt_id="att-9", goal="history:"))
    assert "rejected" not in first["text"], "an attempt that is not a retry hears nothing of it"


async def test_the_result_names_the_checkpoint_that_was_written(tmp_path: Path) -> None:
    written = "sha256:" + "a" * 64
    store = _Store(tmp_path, ref=written)
    task_activities.set_task_store(store)
    set_runtime(AgentRuntime(MemoryStateStore()))
    outcome = await ActivityEnvironment().run(task_activities.agent_turn, _payload(goal="hello"))
    assert outcome["checkpoint_ref"] == written
    # A turn that wrote nothing durable (no database) still names a reference, as before.
    store.ref = None
    outcome = await ActivityEnvironment().run(task_activities.agent_turn, _payload(attempt_id="att-2", goal="hello"))
    assert outcome["checkpoint_ref"] == task_activities._ref("att-2")


# ---- the model's context window -----------------------------------------------------------------------------------

_ENV = {
    "ORBIT_MODEL_MODE": "real", "ORBIT_MODEL_BASE_URL": "https://example.invalid/v1",
    "ORBIT_MODEL_API_KEY": "sk-test-key-0123456789", "ORBIT_MODEL_NAME": "m",
}


def test_the_context_size_comes_from_the_environment_and_reaches_the_chat_model() -> None:
    config = resolve_model_config({**_ENV, "ORBIT_MODEL_CONTEXT_SIZE": "200000"})
    assert config.context_size == 200000
    model = build_chat_model(config, _ENV)
    assert isinstance(model, RealChatModel) and model.context_size == 200000
    # Unset: the chat model keeps its own default.
    unset = resolve_model_config(_ENV)
    assert unset.context_size is None
    default = inspect.signature(OpenAIChatModel.__init__).parameters["context_size"].default
    assert build_chat_model(unset, _ENV).context_size == default


@pytest.mark.parametrize("value", ["lots", "0", "-5", "100"])
def test_a_context_size_that_is_not_a_window_is_refused(value: str) -> None:
    with pytest.raises(ModelConfigError) as caught:
        resolve_model_config({**_ENV, "ORBIT_MODEL_CONTEXT_SIZE": value})
    assert "ORBIT_MODEL_CONTEXT_SIZE" in str(caught.value)


def test_a_profile_can_override_the_context_size() -> None:
    assert agent_config_from_spec({"model_params": {"context_size": 65536}}).context_size == 65536
    for bad in ({"context_size": "big"}, {"context_size": 12}, {"context_size": True}, "nope", None):
        assert agent_config_from_spec({"model_params": bad}).context_size is None
    runtime = AgentRuntime(MemoryStateStore(), model_config=ModelConfig(mode="real", name="base", context_size=32768))
    assert runtime.model_config_for(AgentConfig(context_size=65536)).context_size == 65536
    assert runtime.model_config_for(AgentConfig()).context_size == 32768
    mock = AgentRuntime(MemoryStateStore(), model_config=ModelConfig())
    assert mock.model_config_for(AgentConfig(context_size=65536)).context_size is None


# ---- what a carried attempt is told -----------------------------------------------------------------------------------


def _prompt(**fields: Any) -> str:
    payload = {"goal": "echo:once", "attempt_no": 1, **fields}
    messages = "\n".join(m["text"] for m in payload.get("messages", []))
    return task_activities.turn_prompt(payload, messages)


def test_a_fresh_node_is_given_its_goal_and_the_messages() -> None:
    assert _prompt() == "echo:once"
    assert _prompt(messages=[{"text": "one"}, {"text": "two"}]) == "echo:once\n\nUser messages:\none\ntwo"


def test_a_follow_up_is_given_its_goal_the_message_it_was_made_from() -> None:
    # A follow-up carries on a session too, but it is the node's first attempt: its goal is the user's message.
    assert _prompt(goal="and then?", continue_from="att-0") == "and then?"


def test_a_replacement_attempt_gets_only_the_new_messages_never_the_goal() -> None:
    prompt = _prompt(attempt_no=2, continue_from="att-1", messages=[{"text": "history:"}])
    assert prompt == "history:"
    assert "echo:once" not in prompt


def test_a_retry_gets_the_reason_and_the_new_messages_never_the_goal() -> None:
    prompt = _prompt(attempt_no=2, continue_from="att-1", retry_reason="pytest exited 1", messages=[{"text": "mind the tests"}])
    assert prompt.startswith("mind the tests\n\nYour previous attempt was rejected: pytest exited 1")
    assert "echo:once" not in prompt
    only_reason = _prompt(attempt_no=3, continue_from="att-2", retry_reason="pytest exited 1")
    assert only_reason.startswith("Your previous attempt was rejected: pytest exited 1") and "echo:once" not in only_reason


def test_a_resumed_attempt_with_nothing_new_is_asked_to_go_on() -> None:
    assert _prompt(attempt_no=2, continue_from="att-1") == task_activities.CONTINUE_PROMPT == "Continue the task from where you stopped."


async def test_the_agent_of_a_carried_attempt_does_not_hear_the_goal_again(tmp_path: Path) -> None:
    store = _Store(tmp_path)
    task_activities.set_task_store(store)
    set_runtime(AgentRuntime(MemoryStateStore()))
    run = ActivityEnvironment().run
    await run(task_activities.agent_turn, _payload(attempt_id="att-1", goal="first goal"))
    second = await run(
        task_activities.agent_turn,
        _payload(attempt_id="att-2", attempt_no=2, goal="first goal", continue_from="att-1", messages=[{"text": "history:"}]),
    )
    heard = second["text"]
    assert heard.count("first goal") == 1, "the goal is in the carried session once, not said again"
    resumed = await run(
        task_activities.agent_turn,
        _payload(attempt_id="att-3", attempt_no=2, goal="first goal", continue_from="att-2", messages=[{"text": "history:"}]),
    )
    assert resumed["text"].count("first goal") == 1


async def test_an_attempt_whose_earlier_session_is_gone_starts_with_its_goal_like_a_fresh_node(tmp_path: Path) -> None:
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(AgentRuntime(MemoryStateStore()))
    run = ActivityEnvironment().run
    # att-9 does not exist: the replacement was to carry on it, and starts empty.
    lost = await run(
        task_activities.agent_turn,
        _payload(attempt_id="att-2", attempt_no=2, goal="history:", continue_from="att-9", messages=[{"text": "from the user"}],
                 retry_reason="pytest exited 1"),
    )
    assert lost["text"].startswith("history="), "it was given its goal"
    assert "Your previous attempt was rejected: pytest exited 1" in lost["text"]
    assert "from the user" in lost["text"]
    # When the session is there, the same attempt is given only what is new.
    await run(task_activities.agent_turn, _payload(attempt_id="att-3", goal="the original goal"))
    carried = await run(
        task_activities.agent_turn,
        _payload(attempt_id="att-4", attempt_no=2, goal="history:", continue_from="att-3", messages=[{"text": "history:"}]),
    )
    # Only the new message was sent: the session's history is the first goal and that message, not the goal twice.
    assert carried["text"].count("the original goal") == 1 and carried["text"].count("history:") == 1


async def test_open_session_says_whether_state_was_carried() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await open_session(runtime, "att-1")
    with streaming_for(context("att-2")):
        carried = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="att-2:open", continue_from="att-1"))
        again = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="att-2:open", continue_from="att-1"))
    with streaming_for(context("att-3")):
        gone = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="att-3:open", continue_from="att-missing"))
        none_asked = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="att-4:open"))
    assert (carried.carried, again.carried, gone.carried, none_asked.carried) == (True, True, False, False)

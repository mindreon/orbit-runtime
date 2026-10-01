"""A follow-up carries on the agent session of the attempt before it (the task stays open).

How it can go wrong, written down before the code:
  - the follow-up starts from nothing, so the agent has forgotten what was said;
  - the new session shares state with the old one, so a later turn of one changes the other;
  - the new session keeps the old session's permission preset, so a task switched to read-only still writes;
  - the old session is gone or unreadable and the follow-up fails instead of starting fresh;
  - opening again after a retry forks over the progress the new session has already made.
"""

import logging

import pytest
from orbit_contracts.models import OpenSessionInput, RunTurnInput
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore
from orbit_worker.task_stream import TaskStreamContext, streaming_for


def context(attempt_id: str) -> TaskStreamContext:
    return TaskStreamContext(tenant_id="t", task_id="task-1", attempt_id=attempt_id, activity_attempt=1)


async def open_and_say(runtime: AgentRuntime, attempt_id: str, text: str, *, continue_from: str = "", preset: str = "workspace-write") -> str:
    """Open the attempt's session (carrying on another, if asked), say one thing, return what the agent answered."""
    with streaming_for(context(attempt_id)):
        opened = await runtime.open_session(
            OpenSessionInput(room_id="task-1", turn_id=f"{attempt_id}:open", permission_preset=preset, continue_from=continue_from)  # type: ignore[arg-type]
        )
        return await say(runtime, attempt_id, opened.session_id, text)


async def say(runtime: AgentRuntime, attempt_id: str, session_id: str, text: str) -> str:
    with streaming_for(context(attempt_id)):
        blob = await runtime._store.get(session_id)
        assert blob is not None
        result = await runtime.run_turn(
            RunTurnInput(room_id="task-1", session_id=session_id, turn_id=f"{attempt_id}:{blob.state_version}", message=text, state_version=blob.state_version)
        )
    return getattr(result, "text", "") or ""


async def test_the_follow_up_remembers_what_was_said_before() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await open_and_say(runtime, "att-1", "first thing")
    heard = await open_and_say(runtime, "att-2", "history:", continue_from="att-1")
    assert "first thing" in heard, heard


async def test_without_continue_from_the_session_starts_fresh() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await open_and_say(runtime, "att-1", "first thing")
    heard = await open_and_say(runtime, "att-2", "history:")
    assert "first thing" not in heard, heard


async def test_the_two_sessions_do_not_share_state() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await open_and_say(runtime, "att-1", "first thing")
    await open_and_say(runtime, "att-2", "second thing", continue_from="att-1")
    old = await runtime._store.get("att-1")
    assert old is not None
    heard = await say(runtime, "att-1", "att-1", "history:")
    assert "second thing" not in heard, "what the follow-up said did not leak back into the session it came from"


async def test_the_new_session_takes_its_own_permission_preset() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await open_and_say(runtime, "att-1", "hello", preset="workspace-write")
    await open_and_say(runtime, "att-2", "hello", continue_from="att-1", preset="read-only")
    first = await runtime._store.get("att-1")
    second = await runtime._store.get("att-2")
    assert first is not None and second is not None
    assert first.agent_state["permission_context"]["mode"] != second.agent_state["permission_context"]["mode"]


@pytest.mark.parametrize("previous", ["att-missing", "att-2", ""])
async def test_a_missing_or_self_referencing_session_starts_fresh(previous: str, caplog: pytest.LogCaptureFixture) -> None:
    runtime = AgentRuntime(MemoryStateStore())
    with caplog.at_level(logging.WARNING):
        heard = await open_and_say(runtime, "att-2", "history:", continue_from=previous)
    assert "history=" in heard


async def test_opening_again_after_a_retry_keeps_the_progress_made() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    await open_and_say(runtime, "att-1", "old")
    with streaming_for(context("att-2")):
        opened = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="att-2:open", continue_from="att-1"))
    await say(runtime, "att-2", opened.session_id, "new progress")
    with streaming_for(context("att-2")):
        again = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="att-2:open", continue_from="att-1"))
    assert again.state_version > 1, "the retry found the session where it was, not a fresh fork of att-1"
    heard = await say(runtime, "att-2", again.session_id, "history:")
    assert "new progress" in heard and "old" in heard

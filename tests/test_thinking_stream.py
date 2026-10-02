"""The thinking stream reaches the task page: TurnEvents maps AgentScope thinking blocks to
``assistant.thinking``, ``to_v3`` turns those into ``agent.thinking_delta``, and the mock model
can drive the whole path with a "reason:" message."""

import pytest
from agentscope.event import (
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    ThinkingBlockDeltaEvent,
    ThinkingBlockEndEvent,
)
from orbit_contracts.models import OpenSessionInput, OrbitEvent, RunTurnInput
from orbit_worker.events import MemoryEventIngest
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore
from orbit_worker.task_stream import TaskStreamContext, to_v3
from orbit_worker.turn_events import TurnEvents

CONTEXT = TaskStreamContext(
    tenant_id="tenant-a", task_id="task-1", attempt_id="att_1", activity_attempt=1, node_id="n_1"
)


def _thinking(delta: str, block_id: str = "think-1") -> ThinkingBlockDeltaEvent:
    return ThinkingBlockDeltaEvent(reply_id="r", block_id=block_id, delta=delta)


def test_thinking_deltas_coalesce_like_text() -> None:
    """Thinking blocks stream as assistant.thinking with their own per-block seq."""

    events = TurnEvents({}, clock=lambda: 0.0)
    streamed = []
    for _ in range(3):
        streamed += events.observe(_thinking("模型正在逐字输出一段没有任何空格的思考内容，" * 10))
    streamed += events.observe(ThinkingBlockEndEvent(reply_id="r", block_id="think-1"))
    assert all(kind == "assistant.thinking" for kind, _ in streamed)
    assert [fields["seq"] for _, fields in streamed] == list(range(len(streamed)))
    assert "".join(str(fields["delta"]) for _, fields in streamed) == "模型正在逐字输出一段没有任何空格的思考内容，" * 30


def test_thinking_and_text_stay_separate_in_one_turn() -> None:
    """A thinking block and a text block with different ids stream as different kinds."""

    events = TurnEvents({}, clock=lambda: 0.0)
    streamed = [
        *events.observe(_thinking("thinking part")),
        *events.observe(TextBlockDeltaEvent(reply_id="r", block_id="text-1", delta="text part")),
        *events.observe(ThinkingBlockEndEvent(reply_id="r", block_id="think-1")),
        *events.observe(TextBlockEndEvent(reply_id="r", block_id="text-1")),
    ]
    kinds = {fields["block_id"]: kind for kind, fields in streamed}
    assert kinds["think-1"] == "assistant.thinking"
    assert kinds["text-1"] == "assistant.delta"


def test_to_v3_maps_thinking_to_thinking_delta() -> None:
    """assistant.thinking becomes an ephemeral agent.thinking_delta with a distinct deterministic id."""

    event = OrbitEvent(
        type="assistant.thinking",
        session_id="s",
        room_id="task-1",
        turn_id="turn-1",
        block_id="b",
        seq=2,
        delta="为什么",
    )
    envelope = to_v3(event, CONTEXT)
    assert envelope is not None
    assert envelope["type"] == "agent.thinking_delta"
    assert envelope["retention"] == "ephemeral"
    assert envelope["payload"] == {"attempt_id": "att_1", "text": "为什么", "block_id": "b"}
    token = to_v3(
        OrbitEvent(
            type="assistant.delta",
            session_id="s",
            room_id="task-1",
            turn_id="turn-1",
            block_id="b",
            seq=2,
            delta="为什么",
        ),
        CONTEXT,
    )
    assert token is not None
    assert token["type"] == "agent.token_delta"
    assert envelope["event_id"] != token["event_id"]


@pytest.mark.asyncio
async def test_the_mock_model_drives_the_whole_thinking_path() -> None:
    ingest = MemoryEventIngest()
    runtime = AgentRuntime(MemoryStateStore(), ingest=ingest)
    opened = await runtime.open_session(OpenSessionInput(room_id="room-1", turn_id="open-1"))

    result = await runtime.run_turn(
        RunTurnInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="turn-1",
            message="reason:先核对账本再回答|先看账本",
            state_version=opened.state_version,
        )
    )

    assert result.status == "completed"
    thinking = [event for event in ingest.events if event.type == "assistant.thinking"]
    assert thinking, "the turn must stream thinking events"
    assert "".join(event.delta for event in thinking) == "先核对账本再回答"
    assert all(event.turn_id == "turn-1" for event in thinking)
    assert result.text == "先看账本"

    envelopes = [to_v3(event, CONTEXT) for event in thinking]
    assert all(e is not None and e["type"] == "agent.thinking_delta" for e in envelopes)
    assert "".join(str(e["payload"]["text"]) for e in envelopes if e) == "先核对账本再回答"

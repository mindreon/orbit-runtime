"""The adapter parks on approval, drops the agent object, and resumes from the blob."""

import pytest
from orbit_contracts.models import OpenSessionInput, ResolveApprovalInput, RunTurnInput
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore


@pytest.mark.asyncio
async def test_approval_survives_rebuilding_the_agent() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    opened = await runtime.open_session(
        OpenSessionInput(room_id="room-1", turn_id="open-1")
    )

    parked = await runtime.run_turn(
        RunTurnInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="turn-1",
            message="please gated this",
            state_version=opened.state_version,
        )
    )
    assert parked.status == "needs_approval"
    assert parked.approval is not None
    assert parked.approval.tool_name == "gated_echo"

    # A second process would build a new AgentRuntime over the same store.
    resumed = AgentRuntime(runtime._store)
    done = await resumed.resolve_approval(
        ResolveApprovalInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="approve-1",
            approval_request_id=parked.approval.approval_request_id,
            outcome="allowed-once",
        )
    )
    assert done.status == "completed"
    assert done.text == "done"
    assert done.state_version > parked.state_version


@pytest.mark.asyncio
async def test_repeated_turn_id_does_not_run_twice() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    opened = await runtime.open_session(
        OpenSessionInput(room_id="room-1", turn_id="open-1")
    )
    again = await runtime.open_session(
        OpenSessionInput(room_id="room-1", turn_id="open-1")
    )
    assert again.session_id == opened.session_id

    call = RunTurnInput(
        room_id="room-1",
        session_id=opened.session_id,
        turn_id="turn-1",
        message="please gated this",
        state_version=opened.state_version,
    )
    first = await runtime.run_turn(call)
    second = await runtime.run_turn(call)
    assert second == first

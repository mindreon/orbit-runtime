"""State, isolation, approval, and tracing outside Temporal."""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from orbit_contracts.models import (
    OpenSessionInput,
    ResolveApprovalInput,
    RunTurnInput,
)
from orbit_worker.events import MemoryEventIngest
from orbit_worker.isolation import prepare_isolation
from orbit_worker.runtime import AgentRuntime
from orbit_worker.secrets import reject_secret_values
from orbit_worker.store import MemoryStateStore
from orbit_worker.tracing import configure_tracing

# Long enough that random Fernet (base64) ciphertext never contains it by chance.
PLAINTEXT_MARKER = "orbit-plaintext-marker-visible-in-clear"


@pytest.mark.asyncio
async def test_older_state_version_is_rejected() -> None:
    runtime = AgentRuntime(MemoryStateStore())
    opened = await runtime.open_session(OpenSessionInput(room_id="room-1", turn_id="open-1"))
    with pytest.raises(ValueError, match="state version"):
        await runtime.run_turn(
            RunTurnInput(
                room_id="room-1",
                session_id=opened.session_id,
                turn_id="turn-1",
                message="hello",
                state_version=opened.state_version - 1,
            )
        )


@pytest.mark.asyncio
async def test_approval_survives_a_new_runtime() -> None:
    """A parked tool call is decided by a runtime that never saw it: the state lives in the store, not the process."""
    store = MemoryStateStore()
    first = AgentRuntime(store)
    opened = await first.open_session(OpenSessionInput(room_id="room-1", turn_id="open-1"))
    parked = await first.run_turn(
        RunTurnInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="turn-1",
            message="echo:hello",
            state_version=opened.state_version,
        )
    )
    assert parked.status == "needs_approval"
    assert parked.approval is not None
    assert parked.approval.tool_name == "gated_echo"

    resumed = await AgentRuntime(store).resolve_approval(
        ResolveApprovalInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="approve-1",
            approval_request_id=parked.approval.approval_request_id,
            outcome="allowed-once",
        )
    )
    assert resumed.status == "completed"
    assert resumed.text == "done"


@pytest.mark.asyncio
async def test_open_session_emits_runtime_events() -> None:
    ingest = MemoryEventIngest()
    runtime = AgentRuntime(MemoryStateStore(), ingest=ingest)
    await runtime.open_session(OpenSessionInput(room_id="room-1", turn_id="open-1"))
    kinds = [event.type for event in ingest.events]
    assert "session.status" in kinds
    assert "agent.started" in kinds
    assert ingest.events[0].event_id
    assert ingest.events[0].runtime == "agentscope"


def test_secret_values_never_enter_the_blob() -> None:
    with pytest.raises(ValueError, match="secret"):
        reject_secret_values({"api_key": "super-secret"})
    with pytest.raises(ValueError, match="secret"):
        reject_secret_values({"note": "sk-live-example"})


def test_bwrap_refuses_shared_networking(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="share_net=False"):
        prepare_isolation(mode="bwrap", share_net=True, strict=False, root=tmp_path)


def test_bwrap_backend_is_constructed_offline(tmp_path) -> None:
    snapshot = prepare_isolation(
        mode="bwrap",
        share_net=False,
        strict=False,
        root=tmp_path,
        cpu_max="100000 100000",
        memory_max="268435456",
        apply_cgroup=True,
    )
    assert snapshot.backend == "bwrap"
    assert snapshot.share_net is False
    assert snapshot.cgroup_applied is True
    assert (tmp_path / "cgroup" / "memory.max").read_text(encoding="utf-8").strip() == "268435456"


def test_docker_mode_requires_a_prebaked_image(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="pre-baked"):
        prepare_isolation(mode="docker", share_net=False, strict=True, root=tmp_path)


@pytest.mark.asyncio
async def test_tracing_middleware_records_a_span() -> None:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter)
    runtime = AgentRuntime(MemoryStateStore())
    opened = await runtime.open_session(OpenSessionInput(room_id="room-trace", turn_id="open-1"))
    await runtime.run_turn(
        RunTurnInput(
            room_id="room-trace",
            session_id=opened.session_id,
            turn_id="turn-1",
            message="hello",
            state_version=opened.state_version,
        )
    )
    assert exporter.get_finished_spans()

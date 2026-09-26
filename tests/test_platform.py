"""State, isolation, gateway approval, and tracing outside Temporal."""

import os
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from orbit_contracts.models import (
    DeliverToolResultInput,
    OpenSessionInput,
    ResolveApprovalInput,
    RunTurnInput,
)
from orbit_contracts.schema_export import export_schemas
from orbit_worker.events import MemoryEventIngest
from orbit_worker.isolation import prepare_isolation
from orbit_worker.postgres_store import PostgresStateStore, decode_blob, encode_blob
from orbit_worker.runtime import AgentRuntime
from orbit_worker.secrets import reject_secret_values
from orbit_worker.store import MemoryStateStore, SessionBlob
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
async def test_gateway_approval_survives_a_new_runtime() -> None:
    store = MemoryStateStore()
    first = AgentRuntime(store)
    opened = await first.open_session(OpenSessionInput(room_id="room-1", turn_id="open-1"))
    parked = await first.run_turn(
        RunTurnInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="turn-1",
            message="please charge the account",
            state_version=opened.state_version,
        )
    )
    assert parked.status == "needs_approval"
    assert parked.approval is not None
    assert parked.approval.tool_name == "gateway_charge"

    resumed = AgentRuntime(store)
    external = await resumed.resolve_approval(
        ResolveApprovalInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="approve-1",
            approval_request_id=parked.approval.approval_request_id,
            outcome="allowed-once",
        )
    )
    assert external.status == "needs_external"
    assert external.external is not None

    delivered = await AgentRuntime(store).deliver_tool_result(
        DeliverToolResultInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="deliver-1",
            state_version=external.state_version,
            tool_name=external.external.tool_name,
            call_id=external.external.call_id,
            output="charged 1",
            metadata={"ok": "true"},
        )
    )
    assert delivered.status == "completed"
    assert delivered.text == "done"


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


def test_fernet_blob_hides_agent_state() -> None:
    key = Fernet.generate_key().decode("utf-8")
    blob = SessionBlob(
        session_id="s",
        room_id="r",
        state_version=1,
        agent_state={"context": "visible-conversation"},
        permission_preset="workspace-write",
    )
    payload = encode_blob(blob, key)
    assert b"visible-conversation" not in payload
    assert decode_blob(payload, key) == blob


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


def test_schema_files_cover_the_turn_contract(tmp_path) -> None:
    written = {path.name for path in export_schemas(tmp_path)}
    assert "TurnResult.json" in written
    assert "RoomWorkflowInput.json" in written
    assert "CloudAgentJobInput.json" in written


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


@pytest.mark.asyncio
async def test_postgres_roundtrip_rejects_an_older_version() -> None:
    url = os.environ.get("ORBIT_TEST_POSTGRES_URL", "")
    if not url:
        pytest.skip("ORBIT_TEST_POSTGRES_URL is not set")
    import asyncpg

    key = Fernet.generate_key().decode("utf-8")

    async def connect() -> asyncpg.Connection:
        return await asyncpg.connect(url)

    store = PostgresStateStore(connect, key)
    await store.ensure_schema()
    session_id = uuid4().hex
    blob = SessionBlob(
        session_id=session_id,
        room_id=f"pg-room-{session_id}",
        state_version=1,
        agent_state={"marker": PLAINTEXT_MARKER},
        permission_preset="workspace-write",
    )
    await store.put(blob)
    loaded = await store.get(session_id)
    assert loaded is not None
    assert loaded.agent_state["marker"] == PLAINTEXT_MARKER
    found = await store.find_by_idempotency(blob.room_id, "missing")
    assert found is None
    blob.idempotency["open-1:openSession"] = {"session_id": session_id}
    blob.state_version = 2
    await store.put(blob)
    found = await store.find_by_idempotency(blob.room_id, "open-1:openSession")
    assert found is not None
    assert found.state_version == 2
    stale = blob.model_copy(deep=True)
    stale.state_version = 1
    with pytest.raises(ValueError, match="older"):
        await store.put(stale)
    conn = await connect()
    try:
        row = await conn.fetchrow(
            "SELECT blob FROM orbit_agent_state WHERE session_id = $1",
            session_id,
        )
    finally:
        await conn.close()
    assert PLAINTEXT_MARKER.encode("utf-8") not in bytes(row["blob"])

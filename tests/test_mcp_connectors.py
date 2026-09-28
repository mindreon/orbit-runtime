"""MCP connector validation and attach behavior.

A missing binary must not fail the turn. Secret values are not stored.
"""

import logging

import pytest
from orbit_contracts.models import McpConnectorSpec, McpHeaderRef, OpenSessionInput, RunTurnInput
from orbit_worker.mcp_connectors import (
    normalize_spec,
    specs_for_storage,
    validate_mcp_http_url,
)
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore


def test_public_http_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="https"):
        validate_mcp_http_url("http://mcp.example.com/mcp")


def test_local_http_url_is_allowed() -> None:
    assert validate_mcp_http_url("http://127.0.0.1:9/mcp") == "http://127.0.0.1:9/mcp"
    assert validate_mcp_http_url("https://mcp.example.com/mcp") == "https://mcp.example.com/mcp"


def test_url_must_not_carry_a_secret() -> None:
    with pytest.raises(ValueError, match="secret"):
        validate_mcp_http_url("https://mcp.example.com/mcp?token=hidden-value")
    with pytest.raises(ValueError, match="password"):
        validate_mcp_http_url("https://user:hidden@mcp.example.com/mcp")


def test_header_env_must_be_a_name() -> None:
    spec = McpConnectorSpec(
        id="mcp_docs",
        name="Docs",
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
        header_refs=[McpHeaderRef(name="Authorization", env="Bearer hidden")],
    )
    with pytest.raises(ValueError, match="name"):
        normalize_spec(spec)


def test_storage_keeps_names_only() -> None:
    stored = specs_for_storage(
        [
            McpConnectorSpec(
                id="mcp_docs",
                name="Docs",
                command="npx",
                args=["-y", "docs"],
                env_refs=["DOCS_TOKEN"],
            )
        ]
    )
    assert stored[0]["env_refs"] == ["DOCS_TOKEN"]
    assert "hidden" not in str(stored)


@pytest.mark.asyncio
async def test_unreachable_connector_does_not_fail_the_turn(caplog: pytest.LogCaptureFixture) -> None:
    runtime = AgentRuntime(MemoryStateStore())
    opened = await runtime.open_session(
        OpenSessionInput(
            room_id="room-mcp",
            turn_id="open-1",
            mcp_connectors=[
                McpConnectorSpec(
                    id="mcp_missing",
                    name="Missing",
                    command="orbit-mcp-missing-binary",
                    env_refs=["DOCS_TOKEN"],
                )
            ],
        )
    )
    blob = await runtime._store.get(opened.session_id)
    assert blob is not None
    assert blob.mcp_connectors[0]["command"] == "orbit-mcp-missing-binary"
    assert "DOCS_TOKEN" in blob.mcp_connectors[0]["env_refs"]
    with caplog.at_level(logging.WARNING):
        result = await runtime.run_turn(
            RunTurnInput(
                room_id="room-mcp",
                session_id=opened.session_id,
                turn_id="turn-1",
                message="hello",
                state_version=opened.state_version,
            )
        )
    assert result.status == "completed"
    leaked = [record.getMessage() for record in caplog.records if "hidden" in record.getMessage()]
    assert leaked == []
    await runtime._mcp.close_all()

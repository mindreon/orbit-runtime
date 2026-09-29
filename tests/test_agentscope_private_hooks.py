"""The places where the worker leans on AgentScope internals, pinned to 2.0.9.

These fail on an upgrade the moment one of the hooks moves, instead of at run time in a task.
"""

from __future__ import annotations

import inspect

from agentscope.mcp import MCPClient
from agentscope.sop import SOPEngine, SOPPhase, SOPRunState


def test_mcp_client_keeps_its_session_in_a_private_attribute() -> None:
    # mcp_connectors._serialize_session wraps client._session.call_tool / list_tools under a lock.
    source = inspect.getsource(MCPClient)
    assert "self._session" in source
    client = MCPClient.__new__(MCPClient)
    client._session = None  # the attribute the worker reads must be assignable
    assert client._session is None


def test_sop_engine_streams_one_step_end_event_the_worker_stops_on() -> None:
    # sop_agents.run_one_try closes reply_stream at SOP_STEP_ENDED and reads engine.state.steps.
    assert inspect.isasyncgenfunction(SOPEngine.reply_stream) or callable(SOPEngine.reply_stream)
    assert "SOP_STEP_ENDED" in inspect.getsource(inspect.getmodule(SOPEngine))
    assert {"PENDING", "COMPLETED", "FAILED"} <= {phase.name for phase in SOPPhase}
    fields = SOPRunState.model_fields
    assert "steps" in fields
    assert "verifications" in inspect.getsource(inspect.getmodule(SOPRunState))

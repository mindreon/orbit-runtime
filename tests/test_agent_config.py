"""What a profile contributes to a task attempt's Agent (15 T8.3): instructions, model and MCP connectors."""

import asyncio
import logging

import pytest
from agentscope.state import AgentState
from orbit_contracts.models import OpenSessionInput, RunTurnInput
from orbit_worker.agent_config import MAX_INSTRUCTIONS_CHARS, AgentConfig, agent_config_from_spec
from orbit_worker.chat_model import ModelConfig
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore, SessionBlob
from orbit_worker.task_stream import TaskStreamContext, streaming_for

CONNECTOR = {
    "id": "mcp_docs",
    "name": "Docs",
    "transport": "streamable_http",
    "url": "http://127.0.0.1:9/mcp",
}


def test_empty_spec_changes_nothing() -> None:
    assert agent_config_from_spec({}) == AgentConfig()


def test_spec_carries_instructions_model_and_connectors() -> None:
    config = agent_config_from_spec(
        {"instructions": "  Be brief.  ", "model": "gpt-x", "mcp_connectors": [CONNECTOR]}
    )
    assert config.instructions == "Be brief."
    assert config.model == "gpt-x"
    assert [item["id"] for item in config.mcp_connectors] == ["mcp_docs"]


def test_invalid_connector_is_skipped_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    secret = {**CONNECTOR, "id": "mcp_bad", "url": "https://mcp.example.com/mcp?token=hidden-value"}
    with caplog.at_level(logging.WARNING):
        config = agent_config_from_spec({"mcp_connectors": [secret, CONNECTOR]})
    assert [item["id"] for item in config.mcp_connectors] == ["mcp_docs"]
    assert "hidden-value" not in caplog.text


def test_wrong_types_are_ignored() -> None:
    config = agent_config_from_spec(
        {"instructions": 7, "model": ["x"], "mcp_connectors": "nope"}
    )
    assert config == AgentConfig()


def test_instructions_over_the_limit_are_cut(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        config = agent_config_from_spec({"instructions": "x" * (MAX_INSTRUCTIONS_CHARS + 5)})
    assert len(config.instructions) == MAX_INSTRUCTIONS_CHARS
    assert "instructions" in caplog.text


@pytest.mark.parametrize("model", ["has space", "a;b", "", "x" * 200])
def test_model_name_must_look_like_a_model_name(model: str) -> None:
    assert agent_config_from_spec({"model": model}).model == ""


def _runtime(mode: str = "mock") -> AgentRuntime:
    return AgentRuntime(
        store=MemoryStateStore(), model_config=ModelConfig(mode=mode, name="base")  # type: ignore[arg-type]
    )


def _blob() -> SessionBlob:
    return SessionBlob(
        session_id="s1",
        task_id="t1",
        state_version=1,
        agent_state=AgentState().model_dump(mode="json"),
        permission_preset="workspace-write",
    )


def _context(config: AgentConfig) -> TaskStreamContext:
    return TaskStreamContext(
        tenant_id="t", task_id="task", attempt_id="att", activity_attempt=1, agent=config
    )


def _prompt() -> str:
    # AgentScope keeps the prompt private; this is the string its model call is given.
    return asyncio.run(_runtime()._agent(_blob())._get_system_prompt())


def test_agent_outside_a_task_gets_the_base_prompt_with_the_reply_style() -> None:
    prompt = _prompt()
    assert prompt.startswith("You are an Orbit business agent.")
    assert "one or two plain sentences" in prompt and "Never mention internal ids" in prompt
    assert "Do not paste whole files" in prompt and "TodoWrite" in prompt
    assert "in the language the user wrote in" in prompt
    assert "never when you assigned it" in prompt


def test_instructions_extend_the_prompt_inside_a_task() -> None:
    with streaming_for(_context(AgentConfig(instructions="Answer in French."))):
        prompt = _prompt()
    assert prompt.startswith("You are an Orbit business agent.")
    assert prompt.endswith("Answer in French.")


def test_profile_model_applies_only_to_a_real_model() -> None:
    runtime = _runtime("mock")
    assert runtime.model_config_for(AgentConfig(model="gpt-x")).name == "base"
    real = _runtime("real")
    assert real.model_config_for(AgentConfig(model="gpt-x")).name == "gpt-x"
    assert real.model_config_for(AgentConfig()).name == "base"


async def test_profile_connector_is_attached_and_a_dead_one_does_not_fail_the_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dead = {"id": "mcp_dead", "name": "Dead", "command": "orbit-mcp-missing-binary"}
    runtime = _runtime()
    opened = await runtime.open_session(OpenSessionInput(room_id="task-1", turn_id="open-1"))
    with (
        streaming_for(_context(agent_config_from_config(dead))),
        caplog.at_level(logging.WARNING),
    ):
        result = await runtime.run_turn(
            RunTurnInput(
                room_id="task-1",
                session_id=opened.session_id,
                turn_id="turn-1",
                message="hello",
                state_version=opened.state_version,
            )
        )
    assert result.status == "completed"
    # The connector reached the registry: its failed connect is what got logged.
    assert "mcp_dead" in caplog.text or "Dead" in caplog.text
    await runtime._mcp.close_all()


def agent_config_from_config(connector: dict[str, str]) -> AgentConfig:
    return agent_config_from_spec({"mcp_connectors": [connector]})


# ---- the task's own configuration on top of the expert's (15 M8) ---------------------------------------------------
#
# How it can go wrong, written down before the code:
#   - no task config, or one that leaves connectors open, changes what the expert gave;
#   - an explicitly empty connector list is mistaken for "not set" and the expert's connectors come back;
#   - a connector in the task config skips the checks a profile's connectors get (secret values, env prefix);
#   - "ask" (questions only) still runs with write permission;
#   - the expert's instructions or model are lost when the task config replaces the connectors.

from orbit_worker.agent_config import permission_preset_for, with_task_config

EXPERT = agent_config_from_spec(
    {"instructions": "Be brief.", "model": "gpt-x", "mcp_connectors": [CONNECTOR]}
)
OTHER = {"id": "mcp_other", "name": "Other", "command": "orbit-mcp-other"}


def test_no_task_config_changes_nothing() -> None:
    assert with_task_config(EXPERT, None) == EXPERT
    assert with_task_config(EXPERT, {}) == EXPERT


def test_connectors_left_open_keep_the_experts() -> None:
    assert with_task_config(EXPERT, {"config_version": 2, "connectors": None}) == EXPERT


def test_an_empty_connector_list_removes_the_experts() -> None:
    config = with_task_config(EXPERT, {"config_version": 2, "connectors": []})
    assert config.mcp_connectors == ()
    assert (config.instructions, config.model) == ("Be brief.", "gpt-x")


def test_task_connectors_replace_the_experts_and_keep_the_rest() -> None:
    config = with_task_config(EXPERT, {"config_version": 2, "connectors": [OTHER]})
    assert [item["id"] for item in config.mcp_connectors] == ["mcp_other"]
    assert (config.instructions, config.model) == ("Be brief.", "gpt-x")


def test_a_task_connector_gets_the_same_checks_as_a_profiles(caplog: pytest.LogCaptureFixture) -> None:
    secret = {**OTHER, "id": "mcp_secret", "url": "https://mcp.example.com/mcp?token=hidden-value",
              "transport": "streamable_http"}
    with caplog.at_level(logging.WARNING):
        config = with_task_config(EXPERT, {"config_version": 2, "connectors": [secret, OTHER]})
    assert [item["id"] for item in config.mcp_connectors] == ["mcp_other"]
    assert "hidden-value" not in caplog.text


@pytest.mark.parametrize(
    ("mode", "preset"),
    [("default", "workspace-write"), ("plan", "workspace-write"), ("ask", "read-only")],
)
def test_only_ask_drops_write_permission(mode: str, preset: str) -> None:
    assert permission_preset_for({"mode": mode}) == preset


def test_a_missing_or_unknown_mode_keeps_write_permission() -> None:
    assert permission_preset_for(None) == "workspace-write"
    assert permission_preset_for({}) == "workspace-write"


def test_a_task_model_replaces_the_experts() -> None:
    config = with_task_config(EXPERT, {"config_version": 2, "model": "glm-5"})
    assert config.model == "glm-5"
    assert (config.instructions, config.skills) == (EXPERT.instructions, EXPERT.skills)


def test_an_empty_or_absent_task_model_keeps_the_experts() -> None:
    assert with_task_config(EXPERT, {"config_version": 2, "model": ""}) == EXPERT
    assert with_task_config(EXPERT, {"config_version": 2, "model": None}) == EXPERT
    assert with_task_config(EXPERT, {"config_version": 2}) == EXPERT


def test_a_task_model_gets_the_same_checks_as_a_profiles(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        config = with_task_config(EXPERT, {"config_version": 2, "model": "../etc/passwd"})
    assert config.model == "gpt-x"
    assert "not a model name" in caplog.text


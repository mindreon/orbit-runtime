"""ORBIT_MODEL_MODE resolution. No network; the real-endpoint test skips without env."""

import logging
import os

import pytest
from orbit_contracts.models import OpenSessionInput, RunTurnInput
from orbit_worker.chat_model import ModelConfig, ModelConfigError, resolve_model_config
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore

KEY = "sk-test-not-a-real-key"
URL = "https://models.internal.example/v1"
FULL = {
    "ORBIT_MODEL_BASE_URL": URL,
    "ORBIT_MODEL_API_KEY": KEY,
    "ORBIT_MODEL_NAME": "test-model",
}


def test_unset_mode_is_mock() -> None:
    assert resolve_model_config({}) == ModelConfig(mode="mock", name="mock")


def test_mock_with_partial_config_warns_by_name_only(caplog: pytest.LogCaptureFixture) -> None:
    env = {"ORBIT_MODEL_MODE": "mock", "ORBIT_MODEL_API_KEY": KEY, "ORBIT_MODEL_BASE_URL": URL}
    with caplog.at_level(logging.WARNING, logger="orbit_worker.chat_model"):
        config = resolve_model_config(env)
    assert config.mode == "mock"
    assert "ORBIT_MODEL_NAME" in caplog.text
    assert "ORBIT_MODEL_API_KEY" in caplog.text
    assert KEY not in caplog.text
    assert URL not in caplog.text


def test_real_with_missing_vars_names_them_without_values() -> None:
    env = {"ORBIT_MODEL_MODE": "real", "ORBIT_MODEL_API_KEY": KEY}
    with pytest.raises(ModelConfigError) as caught:
        resolve_model_config(env)
    message = str(caught.value)
    assert "ORBIT_MODEL_BASE_URL" in message
    assert "ORBIT_MODEL_NAME" in message
    assert "ORBIT_MODEL_API_KEY" not in message
    assert KEY not in message


def test_real_with_full_config() -> None:
    config = resolve_model_config({"ORBIT_MODEL_MODE": "real", **FULL})
    assert config == ModelConfig(mode="real", name="test-model")
    assert KEY not in repr(config)
    assert URL not in repr(config)


def test_unknown_mode_fails() -> None:
    with pytest.raises(ModelConfigError):
        resolve_model_config({"ORBIT_MODEL_MODE": "fake"})


def _real_env_ready() -> bool:
    if os.environ.get("ORBIT_MODEL_MODE", "").strip().lower() != "real":
        return False
    return all(os.environ.get(name, "").strip() for name in FULL)


@pytest.mark.skipif(not _real_env_ready(), reason="real model env vars are not set")
@pytest.mark.asyncio
async def test_real_model_completes_a_turn() -> None:
    runtime = AgentRuntime(MemoryStateStore(), model_config=resolve_model_config())
    opened = await runtime.open_session(OpenSessionInput(room_id="room-real", turn_id="open"))
    result = await runtime.run_turn(
        RunTurnInput(
            room_id="room-real",
            session_id=opened.session_id,
            turn_id="turn",
            message="Reply with one short word.",
            state_version=opened.state_version,
        )
    )
    assert result.model_mode == "real"
    assert result.status == "completed", result.error

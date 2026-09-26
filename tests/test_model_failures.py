"""A failed real model request ends the turn as failed, offline, with no secret.

Requests go through RealChatModel, the OpenAI SDK, and its retries; only the
HTTP transport is replaced.
"""

import logging

import httpx2
import pytest
from orbit_contracts.models import OpenSessionInput, RunTurnInput
from orbit_worker import runtime as runtime_module
from orbit_worker.chat_model import FAILURE_MESSAGES, build_chat_model, resolve_model_config
from orbit_worker.events import MemoryEventIngest
from orbit_worker.mock_model import MockChatModel
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore

KEY = "sk-test-not-a-real-key"
HOST = "models.internal.example"
URL = f"https://{HOST}/v1"
# Lets the SDK retry 429 and 5xx without a real backoff sleep.
FAST_RETRY = {"retry-after-ms": "1"}


def _unauthorized(request: httpx2.Request) -> httpx2.Response:
    body = {"error": {"message": f"Incorrect API key provided: {KEY} for {URL}"}}
    return httpx2.Response(401, json=body)


def _rate_limited(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(429, json={"error": {"message": "slow down"}}, headers=FAST_RETRY)


def _server_error(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(500, json={"error": {"message": f"boom at {HOST}"}}, headers=FAST_RETRY)


def _timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout(f"timed out reading {request.url}", request=request)


@pytest.mark.parametrize(
    ("handler", "code", "retryable", "message"),
    [
        (_unauthorized, "auth", False, "模型配置有问题，请联系管理员。"),
        (_rate_limited, "rate_limited", True, "模型当前请求太多，请稍后再试。"),
        (_server_error, "provider_error", True, "模型服务暂时出错，这一轮没跑完。"),
        (_timeout, "timeout", True, "模型响应超时，这一轮没跑完。"),
    ],
    ids=["401", "429", "500", "timeout"],
)
@pytest.mark.asyncio
async def test_failed_request_is_an_explicit_secret_free_failure(
    handler,
    code: str,
    retryable: bool,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for name, value in {
        "ORBIT_MODEL_MODE": "real",
        "ORBIT_MODEL_BASE_URL": URL,
        "ORBIT_MODEL_API_KEY": KEY,
        "ORBIT_MODEL_NAME": "test-model",
    }.items():
        monkeypatch.setenv(name, value)

    requests: list[httpx2.Request] = []

    def transport(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    def build_with_transport(config, env=None):
        model = build_chat_model(config, env)
        mock = httpx2.AsyncClient(transport=httpx2.MockTransport(transport))
        model.client = model.client.with_options(http_client=mock)
        return model

    mock_calls: list[object] = []

    async def mock_called(*args: object, **kwargs: object) -> None:
        mock_calls.append(args)
        raise AssertionError("MockChatModel must not run in real mode")

    monkeypatch.setattr(runtime_module, "build_chat_model", build_with_transport)
    monkeypatch.setattr(MockChatModel, "_call_api", mock_called)

    store = MemoryStateStore()
    ingest = MemoryEventIngest()
    runtime = AgentRuntime(store, ingest=ingest, model_config=resolve_model_config())
    opened = await runtime.open_session(OpenSessionInput(room_id="room-1", turn_id="open-1"))
    before = await store.get(opened.session_id)
    assert before is not None

    caplog.set_level(logging.DEBUG)
    result = await runtime.run_turn(
        RunTurnInput(
            room_id="room-1",
            session_id=opened.session_id,
            turn_id="turn-1",
            message="hello",
            state_version=opened.state_version,
        )
    )

    assert requests, "the request must reach the HTTP transport"
    assert result.status == "failed"
    assert result.error_code == code
    assert result.retryable is retryable
    assert result.model_mode == "real"
    assert result.error == message == FAILURE_MESSAGES[code]
    assert result.state_version == opened.state_version

    after = await store.get(opened.session_id)
    assert after is not None
    assert after.state_version == before.state_version
    assert after.agent_state == before.agent_state
    # Only the idempotency entry for the failed turn is new.
    assert after.model_dump(exclude={"idempotency"}) == before.model_dump(exclude={"idempotency"})
    assert set(after.idempotency) - set(before.idempotency) == {"turn-1:runTurn"}

    assert not [event for event in ingest.events if event.type == "assistant.message"]
    failed = [event for event in ingest.events if event.type == "turn.failed"]
    assert len(failed) == 1
    failure = failed[0].failure
    assert failure is not None
    assert failure.turn_id == "turn-1"
    assert failure.agent_id == opened.session_id
    assert failure.error_code == code
    assert failure.retryable is retryable
    assert failure.message == result.error
    assert failure.message == message
    assert failure.message in set(FAILURE_MESSAGES.values())

    surfaces = [result.error, caplog.text, *(event.model_dump_json() for event in ingest.events)]
    for text in surfaces:
        assert KEY not in text
        assert HOST not in text
    assert mock_calls == []

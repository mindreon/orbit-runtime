"""Chat model selection. Endpoint secrets never leave this module.

``ORBIT_MODEL_MODE=mock`` (or unset) runs ``MockChatModel``. ``real`` runs an
OpenAI-compatible endpoint configured only through environment variables.
``ModelConfig`` holds the mode and the model name, which are the only model
facts that reach Temporal payloads, events, and logs. The API key and base
URL are read from the environment when a model is built, and every error
raised here is written without them.
"""

import logging
import os
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import openai
from agentscope.credential import OpenAICredential
from agentscope.message import Msg
from agentscope.model import ChatModelBase, ChatResponse, OpenAIChatModel
from orbit_contracts.models import ModelMode, TurnErrorCode

from orbit_worker.mock_model import MockChatModel

MODE_VAR = "ORBIT_MODEL_MODE"
BASE_URL_VAR = "ORBIT_MODEL_BASE_URL"
API_KEY_VAR = "ORBIT_MODEL_API_KEY"
NAME_VAR = "ORBIT_MODEL_NAME"
TIMEOUT_VAR = "ORBIT_MODEL_TIMEOUT_SECONDS"
REQUIRED_VARS = (BASE_URL_VAR, API_KEY_VAR, NAME_VAR)

_DEFAULT_TIMEOUT_SECONDS = 60.0
_CLIENT_RETRIES = 2
# These loggers print the full request URL at INFO or DEBUG. openai 3.x sends
# through httpx2/httpcore2; the older names cover a downgraded SDK.
_TRANSPORT_LOGGERS = ("httpx2", "httpcore2", "httpx", "httpcore", "openai")

logger = logging.getLogger(__name__)


class ModelConfigError(RuntimeError):
    """The model configuration is unusable. The message names variables only."""


class ModelRequestError(RuntimeError):
    """A real model request failed.

    ``str(exc)`` is ``FAILURE_MESSAGES[code]`` and nothing else. ``log_detail``
    (exception class, HTTP status, model name) is for worker logs only.
    """

    def __init__(self, code: TurnErrorCode, log_detail: str) -> None:
        super().__init__(FAILURE_MESSAGES[code])
        self.code: TurnErrorCode = code
        self.retryable = RETRYABLE[code]
        self.log_detail = log_detail


@dataclass(frozen=True)
class ModelConfig:
    mode: ModelMode = "mock"
    name: str = "mock"
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


def resolve_model_config(env: Mapping[str, str] | None = None) -> ModelConfig:
    """Read the model mode at worker startup.

    ``real`` with a required variable unset raises ``ModelConfigError``.
    ``mock`` with real variables set logs their names and still returns mock.
    """

    source = os.environ if env is None else env
    mode = source.get(MODE_VAR, "").strip().lower() or "mock"
    present = [name for name in REQUIRED_VARS if source.get(name, "").strip()]
    missing = [name for name in REQUIRED_VARS if name not in present]
    if mode == "mock":
        if present:
            logger.warning(
                "%s is not 'real', so the mock chat model is used. "
                "Real model variables set: %s. Missing: %s.",
                MODE_VAR,
                ", ".join(present),
                ", ".join(missing) or "none",
            )
        return ModelConfig()
    if mode != "real":
        raise ModelConfigError(f"{MODE_VAR} must be 'mock' or 'real'")
    if missing:
        raise ModelConfigError(
            f"{MODE_VAR}=real but required variable(s) are not set: {', '.join(missing)}"
        )
    return ModelConfig(
        mode="real",
        name=source[NAME_VAR].strip(),
        timeout_seconds=_timeout(source),
    )


def build_chat_model(config: ModelConfig, env: Mapping[str, str] | None = None) -> ChatModelBase:
    if config.mode == "mock":
        return MockChatModel()
    source = os.environ if env is None else env
    missing = [name for name in (BASE_URL_VAR, API_KEY_VAR) if not source.get(name, "").strip()]
    if missing:
        raise ModelConfigError(f"required variable(s) are not set: {', '.join(missing)}")
    for name in _TRANSPORT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    api_key = source[API_KEY_VAR].strip()
    base_url = source[BASE_URL_VAR].strip()
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ModelConfigError(f"{BASE_URL_VAR} must be an http or https URL")
    try:
        return RealChatModel(
            credential=OpenAICredential(name="orbit", api_key=api_key, base_url=base_url),
            model=config.name,
            stream=False,
            max_retries=0,
            client_kwargs={"timeout": config.timeout_seconds, "max_retries": _CLIENT_RETRIES},
        )
    # Any error here, including pydantic's, can echo the key or the URL.
    except Exception:  # noqa: BLE001
        raise ModelConfigError(
            f"could not build the real chat model; check {BASE_URL_VAR} and {API_KEY_VAR}"
        ) from None


class RealChatModel(OpenAIChatModel):
    """OpenAI-compatible chat model whose failures are safe to surface.

    AgentScope retries are off (``max_retries=0``) because its retry loop logs
    the provider's error text, and a 401 body can echo part of the key. The
    OpenAI client still retries timeouts, connection errors, 429, and 5xx.
    """

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **generate_kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        try:
            return await super()._call_api(
                model_name,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                **generate_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            # Every provider error is replaced. ``from None`` keeps the original
            # out of the traceback that tracing and Temporal serialize.
            raise ModelRequestError(
                classify_failure(exc), self._log_detail(exc, model_name)
            ) from None

    def _log_detail(self, exc: Exception, model_name: str) -> str:
        status = getattr(exc, "status_code", None)
        detail = type(exc).__name__
        if isinstance(status, int):
            detail = f"{detail} (HTTP {status})"
        return redact(f"{detail}; model={model_name}", self._secret_values())

    def _secret_values(self) -> list[str]:
        base_url = self.credential.base_url or ""
        return [
            self.credential.api_key.get_secret_value(),
            base_url,
            urlsplit(base_url).netloc,
        ]


# The only text a failed turn may surface (TurnResult.error, turn.failed
# message). Product-approved wording; never add provider, exception, or request
# text here. auth and config share one user text on purpose: the difference is
# in the worker log only.
FAILURE_MESSAGES: dict[TurnErrorCode, str] = {
    TurnErrorCode.TIMEOUT: "模型响应超时，这一轮没跑完。",
    TurnErrorCode.AUTH: "模型配置有问题，请联系管理员。",
    TurnErrorCode.RATE_LIMITED: "模型当前请求太多，请稍后再试。",
    TurnErrorCode.PROVIDER_ERROR: "模型服务暂时出错，这一轮没跑完。",
    TurnErrorCode.CONFIG: "模型配置有问题，请联系管理员。",
}

RETRYABLE: dict[TurnErrorCode, bool] = {
    TurnErrorCode.TIMEOUT: True,
    TurnErrorCode.AUTH: False,
    TurnErrorCode.RATE_LIMITED: True,
    TurnErrorCode.PROVIDER_ERROR: True,
    TurnErrorCode.CONFIG: False,
}


def classify_failure(exc: Exception) -> TurnErrorCode:
    """Map a provider error onto the README's error_code table (uses type and status only)."""

    if isinstance(exc, openai.APITimeoutError):
        return TurnErrorCode.TIMEOUT
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return TurnErrorCode.PROVIDER_ERROR
    if status in (401, 403):
        return TurnErrorCode.AUTH
    if status == 429:
        return TurnErrorCode.RATE_LIMITED
    if status >= 500:
        return TurnErrorCode.PROVIDER_ERROR
    return TurnErrorCode.CONFIG


def redact(text: str, secrets: Sequence[str]) -> str:
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    return text


def _timeout(source: Mapping[str, str]) -> float:
    raw = source.get(TIMEOUT_VAR, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise ModelConfigError(f"{TIMEOUT_VAR} must be a number of seconds") from None
    if value <= 0:
        raise ModelConfigError(f"{TIMEOUT_VAR} must be greater than zero")
    return value

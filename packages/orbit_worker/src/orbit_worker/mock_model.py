"""A chat model with no network, so local runs need no API key.

Scripted behaviour, read from the conversation:

- a user message containing "gated" asks for the ``gated_echo`` tool;
- once a tool result is in context, the model answers and stops;
- anything else is a short text reply.
"""

from collections.abc import AsyncGenerator

from agentscope.credential import CredentialBase
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from pydantic import BaseModel


class MockCredential(CredentialBase):
    """Placeholder credential. It never leaves the process."""

    @classmethod
    def get_chat_model_class(cls):  # type: ignore[no-untyped-def]
        return MockChatModel


class MockChatModel(ChatModelBase):
    """Deterministic stand-in for a provider model."""

    class Parameters(BaseModel):
        pass

    def __init__(self) -> None:
        super().__init__(
            credential=MockCredential(name="mock"),
            model="mock",
            parameters=self.Parameters(),
            stream=False,
            max_retries=0,
            context_size=32768,
        )
        # 2.0.8 reads this before the first model call, to reject media the
        # formatter cannot represent. The mock never sends those blocks.
        self.formatter = DeepSeekChatFormatter()

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: object | None = None,
        **kwargs: object,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        del model_name, tool_choice, kwargs
        if _has_tool_result(messages) or not tools:
            return _done("done")
        user_text = _last_user_text(messages)
        if "gated" in user_text.lower():
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="call-gated",
                        name="gated_echo",
                        input='{"text": "hello"}',
                    )
                ],
                is_last=True,
                finished_reason=FinishedReason.COMPLETED,
            )
        return _done("hello")


def _done(text: str) -> ChatResponse:
    return ChatResponse(
        content=[TextBlock(text=text)],
        is_last=True,
        finished_reason=FinishedReason.COMPLETED,
    )


def _has_tool_result(messages: list[Msg]) -> bool:
    for message in messages:
        for block in message.get_content_blocks():
            if isinstance(block, ToolResultBlock):
                return True
    return False


def _last_user_text(messages: list[Msg]) -> str:
    text = ""
    for message in messages:
        if message.role != "user":
            continue
        parts: list[str] = []
        for block in message.get_content_blocks():
            if isinstance(block, TextBlock):
                parts.append(block.text)
        if parts:
            text = "".join(parts)
    return text

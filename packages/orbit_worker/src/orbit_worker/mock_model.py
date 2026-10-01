"""A chat model with no network, so local runs need no API key.

Scripted behaviour, read from the conversation:

- a user message containing "gated" asks for the ``gated_echo`` tool;
- a message starting with "stream:" streams the rest back as provider
  deltas, one per part split at ``CHUNK_SEPARATOR``;
- a message starting with "echo:" asks for ``gated_echo`` with the rest;
- a message starting with "extend:" calls ``slow_echo`` for the "|"-separated texts, asking for more budget once
  when a call is refused;
- a message starting with "unplannable:" declares the task unplannable with the rest as the reason;
- a message starting with "two:" asks for ``gated_echo`` twice in one step, one call per "|"-separated text;
- a message starting with "slow:" calls ``slow_echo`` once per "|"-separated text, one call per model round;
- a message starting with "plan:" creates one task per "|"-separated title, chained one after the other;
- a message starting with "ask:" asks the user the rest through ``ask_user``;
- a message starting with "prompt:" answers with the system prompt it was given, and "tools:" with the names of the
  tools it can call: how a test sees what an expert and a task configuration gave the agent;
- a message starting with "mcp:" calls the tool named before the first "|" with the JSON after it (an object);
- once a tool result is in context, the model answers and stops;
- anything else is a short text reply.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from agentscope.credential import CredentialBase
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from pydantic import BaseModel

from orbit_worker.settings import MockSettings

CHUNK_SEPARATOR = "\x1f"
_STREAM = "stream:"
_ECHO = "echo:"
_ASK = "ask:"
_PLAN = "plan:"
_SLOW = "slow:"
_TWO = "two:"
_UNPLANNABLE = "unplannable:"
_EXTEND = "extend:"
_PROMPT = "prompt:"
_TOOLS = "tools:"
_MCP = "mcp:"


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
        # 2.0.9 reads this before the first model call, to reject media the
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
        user_text = _last_user_text(messages)
        results = _tool_results(messages)
        # Only this turn's tool results. A later "echo:" must still park even if
        # an earlier turn already ran a tool.
        turn_results = _turn_results(messages)
        if user_text.startswith(_STREAM):
            return _stream(user_text[len(_STREAM) :].split(CHUNK_SEPARATOR))
        if user_text.startswith(_ECHO) and tools and not turn_results:
            echoed = user_text[len(_ECHO) :]
            # echo:once stays call-echo. Any other payload gets its own call id,
            # so two agents can park without sharing apr-<call id>.
            call_id = "call-echo" if echoed == "once" else f"call-echo-{echoed}"
            payload = json.dumps({"text": echoed}, ensure_ascii=False)
            return _call(call_id, "gated_echo", payload)
        command = _inspection_command(user_text)
        if command.startswith(_PROMPT):
            return _done("prompt=" + _system_text(messages))
        if command.startswith(_TOOLS):
            return _done("tools=" + ",".join(sorted(_tool_names(tools))))
        if command.startswith(_MCP) and tools:
            name, _, raw_args = command[len(_MCP) :].partition("|")
            if turn_results:
                return _done("mcp-result=" + _last_output(turn_results))
            if name not in _tool_names(tools):
                return _done(f"no tool {name}")
            return _call("call-mcp", name, raw_args or "{}")
        if user_text.startswith(_EXTEND) and tools:
            return _extend_script(user_text[len(_EXTEND) :].split("|"), turn_results)
        if user_text.startswith(_UNPLANNABLE) and tools:
            if turn_results:
                return _done("gave up")
            reason = user_text[len(_UNPLANNABLE) :] or "no reason given"
            return _call("call-unplannable", "orbit_declare_unplannable", json.dumps({"reason": reason}))
        if user_text.startswith(_TWO) and tools:
            if turn_results:
                return _done("two-done")
            texts = user_text[len(_TWO) :].split("|")
            return _calls(
                [(f"call-two-{i}", "gated_echo", json.dumps({"text": text})) for i, text in enumerate(texts)]
            )
        if user_text.startswith(_SLOW) and tools:
            texts = user_text[len(_SLOW) :].split("|")
            done = len(turn_results)
            if done < len(texts):
                return _call(f"call-slow-{done}", "slow_echo", json.dumps({"text": texts[done]}))
            return _done("slowed")
        if user_text.startswith(_PLAN) and tools:
            return _plan_script(user_text[len(_PLAN) :].split("|"), turn_results)
        if user_text.startswith(_ASK) and tools and not turn_results:
            question = json.dumps({"question": user_text[len(_ASK) :]}, ensure_ascii=False)
            return _call("call-ask", "ask_user", question)
        if results or not tools:
            return _done("done" if results else "hello")
        lowered = user_text.lower()
        if "gated" in lowered:
            return _call("call-gated", "gated_echo", '{"text": "hello"}')
        return _done("hello")


def _plan_script(titles: list[str], results: list[ToolResultBlock]) -> ChatResponse:
    """TaskCreate for each title, then TaskUpdate to chain them, then stop."""

    created = [_created_id(block) for block in results if block.name == "TaskCreate"]
    chained = sum(1 for block in results if block.name == "TaskUpdate")
    if len(created) < len(titles):
        title = titles[len(created)].strip()
        return _call(
            f"call-create-{len(created)}",
            "TaskCreate",
            json.dumps({"subject": title, "description": title}),
        )
    if chained < len(created) - 1:
        args = {"task_id": created[chained + 1], "add_blocked_by": [created[chained]]}
        return _call(f"call-chain-{chained}", "TaskUpdate", json.dumps(args))
    return _done("planned")


def _extend_script(texts: list[str], results: list[ToolResultBlock]) -> ChatResponse:
    """slow_echo for each text; when the budget refuses one, ask for more budget once and carry on if it is granted."""

    def state(block: ToolResultBlock) -> str:
        return str(getattr(block.state, "value", block.state))

    slow = [block for block in results if block.name == "slow_echo"]
    extension = [block for block in results if block.name == "orbit_request_budget_extension"]
    done = sum(1 for block in slow if state(block) == "success")
    if extension and state(extension[-1]) != "success":
        return _done("no more budget")
    if done >= len(texts):
        return _done("all done")
    if any(state(block) == "denied" for block in slow) and not extension:
        return _call("call-extend", "orbit_request_budget_extension", json.dumps({"reason": "need more calls"}))
    return _call(f"call-ext-{len(results)}", "slow_echo", json.dumps({"text": texts[done]}))


def _created_id(block: ToolResultBlock) -> str:
    output = block.output if isinstance(block.output, str) else _last_output([block])
    return output.split()[1] if output.startswith("created ") else ""


def _calls(calls: list[tuple[str, str, str]]) -> ChatResponse:
    return ChatResponse(
        content=[ToolCallBlock(id=call_id, name=name, input=payload) for call_id, name, payload in calls],
        is_last=True,
        finished_reason=FinishedReason.COMPLETED,
    )


def _call(call_id: str, name: str, payload: str) -> ChatResponse:
    return ChatResponse(
        content=[ToolCallBlock(id=call_id, name=name, input=payload)],
        is_last=True,
        finished_reason=FinishedReason.COMPLETED,
    )


async def _stream(parts: list[str]) -> AsyncGenerator[ChatResponse, None]:
    # ORBIT_MOCK_STREAM_DELAY_MS spaces the parts out, so a test can watch a reply arrive.
    pause = MockSettings().stream_delay_ms / 1000
    for part in parts:
        if pause:
            await asyncio.sleep(pause)
        # One block id for every delta, as a provider streams one text block.
        yield ChatResponse(content=[TextBlock(id="mock-text", text=part)], is_last=False)


def _done(text: str) -> ChatResponse:
    return ChatResponse(
        content=[TextBlock(text=text)],
        is_last=True,
        finished_reason=FinishedReason.COMPLETED,
    )


def _tool_results(messages: list[Msg]) -> list[ToolResultBlock]:
    return _collect_results(messages)


def _turn_results(messages: list[Msg]) -> list[ToolResultBlock]:
    """Tool results after the latest user text. Earlier turns stay out of this list."""

    last_user = -1
    for index, message in enumerate(messages):
        if message.role != "user":
            continue
        if any(isinstance(block, TextBlock) for block in message.get_content_blocks()):
            last_user = index
    return _collect_results(messages[last_user + 1 :])


def _collect_results(messages: list[Msg]) -> list[ToolResultBlock]:
    found: list[ToolResultBlock] = []
    for message in messages:
        for block in message.get_content_blocks():
            if isinstance(block, ToolResultBlock):
                found.append(block)
    return found


def _last_output(results: list[ToolResultBlock]) -> str:
    if not results:
        return ""
    output = results[-1].output
    if isinstance(output, str):
        return output
    parts: list[str] = []
    for block in output:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "".join(parts)


_USER_MESSAGES = "\n\nUser messages:\n"


def _inspection_command(user_text: str) -> str:
    """The text of a "prompt:", "tools:" or "mcp:" command. After an interrupt the task's goal comes first and the
    person's message last (see agent_turn), so the command may be the last line rather than the start."""
    if user_text.startswith((_PROMPT, _TOOLS, _MCP)) or _USER_MESSAGES not in user_text:
        return user_text
    last = user_text.rsplit(_USER_MESSAGES, 1)[1].splitlines()[-1:]
    return last[0] if last and last[0].startswith((_PROMPT, _TOOLS, _MCP)) else user_text


def _system_text(messages: list[Msg]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.role != "system":
            continue
        parts.extend(b.text for b in message.get_content_blocks() if isinstance(b, TextBlock))
    return "\n".join(parts)


def _tool_names(tools: list[dict] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        name = (tool.get("function") or {}).get("name") or tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


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

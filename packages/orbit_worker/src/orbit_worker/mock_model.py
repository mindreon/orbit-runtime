"""A chat model with no network, so local runs need no API key.

Scripted behaviour, read from the conversation:

- a user message containing "gated" asks for the ``gated_echo`` tool;
- a message starting with "stream:" streams the rest back as provider
  deltas, one per part split at ``CHUNK_SEPARATOR``;
- a message starting with "echo:" asks for ``gated_echo`` with the rest;
- a message starting with "extend:" calls ``slow_echo`` for the "|"-separated texts, asking for more budget once
  when a call is refused;
- a conversation with a user message starting with "fail:" (or, for a retry, the rejection that this failure caused) fails
  every model request with a provider error (retryable), which is how a test sees a node retried, blocked, and the task
  wait for a person;
- a message starting with "unplannable:" declares the task unplannable with the rest as the reason;
- a message starting with "two:" asks for ``gated_echo`` twice in one step, one call per "|"-separated text;
- a message starting with "slow:" calls ``slow_echo`` once per "|"-separated text, one call per model round;
- a message starting with "file:" writes the file named before the first "|" into the workspace with the rest as its
  content, and "sh:" runs the rest as a shell command there (the ``Write`` and ``Bash`` tools; a relative path is
  under /workspace);
- a message starting with "chain:" runs the ";;"-separated steps one after the other, one tool call per model round, each
  step being a "file:", "sh:" or "slow:" command, and answers with what each one returned;
- a message starting with "think:<said>|<reasoning>|<answer>" says <said> and calls a tool, then answers
  "<reasoning></think><answer>": a model that leaks its reasoning into the reply and closes it with a tag nobody opened;
- a message starting with "reason:<thinking>|<answer>" streams <thinking> as thinking-block deltas and <answer> as
  answer deltas: a model that reports its reasoning in the provider's own field;
- a message starting with "plan:" creates one task per "|"-separated title, chained one after the other;
- a message starting with "ask:" asks the user the rest through ``ask_user``;
- a message starting with "prompt:" answers with the system prompt it was given, and "tools:" with the names of the
  tools it can call: how a test sees what an expert and a task configuration gave the agent;
- the leader of a team stage (it has the ``team_assign`` tool) whose goal starts with "team:" posts a note, assigns each
  ";;"-separated task to a member (one starting with "@<role> " goes to that member, the others to the members in turn),
  and when the answers are back says "team-final=" and what each one said; a goal starting with "team-loop:" assigns to
  the first member again after every answer and never finishes (how a test reaches the stage's limits);
- a member of a team stage (it has ``team_note`` and no ``team_assign``) answers "<role> did: <task>", and a task starting
  with "note:" posts the rest as a note for the team first; any of the other scripts below works as a member's task too;
- a member woken by an @mention (its input starts with "团队成员 <name> 在群里 @ 了你:") answers "<role> replied: <text>"; when the text holds
  "ping-pong" it first posts a note "@<name> ping-pong" back (a chain of wakes that only the hop limit stops); a note
  "note:@<role> <text>" is how a member @-mentions another;
- a message starting with "history:" answers with everything the user said in this conversation, which is how a test sees
  that a follow-up carried on its session;
- a message starting with "mcp:" calls the tool named before the first "|" with the JSON after it (an object);
- a verifier prompt (`verify_sop.VERIFIER_PREFIX`) is answered with PASS, except that a step named `flaky-<n>` is refused with
  `FAIL: ...` on the node's first n attempts (the prompt names the step and the attempt);
- when the agent must end in structured output (the ``GenerateStructuredOutput`` tool is there) it calls the tool with the JSON
  after "output:" if the goal starts with that, else with the smallest object that satisfies the schema;
- once a tool result is in context, the model answers and stops;
- anything else is a short text reply.
"""

import asyncio
import json
import re
from collections.abc import AsyncGenerator

from agentscope.credential import CredentialBase
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.message import Msg, TextBlock, ThinkingBlock, ToolCallBlock, ToolResultBlock
from agentscope.model import ChatModelBase, ChatResponse, ChatUsage, FinishedReason
from pydantic import BaseModel

from orbit_worker.settings import MockSettings
from orbit_worker.workspace import WORKSPACE_DIR

CHUNK_SEPARATOR = "\x1f"
_STREAM = "stream:"
_REASON = "reason:"
_ECHO = "echo:"
_ASK = "ask:"
_PLAN = "plan:"
_FILE = "file:"
_SH = "sh:"
_CHAIN = "chain:"
_THINK = "think:"
_SLOW = "slow:"
_TWO = "two:"
_UNPLANNABLE = "unplannable:"
_EXTEND = "extend:"
_PROMPT = "prompt:"
_TOOLS = "tools:"
_MCP = "mcp:"
_HISTORY = "history:"
_FAIL = "fail:"
_TEAM = "team:"
_TEAM_LOOP = "team-loop:"
_NOTE = "note:"
_WOKEN = "团队成员 "
_ASSIGN_TOOL = "team_assign"
_NOTE_TOOL = "team_note"
_MAILBOX = "\n\n团队消息"
_OUTPUT = "output:"
_STRUCTURED = "GenerateStructuredOutput"
# The start of the prompt `orbit_worker.verify_sop` gives a SOP step's verifier (kept here so the mock needs no import of it).
_VERIFIER = "You are the independent verifier of one step of a procedure."


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
        response = await self._answer(model_name, messages, tools, tool_choice, **kwargs)
        # ORBIT_MOCK_TOKENS_PER_CALL: a model that reports what it used, so a token budget has something to count.
        tokens = MockSettings().tokens_per_call
        if tokens and isinstance(response, ChatResponse):
            response.usage = ChatUsage(input_tokens=tokens, output_tokens=tokens, time=0.0)
        return response

    async def _answer(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: object | None = None,
        **kwargs: object,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        del model_name, tool_choice, kwargs
        user_text = _last_user_text(messages)
        if _scripted_failure(_user_texts(messages)):
            # Local import: chat_model imports this module.
            from orbit_worker.chat_model import ModelRequestError

            raise ModelRequestError("provider_error", "mock: scripted failure")
        if user_text.startswith(_VERIFIER):
            return _done(_verifier_verdict(user_text))
        if _STRUCTURED in _tool_names(tools):
            return _call("call-structured", _STRUCTURED, json.dumps(_structured_instance(user_text, tools)))
        results = _tool_results(messages)
        # Only this turn's tool results. A later "echo:" must still park even if
        # an earlier turn already ran a tool.
        turn_results = _turn_results(messages)
        names = _tool_names(tools)
        if _ASSIGN_TOOL in names:
            leading = _leader_script(user_text, tools, results, turn_results)
            if leading is not None:
                return leading
        if _NOTE_TOOL in names and _ASSIGN_TOOL not in names and user_text.startswith(_WOKEN):
            return _woken_script(user_text, messages, turn_results)
        if _NOTE_TOOL in names and _ASSIGN_TOOL not in names and user_text.startswith(_NOTE):
            text = user_text.removeprefix(_NOTE).split(_MAILBOX)[0]
            return _done(f"noted: {text}") if turn_results else _call("call-note", _NOTE_TOOL, json.dumps({"text": text}))
        if user_text.startswith(_STREAM):
            return _stream(user_text[len(_STREAM) :].split(CHUNK_SEPARATOR))
        if user_text.startswith(_REASON):
            thinking, _, answer = user_text[len(_REASON) :].partition("|")
            return _reason_stream(thinking.split(CHUNK_SEPARATOR), answer.split(CHUNK_SEPARATOR))
        if user_text.startswith(_ECHO) and tools and not turn_results:
            echoed = user_text[len(_ECHO) :]
            # echo:once stays call-echo. Any other payload gets its own call id,
            # so two agents can park without sharing apr-<call id>.
            call_id = "call-echo" if echoed == "once" else f"call-echo-{echoed}"
            payload = json.dumps({"text": echoed}, ensure_ascii=False)
            return _call(call_id, "gated_echo", payload)
        command = _inspection_command(user_text)
        if command.startswith(_HISTORY):
            return _done("history=" + " | ".join(_user_texts(messages)))
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
        if command.startswith(_THINK) and tools:
            said, _, rest = command[len(_THINK) :].partition("|")
            reasoning, _, answer = rest.partition("|")
            if not turn_results:
                return _say_and_call(said, "slow_echo", json.dumps({"text": "think"}))
            return _stream([reasoning, "</think>", answer], block="mock-answer")
        if command.startswith(_CHAIN) and tools:
            return _chain_script(command[len(_CHAIN) :].split(";;"), turn_results)
        if command.startswith(_FILE) and tools:
            if turn_results:
                return _done("file-result=" + _last_output(turn_results))
            path, _, content = command[len(_FILE) :].partition("|")
            path = path if path.startswith("/") else f"{WORKSPACE_DIR}/{path}"
            return _call("call-file", "Write", json.dumps({"file_path": path, "content": content}))
        if command.startswith(_SH) and tools:
            if turn_results:
                return _done("sh-result=" + _last_output(turn_results))
            return _call("call-sh", "Bash", json.dumps({"command": command[len(_SH) :]}))
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
        if _NOTE_TOOL in names and _ASSIGN_TOOL not in names:
            role = re.search(r"You are the member (\S+) of a team", _system_text(messages))
            return _done(f"{role.group(1) if role else 'member'} did: {user_text.split(_MAILBOX)[0]}")
        return _done("hello")


def _woken_script(user_text: str, messages: list[Msg], turn_results: list[ToolResultBlock]) -> ChatResponse:
    """A member woken by an @mention: it answers the group; a "ping-pong" text sends a note back to the one who woke it."""
    head, _, body = user_text.split(_MAILBOX)[0].partition(": ")
    sender = (re.search(r"团队成员 (\S+) 在群里", head) or re.search(r"(\S+)", "x")).group(1)  # type: ignore[union-attr]
    role = re.search(r"You are the member (\S+) of a team", _system_text(messages))
    name = role.group(1) if role else "member"
    if "ping-pong" in body and not turn_results:
        return _call("call-pong", _NOTE_TOOL, json.dumps({"text": f"@{sender} ping-pong"}))
    return _done(f"{name} replied: {body}")


def _team_roles(tools: list[dict] | None) -> list[str]:
    """The roles the leader may assign to: the choices of `team_assign`'s `member`."""
    for tool in tools or []:
        function = tool.get("function") or tool
        if function.get("name") == _ASSIGN_TOOL:
            schema = function.get("parameters") or function.get("input_schema") or {}
            return list(schema.get("properties", {}).get("member", {}).get("enum", []))
    return []


def _leader_script(
    user_text: str, tools: list[dict] | None, results: list[ToolResultBlock], turn_results: list[ToolResultBlock]
) -> ChatResponse | None:
    """The leader of a team stage: see the module docstring. None for a goal that is not one of its scripts."""
    roles = _team_roles(tools)
    if user_text.startswith(_TEAM_LOOP):
        answered = sum(1 for block in results if block.name == _ASSIGN_TOOL)
        return _call(f"call-team-loop-{answered}", _ASSIGN_TOOL, json.dumps({"member": roles[0], "task": f"again {answered}"}))
    if not user_text.startswith(_TEAM):
        return None
    answers = [block for block in turn_results if block.name == _ASSIGN_TOOL]
    if answers:
        return _done("team-final=" + " | ".join(_last_output([block]) for block in answers))
    items = [item.strip() for item in user_text.removeprefix(_TEAM).split(_MAILBOX)[0].split(";;")]
    calls = [("call-team-note", _NOTE_TOOL, json.dumps({"text": f"kickoff: {len(items)} task(s)"}))]
    for index, item in enumerate(items):
        member, task = (item[1:].split(" ", 1) + [""])[:2] if item.startswith("@") else (roles[index % len(roles)], item)
        calls.append((f"call-team-{index}", _ASSIGN_TOOL, json.dumps({"member": member, "task": task})))
    return _calls(calls)


def _structured_instance(user_text: str, tools: list[dict] | None) -> object:
    if user_text.startswith(_OUTPUT):
        return json.loads(user_text[len(_OUTPUT) :].splitlines()[0])
    for tool in tools or []:
        function = tool.get("function") or tool
        if function.get("name") == _STRUCTURED:
            return _minimal(function.get("parameters") or function.get("input_schema") or {})
    return {}


def _minimal(schema: dict) -> object:
    """The smallest value that satisfies the common shapes of a JSON Schema (enough for a test fixture)."""
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type")
    kind = next((k for k in kind if k != "null"), "null") if isinstance(kind, list) else kind
    if kind == "object" or (kind is None and "properties" in schema):
        props = schema.get("properties") or {}
        return {name: _minimal(props.get(name) or {}) for name in schema.get("required") or []}
    if kind == "array":
        return [_minimal(schema.get("items") or {})] * int(schema.get("minItems") or 0)
    if kind == "string":
        return "x" * int(schema.get("minLength") or 0)
    if kind in ("integer", "number"):
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    return None


def _verifier_verdict(prompt: str) -> str:
    """PASS, or FAIL for a `flaky-<n>` step on its first n attempts: how a test drives the retry path of a SOP step."""
    lines: dict[str, str] = {}
    for line in prompt.splitlines():
        if ": " in line:
            key, _, value = line.partition(": ")
            lines.setdefault(key, value)  # the first one: the description that follows may hold lines like these
    subject = lines.get("Step", "")
    attempt = int(lines.get("Attempt", "1") or 1) if lines.get("Attempt", "1").isdigit() else 1
    match = re.fullmatch(r"flaky-(\d*)", subject)
    if match and attempt <= int(match.group(1) or 1):
        return f"FAIL: {subject} was refused on attempt {attempt}"
    return "PASS"


def _scripted_failure(texts: list[str]) -> bool:
    """A "fail:" conversation. A failed first turn leaves no session behind, so a retry no longer has the goal; it is told
    why the attempt before it was rejected, and when that was this scripted failure the conversation is still the same one."""
    from orbit_worker.chat_model import FAILURE_MESSAGES

    rejected = "Your previous attempt was rejected: " + FAILURE_MESSAGES["provider_error"]
    return any(text.startswith((_FAIL, rejected)) for text in texts)


def _chain_script(steps: list[str], results: list[ToolResultBlock]) -> ChatResponse:
    """One tool call per step, in order; when the last one is back, what each of them returned."""
    done = len(results)
    if done >= len(steps):
        return _done("chain-result=" + " | ".join(_last_output([block]) for block in results))
    step = steps[done]
    if step.startswith(_FILE):
        path, _, content = step[len(_FILE) :].partition("|")
        path = path if path.startswith("/") else f"{WORKSPACE_DIR}/{path}"
        return _call(f"call-chain-{done}", "Write", json.dumps({"file_path": path, "content": content}))
    if step.startswith(_SH):
        return _call(f"call-chain-{done}", "Bash", json.dumps({"command": step[len(_SH) :]}))
    return _call(f"call-chain-{done}", "slow_echo", json.dumps({"text": step.removeprefix(_SLOW)}))


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


def _say_and_call(text: str, name: str, payload: str) -> ChatResponse:
    """A round that says something and then calls a tool, as a model does before it acts."""
    return ChatResponse(
        content=[TextBlock(id="mock-said", text=text), ToolCallBlock(id="call-said", name=name, input=payload)],
        is_last=True,
        finished_reason=FinishedReason.COMPLETED,
    )


async def _stream(parts: list[str], block: str = "mock-text") -> AsyncGenerator[ChatResponse, None]:
    # ORBIT_MOCK_STREAM_DELAY_MS spaces the parts out, so a test can watch a reply arrive.
    pause = MockSettings().stream_delay_ms / 1000
    for part in parts:
        if pause:
            await asyncio.sleep(pause)
        # One block id for every delta, as a provider streams one text block.
        yield ChatResponse(content=[TextBlock(id=block, text=part)], is_last=False)


async def _reason_stream(thinking: list[str], answer: list[str]) -> AsyncGenerator[ChatResponse, None]:
    """Thinking-block deltas first, then the answer as streamed text, so a test can watch both arrive."""
    pause = MockSettings().stream_delay_ms / 1000
    for part in thinking:
        if pause:
            await asyncio.sleep(pause)
        # One block id for every delta, as a provider streams one thinking block.
        yield ChatResponse(content=[ThinkingBlock(id="mock-think", thinking=part)], is_last=False)
    for part in answer:
        if pause:
            await asyncio.sleep(pause)
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
    if user_text.startswith((_PROMPT, _TOOLS, _MCP, _HISTORY, _FILE, _SH, _CHAIN, _THINK)) or _USER_MESSAGES not in user_text:
        return user_text
    last = user_text.rsplit(_USER_MESSAGES, 1)[1].splitlines()[-1:]
    return last[0] if last and last[0].startswith((_PROMPT, _TOOLS, _MCP, _HISTORY, _FILE, _SH, _CHAIN, _THINK)) else user_text


def _user_texts(messages: list[Msg]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if message.role == "user":
            texts.append("".join(b.text for b in message.get_content_blocks() if isinstance(b, TextBlock)))
    return texts


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

"""Shared helpers: a scripted chat model and counting tools (AgentScope 2.0.9)."""
import json
from collections import Counter

from agentscope.agent import Agent
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.credential import CredentialBase
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock, UserMsg
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit, ToolChunk
from agentscope.message import ToolResultState
from pydantic import BaseModel

EXEC = Counter()
MODEL_CALLS = Counter()


class _Cred(CredentialBase):
    @classmethod
    def get_chat_model_class(cls):
        return ScriptModel


class ScriptModel(ChatModelBase):
    """User text 'one' -> 1 call, 'two' -> 2 calls in one response.
    Once this turn has tool results -> final text 'done'."""

    class Parameters(BaseModel):
        pass

    def __init__(self):
        super().__init__(credential=_Cred(name="x"), model="script",
                         parameters=self.Parameters(), stream=False,
                         max_retries=0, context_size=32768)
        self.formatter = DeepSeekChatFormatter()

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kw):
        MODEL_CALLS["n"] += 1
        last_user, text = -1, ""
        for i, m in enumerate(messages):
            if m.role == "user" and any(isinstance(b, TextBlock) for b in m.get_content_blocks()):
                last_user, text = i, "".join(b.text for b in m.get_content_blocks() if isinstance(b, TextBlock))
        results = [b for m in messages[last_user + 1:] for b in m.get_content_blocks()
                   if isinstance(b, ToolResultBlock)]
        if results or not tools:
            return ChatResponse(content=[TextBlock(text="done")], is_last=True,
                                finished_reason=FinishedReason.COMPLETED)
        n = 2 if "two" in text else 1
        calls = [ToolCallBlock(id=f"call-{i}", name=f"tool_{'ab'[i]}",
                               input=json.dumps({"x": i})) for i in range(n)]
        return ChatResponse(content=calls, is_last=True, finished_reason=FinishedReason.COMPLETED)


def _mk(name):
    def f(x: int) -> ToolChunk:
        """Count one execution."""
        EXEC[name] += 1
        return ToolChunk(content=[TextBlock(text=f"{name} ran")], state=ToolResultState.SUCCESS)
    f.__name__ = name
    return f


async def build(state=None, mode=PermissionMode.BYPASS, middlewares=None, react_config=None):
    st = state or AgentState(permission_context=PermissionContext(mode=mode))
    kw = {}
    if react_config is not None:
        kw["react_config"] = react_config
    a = Agent(name="a", system_prompt="t", model=ScriptModel(), toolkit=Toolkit(),
              state=st, middlewares=middlewares or [], **kw)
    for n in ("tool_a", "tool_b"):
        await a.toolkit.add_tool(FunctionTool(_mk(n), name=n, description="count"))
    return a


def dump(agent):
    return agent.state.model_dump_json()


def load(blob):
    return AgentState.model_validate_json(blob)


def describe(state):
    out = []
    for m in state.context:
        blocks = []
        for b in m.content if isinstance(m.content, list) else []:
            t = type(b).__name__
            if isinstance(b, ToolCallBlock):
                blocks.append(f"call({b.id},{getattr(b.state, 'value', b.state)})")
            elif isinstance(b, ToolResultBlock):
                blocks.append(f"result({b.id},{getattr(b.state, 'value', b.state)})")
            else:
                blocks.append(t)
        out.append(f"{m.role}:{m.id[:8]}:{blocks}")
    return out

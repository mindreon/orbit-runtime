"""S4c: cancel while the tool itself is running (inside toolkit.call_tool)."""
import asyncio, sys, time
sys.path.insert(0, "/tmp/as_verify")
from agentscope.agent import ReActConfig
from agentscope.message import UserMsg, TextBlock
from agentscope.tool import FunctionTool, ToolChunk
from agentscope.message import ToolResultState
import common
from common import build, dump, load, describe, EXEC
def slow(name):
    async def f(x: int) -> ToolChunk:
        EXEC[name + "_start"] += 1
        await asyncio.sleep(3)
        EXEC[name] += 1
        return ToolChunk(content=[TextBlock(text=f"{name} ran")], state=ToolResultState.SUCCESS)
    f.__name__ = name; return f
async def run(raise_err):
    EXEC.clear()
    a = await build(react_config=ReActConfig(interruption_raise_cancelled_error=raise_err))
    for n in ("tool_a", "tool_b"):
        await a.toolkit.remove_tool(n)
        await a.toolkit.add_tool(FunctionTool(slow(n), name=n, description="count"))
    t = asyncio.create_task(a.reply(UserMsg(name="u", content="two")))
    await asyncio.sleep(0.5); t.cancel()
    try:
        r = await t; print("returned normally:", (r.get_text_content() or "")[:80])
    except asyncio.CancelledError:
        print("CancelledError propagated")
    print("EXEC", dict(EXEC), "unfinished", [x.id for x in a.state.get_unfinished_tool_calls(a.name)])
    print("ctx", describe(a.state))
    b = await build(state=load(dump(a)))
    await b.reply(UserMsg(name="u", content="hello"))
    print("next reply ok:", b.state.context[-1].get_text_content(), "EXEC", dict(EXEC))
for f in (True, False):
    print("=== raise =", f); asyncio.run(run(f))

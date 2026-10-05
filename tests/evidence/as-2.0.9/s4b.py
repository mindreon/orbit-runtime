"""S4b: cancel during a slow tool; log timing and whether hooks run."""
import asyncio, sys, time
sys.path.insert(0, "/tmp/as_verify")
from agentscope.agent import ReActConfig
from agentscope.middleware import MiddlewareBase
from agentscope.message import UserMsg
from common import build, dump, load, describe, EXEC
T0 = time.monotonic()
def log(*a): print(f"[{time.monotonic()-T0:5.2f}]", *a)
class Slow(MiddlewareBase):
    async def on_acting(self, agent, input_kwargs, next_handler):
        log("on_acting enter", list(input_kwargs)[:3])
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            log("on_acting got CancelledError"); raise
        async for x in next_handler(**input_kwargs): yield x
async def run(raise_err):
    EXEC.clear()
    a = await build(middlewares=[Slow()], react_config=ReActConfig(interruption_raise_cancelled_error=raise_err))
    t = asyncio.create_task(a.reply(UserMsg(name="u", content="two")))
    await asyncio.sleep(0.5); log("cancel"); t.cancel()
    try:
        r = await t; log("returned normally:", (r.get_text_content() or "")[:80])
    except asyncio.CancelledError:
        log("CancelledError propagated")
    log("EXEC", dict(EXEC), "unfinished", [x.id for x in a.state.get_unfinished_tool_calls(a.name)])
    log("ctx", describe(a.state))
    b = await build(state=load(dump(a)))
    await b.reply(UserMsg(name="u", content="hello"))
    log("next reply ok:", b.state.context[-1].get_text_content())
for f in (True, False):
    print("=== raise =", f); asyncio.run(run(f))

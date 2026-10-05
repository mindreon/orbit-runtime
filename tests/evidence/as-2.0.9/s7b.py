"""S7b: checkpoint after all tool results of a round are written (on_reasoning of the
next round, before next_handler), die, resume with reply(None). Usage: s7b.py crash|resume"""
import asyncio, os, sys
sys.path.insert(0, "/tmp/as_verify")
from agentscope.middleware import MiddlewareBase
from agentscope.message import UserMsg
from common import build, dump, load, describe, EXEC, MODEL_CALLS

BLOB = "/tmp/as_verify/s7b.json"


class DieAtSecondReasoning(MiddlewareBase):
    n = 0

    async def on_reasoning(self, agent, input_kwargs, next_handler):
        self.n += 1
        if self.n == 2:
            open(BLOB, "w").write(dump(agent))
            print("phase1 EXEC", dict(EXEC), "unfinished", len(agent.state.get_unfinished_tool_calls(agent.name)), flush=True)
            os._exit(9)
        async for x in next_handler(**input_kwargs):
            yield x


async def crash():
    a = await build(middlewares=[DieAtSecondReasoning()])
    await a.reply(UserMsg(name="u", content="two"))


async def resume():
    st = load(open(BLOB).read())
    print("ckpt ctx", describe(st))
    b = await build(state=st)
    await b.reply(None)
    print("resume EXEC", dict(EXEC), "model calls", MODEL_CALLS["n"])
    print("final ctx", describe(b.state))


asyncio.run(crash() if sys.argv[1] == "crash" else resume())

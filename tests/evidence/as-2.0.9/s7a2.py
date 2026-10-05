"""S7a-2: second crash during the resumed acting phase (after call-0 result, before call-1).
Usage: s7a2.py first | s7a2.py second | s7a2.py third"""
import asyncio, os, sys
sys.path.insert(0, "/tmp/as_verify")
from agentscope.middleware import MiddlewareBase
from agentscope.message import UserMsg
from common import build, dump, load, describe, EXEC

B1, B2 = "/tmp/as_verify/s7a2_1.json", "/tmp/as_verify/s7a2_2.json"


class DieOn(MiddlewareBase):
    def __init__(self, target, path):
        self.target, self.path = target, path

    async def on_acting(self, agent, input_kwargs, next_handler):
        if input_kwargs["tool_call"].id == self.target:
            open(self.path, "w").write(dump(agent))
            print("dying before", self.target, "EXEC", dict(EXEC), flush=True)
            os._exit(9)
        async for x in next_handler(**input_kwargs):
            yield x


async def first():
    a = await build(middlewares=[DieOn("call-0", B1)])
    await a.reply(UserMsg(name="u", content="two"))


async def second():
    b = await build(state=load(open(B1).read()), middlewares=[DieOn("call-1", B2)])
    await b.reply(None)


async def third():
    st = load(open(B2).read())
    print("ckpt2 ctx", describe(st))
    c = await build(state=st)
    await c.reply(None)
    print("third EXEC", dict(EXEC))
    print("final ctx", describe(c.state))


asyncio.run({"first": first, "second": second, "third": third}[sys.argv[1]]())

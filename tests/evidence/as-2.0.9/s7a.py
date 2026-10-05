"""S7a: checkpoint after reasoning, before acting (on_acting before next_handler);
the process then dies with os._exit (kill -9 stand-in). A second process resumes
with reply(None). Usage: s7a.py crash <schema:0|1> | s7a.py resume"""
import asyncio, json, os, sys
sys.path.insert(0, "/tmp/as_verify")
from pydantic import BaseModel
from agentscope.middleware import MiddlewareBase
from agentscope.message import UserMsg
from common import build, dump, load, describe, EXEC, MODEL_CALLS

BLOB = "/tmp/as_verify/s7a_ckpt.json"


class Out(BaseModel):
    answer: str


class CheckpointThenDie(MiddlewareBase):
    async def on_acting(self, agent, input_kwargs, next_handler):
        open(BLOB, "w").write(dump(agent))
        print("phase1 EXEC", dict(EXEC), "dying before", input_kwargs["tool_call"].id, flush=True)
        os._exit(9)
        yield  # pragma: no cover


async def crash(schema: bool):
    a = await build(middlewares=[CheckpointThenDie()])
    kw = {"structured_schema": Out} if schema else {}
    await a.reply(UserMsg(name="u", content="two"), **kw)


async def resume():
    st = load(open(BLOB).read())
    print("ckpt reply_id", st.reply_id[:8], "cur_iter", st.cur_iter,
          "schema", st.reply_context.structured_schema is not None)
    print("ckpt ctx", describe(st))
    old = st.reply_id
    b = await build(state=st)
    kinds = []
    async for ev in b.reply_stream(None):
        kinds.append(type(ev).__name__)
    print("resume: reply_id changed", b.state.reply_id != old, "cur_iter", b.state.cur_iter,
          "schema", b.state.reply_context.structured_schema is not None)
    print("resume EXEC", dict(EXEC), "model calls", MODEL_CALLS["n"])
    print("final ctx", describe(b.state))
    # can the context be fed to the model again (formatter accepts it)?
    from agentscope.message import UserMsg as U
    await b.reply(U(name="u", content="hello again"))
    print("next reply ok; last text:", b.state.context[-1].get_text_content())


if sys.argv[1] == "crash":
    asyncio.run(crash(sys.argv[2] == "1"))
else:
    asyncio.run(resume())

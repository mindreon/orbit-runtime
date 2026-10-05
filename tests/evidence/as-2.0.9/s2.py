"""S2: two calls parked on ASKING (DEFAULT mode), process 1 saves state and exits;
process 2 approves call-0 and rejects call-1. Usage: s2.py park|decide"""
import asyncio, sys
sys.path.insert(0, "/tmp/as_verify")
from agentscope.event import UserConfirmResultEvent, ConfirmResult, RequireUserConfirmEvent
from agentscope.message import UserMsg
from agentscope.permission import PermissionMode
from common import build, dump, load, describe, EXEC

BLOB = "/tmp/as_verify/s2.json"


async def park():
    a = await build(mode=PermissionMode.DEFAULT)
    asks = []
    async for ev in a.reply_stream(UserMsg(name="u", content="two")):
        if isinstance(ev, RequireUserConfirmEvent):
            asks += [t.id for t in ev.tool_calls]
    print("asks", asks, "EXEC", dict(EXEC), "awaiting",
          [(t.id, t.state) for t in a.state.get_awaiting_tool_calls(a.name)])
    open(BLOB, "w").write(dump(a))


async def decide():
    st = load(open(BLOB).read())
    b = await build(state=st, mode=PermissionMode.DEFAULT)
    calls = {t.id: t for t in st.get_awaiting_tool_calls("a")}
    ev = UserConfirmResultEvent(reply_id=st.reply_id, confirm_results=[
        ConfirmResult(confirmed=True, tool_call=calls["call-0"]),
        ConfirmResult(confirmed=False, tool_call=calls["call-1"]),
    ])
    await b.reply(ev)
    print("decide EXEC", dict(EXEC))
    print("final ctx", describe(b.state))
    for m in b.state.context:
        for blk in m.get_content_blocks("tool_result"):
            print("result", blk.id, blk.state, str(blk.output)[:80])
    # replaying the same decision must be rejected (nothing awaiting)
    try:
        c = await build(state=load(b.state.model_dump_json()), mode=PermissionMode.DEFAULT)
        await c.reply(ev)
        print("replay accepted?! EXEC", dict(EXEC))
    except ValueError as e:
        print("replay rejected:", str(e)[:90])


asyncio.run(park() if sys.argv[1] == "park" else decide())

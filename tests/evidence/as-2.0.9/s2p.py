"""S2-partial: decide only call-0; call-1 stays ASKING. Then decide call-1 in a third process."""
import asyncio, sys
sys.path.insert(0, "/tmp/as_verify")
from agentscope.event import UserConfirmResultEvent, ConfirmResult, RequireUserConfirmEvent
from agentscope.permission import PermissionMode
from common import build, dump, load, describe, EXEC
BLOB, B2 = "/tmp/as_verify/s2.json", "/tmp/as_verify/s2p.json"
async def one():
    st = load(open(BLOB).read()); c = {t.id: t for t in st.get_awaiting_tool_calls("a")}
    b = await build(state=st, mode=PermissionMode.DEFAULT)
    asks = []
    async for ev in b.reply_stream(UserConfirmResultEvent(reply_id=st.reply_id, confirm_results=[ConfirmResult(confirmed=True, tool_call=c["call-0"])])):
        if isinstance(ev, RequireUserConfirmEvent): asks += [t.id for t in ev.tool_calls]
    print("after partial EXEC", dict(EXEC), "re-asked", asks, "awaiting", [t.id for t in b.state.get_awaiting_tool_calls("a")])
    print("ctx", describe(b.state)); open(B2, "w").write(dump(b))
async def two():
    st = load(open(B2).read()); c = {t.id: t for t in st.get_awaiting_tool_calls("a")}
    b = await build(state=st, mode=PermissionMode.DEFAULT)
    await b.reply(UserConfirmResultEvent(reply_id=st.reply_id, confirm_results=[ConfirmResult(confirmed=True, tool_call=c["call-1"])]))
    print("after second EXEC", dict(EXEC)); print("ctx", describe(b.state))
asyncio.run(one() if sys.argv[1] == "one" else two())

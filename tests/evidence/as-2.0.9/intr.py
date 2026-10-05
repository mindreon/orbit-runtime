import asyncio, sys
sys.path.insert(0, "/tmp/as_verify")
from agentscope.event import UserInterruptEvent
from agentscope.permission import PermissionMode
from agentscope.message import UserMsg
from common import build, dump, load, describe, EXEC
async def main():
    st = load(open("/tmp/as_verify/s2.json").read())
    a = await build(state=st, mode=PermissionMode.DEFAULT)
    print("awaiting before", [c.id for c in a.state.get_awaiting_tool_calls(a.name)])
    r = await a.reply(UserInterruptEvent(reply_id=a.state.reply_id))
    print("reply:", (r.get_text_content() or "")[:80])
    print("EXEC", dict(EXEC), "awaiting after", [c.id for c in a.state.get_awaiting_tool_calls(a.name)])
    print("ctx", describe(a.state))
    b = await build(state=load(dump(a)), mode=PermissionMode.BYPASS)
    await b.reply(UserMsg(name="u", content="new direction"))
    print("next reply ok:", b.state.context[-1].get_text_content())
asyncio.run(main())

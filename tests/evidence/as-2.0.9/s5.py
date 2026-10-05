"""S5: s5.py dump (run in 2.0.8) -> writes parked AgentState + SOPRunState; s5.py load (2.0.9) resumes."""
import asyncio, sys, json
sys.path.insert(0, "/tmp/as_verify")
import agentscope
from agentscope.permission import PermissionMode
async def main():
    if sys.argv[1] == "dump":
        from agentscope.message import UserMsg
        from common import build, dump
        a = await build(mode=PermissionMode.DEFAULT)
        await a.reply(UserMsg(name="u", content="two"))
        open("s5_state.json","w").write(dump(a))
        print(agentscope.__version__, "dumped; awaiting", [c.id for c in a.state.get_awaiting_tool_calls(a.name)])
    else:
        from agentscope.event import UserConfirmResultEvent, ConfirmResult
        from common import build, load, describe, EXEC
        st = load(open("s5_state.json").read())
        a = await build(state=st, mode=PermissionMode.DEFAULT)
        calls = a.state.get_awaiting_tool_calls(a.name)
        ev = UserConfirmResultEvent(reply_id=a.state.reply_id, confirm_results=[ConfirmResult(confirmed=True, tool_call=c) for c in calls])
        await a.reply(ev)
        print(agentscope.__version__, "resumed; EXEC", dict(EXEC)); print("ctx", describe(a.state))
asyncio.run(main())

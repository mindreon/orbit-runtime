import asyncio, sys, json
sys.path.insert(0, "/tmp/as_verify")
from agentscope.formatter import OpenAIChatFormatter
from common import load
async def main():
    for p in ["s7a2_2.json", "s7a_ckpt.json"]:
        st = load(open(p).read())
        msgs = await OpenAIChatFormatter().format(st.context)
        for m in msgs:
            print(m["role"], [t["id"] for t in m.get("tool_calls", [])], m.get("tool_call_id"))
asyncio.run(main())

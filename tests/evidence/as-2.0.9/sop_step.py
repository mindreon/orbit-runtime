"""SOP per-step: run step 1 until SOP_STEP_ENDED(COMPLETED), aclose, persist, resume."""
import asyncio, sys
from collections import Counter
from agentscope.sop import SOP, SOPEngine
from agentscope.sop._schema import SOPStepBase
from agentscope.sop._state import SOPPhase, SOPRunState
from agentscope.message import UserMsg, TextBlock
RUNS = Counter()
class Step(SOPStepBase):
    async def reply_stream(self, inputs, state):
        RUNS[self.subject] += 1
        state.phase = SOPPhase.RUNNING
        state.submission = [TextBlock(text=f"{self.subject} done")]
        self.record(state, passed=True, verifier="t")
        state.phase = SOPPhase.COMPLETED
        yield UserMsg(name="x", content=f"{self.subject} out")
def sop(): return SOP("s", [Step("s1", "d"), Step("s2", "d"), Step("s3", "d")])
async def main():
    eng = SOPEngine(sop())
    gen = eng.reply_stream(UserMsg(name="u", content="go"))
    async for ev in gen:
        name = getattr(ev, "name", None)
        if name == "SOP_STEP_ENDED":
            print("step ended", ev.value); await gen.aclose(); break
    blob = eng.state.model_dump_json()
    print("after step1:", RUNS, [s.phase.value for s in eng.state.steps])
    eng2 = SOPEngine(sop(), SOPRunState.model_validate_json(blob))
    gen2 = eng2.reply_stream(None)
    async for ev in gen2:
        if getattr(ev, "name", None) == "SOP_STEP_ENDED":
            print("step ended", ev.value); await gen2.aclose(); break
    print("after step2:", RUNS, [s.phase.value for s in eng2.state.steps])
    print("step2 given:", [m.get_text_content() for m in eng2.state.steps[1].given])
asyncio.run(main())

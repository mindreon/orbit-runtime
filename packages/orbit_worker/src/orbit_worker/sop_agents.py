"""Running an SOP with AgentScope's `SOPEngine` (06 §2).

An SOP is an ordered list of steps; the engine walks them, spends each step's attempt budget and records verdicts in a
`SOPRunState`. Orbit runs it one try at a time: an activity feeds the engine the run state it was given, lets exactly one
try of one step happen, and hands the state back, so every try has its own timeout, heartbeat and checkpoint.

One try of a step is two independent sessions: the executor does the work, then a verifier that has not seen how it was
done judges only the result. A refusal carries the verifier's reason, which the next try's executor is given.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from agentscope.message import Msg, TextBlock, UserMsg
from agentscope.sop import SOP, SOPEngine
from agentscope.sop._schema import SOPStepBase
from agentscope.sop._state import SOPPhase, SOPRunState, SOPStepRunState
from orbit_contracts.models import OpenSessionInput, RunTurnInput

from orbit_worker.sop import Step, Verdict, verify_step

_VERDICT_PROMPT = (
    "You are the verifier of one step of a procedure. Judge only whether the work below achieves the step. "
    "Answer with PASS, or with FAIL: followed by exactly what is wrong and what to change.\n\n"
    "Step: {step}\n\nWork:\n{work}"
)


def parse_verdict(answer: str) -> Verdict:
    """`PASS` accepts; anything else refuses, `FAIL: <why>` giving the reason. Ambiguity refuses: a step the verifier
    did not clearly accept did not pass."""
    text = answer.strip()
    if text.upper().startswith("PASS"):
        return Verdict(True)
    reason = (
        text.split(":", 1)[1].strip() if text.upper().startswith("FAIL") and ":" in text else text
    )
    return Verdict(False, reason or "the verifier gave no reason")


@dataclass(frozen=True)
class RunScope:
    """Where a try runs: the attempt it belongs to, and whether a model or the scripted mock judges it."""

    runtime: Any
    task_id: str
    attempt_id: str
    mock: bool


class OrbitStep(SOPStepBase):
    """A step whose executor and verifier are agent sessions of this worker."""

    def __init__(self, step: Step, scope: RunScope) -> None:
        super().__init__(
            subject=step.subject, description=step.description, max_attempts=step.max_attempts
        )
        self._scope = scope

    async def reply_stream(self, inputs: Any, state: SOPStepRunState) -> AsyncGenerator[Msg, None]:
        tries = len(state.verifications) + 1
        state.phase = SOPPhase.RUNNING
        refused = (
            state.verifications[-1].message
            if state.verifications and not state.verifications[-1].passed
            else ""
        )
        if self._scope.mock:
            work, verdict = f"{self.subject} done", verify_step(self.subject, tries, mock=True)
        else:
            work = await self._execute(inputs, refused, tries)
            verdict = parse_verdict(await self._ask(f"verify:{tries}", self._judge_prompt(work)))
        state.submission = [TextBlock(text=work)]
        self.record(
            state, passed=verdict.passed, message=verdict.message, verifier="orbit-verifier"
        )
        yield UserMsg(name="sop", content=work)

    async def _execute(self, inputs: Any, refused: str, tries: int) -> str:
        prompt = (
            f"Do this step of a procedure: {self.subject}.\nIt must achieve: {self.description}"
        )
        handed = _text(inputs)
        if handed:
            prompt += f"\nWhat you are given:\n{handed}"
        if refused:
            prompt += f"\nYour previous try was refused: {refused}\nFix exactly that."
        return await self._ask(f"exec:{tries}", prompt)

    def _judge_prompt(self, work: str) -> str:
        return _VERDICT_PROMPT.format(step=f"{self.subject}: {self.description}", work=work)

    async def _ask(self, label: str, message: str) -> str:
        scope = self._scope
        turn_id = f"{scope.attempt_id}:sop-{label}:{self.subject}"
        opened = await scope.runtime.open_session(
            OpenSessionInput(
                room_id=scope.task_id,
                turn_id=f"{turn_id}:open",
                permission_preset="workspace-write",
            )
        )
        result = await scope.runtime.run_turn(
            RunTurnInput(
                room_id=scope.task_id,
                session_id=opened.session_id,
                turn_id=turn_id,
                message=message,
                state_version=opened.state_version,
            )
        )
        if result.status == "needs_external" and result.external is not None:
            # A step has nobody to ask: an executor that stops to ask has not done the work, and the verifier says so.
            return f"(the executor stopped to ask the user: {result.external.arguments.get('question', '')})"
        if result.status != "completed":
            raise RuntimeError(f"SOP agent turn ended as {result.status}: {result.error}")
        return result.text


def _text(messages: Any) -> str:
    if not messages:
        return ""
    items = messages if isinstance(messages, list) else [messages]
    return "\n".join(item.get_text_content() or "" for item in items if isinstance(item, Msg))


@dataclass(frozen=True)
class TryOutcome:
    """What one try left behind: the run state to carry on with, the step it was, and how the SOP stands."""

    run_state: str
    step_index: int  # 1-based
    tries: int
    status: str  # "continue" | "completed" | "failed"
    message: str = ""


async def run_one_try(
    steps: tuple[Step, ...], goal: str, run_state: str, scope: RunScope
) -> TryOutcome:
    """Lets the engine make exactly one try of the step that is due, then hands its state back."""
    sop = SOP("orbit", [OrbitStep(step, scope) for step in steps])
    engine = SOPEngine(sop, SOPRunState.model_validate_json(run_state) if run_state else None)
    due = next(
        i for i, record in enumerate(engine.state.steps) if record.phase is not SOPPhase.COMPLETED
    )
    stream = engine.reply_stream(UserMsg(name="user", content=goal) if not run_state else None)
    async for event in stream:
        if getattr(event, "name", None) == "SOP_STEP_ENDED":
            await (
                stream.aclose()
            )  # one try per activity: the engine's next loop turn belongs to the next activity
            break
    return _outcome(engine, steps, due)


def _outcome(engine: SOPEngine, steps: tuple[Step, ...], due: int) -> TryOutcome:
    """How the SOP stands after the try that step `due` (0-based) just had."""
    record = engine.state.steps[due]
    tries = len(record.verifications)
    if all(item.phase is SOPPhase.COMPLETED for item in engine.state.steps):
        status, message = "completed", ""
    elif record.phase is SOPPhase.PENDING and tries >= steps[due].max_attempts:
        # The engine spends the attempt budget only when its own loop goes on; a caller that stops after each try
        # has to do it, and the engine reads FAILED as the end of the run.
        record.phase = SOPPhase.FAILED
        status, message = "failed", f"step {due + 1} was refused {tries} times"
    else:
        status, message = "continue", ""
    return TryOutcome(engine.state.model_dump_json(), due + 1, tries, status, message)

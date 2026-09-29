"""The executor and the verifier of one SOP step, with a real model (06 §2).

One try of a step is two independent sessions: the executor does the work, then a verifier that has not seen how it
was done judges only the result. A refusal carries the verifier's reason, which the next try's executor is given.
"""

from __future__ import annotations

from orbit_contracts.models import OpenSessionInput, RunTurnInput

from orbit_worker.sop import Step, Verdict

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


async def run_step(
    runtime, *, tenant_id: str, task_id: str, attempt_id: str, step: Step, tries: int, feedback: str
) -> tuple[str, Verdict]:
    """Runs one try of `step` and returns the executor's work and the verifier's verdict on it."""
    prompt = f"Do this step of a procedure: {step.subject}.\nIt must achieve: {step.description}"
    if feedback:
        prompt += f"\nYour previous try was refused: {feedback}\nFix exactly that."
    work = await _ask(runtime, task_id, f"{attempt_id}:sop-exec:{step.subject}:{tries}", prompt)
    answer = await _ask(
        runtime,
        task_id,
        f"{attempt_id}:sop-verify:{step.subject}:{tries}",
        _VERDICT_PROMPT.format(step=f"{step.subject}: {step.description}", work=work),
    )
    return work, parse_verdict(answer)


async def _ask(runtime, task_id: str, turn_id: str, message: str) -> str:
    opened = await runtime.open_session(
        OpenSessionInput(
            room_id=task_id, turn_id=f"{turn_id}:open", permission_preset="workspace-write"
        )
    )
    result = await runtime.run_turn(
        RunTurnInput(
            room_id=task_id,
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

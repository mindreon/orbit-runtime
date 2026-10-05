"""SOP definitions the `sop_step` activity runs (06 §2).

DEPRECATED with the old SOP path: a `sop_stage` node is compiled into the plan now (`orbit_orch.task_sop`, `orbit_contracts.v3.sop`).
This stays for attempts that were running on the old path and for replay.

An SOP is an ordered list of step names registered in control (`sop_definitions`); a version is immutable, so a
definition can be cached for the life of the process. The workflow records each completed step in its history, so a
step is never run twice; the activity only ever runs the one step it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class UnknownSopError(LookupError):
    pass


@dataclass(frozen=True)
class Step:
    """One milestone of an SOP: what to call it, what it must achieve, how many refusals it takes."""

    subject: str
    description: str
    max_attempts: int = 3

    @classmethod
    def parse(cls, raw: Any) -> Step:
        if isinstance(raw, str):
            return cls(subject=raw, description=raw)
        subject = str(raw["subject"])
        return cls(
            subject=subject,
            description=str(raw.get("description") or subject),
            max_attempts=int(raw.get("max_attempts") or 3),
        )


class SopSource(Protocol):
    async def get_sop_steps(self, *, tenant_id: str, sop_ref: str) -> tuple[Step, ...] | None: ...


class SopRegistry:
    def __init__(self, source: SopSource) -> None:
        self._source = source
        self._cache: dict[tuple[str, str], tuple[Step, ...]] = {}

    async def steps_of(self, tenant_id: str, sop_ref: str) -> tuple[Step, ...]:
        key = (tenant_id, sop_ref)
        if key not in self._cache:
            steps = await self._source.get_sop_steps(tenant_id=tenant_id, sop_ref=sop_ref)
            if steps is None:
                raise UnknownSopError(f"unknown SOP {sop_ref}")
            self._cache[key] = steps
        return self._cache[key]


class Verdict:
    """A verifier's judgement of one attempt at a step. A refusal carries what to fix, handed to the next attempt."""

    def __init__(self, passed: bool, message: str = "") -> None:
        self.passed = passed
        self.message = message


def verify_step(step_name: str, attempt_no: int, *, mock: bool) -> Verdict:
    """Judges one attempt. With the mock model a step named `flaky-<n>` is refused on its first n attempts, so a
    test can drive the retry path; a real verifier is the model's structured verdict and lands with the model
    wiring."""
    if mock and step_name.startswith("flaky-"):
        refusals = int(step_name.removeprefix("flaky-") or "1")
        if attempt_no <= refusals:
            return Verdict(False, f"{step_name} was refused on attempt {attempt_no}")
    return Verdict(True)

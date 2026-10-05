"""SOP definition v2 (06 §2, 05 §5): the procedure a `sop_stage` node names as `sop_id@version`.

An SOP is registered in control and is immutable per version. When a `sop_stage` node becomes schedulable the
TaskWorkflow does not run an engine: it compiles the definition into a subgraph of the plan (one `agent_turn` node per
step, approval nodes for `human_approval`, edges from `depends_on`), so every step runs the full task path: sandbox
workspace, task configuration, policy, approvals, streaming and budget.

Mapping to AgentScope 2.0.9 (`agentscope.sop`), whose names the definition follows where the concept matches:

| Orbit field | AgentScope | Note |
| --- | --- | --- |
| `name`, `description`, `steps` | `SOP(name, steps, description)` | the registry adds `sop_id` and `version` around them |
| step `subject`, `description`, `max_attempts` | `SOPStep(subject, description, max_attempts)` | the same meaning; `max_attempts` becomes the node's `retry.max_attempts` |
| step `executor` | `SOPStep.executor` | a profile ref (`expert@3`), not an Agent object: it becomes the node's `owner_profile`; None is the task's expert |
| step `verifier` | `SOPStep.verifier` | `{instructions, expert}` and not an Agent object; None is the default verifier (it still judges the step) |
| step `id`, `depends_on`, `output_schema_ref`, `required_artifacts`, `human_approval` | none | Orbit extensions: AgentScope's steps are a linear list |

A v1 entry (`{subject, description, max_attempts}` or a bare string) is a valid v2 step: it has no `id` (the position
names it: `s1`, `s2`, ...) and no `depends_on` (it depends on the step before it), so v1 definitions stay linear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from orbit_contracts.v3.common import ContractModel, VersionedRef
from orbit_contracts.v3.nodes import ArtifactRequirement

MAX_SOP_STEPS = 50
MAX_STEP_ATTEMPTS = 20
STEP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HumanApproval = Literal["before", "after"]


class SopVerifier(ContractModel):
    """Who judges a step and by what: extra criteria on top of the step's own description. A verifier is an independent
    agent session of its own: it does not see how the executor worked, only the result."""

    instructions: str = Field(default="", max_length=8000)
    # The expert whose instructions the verifier is given; None is the default verifier.
    expert: VersionedRef | None = None


class SopStep(ContractModel):
    id: str | None = Field(default=None, pattern=STEP_ID.pattern)
    subject: str = Field(min_length=1, max_length=200)
    # What the step must achieve. Left out, it is the subject.
    description: str = Field(default="", max_length=8000)
    max_attempts: int = Field(default=3, ge=1, le=MAX_STEP_ATTEMPTS)
    executor: VersionedRef | None = None
    verifier: SopVerifier | None = None
    # The steps this one waits for. None is the step before it (linear); an empty list is a step that starts at once.
    depends_on: list[str] | None = None
    output_schema_ref: str | None = Field(default=None, pattern=r"^schema://\S+/[1-9][0-9]*$")
    required_artifacts: list[ArtifactRequirement] = Field(default_factory=list)
    # A person approves before the step starts, or after it finished.
    human_approval: HumanApproval | None = None

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        """A bare string is a v1 step whose subject is that text."""
        return {"subject": value} if isinstance(value, str) else value


class SopDefinition(ContractModel):
    """What control stores for `sop_id@version`, and what the workflow compiles."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=8000)
    steps: list[SopStep] = Field(min_length=1, max_length=MAX_SOP_STEPS)

    @model_validator(mode="after")
    def _is_a_dag(self) -> SopDefinition:
        resolve_steps(self.steps)
        return self


@dataclass(frozen=True)
class ResolvedStep:
    """A step with every default applied: its id, what it waits for, and its description."""

    id: str
    index: int  # 1-based position in the definition
    subject: str
    description: str
    max_attempts: int
    executor: str | None
    verifier_instructions: str
    verifier_expert: str | None
    depends_on: tuple[str, ...]
    output_schema_ref: str | None
    required_artifacts: tuple[ArtifactRequirement, ...]
    human_approval: str | None


def resolve_steps(steps: list[SopStep] | tuple[SopStep, ...]) -> list[ResolvedStep]:
    """The steps with defaults applied, in an order where each step comes after the ones it depends on (a stable
    topological order: definition order wins among steps that are ready). Raises ValueError for an empty list, too many
    steps, a duplicate id, an unknown or self dependency or a cycle."""
    if not steps:
        raise ValueError("an SOP needs at least one step")
    if len(steps) > MAX_SOP_STEPS:
        raise ValueError(f"an SOP has at most {MAX_SOP_STEPS} steps")
    explicit = [step.id for step in steps if step.id]
    if len(set(explicit)) != len(explicit):
        raise ValueError("step ids must be unique")
    taken = set(explicit)
    ids: list[str] = []
    for position, step in enumerate(steps, start=1):
        ids.append(step.id or f"s{position}")
        if not step.id and ids[-1] in taken:
            raise ValueError(f"step {position} has no id and its default id {ids[-1]} is taken")
        taken.add(ids[-1])
    known = set(ids)
    resolved: dict[str, ResolvedStep] = {}
    for position, (step_id, step) in enumerate(zip(ids, steps, strict=True), start=1):
        if step.depends_on is None:
            depends = (ids[position - 2],) if position > 1 else ()
        else:
            depends = tuple(dict.fromkeys(step.depends_on))
        for dependency in depends:
            if dependency not in known:
                raise ValueError(f"step {step_id} depends on unknown step {dependency}")
            if dependency == step_id:
                raise ValueError(f"step {step_id} depends on itself")
        resolved[step_id] = ResolvedStep(
            id=step_id,
            index=position,
            subject=step.subject.strip(),
            description=(step.description or step.subject).strip(),
            max_attempts=step.max_attempts,
            executor=step.executor,
            verifier_instructions=(step.verifier.instructions if step.verifier else "").strip(),
            verifier_expert=step.verifier.expert if step.verifier else None,
            depends_on=depends,
            output_schema_ref=step.output_schema_ref,
            required_artifacts=tuple(step.required_artifacts),
            human_approval=step.human_approval,
        )
        if not resolved[step_id].subject:
            raise ValueError(f"step {step_id} needs a subject")
    ordered: list[ResolvedStep] = []
    done: set[str] = set()
    while len(ordered) < len(ids):
        ready = next(
            (resolved[i] for i in ids if i not in done and all(d in done for d in resolved[i].depends_on)), None
        )
        if ready is None:
            raise ValueError("the steps' dependencies contain a cycle")
        ordered.append(ready)
        done.add(ready.id)
    return ordered

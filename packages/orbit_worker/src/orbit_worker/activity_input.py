"""Validation of activity payloads at the activity boundary (17 G9): a payload that does not fit its model is a
programming error or a version skew, not a transient failure, so it fails the activity without retries."""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from temporalio.exceptions import ApplicationError

Model = TypeVar("Model", bound=BaseModel)

INVALID_INPUT = "invalid_activity_input"


def parse_input(model: type[Model], payload: Any) -> Model:
    """`payload` as `model`, or a non-retryable ApplicationError naming the fields that are wrong."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        fields = ", ".join(
            ".".join(str(part) for part in error["loc"]) or "<payload>" for error in exc.errors()
        )
        raise ApplicationError(
            f"{model.__name__}: invalid {fields}", type=INVALID_INPUT, non_retryable=True
        ) from None


class CheckpointCommitInput(BaseModel):
    """Everything a checkpoint is stored under; nothing is defaulted (17 G9). The kinds are the ones the
    `checkpoints.kind` constraint allows."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    kind: Literal["agent_state", "sop_run_state", "workspace_snapshot", "plan", "team_snapshot"]

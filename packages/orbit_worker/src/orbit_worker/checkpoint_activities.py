"""The commit half of the checkpoint protocol (08 §3), on `orbit.io`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from temporalio import activity

from orbit_worker.activity_input import parse_input
from orbit_worker.task_activities import get_task_store


class CommitCheckpointsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    # The digest the attempt's activity result names, when it names one.
    checkpoint_ref: str | None = None


@activity.defn(name="commit_checkpoints")
async def commit_checkpoints(payload: dict[str, Any]) -> dict[str, Any]:
    """Called by AttemptWorkflow once an activity result is in its history: the checkpoints that result refers to
    are now referenced by history, so the GC keeps them (17 G1)."""
    request = parse_input(CommitCheckpointsInput, payload)
    committed = await get_task_store().commit_checkpoints(
        tenant_id=request.tenant_id,
        attempt_id=request.attempt_id,
        checkpoint_ref=request.checkpoint_ref,
    )
    return {"ok": True, "committed": committed}


CHECKPOINT_ACTIVITIES = [commit_checkpoints]

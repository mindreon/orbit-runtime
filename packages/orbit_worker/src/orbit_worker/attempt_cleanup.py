"""Correct the projection of attempts whose AttemptWorkflow is gone (04 §6, 15 T4.5, 17 G2).

`stage_attempts` is a projection: an attempt parked on an approval or on a user's reply stays STARTING or RUNNING for
days and is perfectly healthy. So age alone never decides anything. Temporal does: only a row whose AttemptWorkflow no
longer exists, or has closed, is stale, and then it is moved to a terminal status with the reason recorded. A row is
never deleted, it is history."""

from __future__ import annotations

from typing import Literal, Protocol

from orbit_contracts.v3 import Failure
from orbit_orch.plan_engine import attempt_workflow_id
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.service import RPCError, RPCStatusCode

from orbit_worker.task_store import Closure, StaleAttempt, TaskStore

WorkflowState = Literal["open", "missing", "completed", "closed"]

# A continued-as-new run is followed by the next run of the same id, so it is still open.
_OPEN = frozenset({WorkflowExecutionStatus.RUNNING, WorkflowExecutionStatus.CONTINUED_AS_NEW})


class AttemptProbe(Protocol):
    """What Temporal says about the AttemptWorkflow with this id: its state, and a short detail for the record."""

    async def state(self, workflow_id: str) -> tuple[WorkflowState, str]: ...


class TemporalAttemptProbe:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def state(self, workflow_id: str) -> tuple[WorkflowState, str]:
        try:
            description = await self._client.get_workflow_handle(workflow_id).describe()
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return "missing", "not found"
            raise
        status = description.status
        name = status.name if status is not None else "UNKNOWN"
        if status in _OPEN:
            return "open", name
        return ("completed" if status == WorkflowExecutionStatus.COMPLETED else "closed"), name


def closure_for(attempt: StaleAttempt, state: WorkflowState, detail: str) -> Closure | None:
    """How to close out a stale row, or None when its workflow is still open."""
    if state == "open":
        return None
    if state == "closed":
        # Failed, cancelled, terminated or timed out: the attempt ended without handing over.
        status, message = "ABORTED", f"the attempt workflow closed as {detail}"
    elif state == "completed":
        # It finished, but the projection never saw it: nothing says whether the result was accepted.
        status, message = "LOST", "the attempt workflow completed but the projection was not updated"
    else:
        status, message = "LOST", "the attempt workflow no longer exists"
    return Closure(
        attempt_id=attempt.attempt_id,
        status=status,
        failure=Failure(failure_class="lost", retryable=False, message=message),
    )


async def cleanup_attempts(store: TaskStore, tenant_id: str, probe: AttemptProbe) -> int:
    """Close out this tenant's stale attempts; returns how many rows changed.

    Every workflow is probed before anything is written, so a Temporal error changes nothing and the activity
    retries."""
    closures: list[Closure] = []
    for attempt in await store.stale_attempts(tenant_id=tenant_id):
        workflow_id = attempt_workflow_id(attempt.task_id, attempt.node_id, attempt.attempt_no)
        state, detail = await probe.state(workflow_id)
        closure = closure_for(attempt, state, detail)
        if closure is not None:
            closures.append(closure)
    return await store.close_out_attempts(tenant_id=tenant_id, closures=closures)

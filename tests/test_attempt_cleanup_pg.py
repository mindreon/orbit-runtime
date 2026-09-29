"""cleanup_attempts (17 G2): Temporal says whether an attempt is orphaned, the row is corrected and never deleted."""

from __future__ import annotations

import json

import pytest
from orbit_orch.maintenance import RuntimeMaintenanceWorkflow
from orbit_orch.plan_engine import attempt_workflow_id
from orbit_worker.attempt_cleanup import WorkflowState, cleanup_attempts
from orbit_worker.maintenance import maintenance_tick, set_maintenance_store
from temporalio import workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

TENANT = "tenant-a"
OTHER = "tenant-b"


class FakeProbe:
    """Answers per workflow id and records what was asked."""

    def __init__(self, answers: dict[str, tuple[WorkflowState, str]] | None = None) -> None:
        self.answers = answers or {}
        self.asked: list[str] = []

    async def state(self, workflow_id: str) -> tuple[WorkflowState, str]:
        self.asked.append(workflow_id)
        return self.answers.get(workflow_id, ("missing", "not found"))


class FailingProbe:
    async def state(self, workflow_id: str) -> tuple[WorkflowState, str]:
        raise ConnectionError("temporal is down")


async def _seed(db, attempt: str, *, tenant: str = TENANT, status: str = "RUNNING", age_h: int = 48) -> str:
    """One projection row (control owns the table, so the owner writes it). Returns its workflow id."""
    conn = await db.owner()
    try:
        await conn.execute(
            """
            INSERT INTO stage_attempts (attempt_id, tenant_id, task_id, node_id, attempt_no, status, profile_ref,
                                        started_at)
            VALUES ($1, $2, 'task_1', $3, 1, $4, 'coder@1', now() - make_interval(hours => $5))
            """,
            attempt,
            tenant,
            f"n_{attempt}",
            status,
            age_h,
        )
    finally:
        await conn.close()
    return attempt_workflow_id("task_1", f"n_{attempt}", 1)


async def _row(db, attempt: str):
    conn = await db.owner()
    try:
        return await conn.fetchrow(
            "SELECT status, failure, finished_at, entity_version FROM stage_attempts WHERE attempt_id=$1",
            attempt,
        )
    finally:
        await conn.close()


async def test_an_attempt_parked_for_two_days_is_left_alone(clean_db, task_store) -> None:
    workflow_id = await _seed(clean_db, "att_parked", age_h=48)
    probe = FakeProbe({workflow_id: ("open", "RUNNING")})
    assert await cleanup_attempts(task_store, TENANT, probe) == 0
    assert probe.asked == [workflow_id]
    row = await _row(clean_db, "att_parked")
    assert (row["status"], row["failure"], row["finished_at"], row["entity_version"]) == (
        "RUNNING",
        None,
        None,
        0,
    )


async def test_a_running_row_of_a_missing_workflow_is_corrected_not_deleted(
    clean_db, task_store
) -> None:
    await _seed(clean_db, "att_gone", status="STARTING")
    assert await cleanup_attempts(task_store, TENANT, FakeProbe()) == 1
    row = await _row(clean_db, "att_gone")
    assert row["status"] == "LOST" and row["finished_at"] is not None and row["entity_version"] == 1
    failure = json.loads(row["failure"])
    assert failure["failure_class"] == "lost" and failure["retryable"] is False
    assert "no longer exists" in failure["message"]
    # Corrected once: the next run finds nothing.
    assert await cleanup_attempts(task_store, TENANT, FakeProbe()) == 0


@pytest.mark.parametrize(
    ("state", "detail", "status"),
    [("completed", "COMPLETED", "LOST"), ("closed", "TERMINATED", "ABORTED"), ("closed", "FAILED", "ABORTED")],
)
async def test_a_closed_workflow_closes_the_row(clean_db, task_store, state, detail, status) -> None:
    workflow_id = await _seed(clean_db, "att_closed")
    await cleanup_attempts(task_store, TENANT, FakeProbe({workflow_id: (state, detail)}))
    row = await _row(clean_db, "att_closed")
    assert row["status"] == status
    if state == "closed":
        assert detail in json.loads(row["failure"])["message"]


async def test_only_stale_running_rows_of_this_tenant_are_asked_about(clean_db, task_store) -> None:
    await _seed(clean_db, "att_young", age_h=2)
    await _seed(clean_db, "att_done", status="ACCEPTED")
    await _seed(clean_db, "att_parked_hitl", status="PARKED_HITL")
    await _seed(clean_db, "att_other", tenant=OTHER)
    stale = await _seed(clean_db, "att_stale")
    probe = FakeProbe()
    assert await cleanup_attempts(task_store, TENANT, probe) == 1
    assert probe.asked == [stale]
    assert (await _row(clean_db, "att_other"))["status"] == "RUNNING"
    assert (await _row(clean_db, "att_done"))["status"] == "ACCEPTED"


async def test_a_temporal_error_changes_nothing(clean_db, task_store) -> None:
    await _seed(clean_db, "att_one")
    with pytest.raises(ConnectionError):
        await cleanup_attempts(task_store, TENANT, FailingProbe())
    assert (await _row(clean_db, "att_one"))["status"] == "RUNNING"


async def test_a_row_that_moved_on_meanwhile_is_not_overwritten(clean_db, task_store) -> None:
    from orbit_contracts.v3 import Failure
    from orbit_worker.task_store import Closure

    await _seed(clean_db, "att_moved", status="ACCEPTED")
    closure = Closure("att_moved", "LOST", Failure(failure_class="lost", retryable=False, message="x"))
    assert await task_store.close_out_attempts(tenant_id=TENANT, closures=[closure]) == 0
    assert (await _row(clean_db, "att_moved"))["status"] == "ACCEPTED"


async def test_the_worker_role_cannot_delete_an_attempt(clean_db, task_store) -> None:
    await _seed(clean_db, "att_history")
    import asyncpg

    async with task_store.pool.acquire() as conn, conn.transaction():  # type: ignore[union-attr]
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", TENANT)
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("DELETE FROM stage_attempts WHERE attempt_id='att_history'")


# --- through Temporal: the maintenance activity with its own client --------------------------------------------------



@workflow.defn(name="AttemptWorkflow")
class _ParkedAttempt:
    """Stands in for an AttemptWorkflow parked on an approval: it waits until it is told to end."""

    def __init__(self) -> None:
        self._done = False

    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: self._done)

    @workflow.signal
    def finish(self) -> None:
        self._done = True


async def test_maintenance_tick_asks_temporal_about_each_attempt(clean_db, task_store) -> None:
    parked = await _seed(clean_db, "att_p")
    ended = await _seed(clean_db, "att_e")
    await _seed(clean_db, "att_never_started")
    set_maintenance_store(task_store)
    async with await WorkflowEnvironment.start_time_skipping() as env, Worker(
        env.client,
        task_queue="attempts",
        workflows=[_ParkedAttempt],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ), Worker(
        env.client, task_queue="orbit.orch", workflows=[RuntimeMaintenanceWorkflow]
    ), Worker(env.client, task_queue="orbit.io", activities=[maintenance_tick]):
        parked_handle = await env.client.start_workflow(
            _ParkedAttempt.run, id=parked, task_queue="attempts"
        )
        ended_handle = await env.client.start_workflow(
            _ParkedAttempt.run, id=ended, task_queue="attempts"
        )
        await ended_handle.terminate("test")
        result = await env.client.execute_workflow(
            RuntimeMaintenanceWorkflow.run,
            {"operation": "cleanup_attempts", "tenant_id": TENANT, "task_queue": "orbit.io"},
            id="maintenance-attempts",
            task_queue="orbit.orch",
        )
        await parked_handle.signal(_ParkedAttempt.finish)
        await parked_handle.result()
    assert result["removed"] == 2
    assert (await _row(clean_db, "att_p"))["status"] == "RUNNING"
    assert (await _row(clean_db, "att_e"))["status"] == "ABORTED"
    assert (await _row(clean_db, "att_never_started"))["status"] == "LOST"

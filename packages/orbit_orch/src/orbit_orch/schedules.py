"""Temporal Schedules that run the runtime maintenance workflow on an interval.

Overlap policy SKIP drops a tick that arrives while the previous run is still open.
"""

import os
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleIntervalSpec,
    ScheduleOverlapPolicy,
    SchedulePolicy,
    ScheduleSpec,
    ScheduleState,
)
from temporalio.service import RPCError, RPCStatusCode

from orbit_orch.maintenance import RuntimeMaintenanceWorkflow

ALL_TENANTS = "all"


def maintenance_schedule(
    operation: str,
    *,
    every: timedelta,
    io_task_queue: str,
    tenant_id: str | None = None,
) -> Schedule:
    """Build one periodic runtime maintenance schedule: for `tenant_id`, or (the default) for every tenant that exists
    when a tick runs, so a tenant created later is covered by the next tick without touching the schedule."""

    if every < timedelta(seconds=1):
        raise ValueError("maintenance interval must be at least one second")
    scope = tenant_id or ALL_TENANTS
    scope_payload = {"tenant_id": tenant_id} if tenant_id else {"all_tenants": True}
    return Schedule(
        action=ScheduleActionStartWorkflow(
            RuntimeMaintenanceWorkflow.run,
            {"operation": operation, "task_queue": io_task_queue, **scope_payload},
            id=f"maintenance:{scope}:{operation}",
            task_queue=io_task_queue,
        ),
        spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=every)]),
        policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
        state=ScheduleState(note=f"Orbit maintenance: {operation}"),
    )


async def ensure_maintenance_schedule(
    client: Client,
    operation: str,
    *,
    every: timedelta,
    io_task_queue: str,
    tenant_id: str | None = None,
) -> None:
    schedule_id = f"orbit-maintenance-{tenant_id or ALL_TENANTS}-{operation}"
    try:
        await client.create_schedule(
            schedule_id,
            maintenance_schedule(
                operation,
                every=every,
                io_task_queue=io_task_queue,
                tenant_id=tenant_id,
            ),
        )
    except ScheduleAlreadyRunningError:
        return
    except RPCError as err:
        if err.status == RPCStatusCode.ALREADY_EXISTS:
            return
        raise


async def ensure_maintenance_schedules_from_env(client: Client, *, io_task_queue: str) -> None:
    """Ensure the three runtime cleanup schedules when enabled. Each covers every tenant (17 G7)."""

    if os.environ.get("ORBIT_MAINTENANCE_ENABLED", "1") != "1":
        return
    intervals = {
        "reap_leases": int(os.environ.get("ORBIT_MAINTENANCE_REAP_SECONDS", "300")),
        "gc_checkpoints": int(os.environ.get("ORBIT_MAINTENANCE_GC_SECONDS", "86400")),
        "cleanup_attempts": int(os.environ.get("ORBIT_MAINTENANCE_ATTEMPTS_SECONDS", "86400")),
    }
    for operation, seconds in intervals.items():
        await ensure_maintenance_schedule(
            client,
            operation,
            every=timedelta(seconds=seconds),
            io_task_queue=io_task_queue,
        )

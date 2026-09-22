"""Temporal Schedules that start a cloud-agent job on an interval.

Each tick starts ``CloudAgentJob``. The server appends the scheduled time
to the workflow id, so successive ticks do not collide. Overlap policy SKIP
drops a tick that arrives while the previous run is still open.
"""

import os
from datetime import timedelta

from orbit_contracts.models import CloudAgentJobInput
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

from orbit_orch.workflows import CloudAgentJob


def recurring_schedule(
    job: CloudAgentJobInput,
    *,
    every: timedelta,
    task_queue: str,
    schedule_id: str,
) -> Schedule:
    """Build the schedule object. Does not talk to Temporal."""

    if every < timedelta(seconds=1):
        raise ValueError("recurring interval must be at least one second")
    return Schedule(
        action=ScheduleActionStartWorkflow(
            CloudAgentJob.run,
            job,
            id=f"recurring:{schedule_id}",
            task_queue=task_queue,
        ),
        spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=every)]),
        policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
        state=ScheduleState(note="Orbit recurring cloud agent job"),
    )


async def ensure_recurring_job(
    client: Client,
    job: CloudAgentJobInput,
    *,
    every: timedelta,
    task_queue: str,
    schedule_id: str,
) -> None:
    """Create the schedule when it is missing. A second call is a no-op."""

    schedule = recurring_schedule(
        job,
        every=every,
        task_queue=task_queue,
        schedule_id=schedule_id,
    )
    try:
        await client.create_schedule(schedule_id, schedule)
    except ScheduleAlreadyRunningError:
        return
    except RPCError as err:
        if err.status == RPCStatusCode.ALREADY_EXISTS:
            return
        raise


async def ensure_recurring_job_from_env(client: Client, *, task_queue: str) -> None:
    """Register the optional recurring job from process environment.

    Unset ``ORBIT_RECURRING_SCHEDULE_ID`` means the orch process does not
    create a schedule. A set id without a repo URL is a configuration error.
    """

    schedule_id = os.environ.get("ORBIT_RECURRING_SCHEDULE_ID", "").strip()
    if not schedule_id:
        return
    repo_url = os.environ.get("ORBIT_RECURRING_REPO_URL", "").strip()
    if not repo_url:
        raise RuntimeError("ORBIT_RECURRING_SCHEDULE_ID requires ORBIT_RECURRING_REPO_URL")
    every_seconds = int(os.environ.get("ORBIT_RECURRING_EVERY_SECONDS", "86400"))
    job = CloudAgentJobInput(
        job_id=schedule_id,
        repo_url=repo_url,
        branch=os.environ.get("ORBIT_RECURRING_BRANCH", "orbit/recurring"),
        prompt=os.environ.get("ORBIT_RECURRING_PROMPT", "recurring run"),
    )
    await ensure_recurring_job(
        client,
        job,
        every=timedelta(seconds=every_seconds),
        task_queue=task_queue,
        schedule_id=schedule_id,
    )

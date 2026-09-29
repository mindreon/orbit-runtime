"""Deterministic workflow used by periodic runtime maintenance schedules."""

from datetime import timedelta
from typing import Any

from temporalio import workflow


@workflow.defn(name="RuntimeMaintenanceWorkflow")
class RuntimeMaintenanceWorkflow:
    @workflow.run
    async def run(self, payload: dict[str, str]) -> dict[str, Any]:
        return await workflow.execute_activity(
            "maintenance_tick",
            payload,
            result_type=dict[str, Any],
            start_to_close_timeout=timedelta(minutes=10),
            task_queue=payload.get("task_queue", "orbit.io"),
        )

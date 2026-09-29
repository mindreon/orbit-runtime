"""Workflow worker entrypoint. Polls the workflow task queue only."""

import asyncio

import structlog
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_orch.logs import configure_logging
from orbit_orch.maintenance import RuntimeMaintenanceWorkflow
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.schedules import ensure_maintenance_schedules_from_env
from orbit_orch.settings import TemporalSettings
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from orbit_orch.versioning import deployment_config_from_env

logger = structlog.get_logger(__name__)


async def _serve() -> None:
    temporal = TemporalSettings()
    queue = temporal.orch_queue
    client = await Client.connect(
        temporal.address,
        namespace=temporal.namespace,
        data_converter=pydantic_data_converter,
    )
    await ensure_maintenance_schedules_from_env(
        client,
        io_task_queue=temporal.io_queue,
    )
    deployment_config = deployment_config_from_env()
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[TaskWorkflow, AttemptWorkflow, RuntimeMaintenanceWorkflow],
        workflow_runner=sandbox_runner(),
        interceptors=[TracingInterceptor()],
        deployment_config=deployment_config,
    )
    # WARNING so the line shows without logging config, like the worker's.
    logger.warning("orbit-orch: polling", task_queue=queue)
    await worker.run()


def main() -> None:
    # One line on stdout, so an empty stdout capture means the capture is broken.
    print("orbit-orch: starting", flush=True)
    configure_logging()
    asyncio.run(_serve())


if __name__ == "__main__":
    main()

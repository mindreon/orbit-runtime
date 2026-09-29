"""Workflow worker entrypoint. Polls the workflow task queue only."""

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_orch.maintenance import RuntimeMaintenanceWorkflow
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.schedules import ensure_maintenance_schedules_from_env
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from orbit_orch.versioning import deployment_config_from_env

logger = logging.getLogger(__name__)


async def _serve() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    queue = os.environ.get("TEMPORAL_TASK_QUEUE", "orbit.orch")
    client = await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )
    await ensure_maintenance_schedules_from_env(
        client,
        io_task_queue=os.environ.get("ORBIT_IO_TASK_QUEUE", "orbit.io"),
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
    logger.warning("orbit-orch: polling task queue %s", queue)
    await worker.run()


def main() -> None:
    # One line on stdout, so an empty stdout capture means the capture is broken.
    print("orbit-orch: starting", flush=True)
    asyncio.run(_serve())


if __name__ == "__main__":
    main()

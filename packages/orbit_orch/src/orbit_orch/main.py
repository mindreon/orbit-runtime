"""Workflow worker entrypoint. Polls the workflow task queue only."""

import asyncio

import structlog
from temporalio.api.enums.v1 import TaskQueueType
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_orch.logs import configure_logging
from orbit_orch.maintenance import RuntimeMaintenanceWorkflow
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.schedules import ensure_maintenance_schedules
from orbit_orch.settings import MaintenanceSettings, TemporalSettings, versioning_settings
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from orbit_orch.versioning import assert_peers_agree_on_versioning, worker_deployment_config

logger = structlog.get_logger(__name__)


async def _serve() -> None:
    # Every setting is read and validated here, before anything connects or polls.
    temporal = TemporalSettings()
    maintenance = MaintenanceSettings()
    versioning = versioning_settings()
    queue = temporal.orch_queue
    client = await Client.connect(
        temporal.address,
        namespace=temporal.namespace,
        data_converter=pydantic_data_converter,
    )
    # The activity workers must be versioned the way this process is (17 G14).
    await assert_peers_agree_on_versioning(
        client,
        versioning,
        [
            (temporal.agent_queue, TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY),
            (temporal.io_queue, TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY),
        ],
    )
    await ensure_maintenance_schedules(
        client,
        io_task_queue=temporal.io_queue,
        settings=maintenance,
    )
    deployment_config = worker_deployment_config(versioning)
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

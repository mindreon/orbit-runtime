"""Workflow worker entrypoint. Polls the workflow task queue only."""

import asyncio
import os

from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_orch.sandbox import sandbox_runner
from orbit_orch.schedules import ensure_recurring_job_from_env
from orbit_orch.workflows import AgentRunWorkflow, CloudAgentJob, RoomWorkflow


async def _serve() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    queue = os.environ.get("TEMPORAL_TASK_QUEUE", "orbit")
    client = await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )
    await ensure_recurring_job_from_env(client, task_queue=queue)
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[RoomWorkflow, AgentRunWorkflow, CloudAgentJob],
        workflow_runner=sandbox_runner(),
        interceptors=[TracingInterceptor()],
    )
    await worker.run()


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()

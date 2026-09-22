"""Activity worker entrypoint. Polls the activity task queue only."""

import asyncio
import os

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_worker.activities import ACTIVITIES, set_runtime
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore


async def _serve() -> None:
    # W1 keeps state in memory. ADR-010's Postgres store replaces this process
    # local dict without changing Activity signatures.
    set_runtime(AgentRuntime(MemoryStateStore()))
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    queue = os.environ.get("TEMPORAL_TASK_QUEUE", "orbit")
    client = await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )
    worker = Worker(client, task_queue=queue, activities=ACTIVITIES)
    await worker.run()


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()

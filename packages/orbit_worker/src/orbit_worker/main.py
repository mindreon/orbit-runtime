"""Activity worker entrypoint. Polls the activity queue and the gateway queue."""

import asyncio
import logging
import os

import asyncpg
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_worker.activities import ACTIVITIES, GATEWAY_ACTIVITIES, set_runtime
from orbit_worker.chat_model import ModelConfigError, build_chat_model, resolve_model_config
from orbit_worker.events import HttpEventIngest, MemoryEventIngest
from orbit_worker.isolation import isolation_from_env
from orbit_worker.postgres_store import PostgresStateStore
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import MemoryStateStore

logger = logging.getLogger(__name__)

async def _open_store() -> MemoryStateStore | PostgresStateStore:
    url = os.environ.get("ORBIT_STATE_STORE_URL", "")
    if not url:
        return MemoryStateStore()
    key = os.environ.get("ORBIT_STATE_KEY", "")

    async def connect() -> asyncpg.Connection:
        return await asyncpg.connect(url)

    store = PostgresStateStore(connect, key)
    await store.ensure_schema()
    return store


async def _health() -> None:
    bind = os.environ.get("ORBIT_WORKER_BIND", "0.0.0.0")
    port = int(os.environ.get("ORBIT_WORKER_PORT", "8090"))

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(1024)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, bind, port)
    await server.serve_forever()


async def _serve() -> None:
    model_config = resolve_model_config()
    # Build once so a malformed endpoint stops the worker before it polls.
    build_chat_model(model_config)
    # WARNING so the line shows without logging config; the worker installs none.
    if model_config.mode == "mock":
        logger.warning("chat model: mock")
    else:
        logger.warning("chat model: real model=%s", model_config.name)
    isolation = isolation_from_env()
    store = await _open_store()
    ingest_url = os.environ.get("ORBIT_EVENT_INGEST_URL", "")
    token = os.environ.get("ORBIT_INTERNAL_TOKEN", "")
    ingest = HttpEventIngest(ingest_url, token) if ingest_url else MemoryEventIngest()
    set_runtime(
        AgentRuntime(store, ingest=ingest, isolation=isolation, model_config=model_config)
    )
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    queue = os.environ.get("TEMPORAL_TASK_QUEUE", "orbit")
    client = await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )
    interceptor = TracingInterceptor()
    activity_worker = Worker(
        client,
        task_queue=queue,
        activities=ACTIVITIES,
        interceptors=[interceptor],
    )
    gateway_worker = Worker(
        client,
        task_queue=f"{queue}-gateway",
        activities=GATEWAY_ACTIVITIES,
        interceptors=[interceptor],
    )
    await asyncio.gather(activity_worker.run(), gateway_worker.run(), _health())


def main() -> None:
    # One line on stdout, so an empty stdout capture means the capture is broken.
    print("orbit-worker: starting", flush=True)
    try:
        asyncio.run(_serve())
    except ModelConfigError as exc:
        raise SystemExit(f"orbit-worker: {exc}") from None


if __name__ == "__main__":
    main()

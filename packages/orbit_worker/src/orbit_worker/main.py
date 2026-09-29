"""Activity worker entrypoint. Polls the agent queue and the io queue."""

import asyncio
import logging
import os
from datetime import timedelta

import asyncpg
from orbit_orch.versioning import deployment_config_from_env
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_worker.chat_model import ModelConfigError, build_chat_model, resolve_model_config
from orbit_worker.events import HttpEventIngest, MemoryEventIngest
from orbit_worker.isolation import isolation_from_env
from orbit_worker.maintenance import MAINTENANCE_ACTIVITIES, set_maintenance_store
from orbit_worker.postgres_store import (
    PLAINTEXT_VAR,
    PostgresStateStore,
    StateConfigError,
    resolve_state_cipher,
)
from orbit_worker.runtime import AgentRuntime
from orbit_worker.runtime_holder import set_runtime
from orbit_worker.store import MemoryStateStore
from orbit_worker.task_activities import (
    AGENT_ACTIVITIES,
    IO_ACTIVITIES,
    set_task_store,
    set_workspace_adapter,
)
from orbit_worker.task_store import TaskStore
from orbit_worker.task_stream import TaskStreamIngest
from orbit_worker.workspace import (
    DockerWorkspaceAdapter,
    LocalWorkspaceAdapter,
    OpenSandboxWorkspaceAdapter,
    PersistentWorkspaceAdapter,
)

logger = logging.getLogger(__name__)


def _workspace_adapter(task_store: TaskStore):
    backend = (
        os.environ.get(
            "ORBIT_WORKSPACE_BACKEND",
            os.environ.get("ORBIT_ISOLATION_MODE", "local"),
        )
        .strip()
        .lower()
    )
    ttl_s = int(os.environ.get("ORBIT_WORKSPACE_TTL_SECONDS", "300"))
    if backend == "local":
        return LocalWorkspaceAdapter(
            os.environ.get("ORBIT_WORKSPACE_ROOT", "/tmp/orbit-workspaces"),
            ttl_s=ttl_s,
        )
    if backend == "docker":
        return DockerWorkspaceAdapter(
            os.environ.get("ORBIT_WORKSPACE_ROOT", "/tmp/orbit-workspaces"),
            os.environ.get("ORBIT_WORKSPACE_IMAGE", "python:3.11-slim"),
            ttl_s=ttl_s,
        )
    if backend == "opensandbox":
        from opensandbox.config import ConnectionConfig

        config = ConnectionConfig(
            domain=os.environ.get("ORBIT_OPENSANDBOX_DOMAIN") or None,
            api_key=os.environ.get("ORBIT_OPENSANDBOX_API_KEY") or None,
            protocol=os.environ.get("ORBIT_OPENSANDBOX_PROTOCOL", "http"),
            use_server_proxy=os.environ.get("ORBIT_OPENSANDBOX_SERVER_PROXY", "0") == "1",
        )
        return OpenSandboxWorkspaceAdapter(
            connection_config=config,
            image=os.environ.get(
                "ORBIT_OPENSANDBOX_IMAGE",
                os.environ.get("ORBIT_SANDBOX_IMAGE", "ghcr.io/mindreon/orbit-sandbox:latest"),
            ),
            snapshot_store=task_store,
            ttl_s=ttl_s,
        )
    raise RuntimeError(f"unsupported ORBIT_WORKSPACE_BACKEND: {backend}")


async def _open_store() -> MemoryStateStore | PostgresStateStore:
    url = os.environ.get("ORBIT_STATE_STORE_URL", "")
    if not url:
        logger.warning("state store: memory")
        return MemoryStateStore()
    # Before connecting, so a bad key stops the worker before it polls.
    cipher = resolve_state_cipher()
    if cipher.allow_plaintext:
        logger.warning(
            "state store: postgres, %s, plaintext state allowed (%s=1)",
            "encrypted" if cipher.fernet is not None else "not encrypted",
            PLAINTEXT_VAR,
        )
    else:
        logger.warning("state store: postgres, encrypted")

    async def connect() -> asyncpg.Connection:
        return await asyncpg.connect(url)

    store = PostgresStateStore(connect, cipher)
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
    task_store = TaskStore()
    await task_store.start()
    set_task_store(task_store)
    set_maintenance_store(task_store)
    workspace = _workspace_adapter(task_store)
    set_workspace_adapter(PersistentWorkspaceAdapter(workspace, task_store))
    ingest_url = os.environ.get("ORBIT_EVENT_INGEST_URL", "")
    token = os.environ.get("ORBIT_INTERNAL_TOKEN", "")
    ingest = TaskStreamIngest(
        HttpEventIngest(ingest_url, token) if ingest_url else MemoryEventIngest(), ingest_url, token
    )
    set_runtime(
        AgentRuntime(
            store,
            ingest=ingest,
            isolation=isolation,
            model_config=model_config,
            tool_ledger=task_store,
        )
    )
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    client = await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )
    interceptor = TracingInterceptor()
    deployment_config = deployment_config_from_env()
    # Activity cancellation (an interrupt, a cancel) reaches a running turn on its next heartbeat, so the
    # throttle bounds how long an interrupt takes to land.
    heartbeat_throttle = timedelta(seconds=float(os.environ.get("ORBIT_HEARTBEAT_THROTTLE_S", "5")))
    agent_worker = Worker(
        client,
        task_queue=os.environ.get("ORBIT_AGENT_TASK_QUEUE", "orbit.agent"),
        activities=AGENT_ACTIVITIES,
        interceptors=[interceptor],
        deployment_config=deployment_config,
        max_heartbeat_throttle_interval=heartbeat_throttle,
        default_heartbeat_throttle_interval=heartbeat_throttle,
    )
    io_worker = Worker(
        client,
        task_queue=os.environ.get("ORBIT_IO_TASK_QUEUE", "orbit.io"),
        activities=IO_ACTIVITIES + MAINTENANCE_ACTIVITIES,
        interceptors=[interceptor],
        deployment_config=deployment_config,
    )
    await asyncio.gather(
        agent_worker.run(), io_worker.run(), _health()
    )


def main() -> None:
    # One line on stdout, so an empty stdout capture means the capture is broken.
    print("orbit-worker: starting", flush=True)
    try:
        asyncio.run(_serve())
    except (ModelConfigError, StateConfigError) as exc:
        raise SystemExit(f"orbit-worker: {exc}") from None


if __name__ == "__main__":
    main()

"""Activity worker entrypoint. Polls the agent queue and the io queue."""

import asyncio
from datetime import timedelta

import structlog
from orbit_orch.logs import configure_logging
from orbit_orch.settings import TemporalSettings, versioning_settings
from orbit_orch.versioning import assert_peers_agree_on_versioning, worker_deployment_config
from temporalio.api.enums.v1 import TaskQueueType
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from orbit_worker.chat_model import ModelConfigError, build_chat_model, resolve_model_config
from orbit_worker.checkpoint_activities import CHECKPOINT_ACTIVITIES
from orbit_worker.checkpoint_state import CheckpointStateStore
from orbit_worker.events import HttpEventIngest, MemoryEventIngest
from orbit_worker.isolation import isolation_from_settings
from orbit_worker.maintenance import (
    MAINTENANCE_ACTIVITIES,
    set_maintenance_store,
    set_maintenance_workspaces,
)
from orbit_worker.runtime import AgentRuntime
from orbit_worker.runtime_holder import set_runtime
from orbit_worker.settings import (
    IsolationSettings,
    McpSettings,
    MockSettings,
    StoreSettings,
    WorkerSettings,
    WorkspaceSettings,
)
from orbit_worker.skills import (
    ChainSkillSource,
    ControlSkillSource,
    DirSkillSource,
    SkillSource,
    set_skill_source,
)
from orbit_worker.task_activities import (
    AGENT_ACTIVITIES,
    IO_ACTIVITIES,
    set_task_store,
    set_workspace_adapter,
)
from orbit_worker.task_store import TaskStore
from orbit_worker.task_stream import TaskStreamIngest
from orbit_worker.verify import DirectorySchemaRegistry
from orbit_worker.verify_activities import set_schema_registry
from orbit_worker.workspace import (
    DockerLimits,
    DockerWorkspaceAdapter,
    LocalWorkspaceAdapter,
    OpenSandboxWorkspaceAdapter,
    PersistentWorkspaceAdapter,
)

logger = structlog.get_logger(__name__)


def _workspace_adapter(settings: WorkspaceSettings, task_store: TaskStore):
    if settings.backend == "local":
        return LocalWorkspaceAdapter(settings.root, ttl_s=settings.ttl_seconds)
    if settings.backend == "docker":
        limits = DockerLimits(
            cpus=settings.docker_cpus, memory=settings.docker_memory, pids_limit=settings.docker_pids_limit
        )
        return DockerWorkspaceAdapter(
            settings.root, settings.image, ttl_s=settings.ttl_seconds, limits=limits, network=settings.docker_network
        )
    from opensandbox.config import ConnectionConfig

    config = ConnectionConfig(
        domain=settings.opensandbox_domain,
        api_key=settings.opensandbox_api_key,
        protocol=settings.opensandbox_protocol,
        use_server_proxy=settings.opensandbox_server_proxy,
    )
    return OpenSandboxWorkspaceAdapter(
        connection_config=config,
        image=settings.opensandbox_image,
        snapshot_store=task_store,
        ttl_s=settings.ttl_seconds,
    )


async def _health(bind: str, port: int) -> None:

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(1024)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, bind, port)
    await server.serve_forever()


async def _serve() -> None:
    # Every setting is read and validated here, before anything connects or polls. Mock and MCP settings are read
    # again where they are used; reading them now makes a bad value stop the worker at startup.
    settings = WorkerSettings()
    workspace_settings = WorkspaceSettings()
    isolation_settings = IsolationSettings()
    store_settings = StoreSettings()
    temporal = TemporalSettings()
    versioning = versioning_settings()
    MockSettings()
    McpSettings()
    model_config = resolve_model_config()
    # Build once so a malformed endpoint stops the worker before it polls.
    build_chat_model(model_config)
    # WARNING so the line shows without logging config; the worker installs none.
    if model_config.mode == "mock":
        logger.warning("chat model", mode="mock")
    else:
        logger.warning("chat model", mode="real", model=model_config.name)
    isolation = isolation_from_settings(isolation_settings)
    task_store = TaskStore(settings=store_settings)
    await task_store.start()
    store = CheckpointStateStore(task_store)
    set_task_store(task_store)
    set_maintenance_store(task_store)
    if settings.output_schema_dir:
        set_schema_registry(DirectorySchemaRegistry(settings.output_schema_dir))
    workspace = _workspace_adapter(workspace_settings, task_store)
    set_maintenance_workspaces(workspace)
    set_workspace_adapter(
        PersistentWorkspaceAdapter(workspace, task_store, commit_wait_s=workspace_settings.commit_wait_seconds)
    )
    ingest_url = settings.event_ingest_url
    token = settings.internal_token
    # Skills: the mounted library first, then control's internal listener (the one the live events go to) for what the
    # library does not have (15 T8.4).
    sources: list[SkillSource] = []
    if settings.skills_dir:
        sources.append(DirSkillSource(settings.skills_dir))
    if ingest_url:
        sources.append(ControlSkillSource(ingest_url.rsplit("/internal/events", 1)[0], token))
    if sources:
        set_skill_source(ChainSkillSource(*sources))
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
    client = await Client.connect(
        temporal.address,
        namespace=temporal.namespace,
        data_converter=pydantic_data_converter,
    )
    # The workflow worker must be versioned the way this process is (17 G14).
    await assert_peers_agree_on_versioning(
        client, versioning, [(temporal.orch_queue, TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW)]
    )
    interceptor = TracingInterceptor()
    deployment_config = worker_deployment_config(versioning)
    heartbeat_throttle = timedelta(seconds=settings.heartbeat_throttle_s)
    agent_worker = Worker(
        client,
        task_queue=temporal.agent_queue,
        activities=AGENT_ACTIVITIES,
        interceptors=[interceptor],
        deployment_config=deployment_config,
        max_heartbeat_throttle_interval=heartbeat_throttle,
        default_heartbeat_throttle_interval=heartbeat_throttle,
    )
    io_worker = Worker(
        client,
        task_queue=temporal.io_queue,
        activities=IO_ACTIVITIES + CHECKPOINT_ACTIVITIES + MAINTENANCE_ACTIVITIES,
        interceptors=[interceptor],
        deployment_config=deployment_config,
    )
    await asyncio.gather(agent_worker.run(), io_worker.run(), _health(settings.bind, settings.port))


def main() -> None:
    # One line on stdout, so an empty stdout capture means the capture is broken.
    print("orbit-worker: starting", flush=True)
    configure_logging()
    try:
        asyncio.run(_serve())
    except ModelConfigError as exc:
        raise SystemExit(f"orbit-worker: {exc}") from None


if __name__ == "__main__":
    main()

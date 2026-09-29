"""How Orbit versions Temporal workflow code and worker deployments."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from temporalio.api.enums.v1 import TaskQueueKind, TaskQueueType, WorkerVersioningMode
from temporalio.api.taskqueue.v1 import PollerInfo, TaskQueue
from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest
from temporalio.client import Client
from temporalio.common import VersioningBehavior
from temporalio.service import RPCError
from temporalio.worker import WorkerDeploymentConfig, WorkerDeploymentVersion

from orbit_orch.settings import VersioningSettings

logger = logging.getLogger(__name__)

ROOM_CONTROL_SURFACE = "orbit-room-control-surface"
AGENT_RUN_SURFACE = "orbit-agent-run-surface"
CLOUD_JOB_SURFACE = "orbit-cloud-job-surface"
ROOM_STAY_OPEN = "orbit-room-stay-open"
ROOM_MCP_CONNECTORS = "orbit-room-mcp-connectors"


def worker_deployment_config(settings: VersioningSettings) -> WorkerDeploymentConfig | None:
    if not settings.enabled:
        return None
    return WorkerDeploymentConfig(
        version=WorkerDeploymentVersion(settings.deployment, settings.build_id),
        use_worker_versioning=True,
        default_versioning_behavior=VersioningBehavior.AUTO_UPGRADE,
    )


class VersioningMismatchError(RuntimeError):
    """A process that polls one of the queues we depend on is set up for a different versioning mode."""


def versioning_mismatches(
    settings: VersioningSettings, queue: str, pollers: Iterable[PollerInfo]
) -> list[str]:
    """What is wrong with the pollers of `queue`, given this process's settings. A worker that is not registered
    for deployment versioning sends no deployment options, so a poller without them is an unversioned one."""
    problems: list[str] = []
    for poller in pollers:
        options = poller.deployment_options
        versioned = (
            poller.HasField("deployment_options")
            and options.worker_versioning_mode == WorkerVersioningMode.WORKER_VERSIONING_MODE_VERSIONED
        )
        if versioned != settings.enabled:
            problems.append(
                f"queue {queue!r} has {'a versioned' if versioned else 'an unversioned'} poller"
                f" ({poller.identity}), but ORBIT_USE_WORKER_VERSIONING is {'1' if settings.enabled else '0'} here"
            )
        elif versioned and options.deployment_name != settings.deployment:
            problems.append(
                f"queue {queue!r} has a poller in deployment {options.deployment_name!r},"
                f" but ORBIT_WORKER_DEPLOYMENT is {settings.deployment!r} here"
            )
    return problems


async def assert_peers_agree_on_versioning(
    client: Client,
    settings: VersioningSettings,
    queues: Iterable[tuple[str, TaskQueueType.ValueType]],
) -> None:
    """Refuse to start when a peer that polls one of `queues` is versioned differently from this process.

    orbit-orch and orbit-worker read the same `ORBIT_USE_WORKER_VERSIONING` from their own environments. If they
    differ, a pinned workflow waits for activities on a build that never polls, with no error. Peers that are not up
    yet cannot be checked; when Temporal cannot be asked, the check is skipped with a warning.
    """
    problems: list[str] = []
    for queue, queue_type in queues:
        request = DescribeTaskQueueRequest(
            namespace=client.namespace,
            task_queue=TaskQueue(name=queue, kind=TaskQueueKind.TASK_QUEUE_KIND_NORMAL),
            task_queue_type=queue_type,
            report_pollers=True,
        )
        try:
            response = await client.workflow_service.describe_task_queue(request)
        except RPCError as err:
            logger.warning("versioning check skipped: could not describe queue %s: %s", queue, err)
            return
        problems.extend(
            versioning_mismatches(settings, queue, response.pollers)
        )
    if problems:
        raise VersioningMismatchError(
            "orbit-orch and orbit-worker disagree on worker versioning: " + "; ".join(dict.fromkeys(problems))
        )

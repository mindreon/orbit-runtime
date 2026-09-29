"""How Orbit versions Temporal workflow code and worker deployments."""

from __future__ import annotations

import os

from temporalio.common import VersioningBehavior
from temporalio.worker import WorkerDeploymentConfig, WorkerDeploymentVersion

ROOM_CONTROL_SURFACE = "orbit-room-control-surface"
AGENT_RUN_SURFACE = "orbit-agent-run-surface"
CLOUD_JOB_SURFACE = "orbit-cloud-job-surface"
ROOM_STAY_OPEN = "orbit-room-stay-open"
ROOM_MCP_CONNECTORS = "orbit-room-mcp-connectors"


def deployment_config_from_env() -> WorkerDeploymentConfig | None:
    if os.environ.get("ORBIT_USE_WORKER_VERSIONING", "0") != "1":
        return None
    deployment = os.environ.get("ORBIT_WORKER_DEPLOYMENT", "orbit")
    build_id = os.environ.get("ORBIT_WORKER_BUILD_ID", "dev")
    return WorkerDeploymentConfig(
        version=WorkerDeploymentVersion(deployment, build_id),
        use_worker_versioning=True,
        default_versioning_behavior=VersioningBehavior.AUTO_UPGRADE,
    )

from __future__ import annotations

import pytest
from orbit_orch.settings import VersioningSettings
from orbit_orch.versioning import (
    VersioningMismatchError,
    assert_peers_agree_on_versioning,
    versioning_mismatches,
    worker_deployment_config,
)
from temporalio.api.deployment.v1 import WorkerDeploymentOptions
from temporalio.api.enums.v1 import TaskQueueType, WorkerVersioningMode
from temporalio.api.taskqueue.v1 import PollerInfo
from temporalio.api.workflowservice.v1 import DescribeTaskQueueResponse
from temporalio.service import RPCError, RPCStatusCode

ON = VersioningSettings(enabled=True, deployment="orbit", build_id="v2")
OFF = VersioningSettings(enabled=False)


def _versioned(deployment: str = "orbit", build_id: str = "v1") -> PollerInfo:
    return PollerInfo(
        identity="versioned",
        deployment_options=WorkerDeploymentOptions(
            deployment_name=deployment,
            build_id=build_id,
            worker_versioning_mode=WorkerVersioningMode.WORKER_VERSIONING_MODE_VERSIONED,
        ),
    )


def _unversioned() -> PollerInfo:
    return PollerInfo(identity="unversioned")


def test_deployment_config_follows_the_switch() -> None:
    assert worker_deployment_config(OFF) is None
    config = worker_deployment_config(ON)
    assert config is not None
    assert (config.version.deployment_name, config.version.build_id) == ("orbit", "v2")


def test_matching_pollers_and_an_empty_queue_agree() -> None:
    assert versioning_mismatches(ON, "q", [_versioned(build_id="v1"), _versioned(build_id="v2")]) == []
    assert versioning_mismatches(OFF, "q", [_unversioned()]) == []
    assert versioning_mismatches(ON, "q", []) == []


def test_a_peer_versioned_the_other_way_is_a_mismatch() -> None:
    assert "unversioned poller" in versioning_mismatches(ON, "q", [_unversioned()])[0]
    assert "versioned poller" in versioning_mismatches(OFF, "q", [_versioned()])[0]


def test_a_peer_in_another_deployment_is_a_mismatch() -> None:
    (problem,) = versioning_mismatches(ON, "q", [_versioned(deployment="other")])
    assert "'other'" in problem
    assert "ORBIT_WORKER_DEPLOYMENT" in problem


class _Service:
    def __init__(self, pollers: list[PollerInfo] | None = None, error: RPCError | None = None) -> None:
        self.requests: list[object] = []
        self._pollers = pollers or []
        self._error = error

    async def describe_task_queue(self, request: object) -> DescribeTaskQueueResponse:
        self.requests.append(request)
        if self._error:
            raise self._error
        return DescribeTaskQueueResponse(pollers=self._pollers)


class _Client:
    namespace = "default"

    def __init__(self, service: _Service) -> None:
        self.workflow_service = service


QUEUES = [("orbit.agent", TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY)]


async def test_startup_fails_with_a_message_naming_both_sides() -> None:
    client = _Client(_Service([_unversioned()]))
    with pytest.raises(VersioningMismatchError, match="ORBIT_USE_WORKER_VERSIONING is 1"):
        await assert_peers_agree_on_versioning(client, ON, QUEUES)  # type: ignore[arg-type]
    assert client.workflow_service.requests[0].task_queue.name == "orbit.agent"


async def test_startup_passes_when_peers_agree_or_are_not_up_yet() -> None:
    await assert_peers_agree_on_versioning(_Client(_Service([_versioned()])), ON, QUEUES)  # type: ignore[arg-type]
    await assert_peers_agree_on_versioning(_Client(_Service([])), ON, QUEUES)  # type: ignore[arg-type]


async def test_startup_skips_the_check_when_temporal_cannot_answer() -> None:
    error = RPCError("unimplemented", RPCStatusCode.UNIMPLEMENTED, b"")
    await assert_peers_agree_on_versioning(_Client(_Service(error=error)), ON, QUEUES)  # type: ignore[arg-type]

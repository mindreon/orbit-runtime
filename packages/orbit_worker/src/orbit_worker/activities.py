"""Temporal Activity bodies. They adapt Orbit contracts onto AgentRuntime."""

import logging

from orbit_contracts.models import (
    AbortSessionInput,
    CloneRepoInput,
    CloneRepoOutput,
    CloseSessionInput,
    CloseSessionOutput,
    DeliverToolResultInput,
    GatewayExecuteInput,
    GatewayExecuteOutput,
    OpenPrInput,
    OpenPrOutput,
    OpenSessionInput,
    OpenSessionOutput,
    PushBranchInput,
    PushBranchOutput,
    ResolveApprovalInput,
    RunTurnInput,
    SteerInput,
    TurnErrorCode,
    TurnResult,
)
from temporalio import activity
from temporalio.exceptions import ApplicationError

from orbit_worker.cloud import clone_repo, open_pr, push_branch
from orbit_worker.gateway import execute_gateway
from orbit_worker.runtime import AgentRuntime
from orbit_worker.store import StateUnreadableError

logger = logging.getLogger(__name__)

_runtime: AgentRuntime | None = None


def set_runtime(runtime: AgentRuntime) -> None:
    """One runtime per worker process. Tests install their own before polling."""

    global _runtime
    _runtime = runtime


def get_runtime() -> AgentRuntime:
    if _runtime is None:
        raise RuntimeError("AgentRuntime is not installed on this worker")
    return _runtime


@activity.defn(name="openSession")
async def open_session(inp: OpenSessionInput) -> OpenSessionOutput:
    # Only a reopen by the same turn id reads an existing blob. There is no
    # turn to fail, so an unreadable one fails the Activity once, no retry.
    try:
        return await get_runtime().open_session(inp)
    except StateUnreadableError as exc:
        code = TurnErrorCode.STATE_UNREADABLE
        logger.warning("room %s openSession [%s]: %s", inp.room_id, code.value, exc.reason)
        raise ApplicationError(str(exc), type=code.value, non_retryable=True) from None


@activity.defn(name="runTurn")
async def run_turn(inp: RunTurnInput) -> TurnResult:
    return await get_runtime().run_turn(inp)


@activity.defn(name="resolveApproval")
async def resolve_approval(inp: ResolveApprovalInput) -> TurnResult:
    return await get_runtime().resolve_approval(inp)


@activity.defn(name="deliverToolResult")
async def deliver_tool_result(inp: DeliverToolResultInput) -> TurnResult:
    return await get_runtime().deliver_tool_result(inp)


@activity.defn(name="steer")
async def steer(inp: SteerInput) -> TurnResult:
    return await get_runtime().steer(inp)


@activity.defn(name="abort")
async def abort(inp: AbortSessionInput) -> CloseSessionOutput:
    return await get_runtime().abort_session(inp)


@activity.defn(name="closeSession")
async def close_session(inp: CloseSessionInput) -> CloseSessionOutput:
    version = await get_runtime().close_session(inp.session_id, inp.turn_id)
    return CloseSessionOutput(closed=True, state_version=version)


@activity.defn(name="gatewayExecute")
async def gateway_execute(inp: GatewayExecuteInput) -> GatewayExecuteOutput:
    return await execute_gateway(inp)


@activity.defn(name="cloneRepo")
async def clone_repo_activity(inp: CloneRepoInput) -> CloneRepoOutput:
    return await clone_repo(inp)


@activity.defn(name="pushBranch")
async def push_branch_activity(inp: PushBranchInput) -> PushBranchOutput:
    return await push_branch(inp)


@activity.defn(name="openPr")
async def open_pr_activity(inp: OpenPrInput) -> OpenPrOutput:
    return await open_pr(inp)


ACTIVITIES = [
    open_session,
    run_turn,
    resolve_approval,
    deliver_tool_result,
    steer,
    abort,
    close_session,
    clone_repo_activity,
    push_branch_activity,
    open_pr_activity,
]

GATEWAY_ACTIVITIES = [gateway_execute]

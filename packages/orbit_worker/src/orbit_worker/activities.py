"""Temporal Activity bodies. They adapt Orbit contracts onto AgentRuntime."""

from orbit_contracts.models import (
    CloseSessionInput,
    CloseSessionOutput,
    OpenSessionInput,
    OpenSessionOutput,
    ResolveApprovalInput,
    RunTurnInput,
    TurnResult,
)
from temporalio import activity

from orbit_worker.runtime import AgentRuntime

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
    return await get_runtime().open_session(inp)


@activity.defn(name="runTurn")
async def run_turn(inp: RunTurnInput) -> TurnResult:
    return await get_runtime().run_turn(inp)


@activity.defn(name="resolveApproval")
async def resolve_approval(inp: ResolveApprovalInput) -> TurnResult:
    return await get_runtime().resolve_approval(inp)


@activity.defn(name="closeSession")
async def close_session(inp: CloseSessionInput) -> CloseSessionOutput:
    version = await get_runtime().close_session(inp.session_id, inp.turn_id)
    return CloseSessionOutput(closed=True, state_version=version)


ACTIVITIES = [open_session, run_turn, resolve_approval, close_session]

"""Live output of a running task attempt, as v3 ephemeral events (09 §4).

The runtime emits its own event kinds. While an `agent_turn` activity runs, this adapter turns the ones a task page
shows live into `agent.token_delta` / `agent.thinking_delta` / `tool.call_started` and posts them to control's ingest.
Everything else is dropped: durable events reach control through `runtime_outbox`, never through this path.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiohttp
from agentscope.skill import Skill
from orbit_contracts.models import OrbitEvent
from orbit_contracts.v3 import Policy
from orbit_orch.plan_engine import deterministic_id

from orbit_worker.agent_config import AgentConfig

if TYPE_CHECKING:
    from orbit_worker.budget_middleware import BudgetMeter


@dataclass(frozen=True)
class TeamTurn:
    """Which agent of a team stage a turn is (07): the leader, who gives work out to `members` (role and what each is for), or
    one of the members, who does it and answers the leader."""

    role: str
    leader: bool
    leader_role: str = ""
    leader_label: str = ""
    members: tuple[tuple[str, str], ...] = ()
    label: str = ""
    # The attempt the stage runs as: what its events say (the agents run under attempt ids of their own, for their sessions and
    # ledger keys, which only the worker uses).
    stage_attempt_id: str = ""


@dataclass(frozen=True)
class TaskStreamContext:
    tenant_id: str
    task_id: str
    attempt_id: str
    activity_attempt: int
    node_id: str = ""
    profile: str = ""
    task_policy: Policy = field(default_factory=Policy)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # Skills staged for this attempt: AgentScope reads them from their directories until the attempt ends.
    skills: tuple[Skill, ...] = ()
    # What this activity has spent and the budget it may spend (05 §4); None outside a task attempt.
    meter: BudgetMeter | None = None
    # The JSON Schema the attempt's final reply must satisfy (the node's `output_schema_ref`); None asks for no structure.
    output_schema: dict[str, Any] | None = None
    # Set when the turn is one of a team stage's agents: it has the team's tools and none of the plan's.
    team: TeamTurn | None = None


def event_attempt_id(context: TaskStreamContext) -> str:
    """The attempt an event of this turn is about: the team stage's for an agent of a stage, else the turn's own."""
    return context.team.stage_attempt_id if context.team is not None and context.team.stage_attempt_id else context.attempt_id


def team_stamp(context: TaskStreamContext) -> dict[str, str]:
    """`team_role`, `team_label` and `team_session` of an event made by an agent of a team stage (empty otherwise)."""
    team = context.team
    if team is None:
        return {}
    return {
        "team_role": team.role,
        **({"team_label": team.label} if team.label else {}),
        "team_session": context.attempt_id,
    }


_current: ContextVar[TaskStreamContext | None] = ContextVar("orbit_task_stream", default=None)


@contextlib.contextmanager
def streaming_for(context: TaskStreamContext) -> Iterator[None]:
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)


_tool_call_id: ContextVar[str] = ContextVar("orbit_tool_call_id", default="")


@contextlib.contextmanager
def executing_tool_call(call_id: str) -> Iterator[None]:
    """Marks which tool call is running, so a tool can derive an idempotent command id from it.

    Restores the previous value by assignment, not by token: this runs inside an async generator, which may be
    finalized from a different context than the one that entered it.
    """
    previous = _tool_call_id.get()
    _tool_call_id.set(call_id)
    try:
        yield
    finally:
        _tool_call_id.set(previous)


def current_tool_call_id() -> str:
    return _tool_call_id.get()


def current_task_context() -> TaskStreamContext | None:
    """The task attempt the running activity belongs to, or None outside one."""
    return _current.get()


class TaskStreamIngest:
    """Wraps the runtime's ingest: task activities stream as v3, anything else passes through unchanged."""

    def __init__(self, inner: Any, url: str, token: str) -> None:
        self._inner = inner
        self._url = url
        self._token = token
        self.failures = 0

    async def emit(self, event: OrbitEvent) -> None:
        context = _current.get()
        if context is None:
            # Progress of a tool call's arguments is for a task page only.
            if event.type != "tool.call_progress":
                await self._inner.emit(event)
            return
        envelope = to_v3(event, context)
        if envelope is not None and self._url:
            await self._post(envelope)

    async def _post(self, envelope: dict[str, Any]) -> None:
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    self._url, json=envelope, headers={"Authorization": f"Bearer {self._token}"}
                ) as response,
            ):
                if response.status >= 300:
                    self.failures += 1
        except aiohttp.ClientError:
            self.failures += 1


def to_v3(event: OrbitEvent, context: TaskStreamContext) -> dict[str, Any] | None:
    shown = event_attempt_id(context)
    stamp = team_stamp(context)
    if event.type == "assistant.delta":
        kind, payload = "agent.token_delta", {"attempt_id": shown, "text": event.delta, "block_id": event.block_id, **stamp}
        seed = f"{context.attempt_id}:{event.turn_id}:{event.block_id}:{event.seq}"
    elif event.type == "assistant.thinking":
        kind, payload = "agent.thinking_delta", {"attempt_id": shown, "text": event.delta, "block_id": event.block_id, **stamp}
        seed = f"{context.attempt_id}:{event.turn_id}:{event.block_id}:{event.seq}:thinking"
    elif event.type in ("tool.call", "tool.call_progress"):
        kind = "tool.call_started"
        payload = {
            "attempt_id": shown,
            "tool_call_id": event.call_id,
            "tool_name": event.tool_name,
            "args_preview": event.args_preview,
            **stamp,
        }
        seed = f"{context.attempt_id}:{event.call_id}:" + (f"progress{event.seq}" if event.type == "tool.call_progress" else "started")
    else:
        return None
    return {
        "schema": "orbit.event/3",
        "event_id": deterministic_id(f"{seed}:{context.activity_attempt}", "evt"),
        "tenant_id": context.tenant_id,
        "task_id": context.task_id,
        "type": kind,
        "retention": "ephemeral",
        "source": {"kind": "worker", "id": "orbit-worker", "attempt_id": shown},
        "entity": {"kind": "attempt", "id": shown, "version": 0},
        "visibility": "tenant",
        "occurred_at": datetime.now(UTC).isoformat(),
        "payload": payload,
    }

"""Exactly-once tool execution across activity retries (10 §1, 06 §3 S7).

Every tool call an attempt makes is recorded in the idempotency ledger under `(attempt_id, tool_call_id)` before it
runs. A call that already succeeded returns its recorded result; a call that started but has no recorded outcome
(the worker died mid-call) is replayed only if the tool is read-only, because anything else may have taken effect.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, Protocol

from agentscope.message import TextBlock, ToolResultState
from agentscope.middleware import MiddlewareBase
from agentscope.tool import ToolChunk, ToolResponse

from orbit_worker.task_store import REPLAY_APPROVED
from orbit_worker.task_stream import (
    TaskStreamContext,
    current_task_context,
    event_attempt_id,
    executing_tool_call,
    team_stamp,
)
from orbit_worker.todo_tools import TODO_WRITE, todo_list_preview
from orbit_worker.worker_events import publish_attempt_event

SCOPE = "side_effect"


class ToolLedger(Protocol):
    async def publish_events(self, events: list[dict[str, Any]]) -> None: ...

    async def claim_ledger(
        self,
        *,
        tenant_id: str,
        scope: str,
        key: str,
        request_hash: str,
        owner: str,
        intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    async def finish_ledger(
        self,
        *,
        tenant_id: str,
        scope: str,
        key: str,
        status: str,
        result_ref: dict[str, Any] | None,
    ) -> None: ...


class OrbitLedgerMiddleware(MiddlewareBase):
    def __init__(self, ledger: ToolLedger) -> None:
        self._ledger = ledger

    async def on_acting(
        self, agent: Any, input_kwargs: dict, next_handler: Callable[..., AsyncGenerator]
    ) -> AsyncGenerator:
        context = current_task_context()
        if context is None:  # not a task attempt: nothing to make idempotent
            async for item in next_handler(**input_kwargs):
                yield item
            return

        call = input_kwargs["tool_call"]
        key = f"{context.attempt_id}:{call.id}"
        request_hash = hashlib.sha256(f"{call.name}:{call.input}".encode()).hexdigest()
        read_only = await _is_read_only(agent, call.name)
        claim = await self._ledger.claim_ledger(
            tenant_id=context.tenant_id,
            scope=SCOPE,
            key=key,
            request_hash=request_hash,
            owner=context.attempt_id,
            intent={"tool": call.name, "read_only": read_only},
        )

        if claim["status"] == "succeeded":
            recorded = _recorded(claim["result_ref"])
            await self._finished(context, call, recorded)
            yield recorded
            return
        if claim["status"] == "failed_permanent":
            failed = _error(f"{call.name} failed earlier and is not retried")
            await self._finished(context, call, failed)
            yield failed
            return
        if not claim["claimed"] and not read_only and claim.get("owner") != REPLAY_APPROVED:
            # Started before, outcome unknown: replaying could repeat a side effect. The activity asks a person first
            # (A12); reaching this point means that step was skipped, so refuse rather than run it.
            refused = _error(f"{call.name} may already have run; it needs approval to run again")
            await self._finished(context, call, refused)
            yield refused
            return

        # A crash or cancel mid-call leaves the row `started`, which is the honest "outcome unknown".
        final: ToolResponse | None = None
        try:
            with executing_tool_call(call.id):
                async for item in next_handler(**input_kwargs):
                    if isinstance(item, ToolResponse):
                        final = item
                    yield item
        except asyncio.CancelledError:
            # The activity was cancelled (an interrupt, a cancel): this call ends as INTERRUPTED in the transcript, and
            # the ledger row stays `started`, the honest "outcome unknown" (06 §3 S4).
            yield ToolResponse(
                content=[TextBlock(text=f"{call.name} was interrupted")],
                state=ToolResultState.INTERRUPTED,
            )
            raise
        if final is None:
            return
        await self._finished(context, call, final)
        await self._ledger.finish_ledger(
            tenant_id=context.tenant_id,
            scope=SCOPE,
            key=key,
            status="succeeded" if final.state == ToolResultState.SUCCESS else "failed_permanent",
            result_ref={
                "tool": call.name,
                "text": "".join(b.text for b in final.content if isinstance(b, TextBlock)),
                "state": final.state.value,
                # Kept with the outcome, so what an agent that takes over is told of leaves out what was only read.
                "read_only": read_only,
            },
        )

    async def _finished(
        self, context: TaskStreamContext, call: Any, response: ToolResponse
    ) -> None:
        """tool.call_finished (durable): how the call ended, with a short look at its result."""
        body = {
            "attempt_id": event_attempt_id(context),
            **team_stamp(context),
            "tool_call_id": call.id,
            "tool_name": call.name,
            "state": response.state.value,
            "result_preview": "".join(
                b.text for b in response.content if isinstance(b, TextBlock)
            )[:200],
        }
        if call.name == TODO_WRITE:
            # The one input a page needs whole: its checklist. `tool.call_started` cuts the arguments to 256 characters and
            # is not durable, so the list rides on this event (a client reads `args_preview` of the latest successful one).
            body["args_preview"] = todo_list_preview(call.input)
        await publish_attempt_event(
            self._ledger,
            tenant_id=context.tenant_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            shown_attempt_id=event_attempt_id(context),
            event_type="tool.call_finished",
            body=body,
            seed=f"finished:{call.id}",
        )


async def _is_read_only(agent: Any, name: str) -> bool:
    tool = await agent.toolkit.get_tool(name)
    return bool(tool is not None and getattr(tool, "is_read_only", False))


def _recorded(result_ref: dict[str, Any] | str | None) -> ToolResponse:
    data = json.loads(result_ref) if isinstance(result_ref, str) else (result_ref or {})
    return ToolResponse(
        content=[TextBlock(text=str(data.get("text", "")))],
        state=ToolResultState(data.get("state", "success")),
    )


def _error(message: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(text=message)], state=ToolResultState.ERROR)


__all__ = ["OrbitLedgerMiddleware", "ToolChunk", "ToolLedger"]

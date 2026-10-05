"""The budget an attempt may spend (05 §4): what the parent reserved for it, enforced where the agent works.

The parent reserves the node's budget from the task's remaining budget and hands the attempt what is left of it for each
turn. The meter counts what the turn spends: tokens (and cost, when the model has a known price) from the usage a model call
reports, tool calls as they start, wall time from the start of the activity. The middleware stops the turn between two
steps, never inside one:

* before a model call, a spent token, cost or wall budget ends the turn with `BudgetExceeded`;
* a tool call that would go past the tool-call budget (or start after the others are spent) is not run: it comes back to
  the agent as an error result, so every call of the batch has a result and the saved state is whole (08 §3, A27); the turn
  then ends before the next model call.

A call that is running is never cut off, so a turn can overrun by the last model call or the last tool call; the overrun is
reported as spent and settled like the rest. Cost is only enforced when the model has a known price: without one it is
recorded as unknown (None), never as zero.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

from agentscope.event import ModelCallEndEvent
from agentscope.message import TextBlock, ToolResultState
from agentscope.middleware import MiddlewareBase
from agentscope.tool import ToolResponse
from orbit_contracts.v3.common import Budget, Usage

from orbit_worker.task_stream import current_task_context

MICROS_PER_MTOK = 1_000_000


class BudgetExceeded(Exception):
    """The attempt spent what was reserved for it. `dimension` names the limit (`tokens`, `tool_calls`, `wall_s`,
    `cost_usd_micros`)."""

    def __init__(self, dimension: str, used: int, limit: int) -> None:
        super().__init__(f"the attempt's {_NAMES.get(dimension, dimension)} budget is spent ({used} of {limit})")
        self.dimension = dimension


_NAMES = {
    "tokens": "token",
    "tool_calls": "tool call",
    "wall_s": "time",
    "cost_usd_micros": "cost",
}


@dataclass(frozen=True)
class ModelPrice:
    """What the model costs, in micro-dollars per million tokens, as the provider charges input and output apart."""

    input_per_mtok: int
    output_per_mtok: int

    def cost(self, tokens_in: int, tokens_out: int) -> int:
        """Micro-dollars for a call, rounded up so a cheap call is never free."""
        return math.ceil((tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok) / MICROS_PER_MTOK)


@dataclass
class BudgetMeter:
    """What one activity of an attempt has spent, and the limits it spends against. A limit that is None is no limit."""

    limit: Budget = field(default_factory=Budget)
    price: ModelPrice | None = None
    clock: Callable[[], float] = time.monotonic
    started: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    cost_usd_micros: int | None = None
    # The first limit a tool call was refused for: the turn ends before it asks the model for anything else.
    refused: str = ""
    # Called after every change of what was spent: the activity puts the snapshot in its heartbeat, so a retry of the
    # activity (a crashed worker) carries on from what was spent instead of spending the reservation again.
    on_change: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.started = self.clock()
        self.set_price(self.price)

    def set_price(self, price: ModelPrice | None) -> None:
        """The model's price, known once the attempt's profile is read. Without one the cost stays unknown."""
        self.price = price
        self.cost_usd_micros = None if price is None else (self.cost_usd_micros or 0)

    @property
    def wall_s(self) -> int:
        return max(0, math.ceil(self.clock() - self.started))

    def add_model_call(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if self.price is not None:
            self.cost_usd_micros = (self.cost_usd_micros or 0) + self.price.cost(tokens_in, tokens_out)
        self.changed()

    def snapshot(self) -> dict[str, Any]:
        """What was spent, as heartbeat details: JSON, small, and enough for `restore`."""
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_calls": self.tool_calls,
            "wall_s": self.wall_s,
            "cost_usd_micros": self.cost_usd_micros,
            "refused": self.refused,
        }

    def restore(self, saved: dict[str, Any]) -> None:
        """Carry on from what an earlier run of this activity spent: the wall clock keeps what it had counted."""
        self.tokens_in = int(saved.get("tokens_in", 0))
        self.tokens_out = int(saved.get("tokens_out", 0))
        self.tool_calls = int(saved.get("tool_calls", 0))
        self.refused = str(saved.get("refused", ""))
        cost = saved.get("cost_usd_micros")
        self.cost_usd_micros = None if cost is None else int(cost)  # `set_price` drops it when the model has no price
        self.started = self.clock() - int(saved.get("wall_s", 0))

    def changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def usage(self) -> Usage:
        return Usage(
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            tool_calls=self.tool_calls,
            wall_s=self.wall_s,
            cost_usd_micros=self.cost_usd_micros,
        )

    def used(self, dimension: str) -> int:
        if dimension == "tokens":
            return self.tokens_in + self.tokens_out
        if dimension == "wall_s":
            return self.wall_s
        if dimension == "cost_usd_micros":
            return self.cost_usd_micros or 0
        return self.tool_calls

    def spent(self, *, tools: bool = False) -> tuple[str, int, int] | None:
        """The first limit that is reached (what was used is at least what was reserved): its name, use and limit. Tool calls
        count only when `tools` says so: a spent tool budget does not stop a model that has nothing more to call."""
        for dimension in ("tokens", "cost_usd_micros", "wall_s", *(("tool_calls",) if tools else ())):
            limit = getattr(self.limit, dimension)
            if dimension == "cost_usd_micros" and self.price is None:
                continue  # an unknown cost is not enforced
            if limit is not None and self.used(dimension) >= limit:
                return dimension, self.used(dimension), limit
        return None


class OrbitBudgetMiddleware(MiddlewareBase):
    """Counts what a turn spends and stops it between two steps when the attempt's reserved budget is gone. It sits outside
    the policy and ledger middleware, so a call it refuses is neither checkpointed nor recorded as started."""

    async def on_reply(self, agent: Any, input_kwargs: dict, next_handler: Callable[..., AsyncGenerator]) -> AsyncGenerator:
        context = current_task_context()
        meter = context.meter if context is not None else None
        async for event in next_handler(**input_kwargs):
            if meter is not None and isinstance(event, ModelCallEndEvent):
                meter.add_model_call(event.input_tokens, event.output_tokens)
            yield event

    async def on_reasoning(
        self, agent: Any, input_kwargs: dict, next_handler: Callable[..., AsyncGenerator]
    ) -> AsyncGenerator:
        context = current_task_context()
        meter = context.meter if context is not None else None
        if meter is not None:
            # Only here, between a step's results and the next model call, is the state of the agent whole.
            reached = meter.spent(tools=bool(meter.refused))
            if reached is not None:
                raise BudgetExceeded(*reached)
            if meter.refused:
                raise BudgetExceeded(meter.refused, meter._used(meter.refused), getattr(meter.limit, meter.refused) or 0)
        async for event in next_handler(**input_kwargs):
            yield event

    async def on_acting(self, agent: Any, input_kwargs: dict, next_handler: Callable[..., AsyncGenerator]) -> AsyncGenerator:
        context = current_task_context()
        meter = context.meter if context is not None else None
        if meter is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return
        # Counted before anything awaits, so calls of one batch that start together cannot all pass the last slot.
        reached = meter.spent(tools=True)
        if reached is not None:
            meter.refused = meter.refused or reached[0]
            call = input_kwargs["tool_call"]
            yield ToolResponse(
                content=[TextBlock(text=f"{call.name} was not run: {BudgetExceeded(*reached)}.")],
                state=ToolResultState.ERROR,
            )
            return
        meter.tool_calls += 1
        meter.changed()
        async for item in next_handler(**input_kwargs):
            yield item


def model_price(input_per_mtok: int | None, output_per_mtok: int | None) -> ModelPrice | None:
    """The price of a model, when both what it charges for input and for output are known."""
    if input_per_mtok is None or output_per_mtok is None:
        return None
    return ModelPrice(input_per_mtok, output_per_mtok)

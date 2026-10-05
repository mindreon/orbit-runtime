"""The arithmetic of a task's budget (05 §4): what is left, what an attempt reserves, what it spent.

Pure and deterministic: the workflow keeps the state (limits, settled usage, the reservations of live attempts) and asks
here. A limit that is missing (None) is no limit, and nothing is reserved against it.
"""

from __future__ import annotations

from collections.abc import Iterable

from orbit_contracts.v3.common import Budget, Usage

FIELDS = ("tokens", "tool_calls", "wall_s", "cost_usd_micros")


def spent(usage: Usage, field: str) -> int:
    """What `usage` counts against the budget's `field`. An unknown cost counts as nothing."""
    if field == "tokens":
        return usage.tokens_in + usage.tokens_out
    value = getattr(usage, field)
    return 0 if value is None else int(value)


def usage_add(left: Usage, right: Usage) -> Usage:
    """The sum of two usages. A cost is unknown only while both sides are unknown."""
    cost = None if left.cost_usd_micros is None and right.cost_usd_micros is None else (
        (left.cost_usd_micros or 0) + (right.cost_usd_micros or 0)
    )
    return Usage(
        tokens_in=left.tokens_in + right.tokens_in,
        tokens_out=left.tokens_out + right.tokens_out,
        tool_calls=left.tool_calls + right.tool_calls,
        wall_s=left.wall_s + right.wall_s,
        cost_usd_micros=cost,
    )


def is_empty(usage: Usage) -> bool:
    return not any(spent(usage, field) for field in FIELDS)


def reserved_total(reservations: Iterable[Budget]) -> dict[str, int]:
    totals = dict.fromkeys(FIELDS, 0)
    for reservation in reservations:
        for field in FIELDS:
            value = getattr(reservation, field)
            if value is not None:
                totals[field] += value
    return totals


def remaining(limits: Budget, usage: Usage, reservations: Iterable[Budget]) -> dict[str, int | None]:
    """What the task can still hand to an attempt: its limit less what was spent and what running attempts hold. None
    where the task has no limit."""
    held = reserved_total(reservations)
    out: dict[str, int | None] = {}
    for field in FIELDS:
        limit = getattr(limits, field)
        out[field] = None if limit is None else max(0, limit - spent(usage, field) - held[field])
    return out


def reserve(
    node_budget: Budget, left: dict[str, int | None], share: int = 1, pool: dict[str, int | None] | None = None
) -> tuple[Budget | None, str]:
    """The budget an attempt of a node reserves, or None and why when the task cannot cover it. A field the node names is
    reserved as it is (and must fit what is left); a field it leaves open takes `1/share` of `pool` (what was left when
    the attempts that start together were chosen; `left` when not given), but never more than is left now, so attempts that
    start together do not take it all. A field the task has no limit for is no limit."""
    pool = left if pool is None else pool
    values: dict[str, int | None] = {}
    for field in FIELDS:
        wanted, room = getattr(node_budget, field), left[field]
        if wanted is not None:
            if room is not None and wanted > room:
                return None, f"{field}: the node needs {wanted} and {room} is left"
            values[field] = wanted
        elif room is None:
            values[field] = None
        elif room <= 0:
            return None, f"{field}: nothing is left"
        else:
            values[field] = min(room, ((pool[field] or 0) // max(1, share)) or room)
    return Budget(**values), ""


def after(budget: Budget, usage: Usage) -> Budget:
    """What is left of an attempt's reserved budget once `usage` is spent."""
    return Budget(**{
        field: None if getattr(budget, field) is None else max(0, int(getattr(budget, field)) - spent(usage, field))
        for field in FIELDS
    })


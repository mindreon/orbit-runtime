"""Durable events a worker owns, written to the runtime outbox (09 §3).

Entity version 0 keeps them out of the workflow's version sequence for the attempt: control ignores it there.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from orbit_orch.plan_engine import deterministic_id


class EventStore(Protocol):
    async def publish_events(self, events: list[dict[str, Any]]) -> None: ...


async def publish_attempt_event(
    store: EventStore,
    *,
    tenant_id: str,
    task_id: str,
    attempt_id: str,
    event_type: str,
    body: dict[str, Any],
    seed: str,
    shown_attempt_id: str | None = None,
) -> None:
    """Appends one durable event of an attempt. The same `seed` is the same event, so a retry cannot duplicate it. `attempt_id`
    makes the event's id; `shown_attempt_id` is the attempt it is about when that is another (an agent of a team stage runs under
    an attempt id of its own, and the stage's is what a consumer knows)."""
    shown = shown_attempt_id or attempt_id
    await store.publish_events([{
        "schema": "orbit.event/3",
        "event_id": deterministic_id(f"{attempt_id}:{seed}", "evt"),
        "tenant_id": tenant_id,
        "task_id": task_id,
        "type": event_type,
        "retention": "durable",
        "source": {"kind": "worker", "id": "orbit-worker", "attempt_id": shown},
        "entity": {"kind": "attempt", "id": shown, "version": 0},
        "visibility": "tenant",
        "occurred_at": datetime.now(UTC).isoformat(),
        "payload": body,
    }])

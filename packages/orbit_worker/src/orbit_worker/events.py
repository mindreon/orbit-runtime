"""Normalized Orbit events. Token streams stay out of Temporal history."""

from datetime import UTC, datetime
from uuid import uuid4

import aiohttp
from orbit_contracts.models import OrbitEvent


class MemoryEventIngest:
    """Records events in process. Tests read this list."""

    def __init__(self) -> None:
        self.events: list[OrbitEvent] = []

    async def emit(self, event: OrbitEvent) -> None:
        self.events.append(_complete(event))


class HttpEventIngest:
    """POST one event to control's internal ingest. Failures are counted.

    A down control plane must not drop an already persisted AgentState, so
    emit logs the failure on this object and returns.
    """

    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._token = token
        self.failures = 0

    async def emit(self, event: OrbitEvent) -> None:
        if not self._url:
            return
        body = _complete(event).model_dump(mode="json")
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session, session.post(
                self._url,
                json=body,
                headers={"Authorization": f"Bearer {self._token}"},
            ) as response:
                if response.status >= 300:
                    self.failures += 1
        except aiohttp.ClientError:
            self.failures += 1


def _complete(event: OrbitEvent) -> OrbitEvent:
    if event.event_id and event.occurred_at:
        return event
    return event.model_copy(
        update={
            "event_id": event.event_id or uuid4().hex,
            "occurred_at": event.occurred_at or datetime.now(UTC).isoformat(),
        }
    )

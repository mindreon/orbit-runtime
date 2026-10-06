"""Waits for a condition, bounded by the wall clock and never by a count of polls.

A count of polls is a bound in the speed of the machine: a runner that answers each query faster than the one the test was
written on reaches the count before what is waited for (a retry that sits out its 5s backoff, a worker that starts late)
has happened. A deadline measures the time instead. No time is skipped here: a test that must cross a backoff or a timer
faster says so itself, and only while no gated activity is held.
"""

import asyncio
from collections.abc import AsyncIterator

DEFAULT_TIMEOUT_S = 60.0
INTERVAL_S = 0.05


async def polls(timeout: float = DEFAULT_TIMEOUT_S, interval: float = INTERVAL_S) -> AsyncIterator[int]:
    """Yields until the deadline, sleeping between yields: `async for _ in polls(): if ready(): break`. The loop ends without
    a break when the deadline passes, so a caller follows it with its assertion or `raise`."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    count = 0
    while loop.time() < end:
        yield count
        count += 1
        await asyncio.sleep(interval)

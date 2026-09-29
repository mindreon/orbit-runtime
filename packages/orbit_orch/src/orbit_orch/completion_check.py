"""The completion check of an attempt that reports `completed` (04 §5, 17 G15).

`attemptFinished(completed)` is the production path to a completed node, so it is checked like a `proposeCompletion`:
the proposal is built from what the attempt reported, and the reasons a check gave are cut down to what fits in a
workflow's state and an event. Pure functions over contract types; nothing here does I/O.
"""

from __future__ import annotations

import hashlib
from typing import Any

from orbit_contracts.v3 import AttemptFinishedSignal, CompletionProposal

MAX_REASONS = 10
MAX_REASON_CHARS = 300
MAX_MESSAGE_CHARS = 2000


def finished_proposal(signal: AttemptFinishedSignal) -> CompletionProposal | None:
    """What the attempt claims: its manifest and checkpoint. The attempt reports no structured output yet, so the
    output is empty and a contract that asks for a schema-valid output cannot be met until the worker produces one.
    None when the attempt reported no result at all."""
    if signal.result is None:
        return None
    return CompletionProposal(
        # One completion per attempt: the same id on a redelivered report, a different one for every attempt.
        command_id=hashlib.sha256(f"{signal.attempt_id}|attemptFinished".encode()).hexdigest(),
        node_id=signal.node_id,
        attempt_id=signal.attempt_id,
        output={},
        artifact_manifest_id=signal.result.manifest_id,
        checkpoint_ref=signal.result.checkpoint_ref,
    )


def missing_result() -> list[dict[str, Any]]:
    return [
        {
            "check": "completion",
            "code": "attempt_result_missing",
            "message": "the attempt reported completed without a result",
            "detail": {},
        }
    ]


def summarize(failures: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The structured reasons without their `detail` (a command's output tail can be kilobytes): check, code and a
    bounded message, at most MAX_REASONS of them."""
    return [
        {
            "check": str(item.get("check", "")),
            "code": str(item.get("code", "")),
            "message": str(item.get("message", ""))[:MAX_REASON_CHARS],
        }
        for item in failures[:MAX_REASONS]
    ]


def failure_message(reasons: list[dict[str, str]]) -> str:
    """The text of `Failure.message` and of the node's status change: every reason as `[check/code] message`."""
    lines = "; ".join(f"[{item['check']}/{item['code']}] {item['message']}" for item in reasons)
    return f"completion verification failed: {lines}"[:MAX_MESSAGE_CHARS]

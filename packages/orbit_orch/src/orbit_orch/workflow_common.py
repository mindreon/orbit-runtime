"""What the task and attempt workflow modules share: timeouts, limits, the versioning switch and pure helpers.

Everything here is deterministic. `versioning_behavior` reads the process's versioning switch once, when a workflow
module is imported to build its `@workflow.defn`; it is never called while a workflow runs.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy, VersioningBehavior

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import Budget

    from orbit_orch.settings import versioning_settings

RETRY = RetryPolicy(maximum_attempts=3)
IO_TIMEOUT = timedelta(minutes=2)
AGENT_TIMEOUT = timedelta(hours=1)
# Restoring the snapshot and starting the sandbox come on top of a command's own timeout.
COMMAND_SETUP_S = 300
HEARTBEAT = timedelta(seconds=30)
OPERATOR_HELD = frozenset({"PAUSED", "PAUSED_NEEDS_REVIEW", "TAKEN_OVER"})
MAX_UPDATES_BEFORE_CAN = 1000
MAX_COMPLETIONS_BEFORE_CAN = 50
# An attempt that ends `completed` goes through the node's completion checks before the node is frozen, and an attempt
# that was rejected or replaced cannot complete the node by reporting late (17 G15).
VERIFY_FINISHED_ATTEMPTS = "task-attempt-completion-verification"


def versioning_behavior(behavior: VersioningBehavior) -> VersioningBehavior:
    """Use deployment versioning only when the worker is registered for it. Evaluated when the module is imported,
    from the process's one reading of the switch; never while a workflow runs."""
    if versioning_settings().enabled:
        return behavior
    return VersioningBehavior.UNSPECIFIED


def sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def reasons(result: dict[str, Any], activity_name: str) -> list[dict[str, Any]]:
    """The structured failure reasons of a verification activity's answer. An activity that says `ok: false` without
    naming a reason still refuses the completion."""
    failures = list(result.get("failures", []))
    if not failures and not result.get("ok", True):
        failures = [{"check": "verification", "code": "verification_failed", "message": f"{activity_name} refused the completion", "detail": {}}]
    return failures


def closed(status: str) -> bool:
    return status in {"COMPLETED", "FAILED", "CANCELLED"}


def budget_add(a: Budget, b: Budget) -> Budget:
    values: dict[str, int | None] = {}
    for name in ("tokens", "tool_calls", "wall_s", "cost_usd_micros"):
        left, right = getattr(a, name), getattr(b, name)
        values[name] = None if left is None and right is None else (left or 0) + (right or 0)
    return Budget(**values)

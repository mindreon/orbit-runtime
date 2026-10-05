"""What happens to a node when its attempt does not complete it (04 §2): retries with a backoff, a limit, and the hand-over
to a person when they are used up.

A failed attempt counts against its node. While the node has tries left it is RETRY_PENDING and waits out a backoff; the
next attempt carries on the agent session of the one that failed and is told why it was rejected. When the failure cannot
be retried, or the tries are used up, the node is BLOCKED and the task asks for a review. A person's `resume`, or a plan
change on the node, puts it back in play with the count at zero. An attempt that was cancelled (interrupt, stop) did not
fail: it does not count and does not wait, and its replacement carries on its session.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orbit_orch.workflow_common import (
        DEFAULT_MAX_ATTEMPTS,
        INTERRUPT_CONTINUES_SESSION,
        OPERATOR_HELD,
        RETRY_BACKOFF_S,
        RETRY_POLICY,
        closed,
    )

from orbit_orch.task_budget import TaskBudget


class TaskRetry(TaskBudget):
    def _entry(self, node_id: str) -> dict[str, Any]:
        return self._node_retry.setdefault(
            node_id, {"failed": 0, "blocked": False, "retry_at": None, "continue_from": "", "reason": ""}
        )

    def _max_attempts(self, node_id: str) -> int:
        retry = self._require_node(node_id).draft.retry
        return retry.max_attempts if retry is not None else DEFAULT_MAX_ATTEMPTS

    def _next_backoff(self) -> float | None:
        """How long the main loop may sleep before a node's backoff ends, or None when none is waiting."""
        if self._plan is None or self._status in OPERATOR_HELD or closed(self._status):
            return None
        waits = [
            self._backoff_remaining(node_id)
            for node_id, state in self._plan.nodes.items()
            if state.status == "RETRY_PENDING" and node_id not in self._attempts
        ]
        waits = [seconds for seconds in waits if seconds > 0]
        return min(waits) if waits else None

    def _retry_or_block(
        self,
        node_id: str,
        attempt: dict[str, Any] | None,
        *,
        failure_class: str,
        retryable: bool,
        message: str,
    ) -> None:
        """The attempt ended without completing the node. Without the retry policy (a history from before it) the node
        simply tries again at once."""
        if not workflow.patched(RETRY_POLICY):
            self._set_node_status(node_id, "RETRY_PENDING", reason=message)
            return
        if attempt is not None:
            if attempt.get("counted"):
                return  # a rejected proposal was counted when it was judged; the attempt's own end is the same failure
            attempt["counted"] = True
        entry = self._entry(node_id)
        entry["failed"] = int(entry["failed"]) + 1
        entry["continue_from"] = str((attempt or {}).get("attempt_id") or entry.get("continue_from", ""))
        entry["reason"] = message[:2000]
        tries = self._max_attempts(node_id)
        if not retryable or entry["failed"] >= tries:
            entry["blocked"], entry["retry_at"] = True, None
            # What blocked it, so a grant of budget frees the nodes that waited for it and no others.
            entry["blocked_for"] = failure_class
            why = (
                f"{failure_class} failure that cannot be retried: {message}"
                if not retryable
                else f"{entry['failed']} of {tries} attempts failed, retries are used up: {message}"
            )
            self._set_node_status(node_id, "BLOCKED", reason=why)
            if self._status not in OPERATOR_HELD and not closed(self._status):
                self._set_status("PAUSED_NEEDS_REVIEW", f"node {node_id} is blocked: {why}")
                self._budget_hold = failure_class == "budget"
            return
        delay = RETRY_BACKOFF_S[min(int(entry["failed"]), len(RETRY_BACKOFF_S)) - 1]
        entry["retry_at"] = (workflow.now() + timedelta(seconds=delay)).isoformat()
        self._set_node_status(node_id, "RETRY_PENDING", reason=message)

    def _attempt_cancelled(self, node_id: str, attempt: dict[str, Any] | None) -> None:
        """An interrupted or stopped attempt: the node tries again at once, on the session the attempt had, and the
        attempt is not held against it."""
        if not workflow.patched(INTERRUPT_CONTINUES_SESSION):
            self._set_node_status(node_id, "RETRY_PENDING")
            return
        if attempt is not None and attempt.get("counted"):
            return  # already counted as failed: the node is waiting out its backoff, or blocked
        if attempt is not None:
            entry = self._entry(node_id)
            entry["continue_from"], entry["reason"], entry["retry_at"] = str(attempt.get("attempt_id", "")), "", None
        if self._require_node(node_id).status not in {"BLOCKED", "COMPLETED"}:
            self._set_node_status(node_id, "RETRY_PENDING")

    def _take_retry_input(self, node_id: str) -> tuple[str, str]:
        """The attempt a new attempt of the node carries on, and why the one before it was rejected."""
        entry = self._node_retry.get(node_id)
        if not entry:
            return "", ""
        reason, entry["reason"] = str(entry.get("reason", "")), ""
        return str(entry.get("continue_from", "")), reason

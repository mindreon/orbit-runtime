"""The budget of a TaskWorkflow (05 §4): reserving it for an attempt, settling what the attempt spent, and the hold a task
is in when it cannot pay for the next node.

When an attempt starts, the task reserves the node's budget from what it has left (its limits, less what was spent and
what running attempts hold) and hands it to the attempt, which spends only within it. When the attempt ends, what it
spent is added to the task's usage and the reservation is gone, so what it did not use is back. That is what keeps
attempts that run side by side from each spending "what was left when they started". A node the task cannot cover is not
started: the task asks for a review (PAUSED_NEEDS_REVIEW) and only a grant resumes it.
"""

from __future__ import annotations

from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import Budget
    from orbit_contracts.v3.common import Usage

    from orbit_orch import budgets
    from orbit_orch.workflow_common import BUDGET_ENFORCEMENT, OPERATOR_HELD, closed

from orbit_orch.task_plan import TaskPlan


class TaskBudget(TaskPlan):
    def _remaining(self) -> dict[str, int | None]:
        """What the task can still hand to an attempt, per limit (None where it has none)."""
        return budgets.remaining(
            self._budgets,
            self._usage,
            (Budget.model_validate(item["reserved"]) for item in self._attempts.values() if item.get("reserved")),
        )

    def _reserve_for(
        self, node_id: str, share: int, pool: dict[str, int | None] | None
    ) -> tuple[Budget | None, str]:
        """The budget an attempt of the node reserves, or None and why when the task cannot cover it."""
        return budgets.reserve(self._require_node(node_id).draft.budget, self._remaining(), share, pool)

    def _hold_for_budget(self, node_id: str, detail: str) -> None:
        """The task cannot pay for this node: it is not started, and the task asks for a review until budget is granted."""
        self._emit("budget.exhausted", {"scope": "task", "node_id": node_id, "detail": detail})
        if self._status not in OPERATOR_HELD and not closed(self._status):
            self._set_status("PAUSED_NEEDS_REVIEW", f"budget exhausted: node {node_id} cannot start ({detail})")
            self._budget_hold = True

    def _settle_attempt(self, attempt: dict[str, Any], usage: Usage | None) -> None:
        """An attempt has ended: what it spent joins the task's usage (and is recorded as a durable event), and what it held
        is released, so whatever it did not use is the task's again. An attempt that reported nothing (cancelled before its
        activity returned) spent nothing that is known."""
        if not workflow.patched(BUDGET_ENFORCEMENT):
            return
        attempt.pop("reserved", None)
        if usage is None or budgets.is_empty(usage):
            return
        self._usage = budgets.usage_add(self._usage, usage)
        self._emit("usage.recorded", {
            "attempt_id": str(attempt.get("attempt_id", "")),
            "usage": usage.model_dump(mode="json", exclude_none=True),
        })

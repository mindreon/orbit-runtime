"""Task status, events and the activity that publishes them."""

from __future__ import annotations

import json
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import AttemptFinishedSignal, Budget, TaskView
    from orbit_contracts.v3.common import Failure, Usage

    from orbit_orch.budgets import reserved_total
    from orbit_orch.plan_engine import deterministic_id
    from orbit_orch.workflow_common import (
        ATTEMPT_OUTPUT_EVENT,
        ATTEMPT_OUTPUT_EVENT_BYTES,
        BUDGET_ENFORCEMENT,
        EVENT_PAYLOADS,
        IO_TIMEOUT,
        NODE_ENTITY_VERSIONS,
        RETRY,
        sha,
    )

from orbit_orch.task_base import TaskWorkflowBase


class TaskEvents(TaskWorkflowBase):
    def _set_status(self, target: str, reason: str) -> None:
        """Move the task to `target` and record the change as a durable event.

        The projection derives task status only from events, so every transition
        must come through here. `patched` keeps histories recorded before status
        events existed replayable.
        """
        if target == self._status:
            return
        previous, self._status = self._status, target
        if target != "PAUSED_NEEDS_REVIEW":
            self._budget_hold = False  # held for budget only while it waits for a review
        if workflow.patched("task-status-events"):
            self._emit("task.status_changed", {"from_status": previous, "to_status": target, "reason": reason})

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        entity_kind = "task"
        if event_type.startswith("plan."):
            entity_kind = "plan"
        elif event_type.startswith("attempt."):
            entity_kind = "attempt"
        elif event_type.startswith("approval."):
            entity_kind = "approval"
        elif event_type.startswith("message."):
            entity_kind = "message"
        entity_id = self._task_id if entity_kind in {"task", "plan"} else str(
            payload.get("attempt_id") or payload.get("approval_id") or payload.get("message_seq") or self._task_id
        )
        node_events = event_type.startswith("node.") and "node_id" in payload and workflow.patched(NODE_ENTITY_VERSIONS)
        if node_events:
            entity_kind, entity_id = "node", str(payload["node_id"])
        version_key = f"{entity_kind}:{entity_id}"
        if node_events and version_key not in self._entity_versions:
            # A node's first version carries on from the task's counter, which numbered its events until now: a projection
            # that holds the row at one of those versions must still find the next event newer.
            self._entity_versions[version_key] = self._entity_versions.get(f"task:{self._task_id}", 0)
        self._entity_versions[version_key] = self._entity_versions.get(version_key, 0) + 1
        self._events.append({
            "schema": "orbit.event/3",
            "event_id": deterministic_id(str(workflow.uuid4()), "evt"),
            "tenant_id": self._tenant_id,
            "task_id": self._task_id,
            "type": event_type,
            "retention": "ephemeral" if event_type.endswith(".delta") else "durable",
            "source": {"kind": "workflow", "id": workflow.info().workflow_id},
            "entity": {"kind": entity_kind, "id": entity_id, "version": self._entity_versions[version_key]},
            "visibility": "tenant",
            "occurred_at": workflow.now().isoformat(),
            "payload": payload,
        })

    async def _flush_events(self) -> None:
        if not self._events:
            return
        events, self._events = self._events, []
        await self._run_short_activity("publish_events", events)

    async def _run_short_activity(self, name: str, payload: Any) -> dict[str, Any]:
        return await workflow.execute_activity(
            name,
            payload,
            task_queue="orbit.io",
            result_type=dict,
            start_to_close_timeout=IO_TIMEOUT,
            retry_policy=RETRY,
        )

    def _task_view(self) -> TaskView:
        return TaskView(
            task_id=self._task_id,
            status=self._status,  # type: ignore[arg-type]
            plan_version=self._plan.version if self._plan else 1,
            pending_approvals=[
                approval_id for approval_id, item in self._approvals.items() if item.get("status") == "PENDING"
            ],
            budgets=self._budgets,
            usage=self._usage,
            budget_reserved=self._reserved_budget(),
            config=self._config,
        )

    def _reserved_budget(self) -> Budget:
        """What running attempts hold of the task's budget and have not settled (05 §4)."""
        held = reserved_total(
            Budget.model_validate(item["reserved"]) for item in self._attempts.values() if item.get("reserved")
        )
        return Budget(**{field: value or None for field, value in held.items()})

    def _attempt_facts(self, attempt: dict[str, Any] | None) -> dict[str, Any]:
        """Which attempt of its node an ending attempt was, and what it ran as (`task-event-payloads-v2`)."""
        if not attempt or not workflow.patched(EVENT_PAYLOADS):
            return {}
        return {
            key: attempt[key]
            for key in ("attempt_no", "profile", "config_version")
            if attempt.get(key) is not None
        }

    def _emit_attempt_finished(
        self,
        signal: AttemptFinishedSignal,
        outcome: str,
        failure: Failure | None,
        usage: Usage | None = None,
        attempt: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "node_id": signal.node_id,
            "attempt_id": signal.attempt_id,
            "outcome": outcome,
            "failure": failure.model_dump(mode="json") if failure else None,
        }
        payload.update(self._attempt_facts(attempt))
        if signal.result and signal.result.output and workflow.patched(ATTEMPT_OUTPUT_EVENT):
            if len(json.dumps(signal.result.output).encode()) <= ATTEMPT_OUTPUT_EVENT_BYTES:
                payload["output"] = signal.result.output
            else:
                payload["output_truncated"] = True
        if workflow.patched(BUDGET_ENFORCEMENT) and usage is not None:
            payload["usage"] = usage.model_dump(mode="json", exclude_none=True)
        self._emit("attempt.finished", payload)

    def _emit_manifest_created(self, signal: AttemptFinishedSignal) -> None:
        if signal.result and signal.result.manifest_id:
            self._emit("artifact.manifest_created", {
                "manifest_id": signal.result.manifest_id,
                "attempt_id": signal.attempt_id,
                "entries": signal.result.manifest_entries,
                "manifest_hash": signal.result.manifest_hash or sha(signal.result.manifest_id),
            })

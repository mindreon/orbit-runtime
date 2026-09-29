"""Task status, events and the activity that publishes them."""

from __future__ import annotations

from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import AttemptFinishedSignal, TaskView
    from orbit_contracts.v3.common import Failure

    from orbit_orch.plan_engine import deterministic_id
    from orbit_orch.workflow_common import IO_TIMEOUT, RETRY, sha

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
        version_key = f"{entity_kind}:{entity_id}"
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
        )

    def _emit_attempt_finished(self, signal: AttemptFinishedSignal, outcome: str, failure: Failure | None) -> None:
        self._emit("attempt.finished", {
            "node_id": signal.node_id,
            "attempt_id": signal.attempt_id,
            "outcome": outcome,
            "failure": failure.model_dump(mode="json") if failure else None,
        })

    def _emit_manifest_created(self, signal: AttemptFinishedSignal) -> None:
        if signal.result and signal.result.manifest_id:
            self._emit("artifact.manifest_created", {
                "manifest_id": signal.result.manifest_id,
                "attempt_id": signal.attempt_id,
                "entries": signal.result.manifest_entries,
                "manifest_hash": signal.result.manifest_hash or sha(signal.result.manifest_id),
            })

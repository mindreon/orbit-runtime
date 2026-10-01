"""Attempts of a TaskWorkflow: scheduling ready nodes, starting and cancelling AttemptWorkflows, and
what an attempt tells its parent."""

from __future__ import annotations

import asyncio
from typing import Any

from temporalio import workflow
from temporalio.exceptions import TemporalError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        ApprovalDecidedSignal,
        AttemptFinishedSignal,
        AttemptParkedSignal,
        AttemptWorkflowInput,
        DeliverMessagesSignal,
        ExternalEventSignal,
    )

    from orbit_orch.plan_engine import attempt_workflow_id, deterministic_id
    from orbit_orch.workflow_common import VERIFY_FINISHED_ATTEMPTS

from orbit_orch.attempt_workflow import AttemptWorkflow
from orbit_orch.task_completion import TaskCompletion


class TaskAttempts(TaskCompletion):
    async def _schedule_ready(self) -> None:
        if self._status in {"PAUSED", "PAUSED_NEEDS_REVIEW", "TAKEN_OVER", "CANCELLED"}:
            return
        if self._plan is None:
            return
        for node_id, state in sorted(self._plan.nodes.items()):
            if state.status not in {"READY", "RETRY_PENDING"} or node_id in self._attempts:
                continue
            if not self._dependencies_done(node_id):
                continue
            node = state.draft
            if node.type in {"agent_turn", "sop_stage"}:
                if node.workspace_access == "write" and any(
                    item.get("workspace_access") == "write" for item in self._attempts.values()
                ):
                    continue
                await self._start_attempt(node_id)
            elif node.type == "approval":
                self._set_node_status(node_id, "AWAITING_APPROVAL")
            elif node.type == "wait":
                self._set_node_status(node_id, "AWAITING_INPUT")
                wait_key = getattr(node.spec, "wait_key", None)
                timer_s = getattr(node.spec, "timer_s", None)
                if timer_s:
                    asyncio.create_task(self._complete_after(node_id, timer_s))
                elif wait_key and wait_key in self._wait_events:
                    self._set_node_status(node_id, "COMPLETED")
            elif node.type == "checkpoint":
                self._set_node_status(node_id, "VERIFYING")
                await self._run_short_activity("checkpoint_commit", self._checkpoint_payload(node_id))
                self._set_node_status(node_id, "COMPLETED")
                self._completed_nodes += 1

    def _attempt_profile(self, owner_profile: str | None) -> str:
        """A node that names a profile of its own (a team member) keeps it. One that only carries the task's
        profile follows the task's expert, if it has one (15 M8)."""
        if owner_profile and owner_profile != self._profile:
            return owner_profile
        team = self._config.team
        if team is not None:
            return next(member.expert for member in team.members if member.role == team.leader)
        return self._config.expert or self._profile

    async def _start_attempt(self, node_id: str) -> None:
        state = self._require_node(node_id)
        attempt_no = state.attempt_count + 1
        attempt_id = deterministic_id(f"{self._task_id}:{node_id}:{attempt_no}", "att")
        workflow_id = attempt_workflow_id(self._task_id, node_id, attempt_no)
        spec = state.draft.spec
        goal = getattr(spec, "goal", None) or getattr(spec, "sop", None) or state.draft.title
        # A follow-up's goal is the message it was made from; only what came after that message is passed on as messages.
        follow_up = self._follow_ups.get(node_id)
        messages = self._inbox if follow_up is None else [m for m in self._inbox if m.message_seq > int(follow_up["seq"])]
        inp = AttemptWorkflowInput(
            task_id=self._task_id,
            tenant_id=self._tenant_id,
            node_id=node_id,
            attempt_id=attempt_id,
            attempt_no=attempt_no,
            node_type=state.draft.type,
            profile=self._attempt_profile(state.draft.owner_profile),
            goal=goal,
            policy=self._policy,
            workspace_access=state.draft.workspace_access or "none",
            messages=messages,
            config=self._config,
            continue_from=(follow_up or {}).get("from") or None,
        )
        self._last_attempt_id = attempt_id
        handle = await workflow.start_child_workflow(
            AttemptWorkflow.run,
            inp,
            id=workflow_id,
            parent_close_policy=workflow.ParentClosePolicy.ABANDON,
            cancellation_type=workflow.ChildWorkflowCancellationType.WAIT_CANCELLATION_COMPLETED,
        )
        self._attempts[node_id] = {
            "attempt_id": attempt_id,
            "attempt_no": attempt_no,
            "workflow_id": workflow_id,
            "workspace_access": state.draft.workspace_access or "none",
            "status": "RUNNING",
        }
        self._attempt_handles[node_id] = handle
        self._update_node(node_id, current_attempt_id=attempt_id, attempt_count=attempt_no, status="RUNNING")
        self._emit("attempt.started", {
            "node_id": node_id,
            "attempt_id": attempt_id,
            "attempt_no": attempt_no,
            "profile": inp.profile,
            "config_version": self._config.config_version,
        })

    async def _complete_after(self, node_id: str, seconds: int) -> None:
        await workflow.sleep(seconds)
        if self._plan and self._plan.nodes.get(node_id, None) and self._plan.nodes[node_id].status == "AWAITING_INPUT":
            self._set_node_status(node_id, "COMPLETED")
            self._completed_nodes += 1

    @workflow.signal(name="attemptFinished")
    async def attempt_finished(self, signal: AttemptFinishedSignal) -> None:
        self._wake += 1
        attempt = self._attempts.get(signal.node_id)
        if attempt is None or attempt.get("attempt_no") != signal.attempt_no:
            return
        if workflow.patched(VERIFY_FINISHED_ATTEMPTS):
            await self._finish_attempt(signal, attempt)
            return
        attempt["status"] = signal.outcome
        attempt["result"] = signal.result.model_dump(mode="json") if signal.result else None
        node = self._require_node(signal.node_id)
        if workflow.patched("task-event-vocabulary"):
            self._emit_attempt_finished(signal, signal.outcome, signal.failure)
        if signal.outcome == "completed":
            self._emit_manifest_created(signal)
            if self._plan.nodes[signal.node_id].status == "BLOCKED":
                # The agent declared its own node unplannable (05 §2): it stays blocked and a person takes over.
                self._set_status("PAUSED_NEEDS_REVIEW", "the agent declared the task unplannable")
            elif self._exploration_spent_without_plan(signal):
                # 05 §4: the budget ran out and nothing was planned, so a person decides what happens next.
                self._emit("budget.exhausted", {"scope": "exploration", "node_id": signal.node_id})
                self._set_node_status(signal.node_id, "BLOCKED")
                self._set_status("PAUSED_NEEDS_REVIEW", "exploration budget spent without a plan")
            else:
                self._set_node_status(signal.node_id, "COMPLETED")
                self._completed_nodes += 1
        elif signal.outcome == "cancelled":
            self._set_node_status(signal.node_id, "RETRY_PENDING")
        else:
            self._set_node_status(signal.node_id, "RETRY_PENDING")
        if node.current_attempt_id:
            self._update_node(signal.node_id, current_attempt_id=None)
            self._attempts.pop(signal.node_id, None)
            self._attempt_handles.pop(signal.node_id, None)

    @workflow.signal(name="attemptParked")
    async def attempt_parked(self, signal: AttemptParkedSignal) -> None:
        self._wake += 1
        attempt = self._attempts.get(signal.node_id)
        if attempt is None or attempt.get("attempt_no") != signal.attempt_no:
            return
        attempt["status"] = "PARKED_" + signal.reason.upper()
        node_status = "AWAITING_APPROVAL" if signal.reason == "approval" else "AWAITING_INPUT"
        self._set_node_status(signal.node_id, node_status)
        for parked in signal.approvals:
            approval_id = deterministic_id(f"{signal.attempt_id}:{parked.tool_call_id}", "apr")
            self._approvals[approval_id] = {
                "approval_id": approval_id,
                "status": "PENDING",
                "attempt_id": signal.attempt_id,
                "tool_call_id": parked.tool_call_id,
                "subject": parked.subject.model_dump(mode="json"),
            }
            self._emit("approval.requested", {
                "approval_id": approval_id,
                "attempt_id": signal.attempt_id,
                "node_id": signal.node_id,
                "tool_call_id": parked.tool_call_id,
                "subject": parked.subject.model_dump(mode="json"),
            })
        if workflow.patched("task-event-vocabulary"):
            self._emit("attempt.parked", {
                "node_id": signal.node_id,
                "attempt_id": signal.attempt_id,
                "reason": signal.reason,
                "question": signal.question,
            })

    @workflow.signal(name="approvalDecided")
    async def approval_decided(self, signal: ApprovalDecidedSignal) -> None:
        for approval in self._approvals.values():
            if approval.get("approval_id") == signal.approval_id:
                approval["status"] = "APPROVED" if signal.decision == "approve" else "REJECTED"

    @workflow.signal(name="deliverMessages")
    async def deliver_messages(self, signal: DeliverMessagesSignal) -> None:
        self._wake += 1
        for message in signal.messages:
            if message in self._inbox:
                self._inbox.remove(message)

    @workflow.signal(name="externalEvent")
    async def external_event(self, signal: ExternalEventSignal) -> None:
        self._wait_events[signal.wait_key] = signal.payload
        if self._plan is not None:
            for node_id, state in self._plan.nodes.items():
                if state.status != "AWAITING_INPUT" or state.draft.type != "wait":
                    continue
                if getattr(state.draft.spec, "wait_key", None) == signal.wait_key:
                    self._set_node_status(node_id, "COMPLETED", frozen=True)
                    self._completed_nodes += 1
        self._wake += 1

    def _resume_parked(self, attempt_id: str) -> None:
        """A parked attempt got what it waited for: its node and the attempt run again."""
        for node_id, item in self._attempts.items():
            if item.get("attempt_id") == attempt_id and str(item.get("status", "")).startswith("PARKED_"):
                item["status"] = "RUNNING"
                self._set_node_status(node_id, "RUNNING")
                self._wake += 1
                return

    def _active_attempt_id(self) -> str | None:
        return next((item.get("attempt_id") for item in self._attempts.values() if item.get("status") == "RUNNING"), None)

    async def _interrupt_active_attempt(self) -> None:
        for node_id, item in list(self._attempts.items()):
            if item.get("status") in {"RUNNING", "PARKED_INPUT", "PARKED_APPROVAL"}:
                handle = workflow.get_external_workflow_handle(item["workflow_id"])
                try:
                    await handle.cancel()
                except TemporalError:
                    pass
                asyncio.create_task(self._await_cancelled(node_id))

    async def _cancel_active_attempts(self) -> None:
        for node_id, item in list(self._attempts.items()):
            try:
                await workflow.get_external_workflow_handle(item["workflow_id"]).cancel()
            except TemporalError:
                pass
            asyncio.create_task(self._await_cancelled(node_id))

    async def _await_cancelled(self, node_id: str) -> None:
        # A cancelled child cannot always send a final signal because the
        # cancellation also cancels its workflow task. Keep the node in
        # CANCELLING while Temporal propagates the request, then release it
        # after a deterministic grace period so a replacement attempt cannot
        # overlap the cancelled one.
        await workflow.sleep(1)
        if node_id in self._attempts:
            self._mark_attempt_cancelled(node_id)

    def _mark_attempt_cancelled(self, node_id: str) -> None:
        attempt = self._attempts.pop(node_id, None)
        self._attempt_handles.pop(node_id, None)
        if attempt is None or self._plan is None:
            return
        if workflow.patched("task-event-vocabulary"):
            # A cancelled child cannot always signal its own end (see _await_cancelled), so the parent records it.
            self._emit("attempt.finished", {
                "node_id": node_id,
                "attempt_id": attempt.get("attempt_id"),
                "outcome": "cancelled",
                "failure": None,
            })
        state = self._require_node(node_id)
        if state.current_attempt_id == attempt.get("attempt_id"):
            self._update_node(node_id, current_attempt_id=None, status="RETRY_PENDING")

    async def _signal_attempt(self, attempt_id: str, signal_name: str, payload: Any) -> None:
        for item in self._attempts.values():
            if item.get("attempt_id") == attempt_id:
                await workflow.get_external_workflow_handle(item["workflow_id"]).signal(signal_name, payload)
                return

"""The updates a person or the API sends to a TaskWorkflow, and the queue of commands the main loop drains."""

from __future__ import annotations

from typing import Any

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        ApprovalDecidedSignal,
        DecideApprovalInput,
        DecideApprovalResult,
        DeliverMessagesSignal,
        GrantBudgetInput,
        GrantBudgetResult,
        InboxMessage,
        RequestProfileSwitchInput,
        RequestProfileSwitchResult,
        SendMessageInput,
        SendMessageResult,
        TaskControlInput,
        TaskControlResult,
    )

    from orbit_orch.workflow_common import VERIFY_FINISHED_ATTEMPTS, budget_add, closed

from orbit_orch.task_attempts import TaskAttempts


class TaskCommands(TaskAttempts):
    @workflow.update(name="sendMessage")
    async def send_message(self, req: SendMessageInput) -> SendMessageResult:
        if closed(self._status):
            raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        message = InboxMessage(
            message_seq=self._next_message_seq,
            client_message_id=req.client_message_id,
            text=req.text,
            attachments=req.attachments,
            delivery=req.delivery,
        )
        self._next_message_seq += 1
        self._inbox.append(message)
        self._wake += 1
        result = SendMessageResult(message_seq=message.message_seq)
        self._dedup[req.command_id] = result
        self._updates += 1
        self._commands.append(("message", message))
        self._emit("message.user", message.model_dump(mode="json"))
        if req.delivery == "interrupt":
            await self._interrupt_active_attempt()
        return result

    @send_message.validator
    def validate_send_message(self, req: SendMessageInput) -> None:
        if closed(self._status):
            raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)

    @workflow.update(name="decideApproval")
    async def decide_approval(self, req: DecideApprovalInput) -> DecideApprovalResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        approval = self._approvals.get(req.approval_id)
        if approval is None or approval["status"] != "PENDING":
            raise ApplicationError("approval is not pending", type="UNKNOWN_APPROVAL", non_retryable=True)
        approval["status"] = "APPROVED" if req.decision == "approve" else "REJECTED"
        approval["comment"] = req.comment
        result = DecideApprovalResult(approval_id=req.approval_id, status=approval["status"])
        self._dedup[req.command_id] = result
        self._updates += 1
        if workflow.patched("task-event-vocabulary"):
            self._emit("approval.decided", {
                "approval_id": req.approval_id,
                "status": approval["status"],
                "comment": req.comment,
            })
        attempt_id = approval.get("attempt_id")
        if attempt_id:
            await self._signal_attempt(
                attempt_id,
                "approvalDecided",
                ApprovalDecidedSignal(
                    approval_id=req.approval_id,
                    tool_call_id=approval.get("tool_call_id"),
                    decision=req.decision,
                    comment=req.comment,
                ),
            )
            if not any(
                item["status"] == "PENDING" and item.get("attempt_id") == attempt_id for item in self._approvals.values()
            ):
                self._resume_parked(attempt_id)
        return result

    @workflow.update(name="control")
    async def control(self, req: TaskControlInput) -> TaskControlResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        transitions = {
            "pause": "PAUSED",
            "resume": "RUNNING",
            "cancel": "CANCELLED",
            "takeover": "TAKEN_OVER",
            "handback": "RUNNING",
        }
        target = transitions[req.action]
        if req.action == "resume" and self._status not in {"PAUSED", "PAUSED_NEEDS_REVIEW"}:
            raise ApplicationError("task cannot be resumed", type="INVALID_TRANSITION", non_retryable=True)
        if req.action == "cancel":
            await self._cancel_active_attempts()
            self._stop = True
        self._set_status(target, req.reason or req.action)
        self._wake += 1
        if req.action == "cancel" and workflow.patched("task-event-vocabulary"):
            self._emit("task.cancelled", {"reason": req.reason or "cancelled"})
        result = TaskControlResult(status=target)  # type: ignore[arg-type]
        self._dedup[req.command_id] = result
        self._updates += 1
        return result

    @workflow.update(name="grantBudget")
    async def grant_budget(self, req: GrantBudgetInput) -> GrantBudgetResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        self._budgets = budget_add(self._budgets, req.delta)
        result = GrantBudgetResult(budgets=self._budgets)
        self._dedup[req.command_id] = result
        self._updates += 1
        self._set_status("RUNNING", "budget_granted")
        return result

    @workflow.update(name="requestProfileSwitch")
    async def request_profile_switch(self, req: RequestProfileSwitchInput) -> RequestProfileSwitchResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        node = self._require_node(req.node_id)
        if node.frozen:
            raise ApplicationError("node is frozen", type="FROZEN_NODE", non_retryable=True)
        result = RequestProfileSwitchResult(
            effective_attempt_no=int(node.attempt_count) + 1,
            needs_approval=True,
        )
        self._dedup[req.command_id] = result
        self._updates += 1
        return result

    async def _drain_commands(self) -> None:
        while self._commands:
            kind, payload = self._commands.pop(0)
            if kind == "message":
                attempt = next(
                    (
                        item
                        for item in self._attempts.values()
                        if item.get("status") in {"RUNNING", "PARKED_INPUT", "PARKED_APPROVAL"}
                    ),
                    None,
                )
                if attempt is not None:
                    await self._signal_attempt(
                        str(attempt["attempt_id"]),
                        "deliverMessages",
                        DeliverMessagesSignal(messages=[payload]),
                    )
                    if attempt.get("status") == "PARKED_INPUT":
                        self._resume_parked(str(attempt["attempt_id"]))
                continue
            if kind != "completion":
                continue
            node_id = payload.node_id
            if node_id not in self._plan.nodes:  # type: ignore[union-attr]
                continue
            attempt = self._attempts.get(node_id)
            if workflow.patched(VERIFY_FINISHED_ATTEMPTS):
                if attempt is None or attempt.get("attempt_id") != payload.attempt_id or attempt.get("verdict"):
                    continue  # replaced, cancelled or judged since the proposal was accepted
                attempt["verdict"] = "verifying"
            self._set_node_status(node_id, "VERIFYING")
            failures: list[dict[str, Any]] = []
            if workflow.patched("task-completion-verification"):
                failures = await self._verify_proposal(payload)
                accepted = not failures
            else:
                result = await self._run_short_activity(
                    "verify_completion",
                    payload.model_dump(mode="json"),
                )
                accepted = bool(result.get("ok", True))
            if workflow.patched(VERIFY_FINISHED_ATTEMPTS):
                self._settle_proposal(node_id, attempt, failures)
            elif accepted:
                self._set_node_status(node_id, "COMPLETED", frozen=True)
                self._completed_nodes += 1
            else:
                self._set_node_status(node_id, "RETRY_PENDING")

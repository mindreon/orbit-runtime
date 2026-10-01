"""AttemptWorkflow: one attempt at a node. It runs the agent turn (or the SOP) as activities and reports to its parent."""

from __future__ import annotations

import asyncio
from typing import Any

from temporalio import workflow
from temporalio.common import VersioningBehavior
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        ApprovalDecidedSignal,
        AttemptFinishedSignal,
        AttemptParkedSignal,
        AttemptWorkflowInput,
        DeliverMessagesSignal,
        InboxMessage,
    )
    from orbit_contracts.v3.common import Failure
    from orbit_contracts.v3.messages import ParkedToolCall

    from orbit_orch.workflow_common import (
        AGENT_TIMEOUT,
        HEARTBEAT,
        IO_TIMEOUT,
        RETRY,
        sha,
        versioning_behavior,
    )


@workflow.defn(name="AttemptWorkflow", versioning_behavior=versioning_behavior(VersioningBehavior.PINNED))
class AttemptWorkflow:
    def __init__(self) -> None:
        self._decisions: dict[str, ApprovalDecidedSignal] = {}
        self._awaiting: set[str] = set()  # tool call ids the attempt is parked on
        self._messages: list[InboxMessage] = []
        self._session_id = ""
        self._state_version = 0
        self._approval_request_id = ""
        self._external: dict[str, str] | None = None
        self._retry_calls: list[dict[str, Any]] | None = None
        self._continue = False
        self._cancelled = False

    @workflow.run
    async def run(self, inp: AttemptWorkflowInput) -> None:
        self._messages = list(inp.messages)
        try:
            if inp.node_type == "sop_stage":
                await self._run_sop(inp)
                return
            while True:
                delivered = list(self._messages)
                result = await workflow.execute_activity(
                    "agent_turn" if inp.node_type == "agent_turn" else "sop_step",
                    {
                        "task_id": inp.task_id,
                        "tenant_id": inp.tenant_id,
                        "node_id": inp.node_id,
                        "attempt_id": inp.attempt_id,
                        "attempt_no": inp.attempt_no,
                        "profile": inp.profile,
                        "config": inp.config.model_dump(mode="json"),
                        "goal": inp.goal,
                        "checkpoint_ref": inp.checkpoint_ref,
                        "workspace_access": inp.workspace_access,
                        "policy": inp.policy.model_dump(mode="json"),
                        "messages": [message.model_dump(mode="json") for message in delivered],
                        "external": self._external,
                        "retry_calls": self._retry_calls,
                        "approval": self._approval_payload(),
                        "session_id": self._session_id,
                        "state_version": self._state_version,
                    },
                    task_queue="orbit.agent",
                    result_type=dict,
                    start_to_close_timeout=AGENT_TIMEOUT,
                    heartbeat_timeout=HEARTBEAT,
                    cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                    retry_policy=RETRY,
                )
                await self._commit_checkpoints(inp, result)
                if result.get("session_id"):
                    self._session_id = str(result["session_id"])
                if result.get("state_version") is not None:
                    self._state_version = int(result["state_version"])
                self._decisions = {}
                self._external = None
                self._retry_calls = result.get("retry_calls")
                # Only what the activity was given is consumed; a message that arrived meanwhile stays queued.
                self._messages = self._messages[len(delivered):]
                state = result.get("status", "completed")
                if state == "parked_approval":
                    self._approval_request_id = str(result.get("approval_request_id", ""))
                    self._awaiting = {str(item["tool_call_id"]) for item in result.get("approvals", [])}
                    await self._notify_parent_parked(inp, "approval", result)
                    # Every parked call needs its own decision before the attempt goes on.
                    await workflow.wait_condition(lambda: self._awaiting <= self._decisions.keys() or self._cancelled)
                    if self._cancelled:
                        await self._notify_parent_finished(inp, "cancelled", result)
                        return
                    continue
                if state == "parked_input":
                    self._external = result.get("external")
                    await self._notify_parent_parked(inp, "input", result)
                    await workflow.wait_condition(lambda: bool(self._messages) or self._cancelled)
                    if self._cancelled:
                        await self._notify_parent_finished(inp, "cancelled", result)
                        return
                    continue
                outcome = "failed" if state == "failed" else "completed"
                await self._notify_parent_finished(inp, outcome, result)
                return
        except asyncio.CancelledError:
            self._cancelled = True
            await self._notify_parent_finished(inp, "cancelled", {"checkpoint_ref": sha(inp.attempt_id)})
            raise

    async def _run_sop(self, inp: AttemptWorkflowInput) -> None:
        """One `sop_step` activity per try. The activity drives AgentScope's SOPEngine for that try and returns the
        engine's run state, which goes into the next call: a finished step is in history, so a retry never repeats it,
        and the engine, not this loop, decides when a step has used up its attempts."""
        run_state, result = "", {}
        while True:
            result = await workflow.execute_activity(
                "sop_step",
                {
                    "task_id": inp.task_id,
                    "tenant_id": inp.tenant_id,
                    "node_id": inp.node_id,
                    "attempt_id": inp.attempt_id,
                    "attempt_no": inp.attempt_no,
                    "goal": inp.goal,
                    "run_state": run_state,
                },
                task_queue="orbit.agent",
                result_type=dict,
                start_to_close_timeout=AGENT_TIMEOUT,
                heartbeat_timeout=HEARTBEAT,
                cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                retry_policy=RETRY,
            )
            if result.get("status") != "continue":
                break
            run_state = str(result["run_state"])
        await self._commit_checkpoints(inp, result)
        await self._notify_parent_finished(inp, "failed" if result.get("status") == "failed" else "completed", result)

    async def _commit_checkpoints(self, inp: AttemptWorkflowInput, result: dict[str, Any]) -> None:
        """The activity result is in this history now, so the checkpoints it refers to are committed (08 §3). A
        failure is logged and not fatal: the GC keeps the newest checkpoint of an attempt either way, so the only
        cost is that older ones of this attempt are collected a day later than they could be."""
        if not workflow.patched("commit-attempt-checkpoints"):
            return
        try:
            await workflow.execute_activity(
                "commit_checkpoints",
                {
                    "tenant_id": inp.tenant_id,
                    "attempt_id": inp.attempt_id,
                    "checkpoint_ref": result.get("checkpoint_ref"),
                },
                task_queue="orbit.io",
                result_type=dict,
                start_to_close_timeout=IO_TIMEOUT,
                retry_policy=RETRY,
            )
        except ActivityError as exc:
            workflow.logger.warning("commit_checkpoints failed for %s: %s", inp.attempt_id, exc)

    @workflow.signal(name="approvalDecided")
    async def approval_decided(self, signal: ApprovalDecidedSignal) -> None:
        self._decisions[signal.tool_call_id or ""] = signal

    def _approval_payload(self) -> dict[str, Any] | None:
        """What the activity is told about the decisions taken: one per call, and whether all of them allow."""
        if not self._decisions:
            return None
        last = list(self._decisions.values())[-1]
        return {
            "approval_id": last.approval_id,
            "approval_request_id": self._approval_request_id,
            "decision": "approve" if all(item.decision == "approve" for item in self._decisions.values()) else "reject",
            "decisions": {call_id: item.decision for call_id, item in self._decisions.items()},
        }

    @workflow.signal(name="deliverMessages")
    async def deliver_messages(self, signal: DeliverMessagesSignal) -> None:
        self._messages.extend(signal.messages)

    async def _notify_parent_parked(self, inp: AttemptWorkflowInput, reason: str, result: dict[str, Any]) -> None:
        parent = workflow.info().parent
        if parent is None:
            return
        approvals = [ParkedToolCall.model_validate(item) for item in result.get("approvals", [])]
        await workflow.get_external_workflow_handle(parent.workflow_id).signal(
            "attemptParked",
            AttemptParkedSignal(
                node_id=inp.node_id,
                attempt_no=inp.attempt_no,
                attempt_id=inp.attempt_id,
                reason=reason,  # type: ignore[arg-type]
                approvals=approvals,
                question=result.get("question"),
            ),
        )

    async def _notify_parent_finished(
        self,
        inp: AttemptWorkflowInput,
        outcome: str,
        result: dict[str, Any],
    ) -> None:
        parent = workflow.info().parent
        if parent is None:
            return
        failure = None
        if outcome == "failed":
            failure = Failure(
                failure_class="transient",
                retryable=True,
                message=str(result.get("error", "attempt failed")),
            )
        await workflow.get_external_workflow_handle(parent.workflow_id).signal(
            "attemptFinished",
            AttemptFinishedSignal(
                attempt_workflow_id=workflow.info().workflow_id,
                node_id=inp.node_id,
                attempt_no=inp.attempt_no,
                attempt_id=inp.attempt_id,
                outcome=outcome,  # type: ignore[arg-type]
                result=None if failure else {
                    "checkpoint_ref": result.get("checkpoint_ref", sha(inp.attempt_id)),
                    "manifest_id": result.get("manifest_id"),
                    "manifest_entries": result.get("manifest_entries", []),
                    "manifest_hash": result.get("manifest_hash"),
                    "usage": result.get("usage", {}),
                    "handover_summary": result.get("handover_summary", ""),
                    "budget_exhausted": bool(result.get("budget_exhausted", False)),
                },
                failure=failure,
            ),
        )

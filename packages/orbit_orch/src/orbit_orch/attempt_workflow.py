"""AttemptWorkflow: one attempt at a node. It runs the agent turn (or the SOP) as activities and reports to its parent."""

from __future__ import annotations

import asyncio
from typing import Any

from temporalio import workflow
from temporalio.common import VersioningBehavior
from temporalio.exceptions import ActivityError
from temporalio.exceptions import CancelledError as TemporalCancelledError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        ApprovalDecidedSignal,
        AttemptFinishedSignal,
        AttemptParkedSignal,
        AttemptWorkflowInput,
        DeliverMessagesSignal,
        InboxMessage,
    )
    from orbit_contracts.v3.common import Failure, Usage
    from orbit_contracts.v3.messages import ParkedToolCall

    from orbit_orch import budgets
    from orbit_orch.workflow_common import (
        AGENT_TIMEOUT,
        ATTEMPT_BUDGET,
        ATTEMPT_COMMIT_ABANDON,
        ATTEMPT_CONTINUE_AS_NEW,
        ATTEMPT_FAILURE_CLASS,
        ATTEMPT_HEARD_MESSAGES,
        ATTEMPT_RETURNS_MESSAGES,
        ATTEMPT_TEAM_STAGE,
        HEARTBEAT,
        IO_TIMEOUT,
        MAX_HEARD,
        RETRY,
        sha,
        versioning_behavior,
    )


from orbit_orch.attempt_team import AttemptTeam

FAILURE_CLASSES = frozenset({"transient", "model", "tool", "policy", "budget", "verification", "lost"})


@workflow.defn(name="AttemptWorkflow", versioning_behavior=versioning_behavior(VersioningBehavior.PINNED))
class AttemptWorkflow(AttemptTeam):
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
        self._ran_activity = False
        # How many of `_messages` the running activity was given. A cancelled attempt took those with it (its session has
        # them); the ones that came after are not heard yet.
        self._in_flight = 0
        # The messages an activity turn was given and did not fail on (`heard_message_seqs`).
        self._heard: set[int] = set()
        # What the turns of this attempt have spent so far, against the budget the parent reserved for it.
        self._usage = Usage()
        # A team stage's state and the events it has not published yet (`attempt_team`).
        self._team_events = []
        self._team_flush = asyncio.Lock()
        self._round_usage = Usage()
        self._mention_seen = set()

    def _should_continue_as_new(self) -> bool:
        return workflow.info().is_continue_as_new_suggested()

    def _carry(self) -> dict[str, Any]:
        """What the next run needs to go on exactly where this one is: the agent session, the messages that are not
        consumed, what the attempt is parked on and the decisions taken on it."""
        return {
            "session_id": self._session_id,
            "state_version": self._state_version,
            "approval_request_id": self._approval_request_id,
            "external": self._external,
            "retry_calls": self._retry_calls,
            "messages": [message.model_dump(mode="json") for message in self._messages],
            "decisions": {key: item.model_dump(mode="json") for key, item in self._decisions.items()},
            "usage": self._usage.model_dump(mode="json", exclude_none=True),
            "heard": sorted(self._heard)[-MAX_HEARD:],
        }

    def _restore(self, carry: dict[str, Any]) -> None:
        self._session_id = str(carry.get("session_id", ""))
        self._state_version = int(carry.get("state_version", 0))
        self._approval_request_id = str(carry.get("approval_request_id", ""))
        self._external = carry.get("external")
        self._retry_calls = carry.get("retry_calls")
        self._messages = [InboxMessage.model_validate(item) for item in carry.get("messages", [])]
        self._decisions = {
            str(key): ApprovalDecidedSignal.model_validate(item) for key, item in dict(carry.get("decisions", {})).items()
        }
        self._usage = Usage.model_validate(carry.get("usage") or {})
        self._heard = {int(seq) for seq in carry.get("heard", [])}

    @workflow.run
    async def run(self, inp: AttemptWorkflowInput) -> None:
        self._messages = list(inp.messages)
        if inp.carry is not None:
            self._restore(inp.carry)
        try:
            if inp.node_type == "sop_stage":
                # DEPRECATED: a `sop_stage` node is compiled into the plan now (`task_sop`, patch `task-sop-expansion`) and
                # starts no attempt. This path stays to replay histories that ran it and to finish attempts in flight.
                await self._run_sop(inp)
                return
            if inp.node_type == "team_stage" and inp.team is not None and workflow.patched(ATTEMPT_TEAM_STAGE):
                await self._run_team(inp)
                return
            while True:
                if self._ran_activity and self._should_continue_as_new() and workflow.patched(ATTEMPT_CONTINUE_AS_NEW):
                    # Long approval and question loops grow this history, so it is cut like the task's (04 §6). Every
                    # handler is done first, and the parent is not told anything: the workflow id stays the same.
                    await workflow.wait_condition(workflow.all_handlers_finished)
                    workflow.continue_as_new(inp.model_copy(update={"messages": [], "carry": self._carry()}))
                delivered = list(self._messages)
                self._in_flight = len(delivered)
                self._heard |= {message.message_seq for message in delivered}
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
                        "allow_rules": [rule.model_dump(mode="json") for rule in inp.allow_rules],
                        "continue_from": inp.continue_from,
                        "retry_reason": inp.retry_reason,
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
                        "switched_from": inp.switched_from,
                        "handover": inp.handover,
                        "output_schema_ref": inp.output_schema_ref,
                        # What is left of the reserved budget for this turn; the worker stops the turn between two steps
                        # when it is spent. Left out, the turn has no limit.
                        **self._budget_payload(inp),
                    },
                    task_queue="orbit.agent",
                    result_type=dict,
                    start_to_close_timeout=AGENT_TIMEOUT,
                    heartbeat_timeout=HEARTBEAT,
                    cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                    retry_policy=RETRY,
                )
                self._ran_activity = True
                self._in_flight = 0
                self._count_usage(result)
                await self._commit_checkpoints(inp, result)
                if result.get("session_id"):
                    self._session_id = str(result["session_id"])
                if result.get("state_version") is not None:
                    self._state_version = int(result["state_version"])
                self._decisions = {}
                self._external = None
                self._retry_calls = result.get("retry_calls")
                state = result.get("status", "completed")
                # Only what the activity was given is consumed; a message that arrived meanwhile stays queued. A turn that
                # failed did not hear its messages (its state is dropped), so they stay and go back to the task.
                if state != "failed" or not workflow.patched(ATTEMPT_RETURNS_MESSAGES):
                    self._messages = self._messages[len(delivered):]
                else:
                    self._heard -= {message.message_seq for message in delivered}
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
        except ActivityError as exc:
            if not workflow.patched(ATTEMPT_FAILURE_CLASS):
                raise
            if isinstance(exc.cause, TemporalCancelledError):
                self._cancelled = True
                await self._notify_parent_finished(inp, "cancelled", {"checkpoint_ref": sha(inp.attempt_id)})
                raise
            # The activity gave up after its retries (a lost worker, a heartbeat that stopped): the parent is told, so the
            # node does not wait for an attempt that is gone.
            await self._notify_parent_finished(
                inp, "failed", {"error": str(exc.cause or exc), "failure_class": "lost", "retryable": True}
            )

    def _budget_payload(self, inp: AttemptWorkflowInput) -> dict[str, Any]:
        if inp.budget is None or not workflow.patched(ATTEMPT_BUDGET):
            return {}
        return {"budget": budgets.after(inp.budget, self._usage).model_dump(mode="json", exclude_none=True)}

    def _count_usage(self, result: dict[str, Any]) -> None:
        if workflow.patched(ATTEMPT_BUDGET) and result.get("usage"):
            self._usage = budgets.usage_add(self._usage, Usage.model_validate(result["usage"]))

    async def _run_sop(self, inp: AttemptWorkflowInput) -> None:
        """DEPRECATED (see `run`): kept for replay and for attempts that were running before SOPs were compiled into the plan.

        One `sop_step` activity per try. The activity drives AgentScope's SOPEngine for that try and returns the
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

    async def _commit_checkpoints(
        self, inp: AttemptWorkflowInput, result: dict[str, Any], attempt_id: str | None = None
    ) -> None:
        """The activity result is in this history now, so the checkpoints it refers to are committed (08 §3). A
        failure is logged and not fatal: the GC keeps the newest checkpoint of an attempt either way, so the only
        cost is that older ones of this attempt are collected a day later than they could be. `attempt_id` names the
        attempt the checkpoints are stored under when it is not this one (a member of a team stage runs under its own)."""
        if not workflow.patched("commit-attempt-checkpoints"):
            return
        try:
            await workflow.execute_activity(
                "commit_checkpoints",
                {
                    "tenant_id": inp.tenant_id,
                    "attempt_id": attempt_id or inp.attempt_id,
                    "checkpoint_ref": result.get("checkpoint_ref"),
                },
                task_queue="orbit.io",
                result_type=dict,
                start_to_close_timeout=IO_TIMEOUT,
                retry_policy=RETRY,
                **(
                    {"cancellation_type": workflow.ActivityCancellationType.ABANDON}
                    if workflow.patched(ATTEMPT_COMMIT_ABANDON)
                    else {}
                ),
            )
        except ActivityError as exc:
            if isinstance(exc.cause, TemporalCancelledError) and workflow.patched(ATTEMPT_FAILURE_CLASS):
                # The attempt was cancelled while it committed: that is not a commit that failed, and it is not swallowed,
                # or the attempt would go on as if it had not been cancelled and never report its end as cancelled.
                raise asyncio.CancelledError from exc
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
            # What a person allowed for the rest of the task with a decision, per call.
            "rules": {
                call_id: item.rule.model_dump(mode="json")
                for call_id, item in self._decisions.items()
                if item.rule is not None and item.decision == "approve"
            },
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
            failure_class, retryable = "transient", True
            if workflow.patched(ATTEMPT_FAILURE_CLASS):
                # The worker says what kind of failure it was and whether trying again can help (04 §2).
                failure_class = result.get("failure_class", "transient")
                if failure_class not in FAILURE_CLASSES:
                    failure_class = "transient"
                retryable = bool(result.get("retryable", True))
            failure = Failure(
                failure_class=failure_class,  # type: ignore[arg-type]
                retryable=retryable,
                message=str(result.get("error", "attempt failed")),
            )
        # What goes back to the task: what the attempt was handed and its turns did not hear. A cancelled turn heard what it
        # was given (its session is kept with it), so only what came after goes back.
        unconsumed = (
            list(self._messages[self._in_flight if outcome == "cancelled" else 0 :])
            if workflow.patched(ATTEMPT_RETURNS_MESSAGES)
            else []
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
                    "usage": (
                        self._usage.model_dump(mode="json", exclude_none=True)
                        if workflow.patched(ATTEMPT_BUDGET)
                        else result.get("usage", {})
                    ),
                    "handover_summary": result.get("handover_summary", ""),
                    "budget_exhausted": bool(result.get("budget_exhausted", False)),
                    "output": result.get("output") or {},
                },
                failure=failure,
                unconsumed_messages=unconsumed,
                heard_message_seqs=sorted(self._heard)[-MAX_HEARD:] if workflow.patched(ATTEMPT_HEARD_MESSAGES) else None,
                usage=self._usage if workflow.patched(ATTEMPT_BUDGET) else None,
            ),
        )

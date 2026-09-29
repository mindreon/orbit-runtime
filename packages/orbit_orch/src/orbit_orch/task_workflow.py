"""TaskWorkflow and AttemptWorkflow for the v3 runtime.

The workflows contain only deterministic scheduling and state transitions.  All
Postgres, object store, workspace, model and event projection work is delegated
to named activities on the worker queues.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy, VersioningBehavior
from temporalio.exceptions import ActivityError, ApplicationError, TemporalError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        Actor,
        ApprovalDecidedSignal,
        AttemptFinishedSignal,
        AttemptParkedSignal,
        AttemptWorkflowInput,
        Budget,
        CompletionAccepted,
        CompletionProposal,
        CompletionResult,
        DecideApprovalInput,
        DecideApprovalResult,
        DeliverMessagesSignal,
        ExternalEventSignal,
        GrantBudgetInput,
        GrantBudgetResult,
        InboxMessage,
        PlanChangeCommand,
        PlanChangeResult,
        PlanView,
        Policy,
        RequestProfileSwitchInput,
        RequestProfileSwitchResult,
        SendMessageInput,
        SendMessageResult,
        TaskControlInput,
        TaskControlResult,
        TaskView,
        TaskWorkflowInput,
    )
    from orbit_contracts.v3.common import Failure, Usage
    from orbit_contracts.v3.messages import ParkedToolCall
    from orbit_contracts.v3.nodes import TaskNodeDraft
    from orbit_contracts.v3.plan import PlanChangeAccepted, PlanChangeRejected

    from orbit_orch.plan_engine import (
        PlanNodeState,
        PlanPolicy,
        PlanState,
        apply,
        deterministic_id,
        initial_plan,
    )


_RETRY = RetryPolicy(maximum_attempts=3)
_IO_TIMEOUT = timedelta(minutes=2)
_AGENT_TIMEOUT = timedelta(hours=1)
_HEARTBEAT = timedelta(seconds=30)
_OPERATOR_HELD = frozenset({"PAUSED", "PAUSED_NEEDS_REVIEW", "TAKEN_OVER"})
_MAX_UPDATES_BEFORE_CAN = 1000
_MAX_COMPLETIONS_BEFORE_CAN = 50


def _versioning_behavior(behavior: VersioningBehavior) -> VersioningBehavior:
    """Use deployment versioning only when the worker is registered for it."""
    if os.environ.get("ORBIT_USE_WORKER_VERSIONING", "0") == "1":
        return behavior
    return VersioningBehavior.UNSPECIFIED


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _closed(status: str) -> bool:
    return status in {"COMPLETED", "FAILED", "CANCELLED"}


def _budget_add(a: Budget, b: Budget) -> Budget:
    values: dict[str, int | None] = {}
    for name in ("tokens", "tool_calls", "wall_s", "cost_usd_micros"):
        left, right = getattr(a, name), getattr(b, name)
        values[name] = None if left is None and right is None else (left or 0) + (right or 0)
    return Budget(**values)


@workflow.defn(name="TaskWorkflow", versioning_behavior=_versioning_behavior(VersioningBehavior.AUTO_UPGRADE))
class TaskWorkflow:
    def __init__(self) -> None:
        self._task_id = ""
        self._tenant_id = ""
        self._created_by: Actor | None = None
        self._title = ""
        self._goal = ""
        self._profile = ""
        self._status = "CREATED"
        self._plan: PlanState | None = None
        self._budgets = Budget()
        self._policy = Policy()
        self._usage = Usage()
        self._inbox: list[InboxMessage] = []
        self._next_message_seq = 1
        self._dedup: dict[str, Any] = {}
        self._commands: list[tuple[str, Any]] = []
        self._attempts: dict[str, dict[str, Any]] = {}
        self._attempt_handles: dict[str, Any] = {}
        self._approvals: dict[str, dict[str, Any]] = {}
        self._wait_events: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._entity_versions: dict[str, int] = {}
        self._updates = 0
        self._completed_nodes = 0
        self._stop = False
        self._wake = 0

    @workflow.run
    async def run(self, inp: TaskWorkflowInput) -> TaskView:
        workflow.patched("taskworkflow-v3")
        self._load(inp)
        self._status = "PLANNING" if inp.carry is None else self._status
        if self._plan is None:
            self._plan = initial_plan(inp.task_id, inp.goal, inp.profile)
            self._emit("task.created", {"title": inp.title, "goal": inp.goal})
        if inp.carry is None or self._status in {"CREATED", "PLANNING", "RUNNING"}:
            self._set_status("RUNNING", "started")
        while not self._stop:
            await self._drain_commands()
            await self._schedule_ready()
            if self._status in _OPERATOR_HELD:
                pass  # pause and takeover stay until an operator resumes; a running attempt does not undo them
            elif any(item.get("status") == "RUNNING" for item in self._attempts.values()):
                self._set_status("RUNNING", "attempt_running")
            elif self._plan and any(
                state.status in {"AWAITING_APPROVAL", "AWAITING_INPUT"}
                for state in self._plan.nodes.values()
            ):
                self._set_status("WAITING", "awaiting_user")
            await self._flush_events()
            if self._all_nodes_completed():
                self._status = "COMPLETED"
                self._emit("task.completed", {})
                await self._flush_events()
                break
            if self._should_continue_as_new():
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(self._carry_input())
            observed_wake = self._wake
            if not self._commands and not self._has_ready_node():
                await workflow.wait_condition(
                    lambda observed_wake=observed_wake: bool(self._commands)
                    or self._has_ready_node()
                    or bool(self._wait_events)
                    or self._wake != observed_wake
                    or self._stop
                )
        if self._status == "CANCELLED":
            # Every child has to be cancelled before the task ends (04 §6).
            await workflow.wait_condition(lambda: not self._attempts)
        await self._flush_events()
        return self._task_view()

    # ---- updates ---------------------------------------------------------

    @workflow.update(name="sendMessage")
    async def send_message(self, req: SendMessageInput) -> SendMessageResult:
        if _closed(self._status):
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
        if _closed(self._status):
            raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)

    @workflow.update(name="submitPlanChange")
    async def submit_plan_change(self, command: PlanChangeCommand) -> PlanChangeResult:
        previous = self._dedup.get(command.command_id)
        if previous is not None:
            return previous
        if self._plan is None:
            # Temporal may deliver an update in the same activation as the
            # start event, before ``run`` has reached initial planning.
            self._task_id = self._task_id or command.task_id
            self._profile = self._profile or "default@1"
            self._plan = initial_plan(self._task_id, self._goal, self._profile)
        policy = PlanPolicy(
            active_attempt_id=self._active_attempt_id(),
            active_attempt_ids=frozenset(
                str(item["attempt_id"])
                for item in self._attempts.values()
                if item.get("status") == "RUNNING"
            ),
            active_node_ids=frozenset(
                node_id for node_id, state in self._plan.nodes.items() if state.status == "RUNNING"
            ),
            max_budget=self._budgets,
        )
        outcome = apply(self._plan, command, policy)
        self._dedup[command.command_id] = outcome.result
        self._updates += 1
        if isinstance(outcome.result, PlanChangeAccepted):
            self._plan = outcome.plan
            self._refresh_readiness()
            self._emit("plan.version_committed", {
                "plan_version": self._plan.version,
                "hash": self._plan.hash,
                "command_id": command.command_id,
            })
        else:
            self._emit("plan.change_rejected", outcome.result.model_dump(mode="json"))
        return outcome.result

    @workflow.update(name="proposeCompletion")
    async def propose_completion(self, req: CompletionProposal) -> CompletionResult:
        if _closed(self._status):
            raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        attempt = self._attempts.get(req.node_id)
        if attempt is None or attempt.get("attempt_id") != req.attempt_id:
            raise ApplicationError("attempt is stale", type="STALE_ATTEMPT", non_retryable=True)
        self._dedup[req.command_id] = CompletionAccepted()
        self._commands.append(("completion", req))
        self._wake += 1
        self._updates += 1
        return CompletionAccepted()

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
        self._budgets = _budget_add(self._budgets, req.delta)
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

    # ---- signals and queries --------------------------------------------

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

    @workflow.signal(name="attemptFinished")
    async def attempt_finished(self, signal: AttemptFinishedSignal) -> None:
        self._wake += 1
        attempt = self._attempts.get(signal.node_id)
        if attempt is None or attempt.get("attempt_no") != signal.attempt_no:
            return
        attempt["status"] = signal.outcome
        attempt["result"] = signal.result.model_dump(mode="json") if signal.result else None
        node = self._require_node(signal.node_id)
        if workflow.patched("task-event-vocabulary"):
            self._emit("attempt.finished", {
                "node_id": signal.node_id,
                "attempt_id": signal.attempt_id,
                "outcome": signal.outcome,
                "failure": signal.failure.model_dump(mode="json") if signal.failure else None,
            })
        if signal.outcome == "completed":
            if signal.result and signal.result.manifest_id:
                self._emit("artifact.manifest_created", {
                    "manifest_id": signal.result.manifest_id,
                    "attempt_id": signal.attempt_id,
                    "entries": signal.result.manifest_entries,
                    "manifest_hash": signal.result.manifest_hash or _sha(signal.result.manifest_id),
                })
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

    @workflow.query(name="getTaskView")
    def get_task_view(self) -> TaskView:
        return self._task_view()

    @workflow.query(name="getPlan")
    def get_plan(self) -> PlanView:
        if self._plan is None:
            raise ApplicationError("plan is not initialized")
        from orbit_contracts.v3.views import NodeView, PlanEdge

        return PlanView(
            plan_version=self._plan.version,
            hash=self._plan.hash,
            nodes=[
                NodeView(
                    node_id=node_id,
                    type=state.draft.type,
                    title=state.draft.title,
                    status=state.status,
                    depends_on=list(state.draft.depends_on),
                    workspace_access=state.draft.workspace_access or "none",
                    owner_profile=state.draft.owner_profile or self._profile,
                    frozen=state.frozen,
                    current_attempt_id=state.current_attempt_id,
                    attempt_count=state.attempt_count,
                )
                for node_id, state in self._plan.nodes.items()
            ],
            edges=[PlanEdge.model_validate({"from": source, "to": target}) for source, target in self._plan.edges],
        )

    @workflow.query(name="getInbox")
    def get_inbox(self, after_seq: int = 0) -> list[InboxMessage]:
        return [item for item in self._inbox if item.message_seq > after_seq]

    # ---- scheduler -------------------------------------------------------

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
                await self._run_short_activity(
                    "checkpoint_commit",
                    {"tenant_id": self._tenant_id, "task_id": self._task_id, "node_id": node_id},
                )
                self._set_node_status(node_id, "COMPLETED")
                self._completed_nodes += 1

    async def _start_attempt(self, node_id: str) -> None:
        state = self._require_node(node_id)
        attempt_no = state.attempt_count + 1
        attempt_id = deterministic_id(f"{self._task_id}:{node_id}:{attempt_no}", "att")
        workflow_id = f"attempt/{self._task_id}/{node_id}/{attempt_no}"
        spec = state.draft.spec
        goal = getattr(spec, "goal", None) or getattr(spec, "sop", None) or state.draft.title
        inp = AttemptWorkflowInput(
            task_id=self._task_id,
            tenant_id=self._tenant_id,
            node_id=node_id,
            attempt_id=attempt_id,
            attempt_no=attempt_no,
            node_type=state.draft.type,
            profile=state.draft.owner_profile or self._profile,
            goal=goal,
            policy=self._policy,
            workspace_access=state.draft.workspace_access or "none",
            messages=self._inbox,
        )
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
        })

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
            self._set_node_status(node_id, "VERIFYING")
            result = await self._run_short_activity(
                "verify_completion",
                payload.model_dump(mode="json"),
            )
            if result.get("ok", True):
                self._set_node_status(node_id, "COMPLETED", frozen=True)
                self._completed_nodes += 1
            else:
                self._set_node_status(node_id, "RETRY_PENDING")

    async def _complete_after(self, node_id: str, seconds: int) -> None:
        await workflow.sleep(seconds)
        if self._plan and self._plan.nodes.get(node_id, None) and self._plan.nodes[node_id].status == "AWAITING_INPUT":
            self._set_node_status(node_id, "COMPLETED")
            self._completed_nodes += 1

    # ---- helpers ---------------------------------------------------------

    def _load(self, inp: TaskWorkflowInput) -> None:
        self._task_id, self._tenant_id, self._created_by = inp.task_id, inp.tenant_id, inp.created_by
        self._title, self._goal, self._profile, self._budgets = inp.title, inp.goal, inp.profile, inp.budgets
        self._policy = inp.policy
        carry = inp.carry
        if not carry:
            return
        self._status = str(carry.get("status", "RUNNING"))
        self._next_message_seq = int(carry.get("next_message_seq", 1))
        self._completed_nodes = int(carry.get("completed_nodes", 0))
        self._entity_versions = {str(key): int(value) for key, value in dict(carry.get("entity_versions", {})).items()}
        self._attempts = {str(key): dict(value) for key, value in dict(carry.get("attempts", {})).items()}
        self._approvals = {str(key): dict(value) for key, value in dict(carry.get("approvals", {})).items()}
        result_types = {
            "SendMessageResult": SendMessageResult,
            "PlanChangeAccepted": PlanChangeAccepted,
            "PlanChangeRejected": PlanChangeRejected,
            "CompletionAccepted": CompletionAccepted,
            "DecideApprovalResult": DecideApprovalResult,
            "TaskControlResult": TaskControlResult,
            "GrantBudgetResult": GrantBudgetResult,
            "RequestProfileSwitchResult": RequestProfileSwitchResult,
        }
        for command_id, raw in dict(carry.get("dedup", {})).items():
            item = dict(raw)
            result_type = result_types.get(str(item.get("type")))
            if result_type is not None:
                self._dedup[str(command_id)] = result_type.model_validate(item["data"])
        self._inbox = [InboxMessage.model_validate(item) for item in carry.get("inbox", [])]
        self._wait_events = {
            str(key): dict(value) for key, value in dict(carry.get("wait_events", {})).items()
        }
        self._updates = 0
        from pydantic import TypeAdapter

        adapter = TypeAdapter(TaskNodeDraft)
        nodes: dict[str, PlanNodeState] = {}
        for raw in carry.get("nodes", []):
            item = dict(raw)
            draft = adapter.validate_python(item["draft"])
            nodes[str(item["node_id"])] = PlanNodeState(
                draft=draft,
                status=item["status"],
                frozen=bool(item.get("frozen", False)),
                current_attempt_id=item.get("current_attempt_id"),
                attempt_count=int(item.get("attempt_count", 0)),
                created_by=item.get("created_by"),
            )
        self._plan = PlanState.build(int(carry["plan_version"]), nodes, [tuple(edge) for edge in carry.get("edges", [])])

    def _carry_input(self) -> TaskWorkflowInput:
        assert self._plan is not None
        carry = {
            "status": self._status,
            "plan_version": self._plan.version,
            "nodes": [
                {"node_id": node_id, **state.canonical()} for node_id, state in self._plan.nodes.items()
            ],
            "edges": [list(edge) for edge in self._plan.edges],
            "next_message_seq": self._next_message_seq,
            "completed_nodes": self._completed_nodes,
            "inbox": [message.model_dump(mode="json") for message in self._inbox],
            "wait_events": self._wait_events,
            "entity_versions": self._entity_versions,
            "attempts": self._attempts,
            "approvals": self._approvals,
            "dedup": {
                command_id: {"type": type(result).__name__, "data": result.model_dump(mode="json")}
                for command_id, result in self._dedup.items()
                if hasattr(result, "model_dump")
            },
        }
        return TaskWorkflowInput(
            task_id=self._task_id,
            tenant_id=self._tenant_id,
            created_by=self._created_by,  # type: ignore[arg-type]
            title=self._title,
            goal=self._goal,
            profile=self._profile,
            node_type_registry_version=1,
            budgets=self._budgets,
            policy=self._policy,
            carry=carry,
        )

    def _should_continue_as_new(self) -> bool:
        return (
            self._updates >= _MAX_UPDATES_BEFORE_CAN
            or self._completed_nodes >= _MAX_COMPLETIONS_BEFORE_CAN
            or workflow.info().is_continue_as_new_suggested()
        )

    def _all_nodes_completed(self) -> bool:
        return bool(self._plan) and all(state.status in {"COMPLETED", "SKIPPED"} for state in self._plan.nodes.values())

    def _has_ready_node(self) -> bool:
        if self._status in _OPERATOR_HELD or self._status == "CANCELLED":
            return False  # nothing is scheduled now; counting these nodes would spin the main loop
        return bool(self._plan) and any(
            state.status in {"READY", "RETRY_PENDING"} and node_id not in self._attempts
            for node_id, state in self._plan.nodes.items()
        )

    def _dependencies_done(self, node_id: str) -> bool:
        assert self._plan is not None
        return all(
            self._plan.nodes[source].status == "COMPLETED"
            for source, target in self._plan.edges
            if target == node_id
        )

    def _refresh_readiness(self) -> None:
        assert self._plan is not None
        for node_id, state in self._plan.nodes.items():
            if state.status == "PENDING" and self._dependencies_done(node_id):
                self._set_node_status(node_id, "READY")

    def _require_node(self, node_id: str) -> PlanNodeState:
        if self._plan is None or node_id not in self._plan.nodes:
            raise ApplicationError("unknown node", type="SCHEMA_INVALID", non_retryable=True)
        return self._plan.nodes[node_id]

    def _set_node_status(self, node_id: str, status: str, frozen: bool | None = None) -> None:
        assert self._plan is not None
        state = self._require_node(node_id)
        self._update_node(node_id, status=status, frozen=state.frozen if frozen is None else frozen)

    def _update_node(self, node_id: str, **updates: Any) -> None:
        assert self._plan is not None
        state = self._require_node(node_id)
        new_status = updates.get("status", state.status)
        if new_status != state.status and workflow.patched("task-event-vocabulary"):
            self._emit("node.status_changed", {
                "node_id": node_id,
                "from_status": state.status,
                "to_status": new_status,
            })
        self._plan = self._plan.with_updates(
            nodes={
                **self._plan.nodes,
                node_id: PlanNodeState(
                    draft=state.draft,
                    status=updates.get("status", state.status),
                    frozen=updates.get("frozen", state.frozen),
                    current_attempt_id=updates.get("current_attempt_id", state.current_attempt_id),
                    attempt_count=updates.get("attempt_count", state.attempt_count),
                ),
            }
        )
        if updates.get("status") in {"COMPLETED", "RETRY_PENDING"}:
            self._refresh_readiness()

    def _exploration_spent_without_plan(self, signal: AttemptFinishedSignal) -> bool:
        assert self._plan is not None
        exploration = deterministic_id(f"{self._task_id}:exploration", "n")
        return bool(
            signal.node_id == exploration
            and signal.result
            and signal.result.budget_exhausted
            and self._plan.version == 1
            and workflow.patched("task-exploration-budget")
        )

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

    async def _run_short_activity(self, name: str, payload: Any) -> dict[str, Any]:
        return await workflow.execute_activity(
            name,
            payload,
            task_queue="orbit.io",
            result_type=dict,
            start_to_close_timeout=_IO_TIMEOUT,
            retry_policy=_RETRY,
        )

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


@workflow.defn(name="AttemptWorkflow", versioning_behavior=_versioning_behavior(VersioningBehavior.PINNED))
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
                    start_to_close_timeout=_AGENT_TIMEOUT,
                    heartbeat_timeout=_HEARTBEAT,
                    cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                    retry_policy=_RETRY,
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
            await self._notify_parent_finished(inp, "cancelled", {"checkpoint_ref": _sha(inp.attempt_id)})
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
                start_to_close_timeout=_AGENT_TIMEOUT,
                heartbeat_timeout=_HEARTBEAT,
                cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                retry_policy=_RETRY,
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
                start_to_close_timeout=_IO_TIMEOUT,
                retry_policy=_RETRY,
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
                    "checkpoint_ref": result.get("checkpoint_ref", _sha(inp.attempt_id)),
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

"""Attempts of a TaskWorkflow: scheduling ready nodes, starting and cancelling AttemptWorkflows, and
what an attempt tells its parent."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from temporalio import workflow
from temporalio.exceptions import TemporalError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        ApprovalDecidedSignal,
        AttemptFinishedSignal,
        AttemptParkedSignal,
        AttemptWorkflowInput,
        Budget,
        DeliverMessagesSignal,
        ExternalEventSignal,
    )

    from orbit_orch.plan_engine import attempt_workflow_id, deterministic_id
    from orbit_orch.workflow_common import (
        BOUNDED_CANCEL,
        BUDGET_ENFORCEMENT,
        CANCEL_FALLBACK_S,
        CANCEL_REQUEST_S,
        CANCEL_WAITS_FOR_CHILD,
        INBOX_CONSUME,
        PROFILE_SWITCH,
        RETRY_POLICY,
        SCHEDULE_CONCURRENCY,
        SIGNAL_CLOSED_CHILD,
        SOP_EXPANSION,
        TEAM_STAGE,
        VERIFY_FINISHED_ATTEMPTS,
        WAIT_TIMER_CARRY,
    )

from orbit_orch.attempt_workflow import AttemptWorkflow
from orbit_orch.task_completion import TaskCompletion


class TaskAttempts(TaskCompletion):
    async def _schedule_ready(self) -> None:
        if self._status in {"PAUSED", "PAUSED_NEEDS_REVIEW", "TAKEN_OVER", "CANCELLED"}:
            return
        if self._plan is None:
            return
        if not workflow.patched(SCHEDULE_CONCURRENCY):
            await self._schedule_ready_by_id()
            return
        # Ready nodes start in plan order, as many as the task's concurrency leaves room for. Attempts that start together
        # share what the task has left of its budget between them.
        order = self._ready_order()
        pool = self._remaining()
        agent_types = {"agent_turn", "sop_stage", "team_stage"}
        compiled = workflow.patched(SOP_EXPANSION)  # a `sop_stage` node is compiled into the plan and runs no attempt
        share = sum(
            1 for node_id in order
            if self._plan.nodes[node_id].draft.type in agent_types
            and not (compiled and self._plan.nodes[node_id].draft.type == "sop_stage")
        )
        for node_id in order:
            if self._status in {"PAUSED", "PAUSED_NEEDS_REVIEW", "TAKEN_OVER", "CANCELLED"}:
                break  # a command that arrived while an attempt was starting stops the rest
            state = self._plan.nodes.get(node_id)
            if state is None or state.status not in {"READY", "RETRY_PENDING"} or node_id in self._attempts:
                continue
            node = state.draft
            if node.type == "sop_stage" and compiled:
                await self._expand_sop(node_id)
            elif node.type in agent_types:
                if self._running_count() >= self._max_concurrency() or (
                    node.workspace_access == "write"
                    and any(item.get("workspace_access") == "write" for item in self._attempts.values())
                ):
                    continue
                if not await self._start_attempt(node_id, share, pool):
                    break  # the task cannot pay for this node: it is on hold
            else:
                await self._start_waiting_node(node_id)

    async def _schedule_ready_by_id(self) -> None:
        """How ready nodes were started before they had an order and a limit: all at once, by node id."""
        assert self._plan is not None
        for node_id, state in sorted(self._plan.nodes.items()):
            if state.status not in {"READY", "RETRY_PENDING"} or node_id in self._attempts:
                continue
            if self._backoff_remaining(node_id) > 0:
                continue  # a retry waits out its backoff; the main loop wakes when it ends
            if not self._dependencies_done(node_id):
                continue
            node = state.draft
            if node.type in {"agent_turn", "sop_stage"}:
                if node.workspace_access == "write" and any(
                    item.get("workspace_access") == "write" for item in self._attempts.values()
                ):
                    continue
                await self._start_attempt(node_id)
            else:
                await self._start_waiting_node(node_id)

    async def _start_waiting_node(self, node_id: str) -> None:
        """A node that no agent runs: an approval waits for a person, a wait for its timer or event, a checkpoint commits."""
        node = self._require_node(node_id).draft
        if node.type == "approval":
            self._set_node_status(node_id, "AWAITING_APPROVAL")
            self._ask_node_approval(node_id)
        elif node.type == "wait":
            self._set_node_status(node_id, "AWAITING_INPUT")
            wait_key = getattr(node.spec, "wait_key", None)
            timer_s = getattr(node.spec, "timer_s", None)
            if timer_s:
                if workflow.patched(WAIT_TIMER_CARRY):
                    self._wait_deadlines[node_id] = (workflow.now() + timedelta(seconds=timer_s)).isoformat()
                    self._arm_wait_timer(node_id)
                else:
                    asyncio.create_task(self._complete_after(node_id, timer_s))
            elif wait_key and wait_key in self._wait_events:
                if workflow.patched(WAIT_TIMER_CARRY):
                    self._complete_wait(node_id)
                else:
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

    def _profile_for_attempt(self, node_id: str, owner_profile: str | None) -> str:
        """The profile the node's next attempt runs as: the one a switch gave the node, else what it would have had."""
        switched = self._node_profile.get(node_id)
        if switched and workflow.patched(PROFILE_SWITCH):
            return str(switched["to"])
        return self._attempt_profile(owner_profile)

    async def _start_attempt(self, node_id: str, share: int = 1, pool: dict[str, int | None] | None = None) -> bool:
        """Start the node's attempt. False when the task cannot pay for it: nothing was started and the task is on hold
        (`task_budget`). `share` is how many attempts start with this one, among which the `pool` (what the task had left
        when they were chosen) is divided for what a node leaves open."""
        state = self._require_node(node_id)
        reserved: Budget | None = None
        if workflow.patched(BUDGET_ENFORCEMENT):
            reserved, why = self._reserve_for(node_id, share, pool)
            if reserved is None:
                self._hold_for_budget(node_id, why)
                return False
            if not reserved.model_dump(exclude_none=True):
                reserved = None  # the task limits nothing this node could spend: nothing is reserved and the attempt is free
        attempt_no = state.attempt_count + 1
        attempt_id = deterministic_id(f"{self._task_id}:{node_id}:{attempt_no}", "att")
        workflow_id = attempt_workflow_id(self._task_id, node_id, attempt_no)
        spec = state.draft.spec
        goal = getattr(spec, "goal", None) or getattr(spec, "sop", None) or state.draft.title
        goal += self._handover_to(node_id)  # a step of a compiled SOP is told what the steps it depends on handed over
        # A follow-up's goal is the message it was made from; only what came after that message is passed on as messages.
        follow_up = self._follow_ups.get(node_id)
        consume = workflow.patched(INBOX_CONSUME)
        if consume:
            # Whatever the inbox holds goes to this attempt, and leaves the inbox. A follow-up's own message is not in it when
            # it was made the goal; one too long for a goal still is.
            messages = list(self._inbox)
            self._inbox = []
        else:
            messages = self._inbox if follow_up is None else [m for m in self._inbox if m.message_seq > int(follow_up["seq"])]
        # A retry carries on the agent session of the attempt before it and is told why that one was rejected (04 §2).
        retry_from, retry_reason = self._take_retry_input(node_id) if workflow.patched(RETRY_POLICY) else ("", "")
        # A review of a leader's work carries on the session of the leader's attempt it follows.
        continue_from = retry_from or (follow_up or {}).get("from") or self._review_nodes.get(node_id, {}).get("from") or None
        team = state.draft.spec if state.draft.type == "team_stage" and workflow.patched(TEAM_STAGE) else None
        profile = (
            next(member.executor for member in team.members if member.role == team.leader)  # type: ignore[union-attr]
            if team is not None
            else self._profile_for_attempt(node_id, state.draft.owner_profile)
        )
        switched_from, handover = "", ""
        switch = self._node_profile.get(node_id)
        if switch and switch.get("pending") and workflow.patched(PROFILE_SWITCH):
            # The first attempt after a switch runs another model, so the old session is not carried (11 §3): it is told
            # where the node's last attempt left off instead.
            switch["pending"] = False
            switched_from, handover, continue_from = str(switch["from"]), self._handovers.get(node_id, ""), None
        inp = AttemptWorkflowInput(
            task_id=self._task_id,
            tenant_id=self._tenant_id,
            node_id=node_id,
            attempt_id=attempt_id,
            attempt_no=attempt_no,
            node_type=state.draft.type,
            profile=profile,
            goal=goal,
            policy=self._policy,
            workspace_access=state.draft.workspace_access or "none",
            messages=messages,
            config=self._config,
            allow_rules=list(self._allow_rules),
            continue_from=continue_from,
            retry_reason=retry_reason,
            budget=reserved,
            switched_from=switched_from or None,
            handover=handover,
            output_schema_ref=state.draft.completion_contract.output_schema_ref,
            team=team,
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
            "profile": profile,
            "config_version": self._config.config_version,
            **({"reserved": reserved.model_dump(mode="json", exclude_none=True)} if reserved is not None else {}),
        }
        self._attempt_handles[node_id] = handle
        self._update_node(node_id, current_attempt_id=attempt_id, attempt_count=attempt_no, status="RUNNING")
        self._emit("attempt.started", {
            "node_id": node_id,
            "attempt_id": attempt_id,
            "attempt_no": attempt_no,
            "profile": inp.profile,
            "config_version": self._config.config_version,
            **({"switched_from": switched_from} if switched_from else {}),
            **({"budget_reserved": reserved.model_dump(mode="json", exclude_none=True)} if reserved is not None else {}),
        })
        return True

    async def _complete_after(self, node_id: str, seconds: int) -> None:
        await workflow.sleep(seconds)
        if self._plan and self._plan.nodes.get(node_id, None) and self._plan.nodes[node_id].status == "AWAITING_INPUT":
            self._set_node_status(node_id, "COMPLETED")
            self._completed_nodes += 1

    def _arm_wait_timer(self, node_id: str) -> None:
        """Start the timer of a wait node from the deadline in the state: what is left, and none if it has passed."""
        remaining = (datetime.fromisoformat(self._wait_deadlines[node_id]) - workflow.now()).total_seconds()
        self._timers[f"wait:{node_id}"] = asyncio.create_task(self._wait_timer(node_id, remaining))

    async def _wait_timer(self, node_id: str, seconds: float) -> None:
        if seconds > 0:
            await workflow.sleep(seconds)
        if node_id in self._wait_deadlines:
            self._complete_wait(node_id)

    def _complete_wait(self, node_id: str) -> None:
        """A wait node's condition is met, by its timer or by the event it waits for: it completes, is frozen and counts."""
        self._wait_deadlines.pop(node_id, None)
        self._timers.pop(f"wait:{node_id}", None)
        if self._plan is None or node_id not in self._plan.nodes or self._plan.nodes[node_id].status != "AWAITING_INPUT":
            return
        self._set_node_status(node_id, "COMPLETED", frozen=True)
        self._completed_nodes += 1
        self._wake += 1

    def _rearm_after_load(self) -> None:
        """A Continue-As-New drops the timers of the old run. Timers of wait nodes and the fallback of attempts that were
        being cancelled are armed again from what the carried state says."""
        if self._plan is None:
            return
        for node_id in list(self._wait_deadlines):
            if self._plan.nodes.get(node_id) is not None and self._plan.nodes[node_id].status == "AWAITING_INPUT":
                if workflow.patched(WAIT_TIMER_CARRY):
                    self._arm_wait_timer(node_id)
            else:
                del self._wait_deadlines[node_id]
        for node_id, item in self._attempts.items():
            if item.get("cancelling") and workflow.patched(CANCEL_WAITS_FOR_CHILD):
                self._arm_cancel_fallback(node_id, item)

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
                    self._wait_deadlines.pop(node_id, None)
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
                await self._cancel_attempt(node_id, item)

    async def _cancel_active_attempts(self) -> None:
        for node_id, item in list(self._attempts.items()):
            await self._cancel_attempt(node_id, item)

    async def _cancel_attempt(self, node_id: str, item: dict[str, Any]) -> None:
        if workflow.patched(CANCEL_WAITS_FOR_CHILD):
            # Messages are not handed to an attempt that is going away: they stay for its replacement.
            item["cancelling"] = True
        handle = workflow.get_external_workflow_handle(item["workflow_id"])
        if workflow.patched(BOUNDED_CANCEL):
            # The request is answered when the child takes it, and a child that is closing at that very moment may never answer
            # it: the command (a task cancel, a stop, an interrupt) must not wait for ever for it. The child reports its own end;
            # the fallback below covers one that does not.
            request = asyncio.ensure_future(handle.cancel())
            done = False
            try:
                await workflow.wait_condition(lambda: request.done(), timeout=timedelta(seconds=CANCEL_REQUEST_S))
                done = True
            except TimeoutError:
                pass
            if done and not request.cancelled() and isinstance(request.exception(), TemporalError):
                pass  # the child is gone already
            elif not done:
                request.add_done_callback(lambda task: task.cancelled() or task.exception())  # its late answer is not an error
        else:
            try:
                await handle.cancel()
            except TemporalError:
                pass
        self._arm_cancel_fallback(node_id, item)

    def _arm_cancel_fallback(self, node_id: str, item: dict[str, Any]) -> None:
        if item.get("cancelling"):
            self._timers[f"cancel:{item['attempt_id']}"] = asyncio.create_task(
                self._await_cancelled(node_id, str(item["attempt_id"]))
            )
        else:
            asyncio.create_task(self._await_cancelled(node_id))

    async def _await_cancelled(self, node_id: str, attempt_id: str | None = None) -> None:
        # The node stays held while the cancelled attempt winds down: the replacement must not take the workspace's write
        # lease before the old one has let go of it. The child reports its own end once its activity has really stopped
        # (`attemptFinished` with outcome `cancelled`), and that releases the node. This timer is only for a child that
        # never reports, so it is longer than the time an activity that stopped heartbeating takes to be declared lost.
        # Before that (`attempt_id` is None, a history from before it) the node was released after a second.
        await workflow.sleep(1 if attempt_id is None else CANCEL_FALLBACK_S)
        current = self._attempts.get(node_id)
        if current is not None and (attempt_id is None or current.get("attempt_id") == attempt_id):
            self._mark_attempt_cancelled(node_id)

    def _mark_attempt_cancelled(self, node_id: str) -> None:
        attempt = self._attempts.pop(node_id, None)
        self._attempt_handles.pop(node_id, None)
        if attempt is None or self._plan is None:
            return
        self._settle_attempt(attempt, None)  # nothing is known of what it spent: what it held goes back
        self._return_unheard(attempt, [])
        self._attempt_ended(attempt, [])
        if workflow.patched("task-event-vocabulary"):
            # A child that never reported its end cannot be waited for any longer, so the parent records it.
            self._emit("attempt.finished", {
                "node_id": node_id,
                "attempt_id": attempt.get("attempt_id"),
                "outcome": "cancelled",
                "failure": None,
                **self._attempt_facts(attempt),
            })
        state = self._require_node(node_id)
        if state.current_attempt_id == attempt.get("attempt_id"):
            self._attempt_cancelled(node_id, attempt)
            self._update_node(node_id, current_attempt_id=None)

    async def _signal_attempt(self, attempt_id: str, signal_name: str, payload: Any) -> bool:
        """Signal a running attempt. False when it cannot be reached: its workflow has closed (it reported its end, or is
        about to, and the parent has not handled that yet). The attempt is then marked closed, so nothing else is sent to it,
        and the caller keeps what it wanted to deliver."""
        for item in self._attempts.values():
            if item.get("attempt_id") == attempt_id:
                if not workflow.patched(SIGNAL_CLOSED_CHILD):
                    await workflow.get_external_workflow_handle(item["workflow_id"]).signal(signal_name, payload)
                    return True
                if item.get("closed"):
                    return False
                try:
                    await workflow.get_external_workflow_handle(item["workflow_id"]).signal(signal_name, payload)
                except TemporalError:
                    item["closed"] = True
                    return False
                return True
        return False

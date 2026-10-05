"""The updates a person or the API sends to a TaskWorkflow, and the queue of commands the main loop drains."""

from __future__ import annotations

from typing import Any

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        ApprovalDecidedSignal,
        CompleteNodeInput,
        CompleteNodeResult,
        DecideApprovalInput,
        DecideApprovalResult,
        DeliverMessagesSignal,
        GrantBudgetInput,
        GrantBudgetResult,
        InboxMessage,
        PermissionRuleSpec,
        RequestProfileSwitchInput,
        RequestProfileSwitchResult,
        SendMessageInput,
        SendMessageResult,
        TaskConfig,
        TaskControlInput,
        TaskControlResult,
        UpdateTaskConfigInput,
        UpdateTaskConfigResult,
    )
    from orbit_contracts.v3.messages import ApprovalSubject

    from orbit_orch.plan_engine import deterministic_id
    from orbit_orch.workflow_common import (
        BUDGET_ENFORCEMENT,
        HANDED_MESSAGES,
        INBOX_CONSUME,
        MAX_HANDED,
        PROFILE_SWITCH,
        RETRY_POLICY,
        SESSION_STAYS_OPEN,
        SIGNAL_CLOSED_CHILD,
        TAKEOVER_INTERRUPTS,
        VERIFY_FINISHED_ATTEMPTS,
        budget_add,
        closed,
        sha,
    )

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
            mentions=list(dict.fromkeys(req.mentions)),
        )
        self._next_message_seq += 1
        self._inbox.append(message)
        self._wake += 1
        result = SendMessageResult(message_seq=message.message_seq)
        self._remember(req.command_id, result)
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
        # A mention names a role of the task's team (a task without one has none): anything else is refused, not guessed at.
        roles = {member.role for member in self._config.team.members} if self._config.team is not None else set()
        unknown = [role for role in req.mentions if role not in roles]
        if unknown:
            raise ApplicationError(
                f"unknown mention: {', '.join(unknown)}", type="UNKNOWN_MENTION", non_retryable=True
            )

    def _mention_goes_to_members(self) -> bool:
        """Whether a message that @-mentions members is answered by nodes of their own now: yes at plan level, no while a team stage
        runs, which takes the message into its mailbox and wakes them itself. Not while the task is held or over: the message waits
        in the inbox then, as any other does."""
        if self._plan is None or self._status not in {"RUNNING", "WAITING", "COMPLETED", "PLANNING"}:
            return False
        return not any(
            self._plan.nodes[node_id].draft.type == "team_stage"
            and item.get("status") in {"RUNNING", "PARKED_INPUT", "PARKED_APPROVAL"}
            for node_id, item in self._attempts.items()
            if node_id in self._plan.nodes
        )

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
        rule = self._rule_to_allow(approval, req)
        result = DecideApprovalResult(approval_id=req.approval_id, status=approval["status"])
        self._remember(req.command_id, result)
        self._updates += 1
        if workflow.patched("task-event-vocabulary"):
            self._emit("approval.decided", {
                "approval_id": req.approval_id,
                "status": approval["status"],
                "comment": req.comment,
                "always": rule is not None,
            })
        if approval.get("kind") == "profile_switch" and req.decision == "approve" and workflow.patched(PROFILE_SWITCH):
            # The person allowed the switch: the node's next attempt runs as the profile asked for.
            self._apply_profile_switch(
                str(approval["node_id"]), str(approval["to_profile"]), str(approval.get("reason", "")), req.approval_id
            )
        if approval.get("kind") == "node_approval":
            self._settle_node_approval(approval, req.decision)  # an approval node of a compiled SOP
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
                    rule=rule,
                ),
            )
            if not any(
                item["status"] == "PENDING" and item.get("attempt_id") == attempt_id for item in self._approvals.values()
            ):
                self._resume_parked(attempt_id)
        return result

    def _rule_to_allow(self, approval: dict[str, Any], req: DecideApprovalInput) -> PermissionRuleSpec | None:
        """The rule an approval was allowed with "always", added to what the task has allowed; None when the decision does
        not ask for it or the approval offered none."""
        offered = (approval.get("subject") or {}).get("allow_rule")
        if not req.always or req.decision != "approve" or not offered:
            return None
        rule = PermissionRuleSpec.model_validate(offered)
        if rule not in self._allow_rules:
            self._allow_rules.append(rule)
        return rule

    @workflow.update(name="control")
    async def control(self, req: TaskControlInput) -> TaskControlResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        transitions = {
            "pause": "PAUSED",
            # Stop is pause that does not wait: the attempt that is running is ended now. Resume starts the work again.
            "stop": "PAUSED",
            "resume": "RUNNING",
            "cancel": "CANCELLED",
            "takeover": "TAKEN_OVER",
            "handback": "RUNNING",
        }
        target = transitions[req.action]
        if req.action == "resume" and self._status not in {"PAUSED", "PAUSED_NEEDS_REVIEW"}:
            raise ApplicationError("task cannot be resumed", type="INVALID_TRANSITION", non_retryable=True)
        if req.action in {"takeover", "handback"} and workflow.patched(TAKEOVER_INTERRUPTS):
            if req.action == "handback" and self._status != "TAKEN_OVER":
                raise ApplicationError("task is not taken over", type="INVALID_TRANSITION", non_retryable=True)
            if req.action == "takeover" and closed(self._status):
                raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)
            if req.action == "takeover":
                # The person takes the wheel: what the agents are doing stops now, as it does for a stop, and does not count
                # against the nodes. A handback sends them on from the session they had.
                await self._interrupt_active_attempt()
        if req.action == "stop":
            if self._status in {"PAUSED", "PAUSED_NEEDS_REVIEW", "TAKEN_OVER"} or closed(self._status):
                raise ApplicationError("task is not running", type="INVALID_TRANSITION", non_retryable=True)
            await self._interrupt_active_attempt()
        if req.action == "cancel":
            await self._cancel_active_attempts()
            self._stop = True
        self._set_status(target, req.reason or req.action)
        if req.action == "resume" and workflow.patched(RETRY_POLICY):
            # A person looked at what retries gave up on and says go on: those nodes get their tries back.
            self._unblock_nodes(
                [node_id for node_id, entry in self._node_retry.items() if entry.get("blocked")], "resumed by a person"
            )
        self._wake += 1
        if req.action == "cancel" and workflow.patched("task-event-vocabulary"):
            self._emit("task.cancelled", {"reason": req.reason or "cancelled"})
        result = TaskControlResult(status=target)  # type: ignore[arg-type]
        self._remember(req.command_id, result)
        self._updates += 1
        return result

    @workflow.update(name="grantBudget")
    async def grant_budget(self, req: GrantBudgetInput) -> GrantBudgetResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        self._budgets = budget_add(self._budgets, req.delta)
        result = GrantBudgetResult(budgets=self._budgets)
        self._remember(req.command_id, result)
        self._updates += 1
        if not workflow.patched(BUDGET_ENFORCEMENT):
            self._set_status("RUNNING", "budget_granted")
            return result
        self._emit("budget.granted", {"command_id": req.command_id, "delta": req.delta.model_dump(mode="json", exclude_none=True)})
        # Only a task that waits because its budget ran out goes on: one a person paused, took over or cancelled stays as it is.
        if self._status == "PAUSED_NEEDS_REVIEW" and self._budget_hold:
            self._set_status("RUNNING", "budget_granted")
        # The nodes that were blocked for lack of budget get their tries back; the others wait for a person as before.
        self._unblock_nodes(
            [node_id for node_id, entry in self._node_retry.items() if entry.get("blocked_for") == "budget"],
            "budget granted",
        )
        self._wake += 1
        return result

    @workflow.update(name="requestProfileSwitch")
    async def request_profile_switch(self, req: RequestProfileSwitchInput) -> RequestProfileSwitchResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        node = self._require_node(req.node_id)
        if node.frozen:
            raise ApplicationError("node is frozen", type="FROZEN_NODE", non_retryable=True)
        if not workflow.patched(PROFILE_SWITCH):
            result = RequestProfileSwitchResult(effective_attempt_no=int(node.attempt_count) + 1, needs_approval=True)
            self._remember(req.command_id, result)
            self._updates += 1
            return result
        if closed(self._status):
            raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)
        if node.draft.type not in {"agent_turn", "sop_stage"}:
            raise ApplicationError("only a node an agent runs has a profile", type="NOT_ALLOWED", non_retryable=True)
        approval_id = ""
        if self._switch_allowed(req.node_id, req.to_profile):
            # The switch takes effect for the node's next attempt; the one that is running is not touched (11 §3).
            self._apply_profile_switch(req.node_id, req.to_profile, req.reason)
        else:
            approval_id = deterministic_id(f"{self._task_id}:profile_switch:{req.command_id}", "apr")
            subject = ApprovalSubject(
                kind="profile_switch",
                digest=sha(f"{req.node_id}:{req.to_profile}"),
                summary=f"Switch node {req.node_id} to {req.to_profile}: {req.reason}"[:500],
                risk="medium",
                detail=req.to_profile,
            )
            self._approvals[approval_id] = {
                "approval_id": approval_id,
                "status": "PENDING",
                "kind": "profile_switch",
                "node_id": req.node_id,
                "to_profile": req.to_profile,
                "reason": req.reason,
                "subject": subject.model_dump(mode="json"),
            }
            self._emit("approval.requested", {
                "approval_id": approval_id,
                "node_id": req.node_id,
                "subject": subject.model_dump(mode="json"),
            })
        result = RequestProfileSwitchResult(
            effective_attempt_no=int(node.attempt_count) + 1,
            needs_approval=bool(approval_id),
            approval_id=approval_id or None,
        )
        self._remember(req.command_id, result)
        self._updates += 1
        return result

    def _switch_allowed(self, node_id: str, to_profile: str) -> bool:
        """Whether a node may switch to `to_profile` without anyone's say: the task's expert (or profile), a member of its
        team, the profile the node is configured to repair itself with, or the one it was given at the start. Any other
        profile may have permissions or a budget the task was not locked to (11 §3), and needs an approval."""
        draft = self._require_node(node_id).draft
        allowed = {self._profile}
        if self._config.expert:
            allowed.add(self._config.expert)
        if self._config.team is not None:
            allowed |= {member.expert for member in self._config.team.members}
        if draft.retry is not None and draft.retry.repair_profile:
            allowed.add(draft.retry.repair_profile)
        if draft.owner_profile:
            allowed.add(draft.owner_profile)
        return to_profile in allowed

    def _apply_profile_switch(self, node_id: str, to_profile: str, reason: str, approval_id: str = "") -> None:
        """The node's next attempt runs as `to_profile`, and records the profile it ran as before. The switch does not touch
        an attempt that is running, and is not applied to a node that has become frozen or is gone."""
        if self._plan is None or node_id not in self._plan.nodes or self._plan.nodes[node_id].frozen:
            return
        current = self._profile_for_attempt(node_id, self._plan.nodes[node_id].draft.owner_profile)
        if to_profile == current:
            return
        pending = self._node_profile.get(node_id)
        origin = str(pending["from"]) if pending and pending.get("pending") else current
        if to_profile == origin:
            self._node_profile.pop(node_id, None)  # back to what its last attempt ran as: no switch is left to tell
        else:
            self._node_profile[node_id] = {"to": to_profile, "from": origin, "pending": True}
        self._emit("profile.switched", {
            "node_id": node_id,
            "from_profile": current,
            "to_profile": to_profile,
            "reason": reason,
            **({"approval_id": approval_id} if approval_id else {}),
        })

    @workflow.update(name="completeNode")
    async def complete_node(self, req: CompleteNodeInput) -> CompleteNodeResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        self._set_node_status(req.node_id, "COMPLETED", frozen=True, reason=req.reason or "completed by a person")
        self._completed_nodes += 1
        # A node a person finished is no longer waiting for retries, a timer or a decision.
        self._node_retry.pop(req.node_id, None)
        self._cancel_node_approvals(req.node_id)
        self._wait_deadlines.pop(req.node_id, None)
        timer = self._timers.pop(f"wait:{req.node_id}", None)
        if timer is not None:
            timer.cancel()
        result = CompleteNodeResult(node_id=req.node_id)
        self._remember(req.command_id, result)
        self._updates += 1
        self._wake += 1
        return result

    @complete_node.validator
    def validate_complete_node(self, req: CompleteNodeInput) -> None:
        # A person's decision, taken while the agents are not working: the task is taken over or paused, the node is not
        # done and has no attempt running. A command that was already applied is let through to be answered again.
        if req.command_id in self._dedup:
            return
        if self._status not in {"TAKEN_OVER", "PAUSED", "PAUSED_NEEDS_REVIEW"}:
            raise ApplicationError("pause or take over the task first", type="INVALID_TRANSITION", non_retryable=True)
        node = self._require_node(req.node_id)
        if node.frozen or node.status in {"COMPLETED", "SKIPPED"}:
            raise ApplicationError("node is already complete", type="FROZEN_NODE", non_retryable=True)
        if req.node_id in self._attempts or node.status in {"RUNNING", "VERIFYING"}:
            raise ApplicationError("node is running", type="INVALID_TRANSITION", non_retryable=True)

    @workflow.update(name="updateTaskConfig")
    async def update_task_config(self, req: UpdateTaskConfigInput) -> UpdateTaskConfigResult:
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        self._config = TaskConfig(
            config_version=self._config.config_version + 1,
            expert=req.expert,
            skills=req.skills,
            connectors=req.connectors,
            mode=req.mode,
            team=req.team,
        )
        result = UpdateTaskConfigResult(config_version=self._config.config_version)
        self._remember(req.command_id, result)
        self._updates += 1
        self._emit("task.config_changed", {
            "config_version": self._config.config_version,
            "expert": self._config.expert,
            "skills": self._config.skills,
            "connector_ids": None if req.connectors is None else [item.id for item in req.connectors],
            "mode": self._config.mode,
        })
        return result

    @update_task_config.validator
    def validate_update_task_config(self, req: UpdateTaskConfigInput) -> None:
        # Control waits only for the update to be accepted, so a refusal has to come from here to reach it. A command
        # that was already applied is let through: the handler answers it again from the dedup table.
        if req.command_id in self._dedup:
            return
        if closed(self._status):
            raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)
        if req.base_config_version != self._config.config_version:
            raise ApplicationError(
                "the task's configuration changed since it was read",
                type="CONFIG_VERSION_CONFLICT",
                non_retryable=True,
            )

    async def _drain_commands(self) -> None:
        while self._commands:
            kind, payload = self._commands.pop(0)
            if kind == "message":
                consume = workflow.patched(INBOX_CONSUME)
                if consume and all(item.message_seq != payload.message_seq for item in self._inbox):
                    continue  # an attempt that started meanwhile took it with the rest of the inbox
                if payload.mentions and self._mention_goes_to_members() and self._mention_follow_ups(payload):
                    continue  # the members it names answer it, each in a node of its own
                attempt = next(
                    (
                        item
                        for item in self._attempts.values()
                        if item.get("status") in {"RUNNING", "PARKED_INPUT", "PARKED_APPROVAL"}
                        # An attempt that is being cancelled is not handed anything: its replacement gets the message.
                        and not (consume and item.get("cancelling"))
                    ),
                    None,
                )
                if attempt is not None and attempt.get("closed") and workflow.patched(SIGNAL_CLOSED_CHILD):
                    self._unsent = True
                    continue  # the attempt has closed: the message waits in the inbox for what comes after it
                if attempt is not None:
                    delivered = await self._signal_attempt(
                        str(attempt["attempt_id"]),
                        "deliverMessages",
                        DeliverMessagesSignal(messages=[payload]),
                    )
                    if not delivered and workflow.patched(SIGNAL_CLOSED_CHILD):
                        self._unsent = True
                        continue  # not sent: it stays in the inbox, and the end of the attempt hands it on
                    if consume:
                        self._inbox = [item for item in self._inbox if item.message_seq != payload.message_seq]
                    if workflow.patched(HANDED_MESSAGES):
                        # Signalled is not heard: the attempt may have left its loop already. It reports what it heard.
                        # Only the newest few: the race is about what arrives as the attempt closes, and what is older was heard
                        # or comes back as unconsumed. This keeps what a Continue-As-New carries independent of the message count.
                        handed = attempt.setdefault("handed", [])
                        handed.append(payload.model_dump(mode="json"))
                        del handed[:-MAX_HANDED]
                    if attempt.get("status") == "PARKED_INPUT":
                        self._resume_parked(str(attempt["attempt_id"]))
                elif self._status == "COMPLETED" and self._all_nodes_completed() and workflow.patched(SESSION_STAYS_OPEN):
                    self._start_follow_up(payload)
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

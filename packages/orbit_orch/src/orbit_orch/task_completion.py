"""Completion checks (04 §5, 17 G3, G15): what a proposal or an attempt's own report needs before its node freezes."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        AttemptFinishedSignal,
        CompletionAccepted,
        CompletionProposal,
        CompletionResult,
    )
    from orbit_contracts.v3.common import Failure

    from orbit_orch.completion_check import (
        failure_message,
        finished_proposal,
        missing_result,
        summarize,
    )
    from orbit_orch.workflow_common import (
        COMMAND_SETUP_S,
        HEARTBEAT,
        RETRY,
        VERIFY_FINISHED_ATTEMPTS,
        closed,
        reasons,
    )

from orbit_orch.task_plan import TaskPlan


class TaskCompletion(TaskPlan):
    @workflow.update(name="proposeCompletion")
    async def propose_completion(self, req: CompletionProposal) -> CompletionResult:
        if closed(self._status):
            raise ApplicationError("task is closed", type="TASK_CLOSED", non_retryable=True)
        previous = self._dedup.get(req.command_id)
        if previous is not None:
            return previous
        attempt = self._attempts.get(req.node_id)
        if attempt is None or attempt.get("attempt_id") != req.attempt_id:
            raise ApplicationError("attempt is stale", type="STALE_ATTEMPT", non_retryable=True)
        if attempt.get("verdict") and workflow.patched(VERIFY_FINISHED_ATTEMPTS):
            # This attempt's completion was already judged (or is being): it does not get a second proposal.
            raise ApplicationError("attempt is stale", type="STALE_ATTEMPT", non_retryable=True)
        self._dedup[req.command_id] = CompletionAccepted()
        self._commands.append(("completion", req))
        self._wake += 1
        self._updates += 1
        return CompletionAccepted()

    def _settle_proposal(self, node_id: str, attempt: dict[str, Any] | None, failures: list[dict[str, Any]]) -> None:
        """The verdict on a `proposeCompletion`. The attempt is marked with it: a rejected attempt stays refused when it
        reports its own end later, an accepted one is not verified again."""
        if attempt is None or self._attempts.get(node_id) is not attempt:
            return  # cancelled while it was verified: the cancel already released the node
        rejections = summarize(failures)
        attempt["verdict"] = "rejected" if rejections else "accepted"
        attempt["rejections"] = rejections
        if rejections:
            self._set_node_status(node_id, "RETRY_PENDING", reason=failure_message(rejections))
        else:
            self._set_node_status(node_id, "COMPLETED", frozen=True)
            self._completed_nodes += 1

    async def _verify_proposal(self, proposal: CompletionProposal) -> list[dict[str, Any]]:
        """04 §5: the checks of the node's completion contract. `verify_completion` (orbit.io) does the schema,
        artifact and manifest checks; each `command` verification then runs as `verify_command` (orbit.agent) on the
        workspace snapshot the manifest names. It stops at the first failing step and returns the structured reasons;
        an empty list means the completion is accepted. A verification activity that cannot run (its retries are
        used up) rejects the completion instead of failing the task workflow."""
        assert self._plan is not None
        contract = self._plan.nodes[proposal.node_id].draft.completion_contract
        try:
            result = await self._run_short_activity(
                "verify_completion",
                {
                    **proposal.model_dump(mode="json"),
                    "tenant_id": self._tenant_id,
                    "task_id": self._task_id,
                    "completion_contract": contract.model_dump(mode="json"),
                },
            )
            failures = reasons(result, "verify_completion")
            if not failures:
                failures = await self._verify_commands(proposal, contract, result.get("workspace_snapshot_ref"))
        except ActivityError as exc:
            failures = [{"check": "verification", "code": "verification_unavailable", "message": str(exc.cause or exc), "detail": {}}]
        if failures:
            workflow.logger.warning(
                "completion rejected",
                extra={"node_id": proposal.node_id, "codes": [item.get("code") for item in failures]},
            )
        return failures

    async def _verify_commands(
        self, proposal: CompletionProposal, contract: Any, snapshot: str | None
    ) -> list[dict[str, Any]]:
        for verification in contract.verifications:
            if verification.kind != "command":
                continue
            timeout_s = int(verification.spec.get("timeout_s", 600))
            result = await workflow.execute_activity(
                "verify_command",
                {
                    "tenant_id": self._tenant_id,
                    "task_id": self._task_id,
                    "node_id": proposal.node_id,
                    "attempt_id": proposal.attempt_id,
                    "workspace_snapshot_ref": snapshot,
                    "command": verification.spec.get("command"),
                    "timeout_s": timeout_s,
                },
                task_queue="orbit.agent",
                result_type=dict,
                start_to_close_timeout=timedelta(seconds=timeout_s + COMMAND_SETUP_S),
                heartbeat_timeout=HEARTBEAT,
                retry_policy=RETRY,
            )
            failures = reasons(result, "verify_command")
            if failures:
                return failures
        return []

    async def _completion_rejections(self, signal: AttemptFinishedSignal, attempt: dict[str, Any]) -> list[dict[str, str]]:
        """Why the node may not complete on this attempt's report, empty if it may. A proposal of this attempt that was
        judged already stands: an accepted one has completed the node, a rejected one keeps the attempt refused."""
        if attempt.get("verdict") == "verifying":
            await workflow.wait_condition(lambda: attempt.get("verdict") != "verifying")
        if attempt.get("verdict") == "accepted":
            return []
        if attempt.get("verdict") == "rejected":
            return list(attempt.get("rejections", []))
        if self._require_node(signal.node_id).status == "BLOCKED" or self._exploration_spent_without_plan(signal):
            return []  # not a completion: a person takes over (05 §2, §4)
        proposal = finished_proposal(signal)
        attempt["verdict"] = "verifying"
        self._set_node_status(signal.node_id, "VERIFYING")
        rejections = summarize(missing_result() if proposal is None else await self._verify_proposal(proposal))
        attempt["verdict"] = "rejected" if rejections else "accepted"
        attempt["rejections"] = rejections
        return rejections

    def _settle_node(self, signal: AttemptFinishedSignal, outcome: str, rejections: list[dict[str, str]]) -> None:
        node_id = signal.node_id
        if rejections:
            self._set_node_status(node_id, "RETRY_PENDING", reason=failure_message(rejections))
        elif outcome != "completed":
            self._set_node_status(node_id, "RETRY_PENDING")
        elif self._plan.nodes[node_id].status == "BLOCKED":  # type: ignore[union-attr]
            # The agent declared its own node unplannable (05 §2): it stays blocked and a person takes over.
            self._set_status("PAUSED_NEEDS_REVIEW", "the agent declared the task unplannable")
        elif self._exploration_spent_without_plan(signal):
            # 05 §4: the budget ran out and nothing was planned, so a person decides what happens next.
            self._emit("budget.exhausted", {"scope": "exploration", "node_id": node_id})
            self._set_node_status(node_id, "BLOCKED")
            self._set_status("PAUSED_NEEDS_REVIEW", "exploration budget spent without a plan")
        elif self._plan.nodes[node_id].status != "COMPLETED":  # type: ignore[union-attr]
            self._set_node_status(node_id, "COMPLETED", frozen=True)
            self._completed_nodes += 1

    async def _finish_attempt(self, signal: AttemptFinishedSignal, attempt: dict[str, Any]) -> None:
        """An attempt reports its end. The report counts once, and only from the node's current attempt: a report
        naming another attempt, or a second delivery of one being settled, is dropped. An attempt that ends
        `completed` is verified (04 §5) before its node is frozen; if the check refuses it the attempt ends as failed
        with the structured reasons and the node goes to RETRY_PENDING. This is a signal handler, so a Continue-As-New
        waits for it and carries the settled state."""
        node_id = signal.node_id
        if attempt.get("attempt_id") != signal.attempt_id or attempt.get("finishing"):
            return
        attempt["finishing"] = True
        attempt["status"] = signal.outcome
        attempt["result"] = signal.result.model_dump(mode="json") if signal.result else None
        outcome, failure = signal.outcome, signal.failure
        rejections: list[dict[str, str]] = []
        if outcome == "completed":
            self._emit_manifest_created(signal)
            rejections = await self._completion_rejections(signal, attempt)
            if self._attempts.get(node_id) is not attempt or self._status == "CANCELLED":
                return  # cancelled while it was verified: the cancel already released the node
            if rejections:
                outcome = "failed"
                failure = Failure(failure_class="verification", retryable=True, message=failure_message(rejections))
        self._emit_attempt_finished(signal, outcome, failure)
        self._settle_node(signal, outcome, rejections)
        if self._require_node(node_id).current_attempt_id == signal.attempt_id:
            self._update_node(node_id, current_attempt_id=None)
        self._attempts.pop(node_id, None)
        self._attempt_handles.pop(node_id, None)
        self._wake += 1

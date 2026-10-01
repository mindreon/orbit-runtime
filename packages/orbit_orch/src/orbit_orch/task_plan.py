"""The plan of a TaskWorkflow: node state, readiness, and plan changes."""

from __future__ import annotations

import hashlib
from typing import Any

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        Actor,
        AttemptFinishedSignal,
        InboxMessage,
        PlanChangeCommand,
        PlanChangeResult,
        PlanView,
    )
    from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec
    from orbit_contracts.v3.plan import AddNodeOp, PlanChangeAccepted

    from orbit_orch.plan_engine import (
        PlanNodeState,
        PlanPolicy,
        apply,
        deterministic_id,
        initial_plan,
    )
    from orbit_orch.workflow_common import OPERATOR_HELD

from orbit_orch.task_events import TaskEvents


class TaskPlan(TaskEvents):
    def _require_node(self, node_id: str) -> PlanNodeState:
        if self._plan is None or node_id not in self._plan.nodes:
            raise ApplicationError("unknown node", type="SCHEMA_INVALID", non_retryable=True)
        return self._plan.nodes[node_id]

    def _set_node_status(self, node_id: str, status: str, frozen: bool | None = None, reason: str = "") -> None:
        assert self._plan is not None
        state = self._require_node(node_id)
        self._update_node(node_id, reason, status=status, frozen=state.frozen if frozen is None else frozen)

    def _update_node(self, node_id: str, reason: str = "", **updates: Any) -> None:
        assert self._plan is not None
        state = self._require_node(node_id)
        new_status = updates.get("status", state.status)
        if new_status != state.status and workflow.patched("task-event-vocabulary"):
            self._emit("node.status_changed", {
                "node_id": node_id,
                "from_status": state.status,
                "to_status": new_status,
                **({"reason": reason} if reason else {}),
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

    def _all_nodes_completed(self) -> bool:
        return bool(self._plan) and all(state.status in {"COMPLETED", "SKIPPED"} for state in self._plan.nodes.values())

    def _has_ready_node(self) -> bool:
        if self._status in OPERATOR_HELD or self._status == "CANCELLED":
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

    def _checkpoint_payload(self, node_id: str) -> dict[str, Any]:
        """The activity input of a checkpoint node. The activity takes no defaults (17 G9), so the identity it stores
        the checkpoint under is spelled out: the node is its own attempt for storage, so two tasks or two nodes never
        share an (attempt, seq) key."""
        payload: dict[str, Any] = {"tenant_id": self._tenant_id, "task_id": self._task_id, "node_id": node_id}
        if workflow.patched("checkpoint-node-explicit-identity"):
            payload |= {
                "attempt_id": deterministic_id(f"{self._task_id}:{node_id}:checkpoint", "att"),
                "seq": 0,
                "kind": "plan",
            }
        return payload

    def _plan_policy(self) -> PlanPolicy:
        assert self._plan is not None
        return PlanPolicy(
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

    def _start_follow_up(self, message: InboxMessage) -> None:
        """A message that comes when every node is done becomes a node of its own. Its attempt carries on the agent session
        of the one that ran last, so the conversation goes on instead of starting over."""
        assert self._plan is not None
        first_line = next((line.strip() for line in message.text.splitlines() if line.strip()), "Follow-up")
        command = PlanChangeCommand(
            command_id=hashlib.sha256(f"follow-up:{self._task_id}:{message.message_seq}".encode()).hexdigest(),
            task_id=self._task_id,
            base_plan_version=self._plan.version,
            actor=Actor(kind="system", id="task-workflow"),
            ops=[AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title=first_line[:200], spec=AgentTurnSpec(goal=message.text)))],
            reason="follow-up message",
        )
        outcome = apply(self._plan, command, self._plan_policy())
        if not isinstance(outcome.result, PlanChangeAccepted):
            self._emit("plan.change_rejected", outcome.result.model_dump(mode="json"))
            return
        self._plan = outcome.plan
        self._refresh_readiness()
        self._emit("plan.version_committed", {
            "plan_version": self._plan.version,
            "hash": self._plan.hash,
            "command_id": command.command_id,
        })
        self._follow_ups[outcome.result.id_map["tmp:1"]] = {"from": self._last_attempt_id, "seq": message.message_seq}

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
        outcome = apply(self._plan, command, self._plan_policy())
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

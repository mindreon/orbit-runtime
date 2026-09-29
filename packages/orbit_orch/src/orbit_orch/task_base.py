"""The state of a TaskWorkflow, and how it is carried across a Continue-As-New."""

from __future__ import annotations

from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        Actor,
        Budget,
        CompletionAccepted,
        DecideApprovalResult,
        GrantBudgetResult,
        InboxMessage,
        Policy,
        RequestProfileSwitchResult,
        SendMessageResult,
        TaskControlResult,
        TaskWorkflowInput,
    )
    from orbit_contracts.v3.common import Usage
    from orbit_contracts.v3.nodes import TaskNodeDraft
    from orbit_contracts.v3.plan import PlanChangeAccepted, PlanChangeRejected

    from orbit_orch.plan_engine import PlanNodeState, PlanState
    from orbit_orch.workflow_common import MAX_COMPLETIONS_BEFORE_CAN, MAX_UPDATES_BEFORE_CAN


class TaskWorkflowBase:
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
            self._updates >= MAX_UPDATES_BEFORE_CAN
            or self._completed_nodes >= MAX_COMPLETIONS_BEFORE_CAN
            or workflow.info().is_continue_as_new_suggested()
        )

"""The state of a TaskWorkflow, and how it is carried across a Continue-As-New."""

from __future__ import annotations

from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        Actor,
        Budget,
        CompleteNodeResult,
        CompletionAccepted,
        DecideApprovalResult,
        GrantBudgetResult,
        InboxMessage,
        PermissionRuleSpec,
        Policy,
        RequestProfileSwitchResult,
        SendMessageResult,
        TaskConfig,
        TaskControlResult,
        TaskWorkflowInput,
        UpdateTaskConfigResult,
    )
    from orbit_contracts.v3.common import Usage
    from orbit_contracts.v3.nodes import TaskNodeDraft
    from orbit_contracts.v3.plan import PlanChangeAccepted, PlanChangeRejected

    from orbit_orch.plan_engine import PlanNodeState, PlanState
    from orbit_orch.workflow_common import (
        MAX_COMPLETIONS_BEFORE_CAN,
        MAX_DECIDED_APPROVALS,
        MAX_DEDUP,
        MAX_UPDATES_BEFORE_CAN,
        STATE_BOUNDS,
    )


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
        self._config = TaskConfig()
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
        # The attempt that ran last, and for each follow-up node (a message that came after the plan was done) the attempt
        # whose agent session it carries on and the message it was made from.
        self._last_attempt_id = ""
        self._follow_ups: dict[str, dict[str, Any]] = {}
        # What a person allowed for the rest of the task, by "always allow" on an approval.
        self._allow_rules: list[PermissionRuleSpec] = []
        # Per node: failed attempts so far, the backoff before the next try, the attempt a retry carries on and why the
        # last one was rejected (`task_retry`).
        self._node_retry: dict[str, dict[str, Any]] = {}
        # When each timer wait node ends, so a Continue-As-New can arm the timer again with what is left.
        self._wait_deadlines: dict[str, str] = {}
        self._timers: dict[str, Any] = {}
        # What was compacted out of the live plan (`task_plan`).
        self._archive: dict[str, Any] = {"count": 0, "hash": "", "recent": []}
        # The task is in PAUSED_NEEDS_REVIEW because its budget ran out: only a grant resumes it (`task_budget`).
        self._budget_hold = False
        # A message could not be handed to an attempt that had closed and waits in the inbox for the next round.
        self._unsent = False
        # Per node, the profile a switch gave it and whether its next attempt has not yet been told (`task_profile`), and
        # what the node's last attempt left behind for a successor that cannot carry its session.
        self._node_profile: dict[str, dict[str, Any]] = {}
        self._handovers: dict[str, str] = {}
        # SOPs compiled into the plan, by the `sop_stage` node (`task_sop`): the steps' nodes, which are still open, how
        # the steps depend on each other and what finished steps handed over.
        self._sops: dict[str, dict[str, Any]] = {}
        # Reviews of a leader's work (`task_review`): per leader node whose attempt created tasks, the review it waits to open
        # (the tasks, what they produced, the round it will be); per review node, the attempt whose session it carries on and
        # its round; and the round each node belongs to, so a chain of reviews is bounded wherever it goes.
        self._reviews: dict[str, dict[str, Any]] = {}
        self._review_nodes: dict[str, dict[str, Any]] = {}
        self._node_rounds: dict[str, int] = {}
        # The attempt each member of the team last ran a node in (a mention follow-up carries on that session).
        self._member_sessions: dict[str, str] = {}

    def _step_of(self, node_id: str) -> tuple[str, dict[str, Any]] | None:
        """The SOP and the part of it a node is (`role`, `step_id`, `index`, `subject`), for a node a compiled SOP made."""
        for sop_node, entry in self._sops.items():
            part = entry["nodes"].get(node_id)
            if part is not None:
                return sop_node, part
        return None

    def _sop_facts(self, node_id: str) -> dict[str, Any]:
        """`parent_node_id`'s companion in `node.status_changed`: which step of which SOP the node is."""
        if node_id in self._sops:
            entry = self._sops[node_id]
            return {"sop_step": {"sop": entry["ref"], "role": "sop", "total": entry["total"]}}
        found = self._step_of(node_id)
        if found is None:
            return {}
        sop_node, part = found
        entry = self._sops[sop_node]
        return {"sop_step": {
            "sop": entry["ref"], "role": part["role"], "total": entry["total"],
            "step_id": part["step_id"], "index": part["index"], "subject": part["subject"],
        }}

    def _sop_child_changed(self, node_id: str, status: str) -> None:
        """A node's status changed. A no-op here; `TaskSop` keeps the `sop_stage` node in step with its nodes."""

    def _note_leader_done(self, node_id: str, attempt: dict[str, Any]) -> None:
        """A node's attempt completed it. A no-op here; `TaskReview` opens a review of what a leader's attempt created."""

    def _record_review_output(self, signal: Any) -> None:
        """A node's attempt ended completed. A no-op here; `TaskReview` keeps what the tasks a leader created produced."""

    def _open_reviews(self) -> None:
        """The main loop's turn to open the reviews that are due. A no-op here; see `TaskReview`."""

    def _load(self, inp: TaskWorkflowInput) -> None:
        self._task_id, self._tenant_id, self._created_by = inp.task_id, inp.tenant_id, inp.created_by
        self._title, self._goal, self._profile, self._budgets = inp.title, inp.goal, inp.profile, inp.budgets
        self._policy = inp.policy
        self._config = inp.config
        carry = inp.carry
        if not carry:
            return
        self._status = str(carry.get("status", "RUNNING"))
        self._last_attempt_id = str(carry.get("last_attempt_id", ""))
        self._follow_ups = {str(k): dict(v) for k, v in dict(carry.get("follow_ups", {})).items()}
        self._allow_rules = [PermissionRuleSpec.model_validate(item) for item in carry.get("allow_rules", [])]
        self._next_message_seq = int(carry.get("next_message_seq", 1))
        # Completions since the last Continue-As-New (04 §7). Carried as it was, a task past the limit continued as new
        # again at once, for ever.
        self._completed_nodes = 0 if workflow.patched(STATE_BOUNDS) else int(carry.get("completed_nodes", 0))
        self._entity_versions = {str(key): int(value) for key, value in dict(carry.get("entity_versions", {})).items()}
        self._usage = Usage.model_validate(carry.get("usage") or {})
        self._budget_hold = bool(carry.get("budget_hold", False))
        self._unsent = bool(carry.get("unsent", False))
        self._node_profile = {str(k): dict(v) for k, v in dict(carry.get("node_profile", {})).items()}
        self._handovers = {str(k): str(v) for k, v in dict(carry.get("handovers", {})).items()}
        self._sops = {str(k): dict(v) for k, v in dict(carry.get("sops", {})).items()}
        self._reviews = {str(k): dict(v) for k, v in dict(carry.get("reviews", {})).items()}
        self._review_nodes = {str(k): dict(v) for k, v in dict(carry.get("review_nodes", {})).items()}
        self._node_rounds = {str(k): int(v) for k, v in dict(carry.get("node_rounds", {})).items()}
        self._member_sessions = {str(k): str(v) for k, v in dict(carry.get("member_sessions", {})).items()}
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
            "CompleteNodeResult": CompleteNodeResult,
            "UpdateTaskConfigResult": UpdateTaskConfigResult,
        }
        for command_id, raw in dict(carry.get("dedup", {})).items():
            item = dict(raw)
            result_type = result_types.get(str(item.get("type")))
            if result_type is not None:
                self._dedup[str(command_id)] = result_type.model_validate(item["data"])
        self._inbox = [InboxMessage.model_validate(item) for item in carry.get("inbox", [])]
        self._node_retry = {str(k): dict(v) for k, v in dict(carry.get("node_retry", {})).items()}
        self._wait_deadlines = {str(k): str(v) for k, v in dict(carry.get("wait_deadlines", {})).items()}
        self._archive = {"count": 0, "hash": "", "recent": [], **dict(carry.get("archive", {}))}
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

    def _remember(self, command_id: str, result: Any) -> None:
        """Record the answer to a command, so the same command gets the same answer (04 §4). The window keeps the 512 most
        recent: Temporal drops its own update ids at Continue-As-New, and a command repeated that much later is not a retry."""
        self._dedup[command_id] = result
        if workflow.patched(STATE_BOUNDS):
            while len(self._dedup) > MAX_DEDUP:
                del self._dedup[next(iter(self._dedup))]

    def _bound_state(self) -> None:
        """What a Continue-As-New carries must not grow with the life of the task: the pending approvals and a short tail
        of the decided ones, the follow-up bookkeeping of nodes that are not done, and the entity versions of what is
        still live."""
        if self._plan is None or not workflow.patched(STATE_BOUNDS):
            return
        decided = [key for key, item in self._approvals.items() if item.get("status") != "PENDING"]
        for key in decided[: max(0, len(decided) - MAX_DECIDED_APPROVALS)]:
            del self._approvals[key]
        live = {item.get("attempt_id") for item in self._attempts.values()}
        live |= {str(item.get("approval_id")) for item in self._approvals.values()}
        self._follow_ups = {
            node_id: item
            for node_id, item in self._follow_ups.items()
            if node_id in self._plan.nodes and self._plan.nodes[node_id].status not in {"COMPLETED", "SKIPPED"}
        }
        self._node_retry = {
            node_id: item
            for node_id, item in self._node_retry.items()
            if node_id in self._plan.nodes and self._plan.nodes[node_id].status != "COMPLETED"
        }
        self._wait_deadlines = {k: v for k, v in self._wait_deadlines.items() if k in self._plan.nodes}
        self._handovers = {k: v for k, v in self._handovers.items() if k in self._plan.nodes}
        self._node_profile = {k: v for k, v in self._node_profile.items() if k in self._plan.nodes}
        # A finished SOP is history; one that is going keeps what its open steps still need.
        self._sops = {
            node_id: entry
            for node_id, entry in self._sops.items()
            if node_id in self._plan.nodes and self._plan.nodes[node_id].status != "COMPLETED"
        }
        for entry in self._sops.values():
            needed = {dep for step in entry["open"] for dep in entry["deps"].get(step, [])}
            entry["outputs"] = {key: value for key, value in entry["outputs"].items() if key in needed}
        # A review that is waiting keeps what it needs; what is about a node that has left the plan is history.
        self._review_nodes = {k: v for k, v in self._review_nodes.items() if k in self._plan.nodes}
        self._node_rounds = {k: v for k, v in self._node_rounds.items() if k in self._plan.nodes or k in self._reviews}
        # A message and a finished attempt are announced once; only task, plan and live entities still get versions.
        self._entity_versions = {
            key: version
            for key, version in self._entity_versions.items()
            if key.split(":", 1)[0] in {"task", "plan"}
            or key.split(":", 1)[-1] in live
            # A node that is still in the plan keeps its own counter, or its next event could be older than the last one.
            or (key.startswith("node:") and key.split(":", 1)[1] in self._plan.nodes)
        }

    def _carry_input(self) -> TaskWorkflowInput:
        assert self._plan is not None
        self._bound_state()
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
            "last_attempt_id": self._last_attempt_id,
            "follow_ups": self._follow_ups,
            "node_retry": self._node_retry,
            "wait_deadlines": self._wait_deadlines,
            "archive": self._archive,
            "usage": self._usage.model_dump(mode="json", exclude_none=True),
            "budget_hold": self._budget_hold,
            "unsent": self._unsent,
            "node_profile": self._node_profile,
            "handovers": self._handovers,
            "sops": self._sops,
            "reviews": self._reviews,
            "review_nodes": self._review_nodes,
            "node_rounds": self._node_rounds,
            "member_sessions": self._member_sessions,
            "allow_rules": [rule.model_dump(mode="json") for rule in self._allow_rules],
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
            config=self._config,
            carry=carry,
        )

    def _should_continue_as_new(self) -> bool:
        return (
            self._updates >= MAX_UPDATES_BEFORE_CAN
            or self._completed_nodes >= MAX_COMPLETIONS_BEFORE_CAN
            or workflow.info().is_continue_as_new_suggested()
        )

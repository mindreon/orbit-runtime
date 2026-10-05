"""The plan of a TaskWorkflow: node state, readiness, and plan changes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime
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
    from orbit_contracts.v3.plan import AddNodeOp, PlanChangeAccepted, UpdateNodeOp

    from orbit_orch.plan_engine import (
        PlanNodeState,
        PlanPolicy,
        apply,
        compact,
        deterministic_id,
        initial_plan,
        plan_bytes,
    )
    from orbit_orch.workflow_common import (
        ARCHIVE_RECENT_TITLES,
        DEFAULT_MAX_CONCURRENCY,
        DEFAULT_MAX_DEPTH,
        EVENT_PAYLOADS,
        FOLLOW_UP_BOUNDED,
        FOLLOW_UP_GOAL_CHARS,
        FOLLOW_UP_REJECTION_VISIBLE,
        INBOX_CONSUME,
        KEEP_RECENT_COMPLETED,
        MAX_ARCHIVED_IDS,
        MAX_PLAN_CHAIN,
        NODE_KEEPS_AUTHOR,
        OPERATOR_HELD,
        PLAN_COMPACT_BYTES,
        PLAN_COMPACTION,
        PLAN_NESTING,
        PLAN_VISIBILITY,
        RETRY_POLICY,
        SCHEDULE_CONCURRENCY,
        SIGNAL_CLOSED_CHILD,
        SOP_EXPANSION,
        sha,
    )

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
                **(self._node_facts(node_id, state, updates) if workflow.patched(EVENT_PAYLOADS) else {}),
            })
        if workflow.patched(NODE_KEEPS_AUTHOR):
            # Every field is kept: a rebuild that forgot `created_by` made an agent's own node somebody else's the moment
            # it became READY, and the agent was refused when it changed it.
            changed = dataclasses.replace(state, **updates)
        else:
            changed = PlanNodeState(
                draft=state.draft,
                status=updates.get("status", state.status),
                frozen=updates.get("frozen", state.frozen),
                current_attempt_id=updates.get("current_attempt_id", state.current_attempt_id),
                attempt_count=updates.get("attempt_count", state.attempt_count),
            )
        self._plan = self._plan.with_updates(nodes={**self._plan.nodes, node_id: changed})
        if updates.get("status") in {"COMPLETED", "RETRY_PENDING"}:
            self._refresh_readiness()
        if new_status != state.status:
            self._sop_child_changed(node_id, new_status)

    def _node_facts(self, node_id: str, state: PlanNodeState, updates: dict[str, Any]) -> dict[str, Any]:
        """What the node is after the change, for the event that announces it (a projection builds the row from it)."""
        draft = state.draft
        current = updates.get("current_attempt_id", state.current_attempt_id)
        return {
            "node_type": draft.type,
            "title": draft.title,
            "workspace_access": draft.workspace_access or "none",
            "owner_profile": draft.owner_profile or self._profile,
            "depends_on": list(draft.depends_on),
            "frozen": bool(updates.get("frozen", state.frozen)),
            "attempt_count": int(updates.get("attempt_count", state.attempt_count)),
            **({"current_attempt_id": current} if current else {}),
            **({"parent_node_id": draft.parent_node_id} if draft.parent_node_id else {}),
            **self._sop_facts(node_id),
        }

    def _commit_facts(self, command: PlanChangeCommand) -> dict[str, Any]:
        """Who committed this plan version and from which one, under the contract's names (`change_command_id`; the
        workflow keeps sending `command_id` for consumers from before it)."""
        if not workflow.patched(EVENT_PAYLOADS):
            return {}
        assert self._plan is not None
        return {
            "change_command_id": command.command_id,
            "parent_version": self._plan.version - 1,
            "actor": command.actor.model_dump(mode="json", exclude_none=True),
        }

    def _all_nodes_completed(self) -> bool:
        return bool(self._plan) and all(state.status in {"COMPLETED", "SKIPPED"} for state in self._plan.nodes.values())

    def _backoff_remaining(self, node_id: str) -> float:
        """Seconds until the node may try again; zero when it may now."""
        due = (self._node_retry.get(node_id) or {}).get("retry_at")
        if not due:
            return 0.0
        return max(0.0, (datetime.fromisoformat(str(due)) - workflow.now()).total_seconds())

    # What the scheduler knows how to act on. A node of another type (a team stage) is never started, so it must not
    # keep the main loop from waiting.
    _SCHEDULED_TYPES = frozenset({"agent_turn", "sop_stage", "approval", "wait", "checkpoint"})

    def _max_concurrency(self) -> int:
        return self._policy.max_concurrency or DEFAULT_MAX_CONCURRENCY

    def _running_count(self) -> int:
        """Attempts that are running now. One parked on an approval or a question is waiting for a person and does not hold a
        slot; one that is being cancelled still runs until its activity has stopped."""
        return sum(1 for item in self._attempts.values() if item.get("status") == "RUNNING")

    def _ready_order(self) -> list[str]:
        """The nodes the scheduler can act on now, in plan order (the order the nodes were created in): ready, past their
        backoff, with their dependencies done, and for an agent node, with a slot free and the workspace's write lease not
        taken by another node."""
        if self._plan is None:
            return []
        slots = self._max_concurrency() - self._running_count()
        write_busy = any(item.get("workspace_access") == "write" for item in self._attempts.values())
        order: list[str] = []
        for node_id, state in self._plan.nodes.items():
            if (
                state.status not in {"READY", "RETRY_PENDING"}
                or node_id in self._attempts
                or state.draft.type not in self._SCHEDULED_TYPES
                or self._backoff_remaining(node_id) > 0
                or not self._dependencies_done(node_id)
            ):
                continue
            if state.draft.type == "sop_stage" and workflow.patched(SOP_EXPANSION):
                order.append(node_id)  # it is compiled into the plan, not run: it takes no slot
                continue
            if state.draft.type in {"agent_turn", "sop_stage"}:
                if slots <= 0 or (state.draft.workspace_access == "write" and write_busy):
                    continue
                slots -= 1
                write_busy = write_busy or state.draft.workspace_access == "write"
            order.append(node_id)
        return order

    def _has_ready_node(self) -> bool:
        if self._status in OPERATOR_HELD or self._status == "CANCELLED":
            return False  # nothing is scheduled now; counting these nodes would spin the main loop
        if workflow.patched(SCHEDULE_CONCURRENCY):
            return bool(self._ready_order())
        return bool(self._plan) and any(
            state.status in {"READY", "RETRY_PENDING"}
            and node_id not in self._attempts
            and self._backoff_remaining(node_id) <= 0
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

    def _plan_policy(self, command: PlanChangeCommand | None = None) -> PlanPolicy:
        assert self._plan is not None
        limits: dict[str, Any] = {}
        if workflow.patched(PLAN_NESTING):
            team = self._config.team
            limits = {"max_depth": team.max_depth if team else DEFAULT_MAX_DEPTH, "max_chain": MAX_PLAN_CHAIN}
        else:
            limits = {"max_depth": None, "max_chain": 8, "legacy_depth": True}
        if command is not None and command.actor.kind == "agent" and workflow.patched(PLAN_VISIBILITY):
            limits["visible_node_ids"] = self._visible_to(command.actor)
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
            archived_node_ids=frozenset(self._archive.get("ids", [])),
            **limits,
        )

    def _is_leader_node(self, node_id: str) -> bool:
        """A node that plans for the task and so sees all of it (05 §6): the exploration node, a follow-up, and, when the
        task is a team's, a node that runs as the team's leader."""
        assert self._plan is not None
        if node_id == deterministic_id(f"{self._task_id}:exploration", "n") or node_id in self._follow_ups:
            return True
        team = self._config.team
        if team is None:
            return False
        leader = next((member.expert for member in team.members if member.role == team.leader), None)
        return self._attempt_profile(self._plan.nodes[node_id].draft.owner_profile) == leader

    def _visible_to(self, actor: Actor | None) -> frozenset[str] | None:
        """The nodes an actor may see and name in a plan change (05 §6), or None for all of them. A person or the system
        sees the whole plan, and so does an agent whose node plans for the task. Any other agent sees its own node, the nodes
        its node's attempts created, and the direct dependencies of those."""
        if actor is None or actor.kind != "agent" or self._plan is None:
            return None
        own = next((node_id for node_id, item in self._attempts.items() if item.get("attempt_id") == actor.attempt_id), None)
        if own is None or own not in self._plan.nodes:
            return frozenset()  # an attempt that is not running any more sees nothing
        if self._is_leader_node(own):
            return None
        attempts = {
            deterministic_id(f"{self._task_id}:{own}:{number}", "att")
            for number in range(1, self._plan.nodes[own].attempt_count + 1)
        }
        seen = {own} | {node_id for node_id, state in self._plan.nodes.items() if state.created_by in attempts}
        return frozenset(seen | {source for source, target in self._plan.edges if target in seen})

    def _start_follow_up(self, message: InboxMessage) -> None:
        """A message that comes when every node is done becomes a node of its own. Its attempt carries on the agent session
        of the one that ran last, so the conversation goes on instead of starting over."""
        assert self._plan is not None
        first_line = next((line.strip() for line in message.text.splitlines() if line.strip()), "Follow-up")
        # The node keeps a bounded goal, so a long conversation does not fill the plan; the attempt is given the whole
        # message as its input message when it is longer than that (a short one is the goal and is not said twice).
        bounded = workflow.patched(FOLLOW_UP_BOUNDED) and workflow.patched(INBOX_CONSUME)
        goal = message.text[:FOLLOW_UP_GOAL_CHARS] if bounded else message.text
        command = PlanChangeCommand(
            command_id=hashlib.sha256(f"follow-up:{self._task_id}:{message.message_seq}".encode()).hexdigest(),
            task_id=self._task_id,
            base_plan_version=self._plan.version,
            actor=Actor(kind="system", id="task-workflow"),
            ops=[AddNodeOp(node=AgentTurnNode(node_id="tmp:1", title=first_line[:200], spec=AgentTurnSpec(goal=goal)))],
            reason="follow-up message",
        )
        if bounded and self._compact_plan():
            command = command.model_copy(update={"base_plan_version": self._plan.version})
        outcome = apply(self._plan, command, self._plan_policy())
        too_big = not isinstance(outcome.result, PlanChangeAccepted) and outcome.result.code == "TOO_MANY_OPS"
        if bounded and too_big and self._compact_plan(force=True, keep_recent=0):
            # Still too big: whatever is finished goes now, whatever the threshold says and the newest ones too.
            command = command.model_copy(update={"base_plan_version": self._plan.version})
            outcome = apply(self._plan, command, self._plan_policy())
        if not isinstance(outcome.result, PlanChangeAccepted):
            self._emit("plan.change_rejected", outcome.result.model_dump(mode="json"))
            if workflow.patched(SIGNAL_CLOSED_CHILD):
                # It was reported (below) and is not tried again by itself: the person sends it again once they have looked.
                self._inbox = [item for item in self._inbox if item.message_seq != message.message_seq]
            if workflow.patched(FOLLOW_UP_REJECTION_VISIBLE):
                # Never silent: the person sent a message and has to be told it was not taken (and the task asks for a review).
                self._set_status(
                    "PAUSED_NEEDS_REVIEW",
                    f"message {message.message_seq} could not be added to the plan: {outcome.result.code}: {outcome.result.detail}",
                )
            return
        if bounded and len(message.text) <= FOLLOW_UP_GOAL_CHARS:
            self._inbox = [item for item in self._inbox if item.message_seq != message.message_seq]
        self._plan = outcome.plan
        self._refresh_readiness()
        self._emit("plan.version_committed", {
            "plan_version": self._plan.version,
            "hash": self._plan.hash,
            "command_id": command.command_id,
            **self._commit_facts(command),
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
        outcome = apply(self._plan, command, self._plan_policy(command))
        self._remember(command.command_id, outcome.result)
        self._updates += 1
        if isinstance(outcome.result, PlanChangeAccepted):
            self._plan = outcome.plan
            self._refresh_readiness()
            self._emit("plan.version_committed", {
                "plan_version": self._plan.version,
                "hash": self._plan.hash,
                "command_id": command.command_id,
                **self._commit_facts(command),
            })
            self._unblock_changed(command)
            self._compact_plan()
        else:
            self._emit("plan.change_rejected", outcome.result.model_dump(mode="json"))
        return outcome.result

    def _unblock_nodes(self, node_ids: list[str], reason: str) -> None:
        """A person decided about nodes that retries gave up on (`task_retry`): each is READY again, with its failed
        attempts forgotten. It keeps the session the last attempt had, and why that one failed."""
        for node_id in node_ids:
            entry = self._node_retry.get(node_id)
            if entry is None or not entry.get("blocked") or node_id not in (self._plan.nodes if self._plan else {}):
                continue
            self._node_retry[node_id] = {**entry, "failed": 0, "blocked": False, "blocked_for": "", "retry_at": None}
            if self._require_node(node_id).status == "BLOCKED":
                self._set_node_status(node_id, "READY", reason=reason)

    def _unblock_changed(self, command: PlanChangeCommand) -> None:
        """A person's plan change on a node that retries gave up on puts it back in play."""
        if command.actor.kind == "agent" or not workflow.patched(RETRY_POLICY):
            return
        self._unblock_nodes(
            [op.node_id for op in command.ops if isinstance(op, UpdateNodeOp)], "the plan change on this node"
        )

    def _compact_plan(self, force: bool = False, keep_recent: int = KEEP_RECENT_COMPLETED) -> bool:
        """Move completed, frozen nodes that no unfinished node depends on out of the live plan, keeping a bounded archive
        summary (how many, a hash chain, the latest titles). It is a system action, so the plan version goes up and
        `plan.version_committed` says so; the control projection already holds the rows of the nodes it drops. It runs
        when the plan has grown to half of the size a plan change may reach, and at Continue-As-New (`force`). The newest
        few completed nodes stay, so the plan still shows what was just done. Returns whether anything was archived."""
        if self._plan is None or not workflow.patched(PLAN_COMPACTION):
            return False
        if not force and plan_bytes(self._plan) <= PLAN_COMPACT_BYTES:
            return False
        outcome = compact(self._plan, held=self._attempts, keep_recent=keep_recent)
        if outcome is None:
            return False
        previous = self._plan.version
        self._plan = outcome.plan
        self._archive = {
            "count": int(self._archive.get("count", 0)) + len(outcome.removed),
            # The ids stay, bounded, so a later plan change that names one is a dependency already met, not an unknown node.
            "ids": [*self._archive.get("ids", []), *outcome.removed][-MAX_ARCHIVED_IDS:],
            "hash": sha(json.dumps([self._archive.get("hash", ""), outcome.removed, outcome.titles])),
            "recent": [*self._archive.get("recent", []), *(title[:80] for title in outcome.titles)][
                -ARCHIVE_RECENT_TITLES:
            ],
        }
        for node_id in outcome.removed:
            self._follow_ups.pop(node_id, None)
            self._node_retry.pop(node_id, None)
            self._wait_deadlines.pop(node_id, None)
        self._emit("plan.version_committed", {
            "plan_version": self._plan.version,
            "parent_version": previous,
            "hash": self._plan.hash,
            "command_id": hashlib.sha256(f"compact:{self._task_id}:{self._plan.version}".encode()).hexdigest(),
            "actor": {"kind": "system", "id": "task-workflow"},
            "reason": "compaction",
            **({"change_command_id": hashlib.sha256(f"compact:{self._task_id}:{self._plan.version}".encode()).hexdigest()}
               if workflow.patched(EVENT_PAYLOADS) else {}),
            "archived": {"count": self._archive["count"], "hash": self._archive["hash"], "added": len(outcome.removed)},
        })
        return True

    @workflow.query(name="getPlan")
    def get_plan(self, actor: Actor | None = None) -> PlanView:
        """The plan as `actor` may see it: all of it for a person, the system, a caller that names nobody and an agent whose
        node plans for the task; an agent that works on a part of it sees that part. The version and hash are the whole
        plan's, so what an agent proposes against them is judged against the real plan."""
        if self._plan is None:
            raise ApplicationError("plan is not initialized")
        from orbit_contracts.v3.views import NodeView, PlanArchive, PlanEdge

        visible = self._visible_to(actor)
        shown = {
            node_id: state for node_id, state in self._plan.nodes.items() if visible is None or node_id in visible
        }
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
                    parent_node_id=state.draft.parent_node_id,
                    sop_step=self._sop_facts(node_id).get("sop_step"),
                )
                for node_id, state in shown.items()
            ],
            edges=[
                PlanEdge.model_validate({"from": source, "to": target})
                for source, target in self._plan.edges
                if source in shown and target in shown
            ],
            archived=PlanArchive(
                count=int(self._archive["count"]),
                hash=str(self._archive["hash"]),
                recent_titles=list(self._archive["recent"]),
            )
            if self._archive.get("count")
            else None,
        )

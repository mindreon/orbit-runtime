"""The leader's review of what the tasks it created produced (05 §7).

A leader plans by creating tasks (`TaskCreate`) and then has nothing more to say until a person writes again. For a team that
is not enough: the leader should see what its members brought back, and either plan the next round or answer. So when an
attempt of a leader node (the exploration node, a follow-up, a node the team's leader owns, or a review itself) completes and
created tasks, the workflow remembers them (`_reviews`). When every one of them is done (completed or skipped; one that is
blocked keeps the review waiting until a person resolves it) it adds, as the system actor and with deterministic ids, a node
"领队复盘" that depends on them and runs as the leader's expert, carrying on the leader's own session. Its prompt holds each
task's title, how it ended, what it reported (cut) and the names of its files. The review can create the next round, which gets
its own review when it is done, or finish with its answer.

The chain is bounded: `Policy.max_review_rounds` (default 5; 0 turns the reviews off). A round is counted along the chain of
who created what: a review's round is one more than the node whose attempt created the tasks it follows, and the tasks
inherit it. When the chain reaches the limit no further review is added; the workflow says so with `plan.review_limit_reached`
and asks for a review of its own (`PAUSED_NEEDS_REVIEW`), so a plan that keeps growing is never left to loop silently.
"""

from __future__ import annotations

import hashlib
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import Actor, AttemptFinishedSignal, InboxMessage, PlanChangeCommand
    from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec
    from orbit_contracts.v3.plan import AddNodeOp, PlanChangeAccepted

    from orbit_orch.plan_engine import apply, deterministic_id
    from orbit_orch.workflow_common import (
        DEFAULT_MAX_REVIEW_ROUNDS,
        FOLLOW_UP_GOAL_CHARS,
        HANDOVER_CHARS,
        LEADER_REVIEW,
        MAX_REVIEW_CHILDREN,
        MENTION_FOLLOW_UPS,
        MENTION_READ_ONLY,
        OPERATOR_HELD,
        REVIEW_PROMPT_CHARS,
        TEAM_MESSAGES,
        closed,
    )

from orbit_orch.task_sop import MAX_HANDOVER_FILES, TaskSop

REVIEW_TITLE = "领队复盘"
# A child that is done: the review no longer waits for it.
_DONE = frozenset({"COMPLETED", "SKIPPED", "CANCELLED"})


def review_goal(round_no: int, max_rounds: int, children: list[dict[str, Any]], more: int) -> str:
    """The prompt of a review: what each task the leader created reported, within `REVIEW_PROMPT_CHARS` in all. `children` are
    the tasks in order, each with `title`, `status`, `summary` and `artifacts`; `more` counts the ones left out of the list."""
    share = max(300, min(HANDOVER_CHARS, REVIEW_PROMPT_CHARS // max(1, len(children))))
    lines = [
        f"{REVIEW_TITLE}（第 {round_no} 轮）: the {len(children) + more} task(s) you created have finished. What each one reported:"
    ]
    for index, child in enumerate(children, 1):
        lines.append(f"\n{index}. {child['title']} [{child['status']}]")
        lines.append(f"   {str(child['summary']).strip()[:share] or '(it reported nothing)'}")
        files = [str(name) for name in child["artifacts"]][:MAX_HANDOVER_FILES]
        if files:
            lines.append("   Files: " + ", ".join(files))
    if more:
        lines.append(f"\n(and {more} more not listed)")
    lines.append(
        "\nWeigh these results against the goal of the task. If something is missing or wrong, create the next tasks with "
        f"TaskCreate (they will be reviewed in turn, up to {max_rounds} rounds in all); if the work is done, answer with the "
        "final result for the user."
    )
    return "\n".join(lines)


class TaskReview(TaskSop):
    def _max_review_rounds(self) -> int:
        configured = self._policy.max_review_rounds
        return DEFAULT_MAX_REVIEW_ROUNDS if configured is None else configured

    def _note_leader_done(self, node_id: str, attempt: dict[str, Any]) -> None:
        """The attempt of `node_id` completed it. If the node is a leader's and the attempt created tasks, a review of them is
        due once they are done."""
        if not workflow.patched(LEADER_REVIEW) or self._plan is None or self._max_review_rounds() == 0:
            return
        state = self._plan.nodes.get(node_id)
        if state is None or state.status != "COMPLETED" or node_id in self._reviews or not self._is_leader_node(node_id):
            return
        # Every attempt of the node counts: a retry may have created tasks before the one that completed it.
        made = {deterministic_id(f"{self._task_id}:{node_id}:{number}", "att") for number in range(1, state.attempt_count + 1)}
        children = [
            child for child, item in self._plan.nodes.items() if child != node_id and item.created_by in made
        ]
        if not children:
            return
        round_no = self._node_rounds.get(node_id, 0) + 1
        kept = children[:MAX_REVIEW_CHILDREN]
        self._reviews[node_id] = {
            "attempt_id": str(attempt["attempt_id"]),
            "profile": str(attempt.get("profile") or self._profile),
            "round": round_no,
            "children": kept,
            "more": len(children) - len(kept),
            "titles": {child: self._plan.nodes[child].draft.title[:80] for child in kept},
            "results": {},
        }
        for child in kept:
            self._node_rounds[child] = round_no

    def _record_review_output(self, signal: AttemptFinishedSignal) -> None:
        """A task a leader created ended completed: what it said and the files it left are what the review is told of it."""
        if signal.result is None:
            return
        for entry in self._reviews.values():
            if signal.node_id in entry["children"]:
                entry["results"][signal.node_id] = {
                    "summary": signal.result.handover_summary[:HANDOVER_CHARS],
                    "artifacts": [
                        str(item.get("name", "")) for item in signal.result.manifest_entries if item.get("name")
                    ][:MAX_HANDOVER_FILES],
                }

    def _open_reviews(self) -> None:
        """Open the reviews that are due: every task of the entry is done. Called by the main loop before it schedules, so a
        task never completes without the review it is owed being in the plan first."""
        if not self._reviews or self._plan is None or self._status in OPERATOR_HELD or closed(self._status):
            return
        for leader_node, entry in list(self._reviews.items()):
            if any(
                child in self._plan.nodes and self._plan.nodes[child].status not in _DONE for child in entry["children"]
            ):
                continue  # a task is still going, or is blocked and waits for a person to resolve it
            del self._reviews[leader_node]
            live = [child for child in entry["children"] if child in self._plan.nodes]
            if not live and not entry["results"]:
                continue  # every task was removed: there is nothing to review
            max_rounds = self._max_review_rounds()
            if entry["round"] > max_rounds:
                self._report_review_limit(leader_node, entry, max_rounds)
                continue
            self._add_review(leader_node, entry, live, max_rounds)

    def _report_review_limit(self, leader_node: str, entry: dict[str, Any], max_rounds: int) -> None:
        self._emit("plan.review_limit_reached", {
            "node_id": leader_node, "round": int(entry["round"]), "max_rounds": max_rounds,
            "children": len(entry["children"]) + int(entry.get("more", 0)),
        })
        self._set_status(
            "PAUSED_NEEDS_REVIEW",
            f"the leader's reviews reached the limit of {max_rounds} rounds: the tasks created in the last round are done "
            "and were not reviewed",
        )

    def _add_review(self, leader_node: str, entry: dict[str, Any], live: list[str], max_rounds: int) -> None:
        assert self._plan is not None
        children: list[dict[str, Any]] = []
        for child in entry["children"]:
            state = self._plan.nodes.get(child)
            result = entry["results"].get(child) or {}
            summary = str(result.get("summary") or "")
            status = "skipped" if state is not None and state.status == "SKIPPED" else "completed"
            children.append({
                "title": entry["titles"].get(child) or child, "status": status, "summary": summary,
                "artifacts": result.get("artifacts", []),
            })
        goal = review_goal(int(entry["round"]), max_rounds, children, int(entry.get("more", 0)))
        command = PlanChangeCommand(
            command_id=hashlib.sha256(f"review:{self._task_id}:{entry['attempt_id']}".encode()).hexdigest(),
            task_id=self._task_id,
            base_plan_version=self._plan.version,
            actor=Actor(kind="system", id="task-workflow"),
            ops=[AddNodeOp(node=AgentTurnNode(
                node_id="tmp:1", title=REVIEW_TITLE, depends_on=live, owner_profile=entry["profile"],
                spec=AgentTurnSpec(goal=goal),
            ))],
            reason=f"leader review, round {entry['round']}",
        )
        outcome = apply(self._plan, command, self._plan_policy())
        too_big = not isinstance(outcome.result, PlanChangeAccepted) and outcome.result.code == "TOO_MANY_OPS"
        if too_big and self._compact_plan(force=True, keep_recent=0):
            # The plan is too big for one more node: what is finished goes, as it does for a follow-up. The tasks the review
            # follows are finished too, so they go with it and it depends on what is left of them.
            node = command.ops[0].node  # type: ignore[union-attr]
            command = command.model_copy(update={
                "base_plan_version": self._plan.version,
                "ops": [AddNodeOp(node=node.model_copy(update={"depends_on": [c for c in live if c in self._plan.nodes]}))],
            })
            outcome = apply(self._plan, command, self._plan_policy())
        if not isinstance(outcome.result, PlanChangeAccepted):
            # Never silent: the leader was owed a review and cannot have it.
            self._emit("plan.change_rejected", outcome.result.model_dump(mode="json"))
            self._set_status(
                "PAUSED_NEEDS_REVIEW",
                f"the review of the tasks of {leader_node} could not be added to the plan: "
                f"{outcome.result.code}: {outcome.result.detail}",
            )
            return
        self._plan = outcome.plan
        review_node = outcome.result.id_map["tmp:1"]
        # Known before the node's first event, which says its round.
        self._review_nodes[review_node] = {"from": entry["attempt_id"], "round": int(entry["round"])}
        self._node_rounds[review_node] = int(entry["round"])
        self._refresh_readiness()
        self._emit("plan.version_committed", {
            "plan_version": self._plan.version,
            "hash": self._plan.hash,
            "command_id": command.command_id,
            **self._commit_facts(command),
            "reason": "leader review",
        })

    # ---- the team's conversation at plan level (07 §5, §6b) -------------------------------------------------------------

    def _team_message(
        self, node_id: str, attempt_id: str | None, kind: str, from_role: str, to_roles: list[str], text: str, round_no: int = 0
    ) -> None:
        """A `team.message` of the plan's members: no stage, so no sequence number; the entity is the attempt, else the node."""
        team = self._config.team
        label = next((m.label for m in (team.members if team else []) if m.role == from_role and m.label), "")
        self._emit("team.message", {
            "node_id": node_id,
            **({"attempt_id": attempt_id} if attempt_id else {}),
            "role": from_role, "from_role": from_role, "to_roles": to_roles, "text": text[:2000], "kind": kind, "round": round_no,
            "hop": 0, **({"label": label, "from_label": label} if label else {}),
        })

    def _emit_team_assigns(self, command: PlanChangeCommand, id_map: dict[str, str]) -> None:
        """The leader's TaskCreate gave a node to a member: said in the group, from the leader to the member."""
        team = self._config.team
        if team is None or command.actor.kind != "agent" or not workflow.patched(TEAM_MESSAGES) or self._plan is None:
            return
        for op in command.ops:
            if not isinstance(op, AddNodeOp):
                continue
            node_id = id_map.get(op.node.node_id, op.node.node_id)
            state = self._plan.nodes.get(node_id)
            owner = self._owner_of(state.draft) if state is not None else None
            if owner is None or owner[0] == team.leader:
                continue
            goal = str(getattr(op.node.spec, "goal", "") or op.node.title)
            self._team_message(
                node_id, command.actor.attempt_id, "assign", team.leader, [owner[0]], f"{op.node.title}\n{goal}" if goal != op.node.title else goal
            )

    def _emit_team_reply(self, signal: AttemptFinishedSignal) -> None:
        """A member's node ended completed: its final text goes back to the group, to the leader (to the user for a node the user
        @-mentioned it into). A leader's review says its answer to everyone."""
        team = self._config.team
        if team is None or signal.result is None or not workflow.patched(TEAM_MESSAGES) or self._plan is None:
            return
        state = self._plan.nodes.get(signal.node_id)
        if state is None:
            return
        text = signal.result.handover_summary
        if signal.node_id in self._review_nodes:
            self._team_message(signal.node_id, signal.attempt_id, "review", team.leader, [], text, int(self._review_nodes[signal.node_id]["round"]))
            return
        owner = self._owner_of(state.draft)
        if owner is None or owner[0] == team.leader:
            return
        to = ["user"] if self._follow_ups.get(signal.node_id, {}).get("mention") else [team.leader]
        self._team_message(signal.node_id, signal.attempt_id, "reply", owner[0], to, text)

    def _remember_member_session(self, node_id: str, attempt_id: str) -> None:
        """The attempt a member last ran a node in: what a later message to it carries on."""
        team = self._config.team
        state = self._plan.nodes.get(node_id) if self._plan else None
        if team is None or state is None:
            return
        owner = self._owner_of(state.draft)
        if owner is not None and owner[0] != team.leader:
            self._member_sessions[owner[0]] = attempt_id

    def _mention_follow_ups(self, message: InboxMessage) -> bool:
        """A user message that @-mentions members of the task's team becomes a node of its own for each of them, owned by the member
        and carrying on the member's last session (a fresh one when it has none), instead of going to the leader. Returns whether
        the message was taken. A message that mentions only the leader, or nobody, is not."""
        team = self._config.team
        if team is None or not message.mentions or not workflow.patched(MENTION_FOLLOW_UPS) or self._plan is None:
            return False
        members = [m for m in team.members if m.role in message.mentions and m.role != team.leader]
        if not members:
            return False
        first_line = next((line.strip() for line in message.text.splitlines() if line.strip()), "Follow-up")
        goal = message.text[:4 * FOLLOW_UP_GOAL_CHARS]
        command = PlanChangeCommand(
            command_id=hashlib.sha256(f"mention:{self._task_id}:{message.message_seq}".encode()).hexdigest(),
            task_id=self._task_id,
            base_plan_version=self._plan.version,
            actor=Actor(kind="system", id="task-workflow"),
            ops=[
                AddNodeOp(node=AgentTurnNode(
                    node_id=f"tmp:{index}", title=f"@{member.label or member.role}: {first_line}"[:200],
                    owner_profile=member.expert, spec=AgentTurnSpec(goal=goal),
                    # Read-only: each works in its own copy of the head snapshot and its files come back as artifacts, so
                    # members mentioned together run in parallel instead of queueing for the single writer.
                    **({"workspace_access": "read"} if workflow.patched(MENTION_READ_ONLY) else {}),
                ))
                for index, member in enumerate(members, 1)
            ],
            reason="message to a member",
        )
        outcome = apply(self._plan, command, self._plan_policy())
        self._inbox = [item for item in self._inbox if item.message_seq != message.message_seq]
        if not isinstance(outcome.result, PlanChangeAccepted):
            self._emit("plan.change_rejected", outcome.result.model_dump(mode="json"))
            self._set_status(
                "PAUSED_NEEDS_REVIEW",
                f"message {message.message_seq} could not be given to {', '.join(m.role for m in members)}: "
                f"{outcome.result.code}: {outcome.result.detail}",
            )
            return True
        self._plan = outcome.plan
        self._refresh_readiness()
        self._emit("plan.version_committed", {
            "plan_version": self._plan.version, "hash": self._plan.hash, "command_id": command.command_id,
            **self._commit_facts(command), "reason": "message to a member",
        })
        first = ""
        for index, member in enumerate(members, 1):
            node_id = outcome.result.id_map[f"tmp:{index}"]
            first = first or node_id
            self._follow_ups[node_id] = {
                "from": self._member_sessions.get(member.role, ""), "seq": message.message_seq, "mention": member.role,
            }
        if workflow.patched(TEAM_MESSAGES):
            self._team_message(first, None, "user", "user", [m.role for m in members], message.text)
        return True

"""A `sop_stage` node compiled into the plan (06 §2, 05 §5).

When a `sop_stage` node becomes schedulable the workflow reads the SOP definition (the `load_sop` activity) and replaces
the node by a subgraph, as the system actor and with deterministic ids: one `agent_turn` node per step, nested under the
node (`parent_node_id`), running as the step's executor with the step's own retry budget; an approval node before or after
a step with `human_approval`; edges from `depends_on`; and a completion contract whose `sop_verifier` verification has an
independent verifier agent judge the step (`verify_sop_step`). Whatever depended on the SOP node now depends on the
subgraph's sinks. The node itself stays in the plan as the group: RUNNING while its nodes are open, COMPLETED when every one
is, BLOCKED when one is. Each step then runs the whole task path (sandbox, configuration, policy, approvals, streaming and
budget), which the old `AttemptWorkflow._run_sop` path did not.

`_sops` is the workflow's own record of the expansion (it is carried across a Continue-As-New): which nodes belong to which
step, which of them are open, how the steps depend on each other and what the finished ones handed over. Completion of the
SOP node follows this record and not the plan, so compaction of finished nodes cannot hide a step from it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import timedelta
from typing import Any

from pydantic import ValidationError
from temporalio import workflow
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import (
        Actor,
        AttemptFinishedSignal,
        CompletionProposal,
        PlanChangeCommand,
    )
    from orbit_contracts.v3.messages import ApprovalSubject
    from orbit_contracts.v3.nodes import (
        AgentTurnNode,
        AgentTurnSpec,
        ApprovalNode,
        ApprovalSpec,
        CompletionContract,
        RetryPolicy,
        Verification,
    )
    from orbit_contracts.v3.plan import AddEdgeOp, AddNodeOp, PlanChangeAccepted, RemoveEdgeOp
    from orbit_contracts.v3.sop import ResolvedStep, SopStep, resolve_steps

    from orbit_orch.plan_engine import apply, deterministic_id
    from orbit_orch.workflow_common import (
        HANDOVER_CHARS,
        HEARTBEAT,
        MAX_PLAN_CHAIN,
        NODE_APPROVALS,
        OPERATOR_HELD,
        RETRY,
        SOP_VERIFIER,
        SOP_VERIFIER_TIMEOUT_S,
        closed,
        reasons,
        sha,
    )

from orbit_orch.task_retry import TaskRetry

# How many file names of a finished step are put in the prompt of the steps after it.
MAX_HANDOVER_FILES = 20
MAX_HANDOVER_CHARS = 12000


class TaskSop(TaskRetry):
    # ---- expansion -----------------------------------------------------------------------------------------------

    async def _expand_sop(self, node_id: str) -> None:
        """Replace the ready `sop_stage` node by the subgraph of its SOP. A SOP that cannot be read or compiled does not
        fail the task: the node is retried (a store that is down) or blocked and the task asks for a review (an unknown or
        invalid SOP, a plan that cannot hold it)."""
        assert self._plan is not None
        ref = str(self._plan.nodes[node_id].draft.spec.sop)  # type: ignore[union-attr]
        try:
            loaded = await self._run_short_activity("load_sop", {"tenant_id": self._tenant_id, "sop_ref": ref})
        except ActivityError as exc:
            self._sop_unusable(node_id, f"SOP {ref} could not be read: {exc.cause or exc}", retryable=True)
            return
        state = self._plan.nodes.get(node_id)
        if (
            state is None
            or state.status not in {"READY", "RETRY_PENDING"}
            or node_id in self._sops
            or self._status in OPERATOR_HELD
            or closed(self._status)
        ):
            return  # something happened while the definition was read: the node is looked at again when it is ready
        if not loaded.get("found"):
            self._sop_unusable(node_id, f"unknown SOP {ref}", retryable=False)
            return
        try:
            steps = resolve_steps([SopStep.model_validate(item) for item in loaded.get("steps", [])])
        except (ValueError, ValidationError) as exc:
            self._sop_unusable(node_id, f"SOP {ref} is not a valid definition: {exc}", retryable=False)
            return
        name = str(loaded.get("name") or ref.rpartition("@")[0])
        self._compact_plan()  # a plan that has grown makes room for the steps first
        built = self._sop_command(node_id, ref, name, steps)
        outcome = apply(self._plan, built[0], built[1])
        if (
            not isinstance(outcome.result, PlanChangeAccepted)
            and outcome.result.code == "TOO_MANY_OPS"
            and self._compact_plan(force=True, keep_recent=0)
        ):
            built = self._sop_command(node_id, ref, name, steps)
            outcome = apply(self._plan, built[0], built[1])
        command, _, parts, agents = built
        if not isinstance(outcome.result, PlanChangeAccepted):
            self._emit("plan.change_rejected", outcome.result.model_dump(mode="json"))
            self._sop_unusable(
                node_id, f"SOP {ref} cannot be added to the plan: {outcome.result.code}: {outcome.result.detail}", retryable=False
            )
            return
        id_map = outcome.result.id_map
        title = self._plan.nodes[node_id].draft.title
        entry: dict[str, Any] = {
            "ref": ref,
            "name": name,
            "title": title,
            "total": len(steps),
            "nodes": {id_map[tmp]: part for tmp, part in parts.items()},
            "open": [id_map[tmp] for tmp in parts],
            "agent": {step.id: id_map[agents[step.id]] for step in steps},
            "deps": {id_map[agents[step.id]]: [id_map[agents[dep]] for dep in step.depends_on] for step in steps},
            "outputs": {},
        }
        self._sops[node_id] = entry
        self._plan = outcome.plan
        self._emit("plan.version_committed", {
            "plan_version": self._plan.version,
            "hash": self._plan.hash,
            "command_id": command.command_id,
            **self._commit_facts(command),
            "reason": "sop expansion",
            "sop_node_id": node_id,
        })
        self._refresh_readiness()
        self._set_node_status(node_id, "RUNNING", reason=f"expanded into {len(steps)} steps")

    def _sop_unusable(self, node_id: str, message: str, *, retryable: bool) -> None:
        self._retry_or_block(node_id, None, failure_class="policy" if not retryable else "transient", retryable=retryable, message=message)

    def _sop_command(
        self, node_id: str, ref: str, name: str, steps: list[ResolvedStep]
    ) -> tuple[PlanChangeCommand, Any, dict[str, dict[str, Any]], dict[str, str]]:
        """The plan change that compiles the SOP: its command, the policy it is judged by, the part each new node is (by
        its temporary id) and the temporary id of each step's agent node. Pure: nothing here changes the workflow."""
        assert self._plan is not None
        title = self._plan.nodes[node_id].draft.title
        total = len(steps)
        ops: list[Any] = []
        parts: dict[str, dict[str, Any]] = {}
        agents: dict[str, str] = {}
        tails: dict[str, str] = {}  # step id -> the node whatever depends on the step waits for
        counter = 0

        def tmp() -> str:
            nonlocal counter
            counter += 1
            return f"tmp:{counter}"

        for step in steps:
            waits = [tails[dependency] for dependency in step.depends_on]
            label = f"{title} {step.index}/{total}: {step.subject}"
            if step.human_approval == "before":
                gate = tmp()
                ops.append(self._approval_op(gate, node_id, f"Approve the start of {label}", waits, f"Start step {step.index}/{total} ({step.subject}) of {name}?"))
                parts[gate] = {"role": "approval_before", "step_id": step.id, "index": step.index, "subject": step.subject}
                waits = [gate]
            agent = tmp()
            agents[step.id] = agent
            ops.append(AddNodeOp(node=self._step_node(agent, node_id, ref, name, step, label, total, waits)))
            parts[agent] = {"role": "step", "step_id": step.id, "index": step.index, "subject": step.subject}
            tails[step.id] = agent
            if step.human_approval == "after":
                gate = tmp()
                ops.append(self._approval_op(gate, node_id, f"Approve the result of {label}", [agent], f"Accept the result of step {step.index}/{total} ({step.subject}) of {name}?"))
                parts[gate] = {"role": "approval_after", "step_id": step.id, "index": step.index, "subject": step.subject}
                tails[step.id] = gate
        depended_on = {dependency for step in steps for dependency in step.depends_on}
        sinks = [tails[step.id] for step in steps if step.id not in depended_on]
        # Whatever waited for the SOP node now waits for the nodes the SOP ends in.
        for source, target in sorted(self._plan.edges):
            if source == node_id:
                ops.append(RemoveEdgeOp(from_node=node_id, to=target))
                ops.extend(AddEdgeOp(from_node=sink, to=target) for sink in sinks)
        command = PlanChangeCommand(
            command_id=hashlib.sha256(f"sop:{self._task_id}:{node_id}".encode()).hexdigest(),
            task_id=self._task_id,
            base_plan_version=self._plan.version,
            actor=Actor(kind="system", id="task-workflow"),
            ops=ops,
            reason=f"expand SOP {ref}",
        )
        policy = self._plan_policy()
        # The nodes sit one level under the SOP node: a node that is nested itself needs the limit to allow one more. This
        # raises the limit for this change only, and the operation count is the one the SOP needs (a definition holds at
        # most 50 steps, each with at most two approvals).
        depth, cursor = 0, self._plan.nodes[node_id].draft.parent_node_id
        while cursor is not None and cursor in self._plan.nodes and depth <= len(self._plan.nodes):
            depth, cursor = depth + 1, self._plan.nodes[cursor].draft.parent_node_id
        policy = dataclasses.replace(
            policy,
            max_ops=max(policy.max_ops, len(ops)),
            max_depth=max(policy.max_depth or 1, depth + 1),
            max_chain=MAX_PLAN_CHAIN,
            legacy_depth=False,
        )
        return command, policy, parts, agents

    @staticmethod
    def _approval_op(ref: str, parent: str, title: str, waits: list[str], summary: str) -> AddNodeOp:
        return AddNodeOp(node=ApprovalNode(
            node_id=ref,
            title=title[:200],
            parent_node_id=parent,
            depends_on=waits,
            spec=ApprovalSpec(summary=summary[:2000], risk="medium"),
        ))

    @staticmethod
    def _step_node(
        ref: str, parent: str, sop: str, name: str, step: ResolvedStep, label: str, total: int, waits: list[str]
    ) -> AgentTurnNode:
        goal = (
            f'You are running step {step.index} of {total} of the procedure "{name}": {step.subject}.\n\n'
            f"What this step must achieve:\n{step.description}"
        )
        verification = {
            "sop": sop,
            "sop_name": name,
            "step_id": step.id,
            "subject": step.subject,
            "description": step.description,
            "instructions": step.verifier_instructions,
            **({"expert": step.verifier_expert} if step.verifier_expert else {}),
        }
        return AgentTurnNode(
            node_id=ref,
            title=label[:200],
            parent_node_id=parent,
            depends_on=waits,
            # The steps share the task's workspace, one writer at a time: the scheduler keeps two of them apart.
            workspace_access="write",
            owner_profile=step.executor,
            retry=RetryPolicy(max_attempts=step.max_attempts),
            completion_contract=CompletionContract(
                output_schema_ref=step.output_schema_ref,
                required_artifacts=list(step.required_artifacts),
                verifications=[Verification(kind="sop_verifier", spec=verification)],
            ),
            spec=AgentTurnSpec(goal=goal),
        )

    # ---- the SOP node follows its nodes --------------------------------------------------------------------------

    def _sop_child_changed(self, node_id: str, status: str) -> None:
        if not self._sops or self._plan is None:
            return
        found = self._step_of(node_id)
        if found is None or found[0] not in self._plan.nodes:
            return
        sop_node, part = found
        entry = self._sops[sop_node]
        sop_status = self._plan.nodes[sop_node].status
        if status in {"COMPLETED", "SKIPPED"}:
            if node_id in entry["open"]:
                entry["open"].remove(node_id)
            if not entry["open"] and sop_status != "COMPLETED":
                self._set_node_status(sop_node, "COMPLETED", frozen=True, reason="every step completed")
                self._completed_nodes += 1
        elif status == "BLOCKED":
            if sop_status != "BLOCKED":
                self._set_node_status(sop_node, "BLOCKED", reason=f"step {part['index']} ({part['subject']}) is blocked")
        elif sop_status == "BLOCKED" and not any(
            self._plan.nodes[open_id].status == "BLOCKED" for open_id in entry["open"] if open_id in self._plan.nodes
        ):
            self._set_node_status(sop_node, "RUNNING", reason="a blocked step is back in play")

    # ---- what a step is told, and what it hands on ---------------------------------------------------------------

    def _handover_to(self, node_id: str) -> str:
        """What the steps this one depends on handed over (their final text, cut, and the names of their files), for the
        prompt of a step's attempt. Empty for a node that is not a step or has no dependency that said anything."""
        found = self._step_of(node_id)
        if found is None or found[1]["role"] != "step":
            return ""
        entry = self._sops[found[0]]
        lines: list[str] = []
        dependencies = entry["deps"].get(node_id, [])
        share = min(HANDOVER_CHARS, max(300, MAX_HANDOVER_CHARS // max(1, len(dependencies))))  # the prompt stays bounded
        for dependency in dependencies:
            part = entry["nodes"].get(dependency)
            output = entry["outputs"].get(dependency) or {}
            summary = str(output.get("summary") or self._handovers.get(dependency, "")).strip()[:share]
            if part is None:
                continue
            lines.append(f'- Step {part["index"]} "{part["subject"]}": {summary or "(it reported nothing)"}')
            files = [str(item) for item in output.get("artifacts", [])][:MAX_HANDOVER_FILES]
            if files:
                lines.append("  Files it left in the workspace: " + ", ".join(files))
        return "\n\nWhat the steps before this one handed over:\n" + "\n".join(lines) if lines else ""

    def _record_step_output(self, signal: AttemptFinishedSignal) -> None:
        """A step's attempt ended completed: what it said and the files it left are what the steps after it are told."""
        found = self._step_of(signal.node_id)
        if found is None or found[1]["role"] != "step" or signal.result is None:
            return
        self._sops[found[0]]["outputs"][signal.node_id] = {
            "summary": signal.result.handover_summary[:HANDOVER_CHARS],
            "artifacts": [str(item.get("name", "")) for item in signal.result.manifest_entries if item.get("name")][:MAX_HANDOVER_FILES],
        }

    # ---- approval nodes ------------------------------------------------------------------------------------------

    def _ask_node_approval(self, node_id: str) -> None:
        """An approval node of a compiled SOP waits for a person: it is an approval like any other (`decideApproval`)."""
        found = self._step_of(node_id)
        if self._plan is None or (found is None and not workflow.patched(NODE_APPROVALS)):
            return  # before `task-node-approvals` only an SOP's approval nodes asked anyone
        node = self._plan.nodes[node_id].draft
        asked = int(self._entry(node_id).get("asked", 0)) + 1
        self._entry(node_id)["asked"] = asked
        approval_id = deterministic_id(f"{self._task_id}:{node_id}:node_approval:{asked}", "apr")
        subject = ApprovalSubject(
            kind="sop_step" if found is not None else "node_approval",
            digest=sha(f"{node_id}:{asked}"),
            summary=str(getattr(node.spec, "summary", node.title))[:500],
            risk=getattr(node.spec, "risk", "medium"),
            detail=found[1]["role"] if found is not None else "approval",
        )
        self._approvals[approval_id] = {
            "approval_id": approval_id,
            "status": "PENDING",
            "kind": "node_approval",
            "node_id": node_id,
            "subject": subject.model_dump(mode="json"),
        }
        self._emit("approval.requested", {
            "approval_id": approval_id, "node_id": node_id, "subject": subject.model_dump(mode="json"),
        })

    def _settle_node_approval(self, approval: dict[str, Any], decision: str) -> None:
        """A person decided an approval node: approved, it completes; rejected, it is blocked and the task asks for a review
        (a `resume` asks again)."""
        node_id = str(approval["node_id"])
        if self._plan is None or node_id not in self._plan.nodes or self._plan.nodes[node_id].status != "AWAITING_APPROVAL":
            return
        if decision == "approve":
            self._set_node_status(node_id, "COMPLETED", frozen=True, reason="approved")
            self._completed_nodes += 1
            return
        entry = self._entry(node_id)
        entry.update(blocked=True, blocked_for="approval", retry_at=None)
        why = f"approval of {self._plan.nodes[node_id].draft.title} was refused"
        self._set_node_status(node_id, "BLOCKED", reason=why)
        if self._status not in OPERATOR_HELD and not closed(self._status):
            self._set_status("PAUSED_NEEDS_REVIEW", f"node {node_id} is blocked: {why}")

    def _cancel_node_approvals(self, node_id: str) -> None:
        """A person completed the node by hand: nobody is asked about it any more."""
        for approval_id, approval in self._approvals.items():
            if approval.get("kind") == "node_approval" and approval.get("node_id") == node_id and approval.get("status") == "PENDING":
                approval["status"] = "CANCELLED"
                self._emit("approval.decided", {
                    "approval_id": approval_id, "status": "CANCELLED", "comment": "the node was completed by a person", "always": False,
                })

    # ---- the verifier --------------------------------------------------------------------------------------------

    async def _verify_sop_steps(
        self, proposal: CompletionProposal, contract: Any, snapshot: str | None
    ) -> list[dict[str, Any]]:
        """Every `sop_verifier` verification of the contract: an independent verifier agent (`verify_sop_step`, on the
        agent queue) judges the step from its description, the attempt's final text, the files it left in the workspace
        snapshot and the verifier's own instructions. A FAIL is a verification failure with the verifier's reason."""
        if not workflow.patched(SOP_VERIFIER):
            return []
        assert self._plan is not None
        draft = self._plan.nodes[proposal.node_id].draft
        attempt = self._attempts.get(proposal.node_id) or {}
        result = attempt.get("result") or {}
        for verification in contract.verifications:
            if verification.kind != "sop_verifier":
                continue
            spec = verification.spec
            outcome = await workflow.execute_activity(
                "verify_sop_step",
                {
                    "tenant_id": self._tenant_id,
                    "task_id": self._task_id,
                    "node_id": proposal.node_id,
                    "attempt_id": proposal.attempt_id,
                    "attempt_no": int(attempt.get("attempt_no", 1)),
                    "workspace_snapshot_ref": snapshot,
                    "sop": spec.get("sop"),
                    "step_id": spec.get("step_id"),
                    "subject": spec.get("subject") or draft.title,
                    "description": spec.get("description") or getattr(draft.spec, "goal", draft.title),
                    "instructions": spec.get("instructions") or "",
                    "expert": spec.get("expert"),
                    "text": str(result.get("handover_summary") or ""),
                    "manifest_entries": [
                        {key: item.get(key) for key in ("name", "media_type", "size_bytes")}
                        for item in result.get("manifest_entries", [])
                    ],
                },
                task_queue="orbit.agent",
                result_type=dict,
                start_to_close_timeout=timedelta(seconds=SOP_VERIFIER_TIMEOUT_S),
                heartbeat_timeout=HEARTBEAT,
                retry_policy=RETRY,
            )
            failures = reasons(outcome, "verify_sop_step")
            if failures:
                return failures
        return []

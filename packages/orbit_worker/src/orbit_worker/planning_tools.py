"""Orbit's TaskCreate / TaskGet / TaskList / TaskUpdate (05 §3).

Same names and parameters as AgentScope's built-in task tools, so the model's habits carry over, but every change is
a `submitPlanChange` Update on TaskWorkflow instead of an edit of `AgentState.tasks_context`. The workflow's PlanEngine
is the only judge of a change; a rejection comes back as text that points the agent at TaskList for the latest graph.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, ClassVar, Protocol

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk
from orbit_contracts.v3.common import Actor
from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec
from orbit_contracts.v3.plan import (
    AddEdgeOp,
    AddNodeOp,
    DeclareBlockedOp,
    NodePatch,
    PlanChangeAccepted,
    PlanChangeCommand,
    PlanChangeResult,
    RemoveNodeOp,
    UpdateNodeOp,
)
from orbit_contracts.v3.views import PlanView
from pydantic import BaseModel, Field, TypeAdapter

from orbit_worker.agent_config import owner_profile_for
from orbit_worker.task_stream import TaskStreamContext, current_task_context, current_tool_call_id
from orbit_worker.team_tools import BRIEF_GUIDE

logger = logging.getLogger(__name__)


class PlanPort(Protocol):
    async def get_plan(self, context: TaskStreamContext) -> PlanView: ...
    async def submit(
        self, context: TaskStreamContext, command: PlanChangeCommand
    ) -> PlanChangeResult: ...


class _CreateParams(BaseModel):
    subject: str = Field(description="A brief title for the task")
    description: str = Field(
        description="What needs to be done. For a task given to a team member it is the member's whole brief: it cannot see "
        "the conversation, so it is complete and detailed (role, background, inputs, existing work, numbered task, output "
        "and done criteria)."
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Arbitrary metadata to attach to the task"
    )


class _GetParams(BaseModel):
    task_id: str = Field(description="The ID of the task to retrieve")


class _ListParams(BaseModel):
    pass


class _UnplannableParams(BaseModel):
    reason: str = Field(min_length=1, description="Why the task cannot be turned into a plan")


class _UpdateParams(BaseModel):
    task_id: str = Field(description="The task id.")
    subject: str | None = Field(default=None, description="New subject for the task")
    description: str | None = Field(default=None, description="New description for the task")
    add_blocks: list[str] | None = Field(default=None, description="Task IDs that this task blocks")
    add_blocked_by: list[str] | None = Field(
        default=None, description="Task IDs that block this task"
    )
    status: str | None = Field(
        default=None, description="pending, in_progress, completed or deleted"
    )


def _text(message: str, state: ToolResultState = ToolResultState.SUCCESS) -> ToolChunk:
    return ToolChunk(content=[TextBlock(text=message)], state=state)


class _PlanTool(ToolBase):
    is_concurrency_safe: bool = False
    is_state_injected: bool = False
    params: ClassVar[type[BaseModel]]

    def __init__(self, plan: PlanPort) -> None:
        super().__init__()
        self._plan = plan
        self.input_schema = self.params.model_json_schema()

    async def check_read_only(self, tool_input: dict[str, Any]) -> bool:
        # The leader of a team may run read-only (a mode "ask"; AgentScope's EXPLORE mode denies whatever is not read-only), yet it plans: a
        # plan change touches the plan, never the workspace. The ledger still goes by `is_read_only`, so replays are as before.
        context = current_task_context()
        return self.is_read_only or (context is not None and context.agent.team is not None)

    async def check_permissions(
        self, tool_input: dict[str, Any], context: PermissionContext
    ) -> PermissionDecision:
        # Whether a change is allowed is the PlanEngine's call (05 §2), not a per-call human prompt.
        del tool_input, context
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW, message=f"{self.name} is judged by the plan engine."
        )

    def _context(self) -> TaskStreamContext:
        context = current_task_context()
        if context is None:
            raise RuntimeError(f"{self.name} only works inside a task attempt")
        return context

    async def _commit(self, ops: list[Any], reason: str) -> PlanChangeAccepted | ToolChunk:
        context = self._context()
        plan = await self._plan.get_plan(context)
        command = PlanChangeCommand(
            command_id=hashlib.sha256(
                f"{context.attempt_id}|{current_tool_call_id()}".encode()
            ).hexdigest(),
            task_id=context.task_id,
            base_plan_version=plan.plan_version,
            actor=Actor(kind="agent", id=context.attempt_id, attempt_id=context.attempt_id),
            ops=ops,
            reason=reason,
        )
        result = await self._plan.submit(context, command)
        if isinstance(result, PlanChangeAccepted):
            return result
        return _text(
            f"Rejected ({result.code}): {result.detail}. The plan is at version {result.latest_plan_version} "
            f"({result.latest_hash}); call TaskList to see the current graph.",
            ToolResultState.ERROR,
        )


class TaskCreateTool(_PlanTool):
    name = "TaskCreate"
    description = (
        "Create a task in the plan. It runs after the task you are working on finishes. When it goes to a team member, the "
        "`description` is that member's brief and is never shortened. " + BRIEF_GUIDE
    )
    is_read_only = False
    params = _CreateParams

    async def call(self, **kwargs: Any) -> ToolChunk:
        args = _CreateParams.model_validate(kwargs)
        context = self._context()
        owner, problem = owner_profile_for(context.agent.team, args.metadata)
        if problem:
            return _text(f"Rejected: {problem}.", ToolResultState.ERROR)
        node = AgentTurnNode(
            node_id="tmp:1",
            title=args.subject,
            # Given to a member of the team, or to nobody: then the node runs as the task's expert (or its profile),
            # as the node that planned it did.
            owner_profile=owner,
            # Work given to a member of the team is nested under the node that gave it (03 §3, depth).
            parent_node_id=context.node_id if owner else None,
            depends_on=[context.node_id],
            spec=AgentTurnSpec(goal=args.description),
        )
        outcome = await self._commit([AddNodeOp(node=node)], f"TaskCreate {args.subject}")
        if isinstance(outcome, ToolChunk):
            return outcome
        return _text(f"created {outcome.id_map['tmp:1']} (plan version {outcome.plan_version})")


class TaskListTool(_PlanTool):
    name = "TaskList"
    description = "List every task in the plan with its status and dependencies."
    is_read_only = True
    is_concurrency_safe = True
    params = _ListParams

    async def call(self, **kwargs: Any) -> ToolChunk:
        del kwargs
        plan = await self._plan.get_plan(self._context())
        lines = [f"plan version {plan.plan_version} ({plan.hash})"]
        for node in plan.nodes:
            blocked_by = ",".join(node.depends_on) or "-"
            lines.append(f"{node.node_id} | {node.status} | {node.title} | blocked_by={blocked_by}")
        return _text("\n".join(lines))


class TaskGetTool(_PlanTool):
    name = "TaskGet"
    description = "Get one task of the plan."
    is_read_only = True
    is_concurrency_safe = True
    params = _GetParams

    async def call(self, **kwargs: Any) -> ToolChunk:
        args = _GetParams.model_validate(kwargs)
        plan = await self._plan.get_plan(self._context())
        node = next((item for item in plan.nodes if item.node_id == args.task_id), None)
        if node is None:
            return _text(f"no task {args.task_id}", ToolResultState.ERROR)
        return _text(
            f"{node.node_id} | {node.status} | {node.title} | type={node.type} | attempts={node.attempt_count} "
            f"| blocked_by={','.join(node.depends_on) or '-'}"
        )


class TaskUpdateTool(_PlanTool):
    name = "TaskUpdate"
    description = "Change a task of the plan: its subject, description, dependencies, or delete it."
    is_read_only = False
    params = _UpdateParams

    async def call(self, **kwargs: Any) -> ToolChunk:
        args = _UpdateParams.model_validate(kwargs)
        if args.status not in (None, "deleted"):
            return _text(
                "A task's status follows its attempts; only status=deleted can be requested.",
                ToolResultState.ERROR,
            )
        ops: list[Any] = []
        patch = NodePatch(
            title=args.subject, spec={"goal": args.description} if args.description else None
        )
        if patch.title or patch.spec:
            ops.append(UpdateNodeOp(node_id=args.task_id, patch=patch))
        ops += [
            AddEdgeOp.model_validate({"from": source, "to": args.task_id})
            for source in args.add_blocked_by or []
        ]
        ops += [
            AddEdgeOp.model_validate({"from": args.task_id, "to": target})
            for target in args.add_blocks or []
        ]
        if args.status == "deleted":
            ops.append(RemoveNodeOp(node_id=args.task_id))
        if not ops:
            return _text("nothing to change")
        outcome = await self._commit(ops, f"TaskUpdate {args.task_id}")
        if isinstance(outcome, ToolChunk):
            return outcome
        return _text(f"updated {args.task_id} (plan version {outcome.plan_version})")


class DeclareUnplannableTool(_PlanTool):
    name = "orbit_declare_unplannable"
    description = (
        "Say that this task cannot be planned. The task then waits for a person to review it."
    )
    is_read_only = False
    params = _UnplannableParams

    async def call(self, **kwargs: Any) -> ToolChunk:
        args = _UnplannableParams.model_validate(kwargs)
        outcome = await self._commit(
            [DeclareBlockedOp(node_id=self._context().node_id, reason=args.reason)], "unplannable"
        )
        if isinstance(outcome, ToolChunk):
            return outcome
        return _text(f"declared unplannable (plan version {outcome.plan_version})")


# The tools that change the plan: the leader's own work node does not have them (it executes, it does not plan).
PLAN_WRITE_TOOLS = frozenset({"TaskCreate", "TaskUpdate", "orbit_declare_unplannable"})


def task_workflow_id(context: TaskStreamContext) -> str:
    return f"task/{context.tenant_id}/{context.task_id}"


async def is_leader_work_node(plan: PlanPort, context: TaskStreamContext) -> bool:
    """Whether the attempt's node is a task the leader gave itself (TaskCreate with its own role as owner). Such a node runs as
    the leader's expert yet is neither the leader's planning node nor a review: the plan nests it under the node that created
    it (`parent_node_id`, set only for a task with an owner), and a review has a `review_round`. The exploration node, a
    follow-up and a member's node have no parent, or run as another expert. A plan that cannot be read is no."""
    try:
        view = await plan.get_plan(context)
    except Exception:
        logger.warning("the plan could not be read to tell the node's kind; treating it as the leader's own", exc_info=True)
        return False
    node = next((item for item in view.nodes if item.node_id == context.node_id), None)
    return node is not None and node.parent_node_id is not None and node.review_round is None


def planning_tools(plan: PlanPort) -> list[ToolBase]:
    return [
        TaskCreateTool(plan),
        TaskGetTool(plan),
        TaskListTool(plan),
        TaskUpdateTool(plan),
        DeclareUnplannableTool(plan),
    ]


class TemporalPlanPort:
    """Reaches TaskWorkflow through the Temporal client of the running activity."""

    def __init__(self, workflow_id_for: Any) -> None:
        self._workflow_id_for = workflow_id_for

    async def get_plan(self, context: TaskStreamContext) -> PlanView:
        from temporalio import activity

        handle = activity.client().get_workflow_handle(self._workflow_id_for(context))
        # The plan as this attempt may see it (05 §6): the asking actor is named, so the workflow answers for it.
        actor = Actor(kind="agent", id=context.attempt_id, attempt_id=context.attempt_id)
        return await handle.query("getPlan", actor, result_type=PlanView)

    async def submit(
        self, context: TaskStreamContext, command: PlanChangeCommand
    ) -> PlanChangeResult:
        from temporalio import activity

        handle = activity.client().get_workflow_handle(self._workflow_id_for(context))
        raw = await handle.execute_update("submitPlanChange", command, result_type=dict)
        return TypeAdapter(PlanChangeResult).validate_python(raw)

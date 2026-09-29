"""Tool policy for task attempts (05 §4): the exploration budget.

The initial node explores and plans. Its profile caps how many tool calls it may make; once they are spent only the
tools that move the task forward stay available. The middleware only ever tightens what AgentScope decided.
"""

from __future__ import annotations

import zlib
from typing import Any, Protocol

from agentscope.middleware import MiddlewareBase
from agentscope.permission import PermissionBehavior, PermissionDecision
from orbit_contracts.v3 import Policy
from orbit_orch.plan_engine import deterministic_id

from orbit_worker.task_stream import TaskStreamContext, current_task_context
from orbit_worker.worker_events import publish_attempt_event

DEFAULT_EXPLORATION_TOOL_CALLS = 50
EXTENSION_TOOL_CALLS = 10  # what one allowed orbit_request_budget_extension adds
# What may still be called after the budget is spent. The other two orbit tools land with the budget-extension and
# unplannable flows.
AFTER_EXHAUSTION = frozenset(
    {"TaskCreate", "orbit_request_budget_extension", "orbit_declare_unplannable"}
)


class PolicyStore(Protocol):
    async def put_checkpoint(
        self,
        *,
        tenant_id: str,
        task_id: str,
        node_id: str,
        attempt_id: str,
        seq: int,
        kind: str,
        payload: bytes,
    ) -> str: ...
    async def publish_events(self, events: list[dict[str, Any]]) -> None: ...
    async def effective_policy(
        self, *, tenant_id: str, profile_ref: str, task_policy: Policy
    ) -> Policy: ...
    async def count_side_effects(self, *, tenant_id: str, attempt_id: str) -> int: ...
    async def granted_extensions(self, *, tenant_id: str, attempt_id: str) -> int: ...


def is_exploration(context: TaskStreamContext) -> bool:
    return context.node_id == deterministic_id(f"{context.task_id}:exploration", "n")


async def effective_policy(store: PolicyStore, context: TaskStreamContext) -> Policy:
    return await store.effective_policy(
        tenant_id=context.tenant_id, profile_ref=context.profile, task_policy=context.task_policy
    )


async def exploration_exhausted(store: PolicyStore, context: TaskStreamContext) -> bool:
    """Whether this attempt is the exploration node and has used up its tool budget."""
    if not is_exploration(context):
        return False
    limit = (await effective_policy(store, context)).exploration_max_tool_calls
    used = await store.count_side_effects(
        tenant_id=context.tenant_id, attempt_id=context.attempt_id
    )
    extra = await store.granted_extensions(
        tenant_id=context.tenant_id, attempt_id=context.attempt_id
    )
    allowed = (
        limit if limit is not None else DEFAULT_EXPLORATION_TOOL_CALLS
    ) + extra * EXTENSION_TOOL_CALLS
    return used >= allowed


class OrbitPolicyMiddleware(MiddlewareBase):
    def __init__(self, store: PolicyStore) -> None:
        self._store = store
        self._checkpointed: set[int] = set()

    async def on_acting(self, agent: Any, input_kwargs: dict, next_handler: Any) -> Any:
        """One checkpoint per batch of tool calls, taken before the batch's first call runs (06 §3 S7). A resume
        from it re-enters the batch with nothing half done: calls that already ran come back from the ledger.
        Nothing is checkpointed between two calls of one batch, and the batch number is the checkpoint's `seq`, so
        entering the same batch again after a resume writes nothing new."""
        context = current_task_context()
        if context is not None:
            batch = _batch_index(agent)
            if batch not in self._checkpointed:
                self._checkpointed.add(batch)
                await self._store.put_checkpoint(
                    tenant_id=context.tenant_id,
                    task_id=context.task_id,
                    node_id=context.node_id,
                    attempt_id=context.attempt_id,
                    seq=batch,
                    kind="agent_state",
                    payload=agent.state.model_dump_json().encode("utf-8"),
                )
        async for item in next_handler(**input_kwargs):
            yield item

    async def on_check_permission(
        self, agent: Any, input_kwargs: dict, next_handler: Any
    ) -> PermissionDecision:
        # AgentScope decides first; this can only turn the answer into a refusal, never into an allowance.
        decision = await next_handler(**input_kwargs)
        context = current_task_context()
        call = input_kwargs["tool_call"]
        # The layers of policy can only take tools away (05 §6): what any of them lists is refused, whatever AgentScope
        # decided.
        if (
            context is not None
            and decision.behavior != PermissionBehavior.DENY
            and call.name in (await effective_policy(self._store, context)).denied_tools
        ):
            return await self._refuse(
                context, call, f"{call.name} is not allowed by the policy of this task."
            )
        if (
            context is None
            or decision.behavior == PermissionBehavior.DENY
            or call.name in AFTER_EXHAUSTION
            or not await exploration_exhausted(self._store, context)
        ):
            return decision
        return await self._refuse(
            context,
            call,
            "The exploration budget is spent. Create the plan with TaskCreate, ask for more budget, or declare the task unplannable.",
        )

    async def _refuse(
        self, context: TaskStreamContext, call: Any, message: str
    ) -> PermissionDecision:
        await publish_attempt_event(
            self._store,
            tenant_id=context.tenant_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            event_type="tool.call_finished",
            body={
                "attempt_id": context.attempt_id,
                "tool_call_id": call.id,
                "tool_name": call.name,
                "state": "denied",
                "result_preview": message,
            },
            seed=f"finished:{call.id}",
        )
        return PermissionDecision(behavior=PermissionBehavior.DENY, message=message)


def _batch_index(agent: Any) -> int:
    """Which batch of tool calls the agent is executing: the reasoning round of its current reply. It is kept in the
    saved state, so entering the same batch after a resume gives the same number; the reply is folded in so a later
    reply of the same attempt does not reuse it."""
    reply = agent.state.reply_context
    return (zlib.crc32(str(agent.state.reply_id).encode()) % 100_000) * 100 + int(reply.cur_iter)

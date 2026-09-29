"""Deterministic B-prime plan evaluation.

The workflow owns the plan state, while this module owns all graph rules.  It
does not import Temporal or any worker dependency, so the same evaluator can
be used by workflow replay and by the control plane's contract checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from orbit_contracts.v3.common import Actor, Budget, NodeId, NodeStatus, Sha256Ref
from orbit_contracts.v3.nodes import TaskNodeDraft
from orbit_contracts.v3.plan import (
    AddEdgeOp,
    AddNodeOp,
    DeclareBlockedOp,
    NodePatch,
    PlanChangeAccepted,
    PlanChangeCommand,
    PlanChangeRejected,
    PlanChangeResult,
    RemoveEdgeOp,
    RemoveNodeOp,
    UpdateNodeOp,
)
from pydantic import TypeAdapter, ValidationError

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DRAFT = TypeAdapter(TaskNodeDraft)


def attempt_workflow_id(task_id: str, node_id: str, attempt_no: int) -> str:
    """The id of the AttemptWorkflow of one attempt (04 §6). Maintenance finds a projection row's workflow by it."""

    return f"attempt/{task_id}/{node_id}/{attempt_no}"


def deterministic_id(seed: str, prefix: str) -> str:
    """Return a stable, valid Orbit id without relying on wall-clock time."""

    value = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:16], "big")
    chars: list[str] = []
    for _ in range(26):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return f"{prefix}_" + "".join(reversed(chars))


@dataclass(frozen=True)
class PlanNodeState:
    draft: TaskNodeDraft
    status: NodeStatus = "PENDING"
    frozen: bool = False
    current_attempt_id: str | None = None
    attempt_count: int = 0
    created_by: str | None = None

    def canonical(self) -> dict[str, object]:
        return {
            "draft": self.draft.model_dump(mode="json", by_alias=True, exclude_none=True),
            "status": self.status,
            "frozen": self.frozen,
            "current_attempt_id": self.current_attempt_id,
            "attempt_count": self.attempt_count,
            "created_by": self.created_by,
        }


@dataclass(frozen=True)
class PlanState:
    version: int
    nodes: dict[str, PlanNodeState]
    edges: frozenset[tuple[str, str]] = frozenset()
    hash: Sha256Ref = "sha256:" + "0" * 64

    @classmethod
    def build(
        cls,
        version: int,
        nodes: dict[str, PlanNodeState],
        edges: Iterable[tuple[str, str]] = (),
    ) -> PlanState:
        edge_set = frozenset(edges)
        canonical = {
            "version": version,
            "nodes": {key: nodes[key].draft.model_dump(mode="json", by_alias=True) for key in sorted(nodes)},
            "edges": [list(edge) for edge in sorted(edge_set)],
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(version=version, nodes=dict(nodes), edges=edge_set, hash=f"sha256:{digest}")

    def with_updates(
        self,
        *,
        version: int | None = None,
        nodes: dict[str, PlanNodeState] | None = None,
        edges: Iterable[tuple[str, str]] | None = None,
    ) -> PlanState:
        return PlanState.build(
            version if version is not None else self.version,
            nodes if nodes is not None else self.nodes,
            edges if edges is not None else self.edges,
        )


@dataclass(frozen=True)
class PlanPolicy:
    """The immutable policy snapshot visible to a PlanEngine invocation."""

    allowed_node_types: frozenset[str] = frozenset(
        {"agent_turn", "sop_stage", "approval", "wait", "checkpoint"}
    )
    max_ops: int = 32
    max_depth: int = 8
    max_serialized_bytes: int = 256 * 1024
    max_budget: Budget = field(default_factory=Budget)
    active_attempt_id: str | None = None
    active_attempt_ids: frozenset[str] = frozenset()
    active_node_ids: frozenset[str] = frozenset()
    visible_node_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class PlanApplyOutcome:
    result: PlanChangeResult
    plan: PlanState


def initial_plan(task_id: str, goal: str, profile: str) -> PlanState:
    node_id = deterministic_id(f"{task_id}:exploration", "n")
    from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec

    node = AgentTurnNode(
        node_id=node_id,
        title="Explore and plan",
        owner_profile=profile,
        spec=AgentTurnSpec(goal=goal),
    )
    return PlanState.build(1, {node_id: PlanNodeState(node, status="READY")})


def apply(plan: PlanState, command: PlanChangeCommand, policy: PlanPolicy) -> PlanApplyOutcome:
    """Apply one command atomically and return either a new plan or the old one."""

    rejected = _reject
    if command.base_plan_version != plan.version:
        return rejected(plan, command, "VERSION_CONFLICT", "base plan version is stale")
    if len(command.ops) > policy.max_ops:
        return rejected(plan, command, "TOO_MANY_OPS", "plan change contains too many operations")
    if command.actor.kind == "agent":
        if command.actor.attempt_id not in policy.active_attempt_ids and policy.active_attempt_id != command.actor.attempt_id:
            return rejected(plan, command, "STALE_ATTEMPT", "agent attempt is no longer active")
        if not command.actor.attempt_id:
            return rejected(plan, command, "STALE_ATTEMPT", "agent commands require an attempt")

    nodes = dict(plan.nodes)
    edges = set(plan.edges)
    id_map: dict[str, NodeId] = {}

    try:
        for index, op in enumerate(command.ops):
            if isinstance(op, AddNodeOp) and op.node.node_id.startswith("tmp:"):
                if op.node.node_id in id_map:
                    raise _PlanError("SCHEMA_INVALID", "duplicate temporary node id")
                id_map[op.node.node_id] = _resolve_new_id(op.node.node_id, command.command_id, index, nodes)
        for index, op in enumerate(command.ops):
            if isinstance(op, AddNodeOp):
                node = op.node
                _check_type(node.type, policy)
                if node.type_version != 1 or (command.actor.kind == "agent" and node.type == "sop_stage"):
                    raise _PlanError("TYPE_NOT_ALLOWED", "node type or version is not available to this actor")
                source_id = node.node_id
                real_id = _resolve_new_id(source_id, command.command_id, index, nodes)
                if source_id.startswith("tmp:"):
                    id_map[source_id] = real_id
                elif real_id in nodes:
                    raise _PlanError("SCHEMA_INVALID", f"node {real_id} already exists")
                draft = node.model_copy(
                    update={
                        "node_id": real_id,
                        "depends_on": [_resolve_ref(ref, id_map) for ref in node.depends_on],
                    }
                )
                for dependency in draft.depends_on:
                    if dependency not in id_map.values():
                        _check_visibility(command.actor, dependency, policy)
                nodes[real_id] = PlanNodeState(draft, status="PENDING", created_by=command.actor.id)
                for dependency in draft.depends_on:
                    edges.add((dependency, real_id))
            elif isinstance(op, UpdateNodeOp):
                node_id = _resolve_ref(op.node_id, id_map)
                current = _require_node(nodes, node_id)
                _check_mutable(current)
                _check_visibility(command.actor, node_id, policy)
                _check_author(command.actor, current)
                nodes[node_id] = replace(current, draft=_patch_node(current.draft, op.patch))
            elif isinstance(op, RemoveNodeOp):
                node_id = _resolve_ref(op.node_id, id_map)
                current = _require_node(nodes, node_id)
                _check_mutable(current)
                _check_visibility(command.actor, node_id, policy)
                _check_author(command.actor, current)
                if any(nodes[target].frozen for source, target in edges if source == node_id):
                    raise _PlanError("FROZEN_NODE", "removing this node changes a frozen dependency")
                del nodes[node_id]
                edges = {edge for edge in edges if node_id not in edge}
            elif isinstance(op, AddEdgeOp):
                source = _resolve_ref(op.from_node, id_map)
                target = _resolve_ref(op.to, id_map)
                _require_node(nodes, source)
                target_node = _require_node(nodes, target)
                if target_node.frozen:
                    raise _PlanError("FROZEN_NODE", f"cannot add a dependency to frozen node {target}")
                for ref in (source, target):
                    if ref not in id_map.values():
                        _check_visibility(command.actor, ref, policy)
                _check_author(command.actor, target_node)
                edges.add((source, target))
            elif isinstance(op, RemoveEdgeOp):
                source = _resolve_ref(op.from_node, id_map)
                target = _resolve_ref(op.to, id_map)
                target_node = _require_node(nodes, target)
                source_node = _require_node(nodes, source)
                if target_node.frozen or source_node.frozen:
                    raise _PlanError("FROZEN_NODE", f"cannot change frozen node {target}")
                edges.discard((source, target))
                _check_visibility(command.actor, source, policy)
                _check_visibility(command.actor, target, policy)
                _check_author(command.actor, target_node)
            elif isinstance(op, DeclareBlockedOp):
                node_id = _resolve_ref(op.node_id, id_map)
                current = _require_node(nodes, node_id)
                _check_visibility(command.actor, node_id, policy)
                _check_mutable(current)
                if command.actor.kind == "agent" and node_id not in policy.active_node_ids:
                    raise _PlanError("POLICY_DENIED", "agent can only block its own node")
                nodes[node_id] = replace(current, status="BLOCKED")
            else:  # pragma: no cover - Pydantic's discriminated union is exhaustive.
                raise _PlanError("SCHEMA_INVALID", "unknown plan operation")

        _validate_graph(nodes, edges, policy)
        nodes = {
            node_id: replace(state, draft=state.draft.model_copy(update={
                "depends_on": sorted(source for source, target in edges if target == node_id),
            })) for node_id, state in nodes.items()
        }
        new_plan = PlanState.build(plan.version + 1, nodes, edges)
        if len(json.dumps({key: state.canonical() for key, state in nodes.items()}).encode()) > policy.max_serialized_bytes:
            raise _PlanError("TOO_MANY_OPS", "plan exceeds the serialized size limit")
        _validate_budget(new_plan.nodes.values(), policy.max_budget)
    except ValidationError:
        return rejected(plan, command, "SCHEMA_INVALID", "node spec does not match its schema")
    except _PlanError as exc:
        return rejected(plan, command, exc.code, str(exc))

    return PlanApplyOutcome(
        PlanChangeAccepted(status="accepted", plan_version=new_plan.version, id_map=id_map),
        new_plan,
    )


class _PlanError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _reject(plan: PlanState, command: PlanChangeCommand, code: str, detail: str) -> PlanApplyOutcome:
    return PlanApplyOutcome(
        PlanChangeRejected(
            status="rejected",
            code=code,  # type: ignore[arg-type]
            detail=detail,
            latest_plan_version=plan.version,
            latest_hash=plan.hash,
        ),
        plan,
    )


def _resolve_new_id(raw: str, command_id: str, index: int, nodes: dict[str, PlanNodeState]) -> str:
    if raw.startswith("tmp:"):
        candidate = deterministic_id(f"{command_id}:{raw}:{index}", "n")
    else:
        candidate = raw
    if candidate in nodes:
        raise _PlanError("SCHEMA_INVALID", f"node {candidate} already exists")
    return candidate


def _resolve_ref(raw: str, id_map: dict[str, str]) -> str:
    if raw.startswith("tmp:"):
        if raw not in id_map:
            raise _PlanError("SCHEMA_INVALID", f"unknown temporary node {raw}")
        return id_map[raw]
    return raw


def _require_node(nodes: dict[str, PlanNodeState], node_id: str) -> PlanNodeState:
    if node_id not in nodes:
        raise _PlanError("SCHEMA_INVALID", f"unknown node {node_id}")
    return nodes[node_id]


def _check_type(node_type: str, policy: PlanPolicy) -> None:
    if node_type not in policy.allowed_node_types:
        raise _PlanError("TYPE_NOT_ALLOWED", f"node type {node_type} is disabled")


def _check_mutable(node: PlanNodeState) -> None:
    if node.frozen or node.status == "COMPLETED":
        raise _PlanError("FROZEN_NODE", "completed nodes are immutable")


def _check_visibility(actor: Actor, node_id: str, policy: PlanPolicy) -> None:
    if policy.visible_node_ids is not None and node_id not in policy.visible_node_ids:
        raise _PlanError("VISIBILITY", f"node {node_id} is outside the actor visibility scope")

def _check_author(actor: Actor, node: PlanNodeState) -> None:
    if actor.kind == "agent" and (node.created_by != actor.id or node.status not in {"PENDING", "READY"}):
        raise _PlanError("POLICY_DENIED", "agent can only change its own unstarted nodes")


def _patch_node(node: TaskNodeDraft, patch: NodePatch) -> TaskNodeDraft:
    changes: dict[str, object] = {}
    if patch.title is not None:
        changes["title"] = patch.title
    if patch.budget is not None:
        changes["budget"] = patch.budget
    if patch.spec is not None:
        changes["spec"] = {**node.spec.model_dump(mode="json"), **patch.spec}
    return _DRAFT.validate_python({**node.model_dump(mode="json", by_alias=True), **changes})


def _validate_graph(nodes: dict[str, PlanNodeState], edges: set[tuple[str, str]], policy: PlanPolicy) -> None:
    for source, target in edges:
        if source not in nodes or target not in nodes:
            raise _PlanError("SCHEMA_INVALID", "an edge references a missing node")
        if source == target:
            raise _PlanError("CYCLE", "a node cannot depend on itself")
    incoming = {node_id: 0 for node_id in nodes}
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for source, target in sorted(edges):
        adjacency[source].append(target)
        incoming[target] += 1
    ready = sorted(node_id for node_id, count in incoming.items() if count == 0)
    depths = {node_id: 0 for node_id in nodes}
    processed = 0
    while ready:
        node_id = ready.pop(0)
        processed += 1
        for child in adjacency[node_id]:
            depths[child] = max(depths[child], depths[node_id] + 1)
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if processed != len(nodes):
        raise _PlanError("CYCLE", "plan graph contains a cycle")
    if max(depths.values(), default=0) > policy.max_depth:
        raise _PlanError("DEPTH_EXCEEDED", "plan depth exceeds policy")


def _validate_budget(nodes: Iterable[PlanNodeState], maximum: Budget) -> None:
    totals = {"tokens": 0, "tool_calls": 0, "wall_s": 0, "cost_usd_micros": 0}
    for node in nodes:
        if node.status in {"COMPLETED", "SKIPPED", "CANCELLED"}:
            continue
        budget = node.draft.budget
        for field_name, total in totals.items():
            value = getattr(budget, field_name)
            if value is not None:
                total += value
                totals[field_name] = total
                limit = getattr(maximum, field_name)
                if limit is not None and totals[field_name] > limit:
                    raise _PlanError("BUDGET_EXCEEDED", f"plan {field_name} budget exceeds policy")

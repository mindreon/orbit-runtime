"""The plan engine's depth and visibility rules (03 §3 invariant 4, 05 §6).

How it can go wrong, written down before the code:
  - "depth" is the length of the chain of dependencies, so an ordinary pipeline of nine steps is refused as too deep, and a
    team's nesting is not limited at all;
  - the nesting limit is checked against nodes that were already there, so lowering it locks the plan;
  - a parent that is missing, or a chain of parents that loops, is accepted and the plan cannot be read afterwards;
  - an agent names or changes a node it was never shown;
  - a plan with no `parent_node_id` gets another hash than it had, so every stored plan version looks changed.
"""

from __future__ import annotations

from orbit_contracts.v3 import Actor, PlanChangeCommand
from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec
from orbit_contracts.v3.plan import (
    AddEdgeOp,
    AddNodeOp,
    PlanChangeAccepted,
    RemoveNodeOp,
    UpdateNodeOp,
)
from orbit_orch.plan_engine import (
    PlanNodeState,
    PlanPolicy,
    PlanState,
    apply,
    compact,
    deterministic_id,
)

USER = Actor(kind="user", id="u")


def _id(name: str) -> str:
    return deterministic_id(name, "n")


def _node(name: str, *, parent: str | None = None, status: str = "PENDING", frozen: bool = False, created_by: str | None = None) -> PlanNodeState:
    draft = AgentTurnNode(
        node_id=_id(name), title=name, spec=AgentTurnSpec(goal=name),
        parent_node_id=None if parent is None else _id(parent),
    )
    return PlanNodeState(draft, status=status, frozen=frozen, created_by=created_by)  # type: ignore[arg-type]


def _change(version: int, *ops, actor: Actor = USER) -> PlanChangeCommand:
    return PlanChangeCommand(
        command_id="01JA" + "0" * 22, task_id="task_" + "0" * 26, base_plan_version=version, actor=actor, ops=list(ops),
    )


def _add(ref: str, *, parent: str | None = None, after: list[str] | None = None) -> AddNodeOp:
    return AddNodeOp(node=AgentTurnNode(
        node_id=ref, title=ref, spec=AgentTurnSpec(goal=ref), parent_node_id=parent, depends_on=after or [],
    ))


def _code(outcome) -> str:
    return outcome.result.code if outcome.result.status == "rejected" else "accepted"


# ---- depth is nesting -------------------------------------------------------------------------------------------------


def test_a_pipeline_of_dependencies_is_not_deep_and_has_only_a_safety_bound() -> None:
    plan = PlanState.build(1, {_id("root"): _node("root", status="READY")})
    chain = [_add("tmp:1", after=[_id("root")])] + [_add(f"tmp:{n}", after=[f"tmp:{n - 1}"]) for n in range(2, 13)]
    accepted = apply(plan, _change(1, *chain), PlanPolicy())
    assert isinstance(accepted.result, PlanChangeAccepted), "twelve steps one after the other: over the old limit of 8"
    # The safety bound is generous but real, and its rejection is not the design's DEPTH_EXCEEDED.
    refused = apply(plan, _change(1, *chain), PlanPolicy(max_chain=10))
    assert _code(refused) == "TOO_MANY_OPS" and "chain of dependencies" in refused.result.detail


def test_before_nesting_the_chain_was_the_depth_and_a_history_that_judged_it_so_still_does() -> None:
    plan = PlanState.build(1, {_id("root"): _node("root", status="READY")})
    chain = [_add("tmp:1", after=[_id("root")])] + [_add(f"tmp:{n}", after=[f"tmp:{n - 1}"]) for n in range(2, 12)]
    legacy = PlanPolicy(max_depth=None, max_chain=8, legacy_depth=True)
    assert _code(apply(plan, _change(1, *chain), legacy)) == "DEPTH_EXCEEDED"


def test_nesting_is_the_length_of_the_parent_chain_and_the_default_limit_is_one() -> None:
    plan = PlanState.build(1, {_id("lead"): _node("lead", status="RUNNING")})
    one_level = apply(plan, _change(1, _add("tmp:1", parent=_id("lead"))), PlanPolicy())
    assert isinstance(one_level.result, PlanChangeAccepted)
    child = one_level.result.id_map["tmp:1"]
    assert one_level.plan.nodes[child].draft.parent_node_id == _id("lead")

    two_levels = apply(one_level.plan, _change(2, _add("tmp:2", parent=child)), PlanPolicy())
    assert _code(two_levels) == "DEPTH_EXCEEDED" and "nesting depth 2" in two_levels.result.detail
    # Within one command too: a parent and its child added together.
    together = apply(plan, _change(1, _add("tmp:1", parent=_id("lead")), _add("tmp:2", parent="tmp:1")), PlanPolicy())
    assert _code(together) == "DEPTH_EXCEEDED"
    # A team that allows two levels takes the same change.
    allowed = apply(one_level.plan, _change(2, _add("tmp:2", parent=child)), PlanPolicy(max_depth=2))
    assert isinstance(allowed.result, PlanChangeAccepted)


def test_a_plan_that_is_already_deeper_than_the_limit_can_still_be_changed() -> None:
    nodes = {_id("a"): _node("a"), _id("b"): _node("b", parent="a"), _id("c"): _node("c", parent="b")}
    plan = PlanState.build(1, nodes)
    other = apply(plan, _change(1, _add("tmp:1")), PlanPolicy(max_depth=1))
    assert isinstance(other.result, PlanChangeAccepted), "only nodes that are new, or moved, are judged"


def test_a_missing_parent_and_a_parent_chain_that_loops_are_refused() -> None:
    plan = PlanState.build(1, {_id("a"): _node("a")})
    ghost = apply(plan, _change(1, _add("tmp:1", parent=_id("ghost"))), PlanPolicy())
    assert _code(ghost) == "SCHEMA_INVALID" and "unknown parent" in ghost.result.detail
    loop = apply(plan, _change(1, _add("tmp:1", parent="tmp:2"), _add("tmp:2", parent="tmp:1")), PlanPolicy(max_depth=5))
    assert _code(loop) == "CYCLE"
    itself = apply(plan, _change(1, _add("tmp:1", parent="tmp:1")), PlanPolicy(max_depth=5))
    assert _code(itself) == "CYCLE"


def test_a_node_that_others_are_nested_under_cannot_be_removed_and_is_not_compacted_while_they_go_on() -> None:
    nodes = {_id("lead"): _node("lead"), _id("member"): _node("member", parent="lead")}
    plan = PlanState.build(1, nodes)
    removal = apply(plan, _change(1, RemoveNodeOp(node_id=_id("lead"))), PlanPolicy())
    assert _code(removal) == "SCHEMA_INVALID" and "nested under it" in removal.result.detail
    # Compaction takes finished nodes nothing needs; a parent of an unfinished node is needed.
    done = {
        _id("lead"): _node("lead", status="COMPLETED", frozen=True),
        _id("member"): _node("member", parent="lead", status="RUNNING"),
    }
    assert compact(PlanState.build(2, done)) is None
    finished = {**done, _id("member"): _node("member", parent="lead", status="COMPLETED", frozen=True)}
    result = compact(PlanState.build(2, finished))
    assert result is not None and sorted(result.removed) == sorted([_id("lead"), _id("member")])


def test_a_child_of_an_archived_parent_is_judged_without_the_parent() -> None:
    plan = PlanState.build(3, {_id("child"): _node("child", parent="lead", status="RUNNING")})
    policy = PlanPolicy(archived_node_ids=frozenset({_id("lead")}))
    outcome = apply(plan, _change(3, _add("tmp:1")), policy)
    assert isinstance(outcome.result, PlanChangeAccepted)


def test_a_plan_without_nesting_keeps_the_hash_it_had() -> None:
    from orbit_orch.plan_engine import _canonical_draft

    plain = _node("a").draft
    assert "parent_node_id" not in _canonical_draft(plain)
    assert "parent_node_id" in _canonical_draft(_node("b", parent="a").draft)
    assert PlanState.build(1, {_id("a"): _node("a")}).hash == PlanState.build(1, {_id("a"): _node("a")}).hash


# ---- visibility -------------------------------------------------------------------------------------------------------


AGENT = Actor(kind="agent", id="att-1", attempt_id="att_" + "0" * 26)


def _policy(visible: set[str] | None) -> PlanPolicy:
    return PlanPolicy(
        active_attempt_ids=frozenset({AGENT.attempt_id}),  # type: ignore[arg-type]
        visible_node_ids=None if visible is None else frozenset(visible),
    )


def test_an_agent_cannot_name_a_node_outside_what_it_may_see() -> None:
    mine = _node("mine", status="READY", created_by="att-1")
    plan = PlanState.build(1, {_id("mine"): mine, _id("hidden"): _node("hidden", status="READY", created_by="att-1")})
    seen = {_id("mine")}
    for op in (
        UpdateNodeOp.model_validate({"op": "update_node", "node_id": _id("hidden"), "patch": {"title": "x"}}),
        RemoveNodeOp(node_id=_id("hidden")),
        AddEdgeOp.model_validate({"op": "add_edge", "from": _id("hidden"), "to": _id("mine")}),
        _add("tmp:1", after=[_id("hidden")]),
        _add("tmp:1", parent=_id("hidden")),
    ):
        outcome = apply(plan, _change(1, op, actor=AGENT), _policy(seen))
        assert _code(outcome) == "VISIBILITY", op
    # What it may see it can change, and a person sees everything.
    own = apply(plan, _change(1, UpdateNodeOp.model_validate({"op": "update_node", "node_id": _id("mine"), "patch": {"title": "y"}}), actor=AGENT), _policy(seen))
    assert isinstance(own.result, PlanChangeAccepted)
    person = apply(plan, _change(1, RemoveNodeOp(node_id=_id("hidden"))), _policy(seen))
    assert isinstance(person.result, PlanChangeAccepted)

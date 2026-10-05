"""`plan_engine.compact`: what leaves a plan when its finished nodes are archived.

How it can go wrong, written down before the code:
  - a node that something unfinished still depends on is removed, and the dependent can never become ready;
  - a node that is not frozen (somebody could still change it) is removed;
  - the nodes a running attempt belongs to, or the newest ones, are removed;
  - an edge to a removed node stays, and the plan no longer validates.
"""

from orbit_contracts.v3.nodes import AgentTurnNode, AgentTurnSpec
from orbit_orch.plan_engine import PlanNodeState, PlanState, compact, deterministic_id, plan_bytes


def _id(name: str) -> str:
    return deterministic_id(name, "n")


def _node(name: str, status: str = "COMPLETED", frozen: bool = True) -> PlanNodeState:
    draft = AgentTurnNode(node_id=_id(name), title=name, spec=AgentTurnSpec(goal=name * 10))
    return PlanNodeState(draft, status=status, frozen=frozen)  # type: ignore[arg-type]


def test_only_finished_frozen_nodes_that_nothing_unfinished_needs_leave() -> None:
    nodes = {
        _id("a"): _node("a"), _id("b"): _node("b"), _id("c"): _node("c", frozen=False),
        _id("d"): _node("d", "RUNNING", frozen=False), _id("e"): _node("e"), _id("f"): _node("f"),
    }
    # d (running) needs b; e needs f, both done: a, e and f leave, b stays for d, c is not frozen.
    plan = PlanState.build(5, nodes, [(_id("b"), _id("d")), (_id("f"), _id("e"))])
    result = compact(plan)
    assert result is not None
    assert sorted(result.removed) == sorted(_id(name) for name in "aef")
    assert set(result.plan.nodes) == {_id(name) for name in "bcd"}
    assert result.plan.version == 6
    assert set(result.plan.edges) == {(_id("b"), _id("d"))}, "edges to removed nodes are gone with them"
    assert result.titles == [plan.nodes[node_id].draft.title for node_id in result.removed]


def test_the_newest_nodes_and_the_held_ones_stay() -> None:
    nodes = {_id(name): _node(name) for name in "abcdef"}
    plan = PlanState.build(2, nodes)
    kept = compact(plan, held={_id("a")}, keep_recent=2)
    assert kept is not None
    assert set(kept.plan.nodes) == {_id(name) for name in "aef"}, "a is held, e and f are the newest two"
    assert compact(PlanState.build(2, {_id(name): _node(name) for name in "ab"}), keep_recent=2) is None


def test_nothing_to_compact_changes_nothing() -> None:
    plan = PlanState.build(3, {_id("a"): _node("a", "RUNNING", frozen=False), _id("b"): _node("b", "PENDING", frozen=False)})
    assert compact(plan) is None
    assert plan_bytes(plan) > 0


# ---- a dependency on a node that was compacted away -------------------------------------------------------------------

from orbit_contracts.v3 import (
    Actor,
    PlanChangeCommand,
)
from orbit_contracts.v3.plan import (
    AddEdgeOp,
    AddNodeOp,
    PlanChangeAccepted,
    RemoveNodeOp,
)
from orbit_orch.plan_engine import PlanPolicy, apply


def _change(version: int, *ops, actor: Actor | None = None) -> PlanChangeCommand:
    return PlanChangeCommand(
        command_id="01JA" + "0" * 22, task_id="task_" + "0" * 26, base_plan_version=version,
        actor=actor or Actor(kind="user", id="u"), ops=list(ops),
    )


def test_naming_an_archived_node_as_a_dependency_is_a_dependency_already_met() -> None:
    plan = PlanState.build(3, {_id("live"): _node("live", "RUNNING", frozen=False)})
    policy = PlanPolicy(archived_node_ids=frozenset({_id("old")}))
    new = AgentTurnNode(node_id="tmp:1", title="next", depends_on=[_id("old"), _id("live")], spec=AgentTurnSpec(goal="g"))
    outcome = apply(plan, _change(3, AddNodeOp(node=new)), policy)
    assert isinstance(outcome.result, PlanChangeAccepted), outcome.result
    added = outcome.result.id_map["tmp:1"]
    assert outcome.plan.nodes[added].draft.depends_on == [_id("live")], "no edge to the archived node is kept"
    assert set(outcome.plan.edges) == {(_id("live"), added)}
    # An edge from an archived node changes nothing; an edge to one, or a change to it, is refused: it is completed.
    edge = apply(outcome.plan, _change(4, AddEdgeOp.model_validate({"op": "add_edge", "from": _id("old"), "to": added})), policy)
    assert isinstance(edge.result, PlanChangeAccepted) and set(edge.plan.edges) == {(_id("live"), added)}
    towards = apply(plan, _change(3, AddEdgeOp.model_validate({"op": "add_edge", "from": _id("live"), "to": _id("old")})), policy)
    assert towards.result.status == "rejected" and towards.result.code == "FROZEN_NODE"
    removal = apply(plan, _change(3, RemoveNodeOp(node_id=_id("old"))), policy)
    assert removal.result.status == "rejected" and removal.result.code == "FROZEN_NODE"


def test_an_id_that_never_existed_is_still_refused() -> None:
    plan = PlanState.build(3, {_id("live"): _node("live", "RUNNING", frozen=False)})
    policy = PlanPolicy(archived_node_ids=frozenset({_id("old")}))
    new = AgentTurnNode(node_id="tmp:1", title="next", depends_on=[_id("ghost")], spec=AgentTurnSpec(goal="g"))
    outcome = apply(plan, _change(3, AddNodeOp(node=new)), policy)
    assert outcome.result.status == "rejected" and outcome.result.code == "SCHEMA_INVALID"

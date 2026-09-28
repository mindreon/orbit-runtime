"""Contract v3 (TaskWorkflow): models, the exported bundle, and the examples.

orbit-control generates Go types from schema/v3/contracts.json and decodes
schema/v3/examples.json in its tests, so these files are the cross-language
contract. See orbit-infra docs/architecture 03, 04, 05 and 09.
"""

import json
import re
from pathlib import Path

import pytest
from orbit_contracts.v3 import (
    DURABLE_EVENT_TYPES,
    EPHEMERAL_EVENT_TYPES,
    Event,
    PlanChangeCommand,
    PlanChangeResult,
    SendMessageInput,
    WaitSpec,
)
from orbit_contracts.v3.examples import contract_examples
from orbit_contracts.v3.export import build_bundle, build_examples
from pydantic import TypeAdapter, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema" / "v3"

TASK = "task_01J9Z3K4M5N6P7Q8R9S0T1V2W3"
NODE = "n_01J9Z3K4M5N6P7Q8R9S0T1V2W4"
ATTEMPT = "att_01J9Z3K4M5N6P7Q8R9S0T1V2W5"
COMMAND = "01J9Z3K4M5N6P7Q8R9S0T1V2W6"
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _plan_change(**overrides: object) -> dict:
    body = {
        "schema_version": "orbit.plan_change/1",
        "command_id": COMMAND,
        "task_id": TASK,
        "base_plan_version": 7,
        "actor": {"kind": "agent", "id": "coder", "attempt_id": ATTEMPT, "profile": "coder@3"},
        "ops": [
            {
                "op": "add_node",
                "node": {
                    "type": "agent_turn",
                    "node_id": "tmp:1",
                    "title": "write report",
                    "depends_on": [NODE],
                    "spec": {"goal": "write the report", "inputs": ["artifact://man_x/report.md"]},
                    "budget": {"tokens": 200000, "tool_calls": 50, "wall_s": 1800},
                },
            },
            {"op": "add_edge", "from": NODE, "to": "tmp:1"},
            {"op": "remove_node", "node_id": NODE},
        ],
        "reason": "split the work",
    }
    body.update(overrides)
    return body


def test_plan_change_command_dispatches_ops_by_kind() -> None:
    command = PlanChangeCommand.model_validate(_plan_change())
    kinds = [type(op).__name__ for op in command.ops]
    assert kinds == ["AddNodeOp", "AddEdgeOp", "RemoveNodeOp"]
    assert command.ops[0].node.spec.goal == "write the report"
    # "from" is a Python keyword; the wire name stays "from".
    dumped = command.model_dump(mode="json", exclude_none=True)
    assert dumped["ops"][1]["from"] == NODE


def test_contract_models_reject_unknown_fields_and_are_frozen() -> None:
    with pytest.raises(ValidationError):
        PlanChangeCommand.model_validate(_plan_change(extra_field=1))
    command = PlanChangeCommand.model_validate(_plan_change())
    with pytest.raises(ValidationError):
        command.base_plan_version = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "task_not-a-ulid"),
        ("command_id", "ABC"),
        ("base_plan_version", 0),
        ("ops", []),
    ],
)
def test_plan_change_command_validates_ids_and_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        PlanChangeCommand.model_validate(_plan_change(**{field: value}))


def test_agent_command_id_may_be_a_sha256_hex() -> None:
    # Agent commands use sha256(attempt_id | tool_call_id), see 04 §4.1.
    PlanChangeCommand.model_validate(_plan_change(command_id="a" * 64))


def test_plan_change_result_is_accepted_or_rejected() -> None:
    adapter = TypeAdapter(PlanChangeResult)
    accepted = adapter.validate_python(
        {"status": "accepted", "plan_version": 8, "id_map": {"tmp:1": NODE}}
    )
    rejected = adapter.validate_python(
        {
            "status": "rejected",
            "code": "VERSION_CONFLICT",
            "detail": "stale",
            "latest_plan_version": 9,
            "latest_hash": "sha256:" + "0" * 64,
        }
    )
    assert type(accepted).__name__ == "PlanChangeAccepted"
    assert type(rejected).__name__ == "PlanChangeRejected"
    with pytest.raises(ValidationError):
        adapter.validate_python({"status": "rejected", "code": "NOPE", "detail": ""})


def test_send_message_defaults_to_queue_delivery() -> None:
    message = SendMessageInput.model_validate(
        {"command_id": COMMAND, "client_message_id": COMMAND, "text": "hi"}
    )
    assert message.delivery == "queue"
    with pytest.raises(ValidationError):
        SendMessageInput.model_validate(
            {"command_id": COMMAND, "client_message_id": COMMAND, "text": ""}
        )


def test_wait_spec_needs_a_timer_or_a_wait_key() -> None:
    WaitSpec.model_validate({"wait_key": "deploy-done"})
    WaitSpec.model_validate({"timer_s": 60})
    with pytest.raises(ValidationError):
        WaitSpec.model_validate({})


def test_event_retention_is_fixed_by_type() -> None:
    # 09 §2: durable events get a seq and survive a reload; ephemeral ones do not.
    assert DURABLE_EVENT_TYPES.isdisjoint(EPHEMERAL_EVENT_TYPES)
    adapter = TypeAdapter(Event)
    base = {
        "schema": "orbit.event/3",
        "event_id": "evt_01J9Z3K4M5N6P7Q8R9S0T1V2W7",
        "task_id": TASK,
        "source": {"kind": "worker", "id": "worker-1", "attempt_id": ATTEMPT},
        "entity": {"kind": "attempt", "id": ATTEMPT, "version": 3},
        "occurred_at": "2026-09-29T00:00:00Z",
    }
    event = adapter.validate_python(
        {**base, "type": "agent.token_delta", "payload": {"attempt_id": ATTEMPT, "text": "he"}}
    )
    assert event.retention == "ephemeral"
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                **base,
                "type": "agent.token_delta",
                "retention": "durable",
                "payload": {"attempt_id": ATTEMPT, "text": "he"},
            }
        )


def test_every_event_type_is_covered_by_the_union() -> None:
    names = set(build_bundle()["x-unions"]["Event"]["discriminator"]["mapping"])
    assert names == DURABLE_EVENT_TYPES | EPHEMERAL_EVENT_TYPES


def _property_names(schema: object) -> set[str]:
    found: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                found.update(value)
            found |= _property_names(value)
    elif isinstance(schema, list):
        for item in schema:
            found |= _property_names(item)
    return found


def test_bundle_uses_snake_case_and_names_every_union() -> None:
    bundle = build_bundle()
    bad = sorted(name for name in _property_names(bundle) if not SNAKE.match(name))
    assert bad == []

    def unions(schema: object) -> list[dict]:
        out: list[dict] = []
        if isinstance(schema, dict):
            if "oneOf" in schema:
                out.append(schema)
            for value in schema.values():
                out.extend(unions(value))
        elif isinstance(schema, list):
            for item in schema:
                out.extend(unions(item))
        return out

    for union in unions(bundle):
        assert "x-go-type" in union and "discriminator" in union


def test_committed_schema_and_examples_match_the_models() -> None:
    # CI also runs the exporter and `git diff --exit-code schema`.
    bundle = json.loads((SCHEMA_DIR / "contracts.json").read_text())
    examples = json.loads((SCHEMA_DIR / "examples.json").read_text())
    assert bundle == build_bundle()
    assert examples == build_examples()


def test_examples_cover_every_union_member_and_round_trip() -> None:
    examples = build_examples()
    bundle = build_bundle()
    for name, union in bundle["x-unions"].items():
        members = set(union["discriminator"]["mapping"])
        seen = {item[union["discriminator"]["propertyName"]] for item in examples[name]}
        assert seen == members, name
    for name, (adapter, items) in contract_examples().items():
        for item in items:
            dumped = adapter.dump_python(item, mode="json", exclude_none=True, by_alias=True)
            again = adapter.validate_python(dumped)
            assert (
                adapter.dump_python(again, mode="json", exclude_none=True, by_alias=True) == dumped
            ), name

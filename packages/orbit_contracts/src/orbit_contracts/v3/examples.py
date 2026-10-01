"""Example instances for every top-level model and every union member.

Exported to schema/v3/examples.json. orbit-control decodes each one into the
generated Go type and encodes it back, which keeps the two languages in step.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter

from orbit_contracts.v3 import events as ev
from orbit_contracts.v3 import messages as msg
from orbit_contracts.v3 import nodes as nd
from orbit_contracts.v3 import plan as pl
from orbit_contracts.v3 import views as vw
from orbit_contracts.v3.common import Actor, Budget, ConnectorSnapshot, Failure, Usage

TASK = "task_01J9Z3K4M5N6P7Q8R9S0T1V2W3"
NODE = "n_01J9Z3K4M5N6P7Q8R9S0T1V2W4"
NODE_2 = "n_01J9Z3K4M5N6P7Q8R9S0T1V2X4"
ATTEMPT = "att_01J9Z3K4M5N6P7Q8R9S0T1V2W5"
COMMAND = "01J9Z3K4M5N6P7Q8R9S0T1V2W6"
AGENT_COMMAND = "3" * 64
APPROVAL = "apr_01J9Z3K4M5N6P7Q8R9S0T1V2W8"
MANIFEST = "man_01J9Z3K4M5N6P7Q8R9S0T1V2W9"
CHECKPOINT = "ckpt_01J9Z3K4M5N6P7Q8R9S0T1V2WA"
REF = "sha256:" + "ab" * 32
WHEN = datetime(2026, 9, 29, 8, 30, tzinfo=UTC)

USER = Actor(kind="user", id="u_1")
AGENT = Actor(kind="agent", id="coder", attempt_id=ATTEMPT, profile="coder@3")
BUDGET = Budget(tokens=200_000, tool_calls=50, wall_s=1800, cost_usd_micros=2_000_000)
USAGE = Usage(tokens_in=1200, tokens_out=300, tool_calls=4, wall_s=42, cost_usd_micros=5100)
SUBJECT = msg.ApprovalSubject(kind="tool_call", digest=REF, summary="rm -rf build/", risk="high")


def _nodes() -> list[Any]:
    common: dict[str, Any] = {"title": "step", "depends_on": [NODE]}
    return [
        nd.AgentTurnNode(
            node_id="tmp:1",
            spec=nd.AgentTurnSpec(goal="write the report", inputs=["artifact://man_x/a.md"]),
            budget=BUDGET,
            completion_contract=nd.CompletionContract(
                output_schema_ref="schema://report/1",
                required_artifacts=[
                    nd.ArtifactRequirement(name="report", media_type="text/markdown")
                ],
                verifications=[nd.Verification(kind="command", spec={"command": "pytest -q"})],
            ),
            **common,
        ),
        nd.SopStageNode(node_id="tmp:2", spec=nd.SopStageSpec(sop="release@2"), **common),
        nd.TeamStageNode(
            node_id="tmp:3",
            spec=nd.TeamStageSpec(objective="review", member_profiles=["reviewer@2"]),
            **common,
        ),
        nd.ApprovalNode(node_id="tmp:4", spec=nd.ApprovalSpec(summary="ship it?"), **common),
        nd.WaitNode(
            node_id="tmp:5", spec=nd.WaitSpec(timer_s=600), workspace_access="none", **common
        ),
        nd.CheckpointNode(
            node_id=NODE_2,
            spec=nd.CheckpointSpec(label="before deploy"),
            retry=nd.RetryPolicy(max_attempts=2, repair_profile="coder-strong@2"),
            timeout=nd.NodeTimeout(attempt_s=600),
            **common,
        ),
    ]


def _ops() -> list[Any]:
    return [
        pl.AddNodeOp(node=_nodes()[0]),
        pl.UpdateNodeOp(node_id=NODE, patch=pl.NodePatch(title="renamed", spec={"goal": "x"})),
        pl.RemoveNodeOp(node_id="tmp:9"),
        pl.AddEdgeOp(from_node=NODE, to=NODE_2),
        pl.RemoveEdgeOp(from_node=NODE, to=NODE_2),
        pl.DeclareBlockedOp(node_id=NODE, reason="needs credentials"),
    ]


def _event(cls: type, payload: Any, *, durable: bool, **extra: Any) -> Any:
    position = {"seq": 42} if durable else {"after_seq": 41}
    return cls(
        event_id="evt_01J9Z3K4M5N6P7Q8R9S0T1V2W7",
        task_id=TASK,
        source=ev.EventSource(kind="worker", id="worker-1", attempt_id=ATTEMPT),
        entity=ev.EntityRef(kind="attempt", id=ATTEMPT, version=3),
        occurred_at=WHEN,
        payload=payload,
        **position,
        **extra,
    )


def _events() -> list[Any]:
    closed = ev.TaskClosedPayload(reason="done")
    tool = {"attempt_id": ATTEMPT, "tool_call_id": "call-0", "tool_name": "shell"}
    durable = [
        (
            ev.TaskCreatedEvent,
            ev.TaskCreatedPayload(title="t", goal="g", profile="coder@3", created_by=USER),
        ),
        (
            ev.TaskStatusChangedEvent,
            ev.TaskStatusChangedPayload(from_status="PLANNING", to_status="RUNNING"),
        ),
        (ev.TaskCompletedEvent, closed),
        (
            ev.TaskFailedEvent,
            ev.TaskClosedPayload(
                failure=Failure(failure_class="model", retryable=False, message="x")
            ),
        ),
        (ev.TaskCancelledEvent, closed),
        (
            ev.PlanVersionCommittedEvent,
            ev.PlanVersionCommittedPayload(
                plan_version=2, parent_version=1, hash=REF, change_command_id=COMMAND, actor=AGENT
            ),
        ),
        (
            ev.PlanChangeRejectedEvent,
            ev.PlanChangeRejectedPayload(
                command_id=AGENT_COMMAND, code="CYCLE", detail="n_a -> n_a", actor=AGENT
            ),
        ),
        (
            ev.NodeStatusChangedEvent,
            ev.NodeStatusChangedPayload(
                node_id=NODE, from_status="RUNNING", to_status="AWAITING_INPUT"
            ),
        ),
        (
            ev.AttemptStartedEvent,
            ev.AttemptStartedPayload(
                node_id=NODE, attempt_id=ATTEMPT, attempt_no=1, profile="coder@3", config_version=1
            ),
        ),
        (
            ev.AttemptResumedEvent,
            ev.AttemptResumedPayload(
                node_id=NODE, attempt_id=ATTEMPT, attempt_no=1, activity_attempt=2
            ),
        ),
        (
            ev.AttemptParkedEvent,
            ev.AttemptParkedPayload(
                node_id=NODE, attempt_id=ATTEMPT, reason="input", question="Which branch?"
            ),
        ),
        (
            ev.AttemptFinishedEvent,
            ev.AttemptFinishedPayload(
                node_id=NODE, attempt_id=ATTEMPT, outcome="completed", usage=USAGE
            ),
        ),
        (
            ev.ApprovalRequestedEvent,
            ev.ApprovalRequestedPayload(
                approval_id=APPROVAL, attempt_id=ATTEMPT, tool_call_id="call-1", subject=SUBJECT
            ),
        ),
        (
            ev.ApprovalDecidedEvent,
            ev.ApprovalDecidedPayload(approval_id=APPROVAL, status="APPROVED", decided_by="u_1"),
        ),
        (
            ev.CheckpointCommittedEvent,
            ev.CheckpointCommittedPayload(
                checkpoint_id=CHECKPOINT, attempt_id=ATTEMPT, kind="agent_state", blob_ref=REF
            ),
        ),
        (
            ev.ManifestCreatedEvent,
            ev.ManifestCreatedPayload(
                manifest_id=MANIFEST, attempt_id=ATTEMPT, entry_count=2, manifest_hash=REF
            ),
        ),
        (
            ev.UserMessageEvent,
            ev.UserMessagePayload(
                message_seq=3, client_message_id=COMMAND, text="use main", delivery="interrupt"
            ),
        ),
        (ev.AgentFinalMessageEvent, ev.AgentFinalMessagePayload(attempt_id=ATTEMPT, text="done")),
        (ev.BudgetExhaustedEvent, ev.BudgetExhaustedPayload(scope="exploration", node_id=NODE)),
        (
            ev.BudgetGrantedEvent,
            ev.BudgetGrantedPayload(command_id=COMMAND, delta=Budget(tokens=50_000)),
        ),
        (
            ev.ProfileSwitchedEvent,
            ev.ProfileSwitchedPayload(
                node_id=NODE,
                from_profile="coder@3",
                to_profile="coder-strong@2",
                reason="two failed checks",
            ),
        ),
        (
            ev.TaskConfigChangedEvent,
            ev.TaskConfigChangedPayload(
                config_version=2,
                expert="writer@2",
                skills=["handle/skill-a"],
                connector_ids=["mcp_docs"],
                mode="plan",
            ),
        ),
        (
            ev.ToolCallFinishedEvent,
            ev.ToolCallFinishedPayload(**tool, state="success", result_preview="ok"),
        ),
        (ev.UsageRecordedEvent, ev.UsagePayload(attempt_id=ATTEMPT, usage=USAGE)),
    ]
    ephemeral = [
        (ev.TokenDeltaEvent, ev.TextDeltaPayload(attempt_id=ATTEMPT, text="Hel")),
        (ev.ThinkingDeltaEvent, ev.TextDeltaPayload(attempt_id=ATTEMPT, text="hmm")),
        (ev.ToolCallStartedEvent, ev.ToolCallStartedPayload(**tool, args_preview="ls")),
        (
            ev.ToolProgressEvent,
            ev.ToolProgressPayload(attempt_id=ATTEMPT, tool_call_id="call-0", text="50%"),
        ),
        (ev.UsageDeltaEvent, ev.UsagePayload(attempt_id=ATTEMPT, usage=USAGE)),
        (
            ev.ExecOutputEvent,
            ev.ExecOutputPayload(attempt_id=ATTEMPT, stream="stderr", text="warn"),
        ),
        (ev.HeartbeatEvent, ev.HeartbeatPayload()),
    ]
    return [_event(c, p, durable=True) for c, p in durable] + [
        _event(c, p, durable=False) for c, p in ephemeral
    ]


def _models() -> dict[str, list[Any]]:
    message = msg.InboxMessage(message_seq=3, client_message_id=COMMAND, text="use main")
    return {
        "TaskWorkflowInput": [
            vw.TaskWorkflowInput(
                task_id=TASK,
                tenant_id="default",
                created_by=USER,
                title="t",
                goal="g",
                profile="coder@3",
                sop="release@2",
                node_type_registry_version=1,
                budgets=BUDGET,
            )
        ],
        "PlanChangeCommand": [
            pl.PlanChangeCommand(
                command_id=AGENT_COMMAND,
                task_id=TASK,
                base_plan_version=7,
                actor=AGENT,
                ops=_ops(),
                reason="split the work",
            )
        ],
        "CompletionProposal": [
            msg.CompletionProposal(
                command_id=AGENT_COMMAND,
                node_id=NODE,
                attempt_id=ATTEMPT,
                output={"pages": 3},
                artifact_manifest_id=MANIFEST,
                checkpoint_ref=REF,
                claimed_side_effects=["side_effect:x"],
            )
        ],
        "SendMessageInput": [
            msg.SendMessageInput(
                command_id=COMMAND,
                client_message_id=COMMAND,
                text="use main",
                delivery="interrupt",
                attachments=[
                    msg.Attachment(
                        uri="artifact://man_x/a.png", name="a.png", media_type="image/png"
                    )
                ],
            )
        ],
        "SendMessageResult": [msg.SendMessageResult(message_seq=3)],
        "DecideApprovalInput": [
            msg.DecideApprovalInput(command_id=COMMAND, approval_id=APPROVAL, decision="approve")
        ],
        "DecideApprovalResult": [msg.DecideApprovalResult(approval_id=APPROVAL, status="APPROVED")],
        "TaskControlInput": [
            msg.TaskControlInput(command_id=COMMAND, action="pause", reason="lunch")
        ],
        "TaskControlResult": [msg.TaskControlResult(status="PAUSED")],
        "GrantBudgetInput": [msg.GrantBudgetInput(command_id=COMMAND, delta=Budget(tokens=50_000))],
        "GrantBudgetResult": [msg.GrantBudgetResult(budgets=BUDGET)],
        "RequestProfileSwitchInput": [
            msg.RequestProfileSwitchInput(
                command_id=COMMAND, node_id=NODE, to_profile="coder-strong@2", reason="stuck"
            )
        ],
        "UpdateTaskConfigInput": [
            msg.UpdateTaskConfigInput(
                command_id=COMMAND,
                base_config_version=1,
                expert="writer@2",
                skills=["handle/skill-a"],
                connectors=[ConnectorSnapshot(id="mcp_docs", name="Docs", command="orbit-mcp-docs")],
                mode="plan",
            )
        ],
        "UpdateTaskConfigResult": [msg.UpdateTaskConfigResult(config_version=2)],
        "RequestProfileSwitchResult": [
            msg.RequestProfileSwitchResult(effective_attempt_no=2, needs_approval=True)
        ],
        "ExternalEventSignal": [
            msg.ExternalEventSignal(wait_key="deploy-done", payload={"ok": True})
        ],
        "AttemptFinishedSignal": [
            msg.AttemptFinishedSignal(
                attempt_workflow_id=f"attempt/{TASK}/{NODE}/1",
                node_id=NODE,
                attempt_no=1,
                attempt_id=ATTEMPT,
                outcome="completed",
                result=msg.AttemptResult(
                    handover_summary="done", manifest_id=MANIFEST, checkpoint_ref=REF, usage=USAGE
                ),
            ),
            msg.AttemptFinishedSignal(
                attempt_workflow_id=f"attempt/{TASK}/{NODE}/2",
                node_id=NODE,
                attempt_no=2,
                attempt_id=ATTEMPT,
                outcome="failed",
                failure=Failure(failure_class="transient", retryable=True, message="timeout"),
            ),
        ],
        "AttemptParkedSignal": [
            msg.AttemptParkedSignal(
                node_id=NODE,
                attempt_no=1,
                attempt_id=ATTEMPT,
                reason="approval",
                approvals=[msg.ParkedToolCall(tool_call_id="call-1", subject=SUBJECT)],
            )
        ],
        "ApprovalDecidedSignal": [
            msg.ApprovalDecidedSignal(
                approval_id=APPROVAL, tool_call_id="call-1", decision="reject", comment="no"
            )
        ],
        "DeliverMessagesSignal": [msg.DeliverMessagesSignal(messages=[message])],
        "TaskView": [
            vw.TaskView(
                task_id=TASK,
                status="WAITING",
                plan_version=2,
                pending_approvals=[APPROVAL],
                budgets=BUDGET,
                usage=USAGE,
            )
        ],
        "PlanView": [
            vw.PlanView(
                plan_version=2,
                hash=REF,
                nodes=[
                    vw.NodeView(
                        node_id=NODE,
                        type="agent_turn",
                        title="explore",
                        status="COMPLETED",
                        workspace_access="write",
                        owner_profile="coder@3",
                        frozen=True,
                        attempt_count=1,
                    )
                ],
                edges=[vw.PlanEdge(from_node=NODE, to=NODE_2)],
            )
        ],
        "InboxView": [vw.InboxView(messages=[message])],
    }


def contract_examples() -> dict[str, tuple[TypeAdapter[Any], list[Any]]]:
    from orbit_contracts.v3.export import MODELS

    by_name = {model.__name__: model for model in MODELS}
    out: dict[str, tuple[TypeAdapter[Any], list[Any]]] = {
        name: (TypeAdapter(by_name[name]), items) for name, items in _models().items()
    }
    out["TaskNodeDraft"] = (TypeAdapter(nd.TaskNodeDraft), _nodes())
    out["PlanOp"] = (TypeAdapter(pl.PlanOp), _ops())
    out["PlanChangeResult"] = (
        TypeAdapter(pl.PlanChangeResult),
        [
            pl.PlanChangeAccepted(plan_version=8, id_map={"tmp:1": NODE_2}),
            pl.PlanChangeRejected(
                code="VERSION_CONFLICT", detail="stale", latest_plan_version=9, latest_hash=REF
            ),
        ],
    )
    out["CompletionResult"] = (
        TypeAdapter(msg.CompletionResult),
        [
            msg.CompletionAccepted(),
            msg.CompletionRejected(code="STALE_ATTEMPT", detail="attempt 1 was replaced"),
        ],
    )
    out["Event"] = (TypeAdapter(ev.Event), _events())
    missing = set(by_name) - set(out)
    if missing:
        raise ValueError(f"no examples for {sorted(missing)}")
    return out

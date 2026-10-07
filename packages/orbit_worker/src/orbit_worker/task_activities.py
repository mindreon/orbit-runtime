"""Activities used by TaskWorkflow and AttemptWorkflow."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import structlog
from agentscope.skill import Skill
from orbit_contracts.v3 import Budget, Policy
from orbit_contracts.v3.messages import ask_user_park
from orbit_orch.plan_engine import deterministic_id
from temporalio import activity

from orbit_worker.activity_input import CheckpointCommitInput, parse_input
from orbit_worker.agent_config import permission_preset_for, permission_spec_for, with_task_config
from orbit_worker.budget_middleware import BudgetMeter, model_price
from orbit_worker.checkpoint_state import SESSION_SEQ_BASE
from orbit_worker.language import detect_language
from orbit_worker.manifest_record import record_manifest
from orbit_worker.planning_tools import TemporalPlanPort, is_leader_work_node, task_workflow_id
from orbit_worker.policy_middleware import exploration_exhausted
from orbit_worker.sandbox import SandboxSession, bind_sandbox, unbind_sandbox
from orbit_worker.settings import MockSettings, WorkerSettings
from orbit_worker.skills import (
    ExpertScopedSource,
    bundle_skill_ids,
    get_skill_source,
    staged_skills,
)
from orbit_worker.sop import SopRegistry, UnknownSopError
from orbit_worker.sop_agents import RunScope, run_one_try
from orbit_worker.task_store import TaskStore
from orbit_worker.task_stream import (
    TaskStreamContext,
    TeamTurn,
    event_attempt_id,
    streaming_for,
    team_stamp,
)
from orbit_worker.verify_activities import VERIFY_AGENT_ACTIVITIES, VERIFY_IO_ACTIVITIES
from orbit_worker.worker_events import publish_attempt_event
from orbit_worker.workspace import WorkspaceAdapter

_store: TaskStore | None = None
_sops: SopRegistry | None = None
_workspace: WorkspaceAdapter | None = None


def set_task_store(store: TaskStore) -> None:
    global _store, _sops
    _store = store
    _sops = SopRegistry(store)


def get_task_store() -> TaskStore:
    if _store is None:
        raise RuntimeError("task store is not installed")
    return _store


def set_workspace_adapter(adapter: WorkspaceAdapter) -> None:
    global _workspace
    _workspace = adapter


def get_workspace_adapter() -> WorkspaceAdapter | None:
    return _workspace


def _ref(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _session_checkpoint_ref(tenant_id: str, attempt_id: str) -> str | None:
    """The reference of the newest `agent_state` checkpoint the attempt wrote: what the workflow records in history and
    commits (08 §3). None when nothing durable was written (no database, or the turn never saved)."""
    return await get_task_store().latest_checkpoint_ref(
        tenant_id=tenant_id, attempt_id=attempt_id, kind="agent_state", min_seq=SESSION_SEQ_BASE
    )


def _failed(error: str, failure_class: str, retryable: bool, **fields: Any) -> dict[str, Any]:
    """A failed result. It says what kind of failure it was and whether trying again can help, which is what decides
    whether the workflow retries the node or gives it to a person (04 §2)."""
    return {"status": "failed", "error": error, "failure_class": failure_class, "retryable": retryable, **fields}


_active_meter: contextvars.ContextVar[BudgetMeter | None] = contextvars.ContextVar("orbit_active_meter", default=None)


def _beat() -> None:
    """A heartbeat. An agent turn puts what it has spent in it: a retry of the activity (a worker that crashed) reads it
    back and carries on from there, instead of spending the attempt's reserved budget again (05 §4)."""
    meter = _active_meter.get()
    if meter is None:
        activity.heartbeat()
    else:
        activity.heartbeat(meter.snapshot())


async def _heartbeat() -> None:
    """Heartbeats for the whole activity. Cancellation reaches the turn on a heartbeat, so this sets how soon."""
    interval = WorkerSettings().heartbeat_throttle_s
    while True:
        _beat()
        await asyncio.sleep(interval)


async def _mock_delay(setting: str = "turn_delay_ms") -> None:
    """Mock-model latency. It heartbeats, so a cancel lands mid-turn like it does on a real model call."""
    settings = MockSettings()
    if not settings.mock:
        return
    delay_ms = getattr(settings, setting)
    remaining = (settings.turn_delay_ms if delay_ms is None else delay_ms) / 1000
    while remaining > 0:
        _beat()
        step = min(0.25, remaining)
        await asyncio.sleep(step)
        remaining -= step


async def _publish_worker_event(
    payload: dict[str, Any], event_type: str, body: dict[str, Any], seed: str, shown_attempt_id: str | None = None
) -> None:
    await publish_attempt_event(
        get_task_store(),
        tenant_id=str(payload.get("tenant_id", "default")),
        task_id=str(payload["task_id"]),
        attempt_id=str(payload["attempt_id"]),
        event_type=event_type,
        body=body,
        seed=seed,
        shown_attempt_id=shown_attempt_id,
    )


async def _announce_resumed(payload: dict[str, Any]) -> None:
    """attempt.resumed: this activity continues an attempt after a retry, an approval or a user reply."""
    activity_attempt = activity.info().attempt
    if activity_attempt <= 1 and not (payload.get("approval") or payload.get("external") or payload.get("team_results")):
        return
    # A member of a team stage runs under an attempt id of its own, but it is the stage's attempt that is parked and resumed.
    team = payload.get("team") or {}
    stage_attempt = str(team.get("stage_attempt_id") or payload["attempt_id"])
    role = str(team.get("role") or "")
    await _publish_worker_event(
        {**payload, "attempt_id": stage_attempt},
        "attempt.resumed",
        {
            "node_id": str(payload["node_id"]),
            "attempt_id": stage_attempt,
            "attempt_no": int(payload.get("attempt_no", 1)),
            "activity_attempt": activity_attempt,
            "state_version": int(payload.get("state_version", 0)),
        },
        f"resumed:{role}:{activity_attempt}:{payload.get('state_version', 0)}" if role else f"resumed:{activity_attempt}:{payload.get('state_version', 0)}",
    )


def _park_for_retry(payload: dict[str, Any], unknown: list[dict[str, Any]]) -> dict[str, Any]:
    """The worker died while side-effecting calls were running: ask a person before any of them runs again (A12)."""
    attempt_id = str(payload["attempt_id"])
    return {
        "status": "parked_approval",
        "checkpoint_ref": _ref(attempt_id),
        "question": "an earlier run may already have made these calls",
        "approval_request_id": f"retry:{attempt_id}",
        "retry_calls": unknown,
        "session_id": str(payload.get("session_id", "")),
        "state_version": int(payload.get("state_version", 0)),
        "approvals": [
            {
                "tool_call_id": call["key"].split(":", 1)[1],
                "subject": {
                    "kind": "non_idempotent_retry",
                    "digest": _ref(call["key"]),
                    "summary": f"{call['tool']} may already have run; run it again?",
                    "risk": "high",
                },
            }
            for call in unknown
        ],
    }


def _bind_log_context(payload: dict[str, Any]) -> None:
    """Every log line of this activity carries the task it belongs to."""

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        tenant_id=str(payload.get("tenant_id", "default")),
        task_id=str(payload.get("task_id", "")),
        attempt_id=str(payload.get("attempt_id", "")),
    )


async def _sandbox_entries(tenant_id: str, task_id: str, sandbox: SandboxSession | None) -> list[dict[str, Any]]:
    """The manifest entries of an attempt: the files in its sandbox workspace. What the agent said is not one."""
    if sandbox is None:
        return []
    store = get_task_store()
    return [
        {
            "name": file.name,
            "media_type": file.media_type,
            "size_bytes": len(file.payload),
            "blob_ref": await store.put_artifact_blob(tenant_id=tenant_id, task_id=task_id, payload=file.payload),
        }
        for file in await sandbox.files()
    ]


async def _language_of(runtime: Any, payload: dict[str, Any], messages: str, session_id: str) -> str:
    """The language the agent of this turn writes in. An attempt that carries on an earlier session (a review of the leader's
    work, a follow-up, a retry) keeps the language that session was set to: its own prompt is mostly the workflow's wording, not
    the user's. Any other starts from what it was given, the goal and the user's messages, and says so only when it is sure."""
    previous = str(payload.get("continue_from") or "")
    inherited = await runtime.session_language(previous) if previous and not session_id else ""
    return str(inherited) or detect_language(str(payload.get("goal", "")), messages)


def _team_turn(raw: Any) -> TeamTurn | None:
    """Which agent of a team stage the turn is, from the workflow's payload; None for any other turn."""
    if not isinstance(raw, dict):
        return None
    return TeamTurn(
        role=str(raw["role"]),
        leader=bool(raw.get("leader")),
        leader_role=str(raw.get("leader_role") or ""),
        leader_label=str(raw.get("leader_label") or ""),
        label=str(raw.get("label") or ""),
        stage_attempt_id=str(raw.get("stage_attempt_id") or ""),
        members=tuple(
            (str(item[0]), str(item[1]), str(item[2]) if len(item) > 2 else "")
            for item in raw.get("members") or []
        ),
    )


MAX_TEAM_FILES = 100


async def _offer_team_files(sandbox: SandboxSession, tenant_id: str, files: list[dict[str, Any]]) -> None:
    """What the members left, for the leader to read and merge: `.team/<role>/<name>` in its workspace (07 §8)."""
    by_role: dict[str, list[tuple[str, bytes]]] = {}
    for item in files[:MAX_TEAM_FILES]:
        try:
            payload = await get_task_store().get_artifact_blob(tenant_id=tenant_id, blob_ref=str(item["blob_ref"]))
        except (OSError, ValueError, KeyError):
            structlog.get_logger(__name__).warning("a member's file could not be read", file=str(item.get("name")))
            continue
        by_role.setdefault(str(item["role"]), []).append((str(item["name"]), payload))
    for role, items in by_role.items():
        sandbox.offer_team_files(role, items)


def _skills_in_sandbox(skills: tuple[Skill, ...], sandbox: SandboxSession | None) -> tuple[Skill, ...]:
    """The skills as the agent will see them. Its tools run on the workspace, so a skill's files have to be there; the
    place it is told is that one, and they are copied in when the workspace is first used."""
    if sandbox is None:
        return skills
    return tuple(
        dataclasses.replace(skill, dir=sandbox.offer_skill(Path(skill.dir).name, Path(skill.dir))) for skill in skills
    )


def _output_schema(ref: Any) -> tuple[dict[str, Any] | None, str]:
    """The JSON Schema `ref` names (`schema://name/N`, read as verification reads it), or why there is none. No ref asks for
    no structured output."""
    from orbit_worker.verify import SchemaInvalidError
    from orbit_worker.verify_activities import get_schema_registry

    if not ref:
        return None, ""
    registry = get_schema_registry()
    if registry is None:
        return None, f"the output schema {ref} cannot be resolved: no schema directory is configured (ORBIT_OUTPUT_SCHEMA_DIR)"
    try:
        schema = registry.get(str(ref))
    except SchemaInvalidError as exc:
        return None, f"the output schema {ref} is not a valid JSON Schema: {exc}"
    if schema is None:
        return None, f"the output schema {ref} is not registered"
    return schema, ""


CONTINUE_PROMPT = "Continue the task from where you stopped."
HANDOVER_CHARS = 2000


MAX_HANDOVER_EFFECTS = 50


def turn_prompt(payload: dict[str, Any], messages: str, carried: bool | None = None) -> str:
    """What the agent is told at the start of an attempt's turn.

    A fresh node, and a follow-up (whose goal is the user's own message), are given the goal and the messages. An attempt
    that carries on the session of an earlier attempt of the same node (a retry, or the replacement of an interrupted or
    stopped one: `continue_from` set and `attempt_no` past the first) already has the goal and the work so far in that
    session, so the goal is not said again, or the agent would start the task over. It is given only what is new: the
    user's messages, and for a retry why the attempt before it was rejected; with neither, a plain request to go on.

    `carried` is what the runtime says when it opened the session: whether state was really carried. An attempt that was to
    carry on a session that was gone or unreadable starts empty, and is told its task like a fresh node. Left None (the
    session was opened by an earlier turn of this attempt) it is taken from the payload."""
    reason = str(payload.get("retry_reason") or "").strip()
    rejected = f"Your previous attempt was rejected: {reason}\nFix what is wrong, then finish the task." if reason else ""
    if carried is None:
        carried = bool(payload.get("continue_from")) and int(payload.get("attempt_no", 1)) > 1
    if not carried:
        prompt = str(payload.get("goal", ""))
        handover = str(payload.get("handover") or "").strip()
        if handover:
            # The node was switched to another agent, whose session is not carried: it is told where the work stands.
            prompt += f"\n\nThis work was handed over to you. Where the agent before you left off:\n{handover}"
        done = [str(line) for line in (payload.get("side_effects") or [])][:MAX_HANDOVER_EFFECTS]
        if done:
            # What the ledger says already happened: the new agent must not take it for undone and do it again.
            prompt += "\n\nThese actions were already carried out and succeeded; do not repeat them:\n" + "\n".join(
                f"- {line}" for line in done
            )
        if rejected:
            # A retry whose session is gone starts from the goal, and is still told why it is a retry.
            prompt += f"\n\n{rejected}"
        return prompt + (f"\n\nUser messages:\n{messages}" if messages else "")
    parts = [part for part in (messages, rejected) if part]
    return "\n\n".join(parts) if parts else CONTINUE_PROMPT


@activity.defn(name="agent_turn")
async def agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one AgentScope turn and return a small, durable handover result."""

    _bind_log_context(payload)
    # What this activity may spend, counted from now (05 §4). No budget in the input means no limit. A retry of the
    # activity carries on from what the run before it had spent (its last heartbeat), before it heartbeats anything new.
    meter = BudgetMeter(limit=Budget.model_validate(payload.get("budget") or {}))
    saved = next((item for item in activity.info().heartbeat_details if isinstance(item, dict) and "tokens_in" in item), None)
    if saved is not None and activity.info().attempt > 1:
        meter.restore(saved)
    meter.on_change = _beat
    _active_meter.set(meter)
    heartbeat = asyncio.create_task(_heartbeat())
    sandbox: SandboxSession | None = None
    sandbox_token = None
    outcome: dict[str, Any] = _failed("agent turn did not return", "transient", True)
    skill_stage = contextlib.AsyncExitStack()
    try:
        from orbit_contracts.models import OpenSessionInput, RunTurnInput

        from orbit_worker.runtime_holder import get_runtime

        task_id = str(payload["task_id"])
        attempt_id = str(payload["attempt_id"])
        await _announce_resumed(payload)
        tenant_id = str(payload.get("tenant_id", "default"))
        team = _team_turn(payload.get("team"))
        member = team is not None and not team.leader
        output_schema, schema_error = _output_schema(payload.get("output_schema_ref"))
        if schema_error:
            # Trying again cannot make the schema appear: the node is blocked and a person decides (not an endless retry).
            outcome = _failed(schema_error, "policy", False)
            return outcome
        retry_calls = payload.get("retry_calls")
        if retry_calls:
            if (payload.get("approval") or {}).get("decision") != "approve":
                outcome = _failed("running a call with an unknown outcome again was rejected", "policy", False)
                return outcome
            await get_task_store().approve_replay(
                tenant_id=tenant_id, keys=[call["key"] for call in retry_calls]
            )
        else:
            unknown = await get_task_store().unknown_side_effects(
                tenant_id=tenant_id, attempt_id=attempt_id
            )
            if unknown:
                outcome = _park_for_retry(payload, unknown)
                return outcome
        if _workspace is not None and not (member and (payload.get("team") or {}).get("workspace") == "none"):
            # A member of a team stage, and a node that declares `read`, work in a copy of the task's workspace that is thrown away
            # (07 §8, 08 §1): what they leave comes back as artifacts, never as the head.
            sandbox = SandboxSession(
                _workspace, get_task_store(), tenant_id=tenant_id, task_id=task_id, holder=attempt_id,
                replica=member or payload.get("workspace_access") == "read",
            )
            sandbox_token = bind_sandbox(sandbox)
            await _offer_team_files(sandbox, tenant_id, (payload.get("team") or {}).get("files") or [])
            if payload.get("workspace_access") == "write":
                # A node that asked for the workspace holds it from the start (and so always has a snapshot to check).
                await sandbox.lease()
        runtime = get_runtime()
        await _mock_delay()
        session_id = str(payload.get("session_id", ""))
        state_version = int(payload.get("state_version", 0))
        messages = "\n".join(str(item.get("text", "")) for item in payload.get("messages", []))
        if payload.get("switched_from") and "side_effects" not in payload:
            # The first attempt after a switch of profile starts without the old session: the ledger tells it what the
            # node's earlier attempts already did.
            payload = {**payload, "side_effects": await get_task_store().succeeded_side_effects(
                tenant_id=tenant_id,
                attempt_ids=[
                    deterministic_id(f"{task_id}:{payload['node_id']}:{number}", "att")
                    for number in range(1, int(payload.get("attempt_no", 1)))
                ],
                limit=MAX_HANDOVER_EFFECTS,
            )}
        prompt = turn_prompt(payload, messages)
        approval = payload.get("approval")
        external = payload.get("external")
        task_config = payload.get("config")
        base_config = await get_task_store().agent_config(
            tenant_id=tenant_id, profile_ref=str(payload.get("profile", ""))
        )
        agent_config = with_task_config(base_config, task_config, str(payload.get("profile", "")))
        model = runtime.model_config_for(agent_config)
        meter.set_price(
            model_price(
                agent_config.price_input_per_mtok
                if agent_config.price_input_per_mtok is not None
                else model.price_input_per_mtok,
                agent_config.price_output_per_mtok
                if agent_config.price_output_per_mtok is not None
                else model.price_output_per_mtok,
            )
        )
        skill_source = get_skill_source()
        skills = (
            await skill_stage.enter_async_context(
                staged_skills(
                    ExpertScopedSource(skill_source, tenant_id, agent_config.bundle_ref),
                    (*agent_config.skills, *bundle_skill_ids(agent_config.bundle_skills)),
                )
            )
            if skill_source is not None
            else ()
        )
        skills = _skills_in_sandbox(skills, sandbox)
        stream = TaskStreamContext(
            tenant_id=str(payload.get("tenant_id", "default")),
            task_id=task_id,
            attempt_id=attempt_id,
            activity_attempt=activity.info().attempt,
            node_id=str(payload["node_id"]),
            profile=str(payload.get("profile", "")),
            task_policy=Policy.model_validate(payload.get("policy") or {}),
            agent=agent_config,
            skills=skills,
            meter=meter,
            output_schema=output_schema,
            team=team,
            language=await _language_of(runtime, payload, messages, session_id),
        )
        if agent_config.team is not None and team is None and await is_leader_work_node(
            TemporalPlanPort(task_workflow_id), stream
        ):
            # A task the leader gave itself: it does the work instead of planning it again.
            agent_config = with_task_config(base_config, task_config, str(payload.get("profile", "")), own_task=True)
            stream = dataclasses.replace(stream, agent=agent_config)
        with streaming_for(stream):
            if not session_id:
                opened = await runtime.open_session(
                    OpenSessionInput(
                        room_id=task_id,
                        turn_id=f"{attempt_id}:open",
                        permission_preset=permission_preset_for(task_config),
                        permissions=permission_spec_for(task_config),
                        continue_from=str(payload.get("continue_from") or ""),
                        allow_rules=[dict(rule) for rule in payload.get("allow_rules") or []],
                    )
                )
                session_id = opened.session_id
                state_version = opened.state_version
                # Told only what is new when state was really carried, and the goal when it was not (the earlier session was
                # gone or unreadable). A follow-up (the node's first attempt) is always given its goal, the user's message.
                prompt = turn_prompt(payload, messages, opened.carried and int(payload.get("attempt_no", 1)) > 1)
            if member and not (approval or external or payload.get("team_results")):
                # A member is given the task of each assignment, on whatever session it carries on: the stage's attempt number
                # is not its own, and a plain request to go on would lose the task.
                prompt = str(payload.get("goal", ""))
            if payload.get("team_results"):
                from orbit_contracts.models import DeliverToolResultsInput, ExternalResult

                # The answers to every external call the agent of a team stage is parked on (07 §5).
                result = await runtime.deliver_tool_results(
                    DeliverToolResultsInput(
                        room_id=task_id,
                        session_id=session_id,
                        turn_id=f"{attempt_id}:team:{state_version}",
                        state_version=state_version,
                        results=[ExternalResult.model_validate(item) for item in payload["team_results"]],
                    )
                )
            elif external:
                from orbit_contracts.models import DeliverToolResultInput

                # The user's reply is the answer to the ask_user call the attempt parked on (04 §2).
                result = await runtime.deliver_tool_result(
                    DeliverToolResultInput(
                        room_id=task_id,
                        session_id=session_id,
                        turn_id=f"{attempt_id}:answer:{state_version}",
                        state_version=state_version,
                        tool_name=str(external["tool_name"]),
                        call_id=str(external["call_id"]),
                        output=messages,
                    )
                )
            elif approval and not retry_calls:
                from orbit_contracts.models import ResolveApprovalInput

                result = await runtime.resolve_approval(
                    ResolveApprovalInput(
                        room_id=task_id,
                        session_id=session_id,
                        turn_id=f"{attempt_id}:approval:{approval.get('approval_id', 'decision')}",
                        approval_request_id=str(approval.get("approval_request_id", "")),
                        outcome="allowed-once"
                        if approval.get("decision") == "approve"
                        else "rejected",
                        # One decision per parked call: a person may allow some and refuse others (A11).
                        decisions={
                            call_id: decision == "approve"
                            for call_id, decision in (approval.get("decisions") or {}).items()
                        },
                        rules={str(call_id): dict(rule) for call_id, rule in (approval.get("rules") or {}).items()},
                    )
                )
            else:
                result = await runtime.run_turn(
                    RunTurnInput(
                        room_id=task_id,
                        session_id=session_id,
                        turn_id=f"{attempt_id}:turn:{state_version}",
                        message=prompt,
                        state_version=state_version,
                    )
                )
        if result.status == "needs_approval":
            outcome = {
                "status": "parked_approval",
                "checkpoint_ref": _ref(attempt_id),
                "question": result.text,
                "session_id": result.session_id,
                "state_version": result.state_version,
                "approval_request_id": result.approval.approval_request_id
                if result.approval
                else "",
                "approvals": [
                    {
                        "tool_call_id": ask.call_id or "call",
                        "subject": {
                            "kind": "tool_call",
                            "digest": _ref(ask.approval_request_id),
                            "summary": ask.tool_name,
                            "risk": ask.risk,
                            "detail": ask.detail,
                            "allow_rule": ask.allow_rule,
                        },
                    }
                    for ask in (result.approvals or ([result.approval] if result.approval else []))
                ],
            }
        elif result.status == "needs_external" and team is not None:
            calls = result.externals or ([result.external] if result.external else [])
            allowed = {"ask_user", "team_assign"} if team.leader else {"ask_user"}
            unknown = [call.tool_name for call in calls if call.tool_name not in allowed]
            if unknown or not calls:
                outcome = _failed(
                    f"external tool {unknown[0] if unknown else 'unknown'} is not available to the agents of a team",
                    "tool", False, checkpoint_ref=_ref(attempt_id),
                )
            else:
                outcome = {
                    "status": "team_external",
                    "checkpoint_ref": _ref(attempt_id),
                    "externals": [call.model_dump(mode="json") for call in calls],
                    "session_id": result.session_id,
                    "state_version": result.state_version,
                }
        elif result.status == "needs_external":
            call = result.external
            if call is None or call.tool_name != "ask_user":
                name = call.tool_name if call else "unknown"
                outcome = _failed(
                    f"external tool {name} is not available to tasks", "tool", False, checkpoint_ref=_ref(attempt_id)
                )
            else:
                # Structured questions ride along when they are valid; otherwise only the plain question goes.
                text, asked = ask_user_park([call.arguments])
                outcome = {
                    "status": "parked_input",
                    "checkpoint_ref": _ref(attempt_id),
                    "question": text,
                    **({"questions": [q.model_dump(mode="json", exclude_none=True) for q in asked]} if asked else {}),
                    "external": {"call_id": call.call_id, "tool_name": call.tool_name},
                    "session_id": result.session_id,
                    "state_version": result.state_version,
                }
        elif result.status == "failed" and result.error_code == "budget":
            # The attempt spent what was reserved for it and stopped between two steps. Trying again would spend the same
            # again: a person decides, with a grant of budget (05 §4).
            outcome = _failed(result.error or "the attempt's budget is spent", "budget", False, checkpoint_ref=_ref(attempt_id))
        elif result.status == "failed":
            # A model that did not answer in time or is overloaded can be asked again; a refused key or a state that cannot be
            # read cannot be helped by trying again.
            outcome = _failed(
                result.error or "agent turn failed",
                "lost" if result.error_code == "state_unreadable" else "model",
                bool(result.retryable),
                checkpoint_ref=_ref(attempt_id),
            )
        elif member:
            # A member's answer goes to the leader as the result of its assignment, with the files it left in its copy of the
            # workspace (names and blobs). It is not the attempt's: no manifest is announced and nothing is snapshotted.
            outcome = {
                "status": "completed",
                "text": result.text or "completed",
                "checkpoint_ref": _ref(attempt_id),
                "team_files": await _sandbox_entries(tenant_id, task_id, sandbox),
                "session_id": result.session_id,
                "state_version": result.state_version,
            }
        else:
            # A reply that ended in structured output says it as the object: that is what the steps after it are handed.
            text = json.dumps(result.output, ensure_ascii=False) if result.output is not None else result.text or "completed"
            entries = await _sandbox_entries(tenant_id, task_id, sandbox)
            await _publish_worker_event(
                payload,
                "message.agent_final",
                {"attempt_id": event_attempt_id(stream), "text": text, **team_stamp(stream)},
                f"final:{payload.get('state_version', 0)}",
                shown_attempt_id=event_attempt_id(stream),
            )
            exhausted = False if team is not None else await exploration_exhausted(get_task_store(), stream)
            manifest_id = deterministic_id(f"{task_id}:{attempt_id}:manifest", "man")
            outcome = {
                "status": "completed",
                "text": text,
                "checkpoint_ref": _ref(attempt_id),
                "manifest_id": manifest_id,
                "manifest_entries": entries,
                "manifest_hash": _ref(json.dumps(entries, sort_keys=True)),
                "handover_summary": text[:HANDOVER_CHARS],
                **({"output": result.output} if result.output is not None else {}),
                "budget_exhausted": exhausted,
                "session_id": result.session_id,
                "state_version": result.state_version,
            }
        if result.notes and outcome.get("status") != "failed":
            outcome["notes"] = list(result.notes)
            outcome["note_mentions"] = [list(item) for item in result.note_mentions]
        return outcome
    finally:
        if sandbox_token is not None:
            unbind_sandbox(sandbox_token)
        await skill_stage.aclose()
        manifest_recorded = False

        async def _record_with_snapshot(ref: str) -> None:
            # Under the task's commit lock: the snapshot becomes the task's head by the manifest that names it, and the next
            # attempt to save must find it there.
            nonlocal manifest_recorded
            outcome["workspace_snapshot_ref"] = ref
            await record_manifest(get_task_store(), payload, outcome)
            manifest_recorded = True

        if sandbox is not None:
            try:
                snapshot = await sandbox.close(on_saved=_record_with_snapshot)
            except Exception:  # noqa: BLE001 - any failure to save fails the attempt, and is logged
                # What the attempt did in the workspace would be lost without a word: the attempt fails instead. The result
                # that is returned is the dict that was built, so it is rewritten in place.
                structlog.get_logger(__name__).exception("the workspace could not be saved")
                outcome.clear()
                outcome.update(_failed("the workspace could not be saved", "transient", True))
            else:
                if snapshot is not None:
                    outcome["workspace_snapshot_ref"] = snapshot
        if "checkpoint_ref" in outcome:
            try:
                outcome["checkpoint_ref"] = (
                    await _session_checkpoint_ref(str(payload.get("tenant_id", "default")), str(payload["attempt_id"]))
                    or outcome["checkpoint_ref"]
                )
            except Exception:  # noqa: BLE001 - the digest of the attempt id stands in; the checkpoint is committed either way
                structlog.get_logger(__name__).warning("the session checkpoint could not be looked up", exc_info=True)
        # What the activity spent, whatever way it ended: the attempt adds it up and the task settles its reservation by it.
        outcome["usage"] = meter.usage().model_dump(mode="json", exclude_none=True)
        if not manifest_recorded:
            await record_manifest(get_task_store(), payload, outcome)
        heartbeat.cancel()


@activity.defn(name="sop_step")
async def sop_step(payload: dict[str, Any]) -> dict[str, Any]:
    """DEPRECATED (a `sop_stage` node is compiled into the plan now, `orbit_orch.task_sop`): run one try of the SOP named by
    `goal` on AgentScope's engine, for attempts that were running before that and for replay."""

    heartbeat = asyncio.create_task(_heartbeat())
    try:
        return await _run_sop_step(payload)
    finally:
        heartbeat.cancel()


async def _run_sop_step(payload: dict[str, Any]) -> dict[str, Any]:
    _bind_log_context(payload)
    await _announce_resumed(payload)
    tenant_id = str(payload.get("tenant_id", "default"))
    try:
        get_task_store()
        assert _sops is not None
        steps = await _sops.steps_of(tenant_id, str(payload.get("goal", "")))
    except UnknownSopError as exc:
        return _failed(str(exc), "policy", False)
    from orbit_worker.runtime_holder import get_runtime

    await _mock_delay("sop_step_delay_ms")
    outcome = await run_one_try(
        steps,
        goal=f"Run the procedure {payload.get('goal', '')}.",
        run_state=str(payload.get("run_state", "")),
        scope=RunScope(
            runtime=get_runtime(),
            task_id=str(payload["task_id"]),
            attempt_id=str(payload["attempt_id"]),
            mock=MockSettings().mock,
        ),
    )
    # The checkpoint is the engine's own run state (06 §2): what a takeover of this attempt resumes from.
    checkpoint = await get_task_store().put_checkpoint(
        tenant_id=tenant_id,
        task_id=str(payload["task_id"]),
        node_id=str(payload["node_id"]),
        attempt_id=str(payload["attempt_id"]),
        seq=outcome.step_index * 10 + outcome.tries,
        kind="sop_run_state",
        payload=outcome.run_state.encode("utf-8"),
    )
    result: dict[str, Any] = {
        "status": outcome.status,
        "checkpoint_ref": checkpoint,
        "run_state": outcome.run_state,
        "step": outcome.step_index,
    }
    if outcome.status == "failed":
        # The engine has retried the step itself; what it gives up on is a step that could not be done.
        result |= _failed(outcome.message, "tool", True)
    return result


@activity.defn(name="load_sop")
async def load_sop(payload: dict[str, Any]) -> dict[str, Any]:
    """The SOP definition `sop_ref` names, for the workflow that compiles a `sop_stage` node into its plan. `found` is
    False for a reference control has no definition of; the steps are as stored and are given their defaults by the
    workflow (`orbit_contracts.v3.sop`), so a v1 definition and a v2 one are read the same way."""
    tenant_id, sop_ref = str(payload.get("tenant_id", "default")), str(payload.get("sop_ref", ""))
    definition = await get_task_store().get_sop_definition(tenant_id=tenant_id, sop_ref=sop_ref)
    if definition is None:
        return {"found": False, "steps": []}
    return {"found": True, **definition}


@activity.defn(name="checkpoint_commit")
async def checkpoint_commit(payload: dict[str, Any]) -> dict[str, Any]:
    """Store the checkpoint of a `checkpoint` node (04 §1). A payload with a field missing fails the activity without
    retries: it is an orchestrator out of step with this worker, and no retry changes that (17 G9)."""
    request = parse_input(CheckpointCommitInput, payload)
    ref = await get_task_store().put_checkpoint(
        tenant_id=request.tenant_id,
        task_id=request.task_id,
        node_id=request.node_id,
        attempt_id=request.attempt_id,
        seq=request.seq,
        kind=request.kind,
        payload=json.dumps(request.model_dump(mode="json"), sort_keys=True).encode("utf-8"),
    )
    return {"ok": True, "checkpoint_ref": ref}


@activity.defn(name="publish_events")
async def publish_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    await get_task_store().publish_events(events)
    return {"ok": True, "count": len(events)}


AGENT_ACTIVITIES = [agent_turn, sop_step, *VERIFY_AGENT_ACTIVITIES]
IO_ACTIVITIES = [checkpoint_commit, load_sop, publish_events, *VERIFY_IO_ACTIVITIES]

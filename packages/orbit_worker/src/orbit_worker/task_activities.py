"""Activities used by TaskWorkflow and AttemptWorkflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any

import structlog
from orbit_contracts.v3 import Policy
from orbit_orch.plan_engine import deterministic_id
from temporalio import activity

from orbit_worker.activity_input import CheckpointCommitInput, parse_input
from orbit_worker.manifest_record import record_manifest
from orbit_worker.policy_middleware import exploration_exhausted
from orbit_worker.sop import SopRegistry, UnknownSopError
from orbit_worker.sop_agents import RunScope, run_one_try
from orbit_worker.task_store import TaskStore
from orbit_worker.task_stream import TaskStreamContext, streaming_for
from orbit_worker.verify_activities import VERIFY_AGENT_ACTIVITIES, VERIFY_IO_ACTIVITIES
from orbit_worker.worker_events import publish_attempt_event
from orbit_worker.workspace import WorkspaceAdapter, keep_lease_alive

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


async def _heartbeat() -> None:
    """Heartbeats for the whole activity. Cancellation reaches the turn on a heartbeat, so this sets how soon."""
    interval = float(os.environ.get("ORBIT_HEARTBEAT_THROTTLE_S", "5"))
    while True:
        activity.heartbeat()
        await asyncio.sleep(interval)


async def _mock_delay(variable: str = "ORBIT_MOCK_TURN_DELAY_MS") -> None:
    """Mock-model latency. It heartbeats, so a cancel lands mid-turn like it does on a real model call."""
    if os.environ.get("ORBIT_MODEL_MODE", "mock") != "mock":
        return
    remaining = (
        int(os.environ.get(variable, os.environ.get("ORBIT_MOCK_TURN_DELAY_MS", "0"))) / 1000
    )
    while remaining > 0:
        activity.heartbeat()
        step = min(0.25, remaining)
        await asyncio.sleep(step)
        remaining -= step


async def _publish_worker_event(
    payload: dict[str, Any], event_type: str, body: dict[str, Any], seed: str
) -> None:
    await publish_attempt_event(
        get_task_store(),
        tenant_id=str(payload.get("tenant_id", "default")),
        task_id=str(payload["task_id"]),
        attempt_id=str(payload["attempt_id"]),
        event_type=event_type,
        body=body,
        seed=seed,
    )


async def _announce_resumed(payload: dict[str, Any]) -> None:
    """attempt.resumed: this activity continues an attempt after a retry, an approval or a user reply."""
    activity_attempt = activity.info().attempt
    if activity_attempt <= 1 and not (payload.get("approval") or payload.get("external")):
        return
    await _publish_worker_event(
        payload,
        "attempt.resumed",
        {
            "node_id": str(payload["node_id"]),
            "attempt_id": str(payload["attempt_id"]),
            "attempt_no": int(payload.get("attempt_no", 1)),
            "activity_attempt": activity_attempt,
        },
        f"resumed:{activity_attempt}:{payload.get('state_version', 0)}",
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


@activity.defn(name="agent_turn")
async def agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one AgentScope turn and return a small, durable handover result."""

    _bind_log_context(payload)
    heartbeat = asyncio.create_task(_heartbeat())
    lease = None
    renewer: asyncio.Task[None] | None = None
    outcome: dict[str, Any] = {"status": "failed", "error": "agent turn did not return"}
    try:
        from orbit_contracts.models import OpenSessionInput, RunTurnInput

        from orbit_worker.runtime_holder import get_runtime

        task_id = str(payload["task_id"])
        attempt_id = str(payload["attempt_id"])
        await _announce_resumed(payload)
        tenant_id = str(payload.get("tenant_id", "default"))
        retry_calls = payload.get("retry_calls")
        if retry_calls:
            if (payload.get("approval") or {}).get("decision") != "approve":
                outcome = {
                    "status": "failed",
                    "error": "running a call with an unknown outcome again was rejected",
                }
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
        if _workspace is not None and payload.get("workspace_access") == "write":
            lease = await _workspace.acquire(tenant_id, task_id, holder=attempt_id)
            renewer = asyncio.create_task(
                keep_lease_alive(_workspace, lease, getattr(_workspace, "ttl_s", 300))
            )
        runtime = get_runtime()
        await _mock_delay()
        session_id = str(payload.get("session_id", ""))
        state_version = int(payload.get("state_version", 0))
        messages = "\n".join(str(item.get("text", "")) for item in payload.get("messages", []))
        prompt = str(payload.get("goal", ""))
        if messages:
            prompt += "\n\nUser messages:\n" + messages
        approval = payload.get("approval")
        external = payload.get("external")
        stream = TaskStreamContext(
            tenant_id=str(payload.get("tenant_id", "default")),
            task_id=task_id,
            attempt_id=attempt_id,
            activity_attempt=activity.info().attempt,
            node_id=str(payload["node_id"]),
            profile=str(payload.get("profile", "")),
            task_policy=Policy.model_validate(payload.get("policy") or {}),
        )
        with streaming_for(stream):
            if not session_id:
                opened = await runtime.open_session(
                    OpenSessionInput(
                        room_id=task_id,
                        turn_id=f"{attempt_id}:open",
                        permission_preset="workspace-write",
                    )
                )
                session_id = opened.session_id
                state_version = opened.state_version
            if external:
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
                            "risk": "medium",
                        },
                    }
                    for ask in (result.approvals or ([result.approval] if result.approval else []))
                ],
            }
        elif result.status == "needs_external":
            call = result.external
            if call is None or call.tool_name != "ask_user":
                name = call.tool_name if call else "unknown"
                outcome = {
                    "status": "failed",
                    "error": f"external tool {name} is not available to tasks",
                    "checkpoint_ref": _ref(attempt_id),
                }
            else:
                outcome = {
                    "status": "parked_input",
                    "checkpoint_ref": _ref(attempt_id),
                    "question": call.arguments.get("question", ""),
                    "external": {"call_id": call.call_id, "tool_name": call.tool_name},
                    "session_id": result.session_id,
                    "state_version": result.state_version,
                }
        elif result.status == "failed":
            outcome = {
                "status": "failed",
                "error": result.error or "agent turn failed",
                "checkpoint_ref": _ref(attempt_id),
            }
        else:
            text = result.text or "completed"
            entries = [
                {
                    "name": "result.txt",
                    "media_type": "text/plain",
                    "size_bytes": len(text.encode("utf-8")),
                    "blob_ref": await get_task_store().put_artifact_blob(
                        tenant_id=str(payload.get("tenant_id", "default")),
                        task_id=task_id,
                        payload=text.encode("utf-8"),
                    ),
                }
            ]
            await _publish_worker_event(
                payload,
                "message.agent_final",
                {"attempt_id": attempt_id, "text": text},
                f"final:{payload.get('state_version', 0)}",
            )
            exhausted = await exploration_exhausted(get_task_store(), stream)
            manifest_id = deterministic_id(f"{task_id}:{attempt_id}:manifest", "man")
            outcome = {
                "status": "completed",
                "text": text,
                "checkpoint_ref": _ref(attempt_id),
                "manifest_id": manifest_id,
                "manifest_entries": entries,
                "manifest_hash": _ref(json.dumps(entries, sort_keys=True)),
                "budget_exhausted": exhausted,
                "session_id": result.session_id,
                "state_version": result.state_version,
            }
        return outcome
    finally:
        if renewer is not None:
            renewer.cancel()
        if lease is not None and _workspace is not None:
            try:
                outcome["workspace_snapshot_ref"] = await _workspace.snapshot(lease)
            finally:
                await _workspace.release(lease)
        await record_manifest(get_task_store(), payload, outcome)
        heartbeat.cancel()


@activity.defn(name="sop_step")
async def sop_step(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one step of the SOP named by `goal`. `done` tells the workflow when the last step has finished."""

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
        return {"status": "failed", "error": str(exc)}
    from orbit_worker.runtime_holder import get_runtime

    await _mock_delay("ORBIT_MOCK_SOP_STEP_DELAY_MS")
    outcome = await run_one_try(
        steps,
        goal=f"Run the procedure {payload.get('goal', '')}.",
        run_state=str(payload.get("run_state", "")),
        scope=RunScope(
            runtime=get_runtime(),
            task_id=str(payload["task_id"]),
            attempt_id=str(payload["attempt_id"]),
            mock=os.environ.get("ORBIT_MODEL_MODE", "mock") == "mock",
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
        result["error"] = outcome.message
    return result


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
IO_ACTIVITIES = [checkpoint_commit, publish_events, *VERIFY_IO_ACTIVITIES]

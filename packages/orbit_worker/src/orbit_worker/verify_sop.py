"""The verifier of a compiled SOP step (06 §2): an independent agent session that judges one step.

The executor's attempt is an ordinary `agent_turn`. When it reports completed, the workflow runs `verify_sop_step`: a fresh
session (no state of the executor's, read-only permissions) is given the step's description, the verifier's own
instructions, the executor's final text and the names of the files it left, and can read those files in a scratch copy of
the attempt's workspace snapshot, which is thrown away. It answers `PASS` or `FAIL: <what to fix>`; the reason of a FAIL
goes back to the executor's next attempt as the reason the completion was rejected.

With the mock model the verdict is scripted by the step's name: `flaky-<n>` is refused on the node's first n attempts (the
same convention the old SOP path had), so a test can drive the retry path.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

from orbit_contracts.models import OpenSessionInput, RunTurnInput

from orbit_worker import verify
from orbit_worker.agent_config import AgentConfig, agent_config_override
from orbit_worker.sandbox import SandboxSession, bind_sandbox, unbind_sandbox
from orbit_worker.sop_agents import parse_verdict

# What a verifier prompt starts with; the mock model recognises it.
VERIFIER_PREFIX = "You are the independent verifier of one step of a procedure."
MAX_TEXT_CHARS = 8000
MAX_FILES_LISTED = 50


def verifier_prompt(payload: dict[str, Any]) -> str:
    """The message the verifier session is asked. Everything in it comes from the workflow's payload; the verifier expert's own
    instructions are the session's system prompt (`judge`)."""
    parts = [
        (
            f"{VERIFIER_PREFIX} You did not do the work and have not seen how it was done. Judge only whether the "
            "result achieves the step. Answer with PASS, or with FAIL: followed by exactly what is wrong and what to change."
        )
    ]
    sop = str(payload.get("sop") or "").strip()
    if sop:
        parts.append(f"Procedure: {payload.get('sop_name') or sop}")
    parts.append(f"Step: {payload['subject']}\nAttempt: {int(payload.get('attempt_no') or 1)}")
    parts.append(f"What the step must achieve:\n{payload.get('description') or payload['subject']}")
    instructions = str(payload.get("instructions") or "").strip()
    if instructions:
        parts.append(f"Also check:\n{instructions}")
    text = str(payload.get("text") or "").strip()[:MAX_TEXT_CHARS]
    parts.append(f"The executor's final report:\n{text or '(it reported nothing)'}")
    entries = [item for item in payload.get("manifest_entries") or [] if "name" in item][:MAX_FILES_LISTED]
    if entries:
        listing = "\n".join(f"- {item.get('name')} ({item.get('media_type')}, {item.get('size_bytes')} bytes)" for item in entries)
        parts.append(
            f"Files the executor left in the workspace (/workspace; you can read them, you cannot change anything):\n{listing}"
        )
    return "\n\n".join(parts)


async def judge(payload: dict[str, Any]) -> list[verify.Failure]:
    """Run the verifier session and turn its answer into verification failures: none for PASS, one with the verifier's
    reason for FAIL, one for a session that ended without an answer."""
    from orbit_worker.runtime_holder import get_runtime
    from orbit_worker.task_activities import get_task_store, get_workspace_adapter

    tenant_id, task_id = str(payload["tenant_id"]), str(payload["task_id"])
    store = get_task_store()
    # The verifier runs as its expert: that profile's instructions and model (the same `model_config_for` path an attempt's
    # agent takes), but none of its tools: no connectors, skills or team, and the session is read-only.
    expert = AgentConfig()
    if payload.get("expert"):
        loaded = await store.agent_config(tenant_id=tenant_id, profile_ref=str(payload["expert"]))
        expert = dataclasses.replace(loaded, mcp_connectors=(), skills=(), team=None)
    adapter, snapshot = get_workspace_adapter(), payload.get("workspace_snapshot_ref")
    sandbox = token = None
    if adapter is not None and snapshot:
        sandbox = SandboxSession(
            adapter, store, tenant_id=tenant_id, task_id=task_id, holder=f"verify:{payload['attempt_id']}", restore_from=str(snapshot)
        )
        token = bind_sandbox(sandbox)
    try:
        runtime = get_runtime()
        override = agent_config_override(expert) if payload.get("expert") else contextlib.nullcontext()
        turn_id = f"{payload['attempt_id']}:sop-verify:{payload['node_id']}"
        # Not the attempt's session: outside a task context a session is its own, opened from this turn alone.
        with override:
            opened = await runtime.open_session(
                OpenSessionInput(room_id=task_id, turn_id=f"{turn_id}:open", permission_preset="read-only")
            )
            result = await runtime.run_turn(
                RunTurnInput(
                    room_id=task_id,
                    session_id=opened.session_id,
                    turn_id=turn_id,
                    message=verifier_prompt(payload),
                    state_version=opened.state_version,
                )
            )
    finally:
        if token is not None:
            unbind_sandbox(token)
        if sandbox is not None:
            await sandbox.close()
    if result.status != "completed":
        return [verify.failure(
            "sop_verifier", "sop_verifier_inconclusive",
            f"the verifier of step {payload['subject']!r} ended as {result.status} without a verdict: {result.error or result.text}"[:300],
        )]
    verdict = parse_verdict(result.text)
    if verdict.passed:
        return []
    return [verify.failure(
        "sop_verifier", "sop_step_failed", f"step {payload['subject']!r} was refused: {verdict.message}"[:2000],
        step=str(payload.get("step_id") or ""),
    )]

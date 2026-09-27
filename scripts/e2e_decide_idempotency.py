"""End-to-end proof of C34 §2.4 decide idempotency (runtime half).

Reuses the ``scripts/e2e_a1_events.py`` harness: a real Temporal dev server,
orbit-orch, orbit-worker, the mock model, and real Postgres. The report's
``commit`` is ``git rev-parse HEAD``. Two runs on one commit write the same
bytes. There is no ``--commit`` override.

    uv run python scripts/e2e_decide_idempotency.py run --out artifacts/e2e-decide-idempotency.json
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncpg
from e2e_a1_events import (
    INGEST_TOKEN,
    Harness,
    Recorder,
    _commit,
    _free_port,
    _of,
    _start,
    _stop,
    _versions,
)
from orbit_contracts.models import (
    DECIDED_APPROVALS_LIMIT_MESSAGE,
    MAX_DECIDED_APPROVALS,
    AgentRunInput,
    DecidedApproval,
    DecideOutcome,
    ResolveSignal,
    RoomCarryOver,
    RoomWorkflowInput,
    resume_turn_id,
)
from orbit_orch.workflows import AgentRunWorkflow, RoomWorkflow
from orbit_worker.postgres_store import PostgresStateStore, StateCipher
from temporalio.api.enums.v1 import EventType
from temporalio.api.workflowservice.v1 import GetSystemInfoRequest
from temporalio.client import (
    Client,
    WorkflowExecutionStatus,
    WorkflowHandle,
    WorkflowUpdateFailedError,
    WorkflowUpdateStage,
)
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment

QUEUES = {
    "id": "orbit-e2e-id",
    "ttl": "orbit-e2e-id-ttl",
    "slow": "orbit-e2e-id-slow",
}
DELIVERY_TIMEOUT_S = 1
RESOLVE_DELAY_S = 8
CASE_TIMEOUT_S = 120
MESSAGE = "echo:once"


def _rows(
    count: int,
    *,
    prefix: str,
    when: datetime,
    state: str,
) -> list[DecidedApproval]:
    rows: list[DecidedApproval] = []
    for index in range(count):
        approval_id = f"{prefix}-{index:04d}"
        outcome = None
        if state == "done":
            outcome = DecideOutcome(
                decision="allow",
                agent_id="main",
                resume_turn_id=resume_turn_id(approval_id),
                turn_status="completed",
            )
        rows.append(
            DecidedApproval(
                approval_request_id=approval_id,
                decided_at=when,
                state=state,  # type: ignore[arg-type]
                outcome=outcome,
            )
        )
    return rows


def _outcome(value: DecideOutcome) -> dict[str, object]:
    return {
        "decision": value.decision,
        "agentId": value.agent_id,
        "resumeTurnId": value.resume_turn_id,
        "turnStatus": value.turn_status,
        "errorCode": value.error_code,
    }


def _error_type(exc: BaseException) -> str:
    current: BaseException | None = exc
    for _ in range(8):
        if current is None:
            break
        if isinstance(current, ApplicationError) and current.type:
            return current.type
        current = current.__cause__ or getattr(current, "cause", None)
    return type(exc).__name__


def _error_info(exc: BaseException) -> dict[str, object]:
    current: BaseException | None = exc
    for _ in range(8):
        if current is None:
            break
        if isinstance(current, ApplicationError) and current.type:
            return {"type": current.type, "nonRetryable": bool(current.non_retryable)}
        current = current.__cause__ or getattr(current, "cause", None)
    return {"type": type(exc).__name__, "nonRetryable": False}


def _usage_count(events: list[dict]) -> int:
    return len(_of(events, "usage"))


def _gated_echo(events: list[dict]) -> int:
    return len(
        [body for body in _of(events, "tool.result") if body.get("toolName") == "gated_echo"]
    )


def _activity_scheduled(history: object, name: str) -> int:
    count = 0
    for event in history.events:  # type: ignore[attr-defined]
        if event.event_type != EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            continue
        scheduled = event.activity_task_scheduled_event_attributes
        if scheduled.activity_type.name == name:
            count += 1
    return count


def _accepted_before_activity(history: object, name: str) -> bool:
    accepted = None
    completed = None
    scheduled_ids: set[int] = set()
    for event in history.events:  # type: ignore[attr-defined]
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED:
            accepted = event.event_id
        elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            scheduled = event.activity_task_scheduled_event_attributes
            if scheduled.activity_type.name == name:
                scheduled_ids.add(event.event_id)
        elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
            done = event.activity_task_completed_event_attributes
            if done.scheduled_event_id in scheduled_ids:
                completed = event.event_id
    return accepted is not None and completed is not None and accepted < completed


def _activity_slower_than(history: object, name: str, seconds: float) -> bool:
    started: dict[int, datetime] = {}
    for event in history.events:  # type: ignore[attr-defined]
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            scheduled = event.activity_task_scheduled_event_attributes
            if scheduled.activity_type.name == name:
                started[event.event_id] = event.event_time.ToDatetime(tzinfo=UTC)
        elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
            done = event.activity_task_completed_event_attributes
            start = started.get(done.scheduled_event_id)
            if start is None:
                continue
            end = event.event_time.ToDatetime(tzinfo=UTC)
            if (end - start).total_seconds() > seconds:
                return True
    return False


class Suite:
    def __init__(self, client: Client, recorder: Recorder, logs: Path) -> None:
        self.client = client
        self.recorder = recorder
        self.logs = logs
        self._harness = Harness(client, recorder, {}, logs)

    async def open_room(
        self,
        room_id: str,
        mode: str,
        carry: RoomCarryOver | None = None,
    ) -> WorkflowHandle:
        handle = await self.client.start_workflow(
            RoomWorkflow.run,
            RoomWorkflowInput(room_id=room_id, carry_over=carry),
            id=f"room:{room_id}",
            task_queue=QUEUES[mode],
        )
        await self._harness.wait_status(handle, "running")
        return handle

    async def park(self, handle: WorkflowHandle) -> str:
        parked = await handle.execute_update("runTurn", {"turnId": "tn-1", "message": MESSAGE})
        approval = parked.get("approval") or {}
        approval_id = approval.get("approvalRequestId") or ""
        if not approval_id:
            raise RuntimeError("turn did not park")
        await self._harness.wait_status(handle, "awaiting_approval")
        return approval_id

    async def decide(
        self,
        handle: WorkflowHandle,
        approval_id: str,
        update_id: str,
    ) -> DecideOutcome:
        return await handle.execute_update(
            "decide",
            {"decision": "allow", "approvalRequestId": approval_id},
            id=update_id,
            result_type=DecideOutcome,
        )

    def warnings(self, mode: str, text: str) -> int:
        err = (self.logs / f"orbit-orch-{mode}.stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )
        return err.count(text)

    async def histories(self, handle: WorkflowHandle) -> list:
        """Every run in the continue-as-new chain, starting at the first run.

        Visibility can lag behind continue-as-new, so this follows
        ``new_execution_run_id`` instead of listing executions.
        """

        found = []
        seen: set[str] = set()
        run_id = handle.result_run_id
        while run_id and run_id not in seen:
            seen.add(run_id)
            pinned = self.client.get_workflow_handle(handle.id, run_id=run_id)
            history = await pinned.fetch_history()
            found.append(history)
            run_id = ""
            for event in history.events:
                if event.event_type != EventType.EVENT_TYPE_WORKFLOW_EXECUTION_CONTINUED_AS_NEW:
                    continue
                run_id = (
                    event.workflow_execution_continued_as_new_event_attributes.new_execution_run_id
                )
                break
        if not found:
            found.append(await handle.fetch_history())
        return found


def _passes(row: dict) -> bool:
    return row["expected"] == row["actual"]


async def case_s_id_2(suite: Suite) -> dict:
    room = "s-id-2"
    handle = await suite.open_room(room, "id")
    approval_id = await suite.park(handle)
    run_before = (await handle.describe()).run_id
    first = await suite.decide(handle, approval_id, approval_id)
    second = await suite.decide(handle, approval_id, approval_id)
    third = await suite.decide(handle, approval_id, approval_id + ":other")
    # N9: a different update id waits while the first resume is still running.
    slow = await suite.open_room("s-id-2-n9", "slow")
    slow_id = await suite.park(slow)
    concurrent = await asyncio.gather(
        suite.decide(slow, slow_id, slow_id),
        suite.decide(slow, slow_id, slow_id + ":other"),
    )
    config = await handle.query(RoomWorkflow.decide_config)
    expected_outcome = {
        "decision": "allow",
        "agentId": "main",
        "resumeTurnId": resume_turn_id(approval_id),
        "turnStatus": "completed",
        "errorCode": None,
    }
    events = suite.recorder.room(room)
    slow_events = suite.recorder.room("s-id-2-n9")
    resume = resume_turn_id(approval_id)
    started = [body for body in _of(events, "turn.started") if body.get("turnId") == resume]
    slow_started = [
        body
        for body in _of(slow_events, "turn.started")
        if body.get("turnId") == resume_turn_id(slow_id)
    ]
    histories = await suite.histories(handle)
    resolves = sum(_activity_scheduled(history, "resolveApproval") for history in histories)
    run_after = (await handle.describe()).run_id
    return {
        "id": "S-ID-2",
        "title": "Repeated decide returns the first outcome and resumes once",
        "steps": [
            f"Park room {room} on the mock worker (Postgres).",
            "decide twice with Update id = approvalRequestId.",
            "decide again with a different Update id.",
            "On the slow worker, send two Update ids concurrently (N9).",
            "Query decideConfig. Count turn.started and resolveApproval.",
        ],
        "expected": {
            "sameRun": True,
            "first": expected_outcome,
            "second": expected_outcome,
            "third": expected_outcome,
            "concurrentEqual": True,
            "turnStarted": 1,
            "slowTurnStarted": 1,
            "resolveApproval": 1,
            "toolResults": 1,
            "slowToolResults": 1,
            "ttlS": 86400,
            "maxDecided": MAX_DECIDED_APPROVALS,
            "ttlAtLeast60": True,
        },
        "actual": {
            "sameRun": run_before == run_after,
            "first": _outcome(first),
            "second": _outcome(second),
            "third": _outcome(third),
            "concurrentEqual": _outcome(concurrent[0]) == _outcome(concurrent[1]),
            "turnStarted": len(started),
            "slowTurnStarted": len(slow_started),
            "resolveApproval": resolves,
            "toolResults": len(_of(events, "tool.result")),
            "slowToolResults": len(_of(slow_events, "tool.result")),
            "ttlS": config.ttl_s,
            "maxDecided": config.max_decided,
            "ttlAtLeast60": config.ttl_s >= 60,
        },
    }


async def case_s_id_7(suite: Suite) -> dict:
    room = "s-id-7"
    handle = await suite.open_room(room, "ttl")
    approval_id = await suite.park(handle)
    before = (await handle.describe()).run_id
    first = await suite.decide(handle, approval_id, approval_id)
    changed = await _wait_run(handle, before)
    second = await suite.decide(handle, approval_id, approval_id)
    histories = await suite.histories(handle)
    resolves = sum(_activity_scheduled(history, "resolveApproval") for history in histories)
    events = suite.recorder.room(room)
    resume = resume_turn_id(approval_id)
    started = [body for body in _of(events, "turn.started") if body.get("turnId") == resume]
    return {
        "id": "S-ID-7",
        "title": "After continue-as-new the same Update id returns the first outcome",
        "steps": [
            "ORBIT_CAN_TURN_THRESHOLD=1 on this worker.",
            "decide once, wait until the workflow run id changes, decide with the same Update id.",
        ],
        "expected": {
            "continued": True,
            "sameOutcome": True,
            "resolveApproval": 1,
            "turnStarted": 1,
            "toolResults": 1,
        },
        "actual": {
            "continued": changed,
            "sameOutcome": _outcome(first) == _outcome(second),
            "resolveApproval": resolves,
            "turnStarted": len(started),
            "toolResults": len(_of(events, "tool.result")),
        },
    }


async def case_s_id_8(suite: Suite) -> dict:
    room = "s-id-8"
    now = datetime.now(UTC)
    carry = RoomCarryOver(
        room_id=room,
        decided_approvals=_rows(MAX_DECIDED_APPROVALS, prefix="pre", when=now, state="done"),
    )
    handle = await suite.open_room(room, "id", carry)
    approval_id = await suite.park(handle)
    try:
        await suite.decide(handle, approval_id, approval_id)
        error = ""
    except WorkflowUpdateFailedError as exc:
        error = _error_type(exc)
    await _wait_closed(handle)
    ids = await handle.query(RoomWorkflow.decided_approval_ids)
    events = suite.recorder.room(room)
    failed = _of(events, "room.failed")
    failure = failed[0].get("failure") if failed else None
    desc = await handle.describe()
    marker = f"decided_approvals_high_watermark{{roomId={room},count={MAX_DECIDED_APPROVALS}}}"
    return {
        "id": "S-ID-8",
        "title": "The 1025th id fails the room; writing running first would have passed the cap",
        "steps": [
            f"Start with {MAX_DECIDED_APPROVALS} unexpired decided rows.",
            "Park a new approval and decide it.",
            "A handler that writes running before the cap check would accept this id.",
        ],
        "expected": {
            "updateError": "DECIDED_APPROVALS_LIMIT",
            "workflowStatus": "FAILED",
            "roomFailed": 1,
            "failure": {
                "code": "DECIDED_APPROVALS_LIMIT",
                "message": DECIDED_APPROVALS_LIMIT_MESSAGE,
            },
            "idCount": MAX_DECIDED_APPROVALS,
            "newIdStored": False,
            "toolResults": 0,
            "watermarkLines": 1,
        },
        "actual": {
            "updateError": error,
            "workflowStatus": desc.status.name,
            "roomFailed": len(failed),
            "failure": failure,
            "idCount": len(ids),
            "newIdStored": approval_id in ids,
            "toolResults": len(_of(events, "tool.result")),
            "watermarkLines": suite.warnings("id", marker),
        },
    }


async def case_s_id_8_order(suite: Suite) -> dict:
    room = "s-id-8-order"
    now = datetime.now(UTC)
    old = now - timedelta(days=2)
    rows = _rows(MAX_DECIDED_APPROVALS - 1, prefix="ord", when=now, state="done")
    rows.append(
        DecidedApproval(
            approval_request_id="ord-running",
            decided_at=old,
            state="running",
        )
    )
    handle = await suite.open_room(room, "id", RoomCarryOver(room_id=room, decided_approvals=rows))
    approval_id = await suite.park(handle)
    try:
        await suite.decide(handle, approval_id, approval_id)
        error = ""
    except WorkflowUpdateFailedError as exc:
        error = _error_type(exc)
    await _wait_closed(handle)
    ids = await handle.query(RoomWorkflow.decided_approval_ids)
    events = suite.recorder.room(room)
    return {
        "id": "S-ID-8-order",
        "title": "A running row is not pruned, so the new id still hits the cap",
        "steps": [
            "Carry 1023 fresh done rows and one running row decided two days ago.",
            "Pruning that running row would leave 1023 and the new decide would succeed.",
        ],
        "expected": {
            "updateError": "DECIDED_APPROVALS_LIMIT",
            "runningKept": True,
            "newIdStored": False,
            "idCount": MAX_DECIDED_APPROVALS,
            "toolResults": 0,
        },
        "actual": {
            "updateError": error,
            "runningKept": "ord-running" in ids,
            "newIdStored": approval_id in ids,
            "idCount": len(ids),
            "toolResults": len(_of(events, "tool.result")),
        },
    }


async def case_s_id_10(suite: Suite) -> dict:
    room = "s-id-10"
    stale = datetime.now(UTC) - timedelta(seconds=6)
    carry = RoomCarryOver(
        room_id=room,
        decided_approvals=_rows(MAX_DECIDED_APPROVALS, prefix="old", when=stale, state="done"),
    )
    handle = await suite.open_room(room, "ttl", carry)
    config = await handle.query(RoomWorkflow.decide_config)
    approval_id = await suite.park(handle)
    before = (await handle.describe()).run_id
    outcome = await suite.decide(handle, approval_id, approval_id)
    continued = await _wait_run(handle, before)
    ids = await handle.query(RoomWorkflow.decided_approval_ids)
    desc = await handle.describe()
    return {
        "id": "S-ID-10",
        "title": "Expired rows are pruned before the cap, then only the new id is carried",
        "steps": [
            "ORBIT_E2E=1 and ORBIT_DECIDED_APPROVAL_TTL_S=5.",
            "Carry 1024 rows decided 6 seconds ago. Decide a new id.",
            "Wait for continue-as-new and read decidedApprovalIds.",
        ],
        "expected": {
            "ttlS": 5,
            "turnStatus": "completed",
            "workflowStatus": "RUNNING",
            "continued": True,
            "ids": [approval_id],
            "toolResults": 1,
        },
        "actual": {
            "ttlS": config.ttl_s,
            "turnStatus": outcome.turn_status,
            "workflowStatus": desc.status.name,
            "continued": continued,
            "ids": ids,
            "toolResults": len(_of(suite.recorder.room(room), "tool.result")),
        },
    }


async def case_s_id_11(suite: Suite) -> dict:
    room = "s-id-11"
    handle = await suite.client.start_workflow(
        AgentRunWorkflow.run,
        AgentRunInput(
            room_id=room,
            prompt=MESSAGE,
            permission_preset="workspace-write",
        ),
        id=f"agent:{room}",
        task_queue=QUEUES["id"],
    )
    approval_id = await _wait_approval(suite, room)
    signal = ResolveSignal(approval_request_id=approval_id)
    await handle.signal("resolve", signal)
    await handle.signal("resolve", signal)
    await handle.result()
    events = suite.recorder.room(room)
    warning = f"duplicate resolve ignored approvalRequestId={approval_id}"
    return {
        "id": "S-ID-11",
        "title": "A child resolve signal for the same approval runs the tool once",
        "steps": [
            "Start AgentRunWorkflow directly and wait until it parks.",
            "Signal resolve twice with the same approvalRequestId.",
        ],
        "expected": {"toolResults": 1, "warnings": 1, "turnStarted": 1},
        "actual": {
            "toolResults": len(_of(events, "tool.result")),
            "warnings": suite.warnings("id", warning),
            "turnStarted": len(_of(events, "turn.started")),
        },
    }


async def case_s_id_13(suite: Suite) -> dict:
    room = "s-id-13"
    handle = await suite.open_room(room, "slow")
    approval_id = await suite.park(handle)
    outcome = await suite.decide(handle, approval_id, approval_id)
    history = await handle.fetch_history()
    snapshot = await handle.query(RoomWorkflow.snapshot)
    return {
        "id": "S-ID-13",
        "title": "A slow resume still saves the turn after the delivery timeout",
        "steps": [
            f"The resolve activity sleeps {RESOLVE_DELAY_S}s.",
            f"The harness treats {DELIVERY_TIMEOUT_S}s as ORBIT_DECISION_DELIVERY_TIMEOUT.",
            "HTTP 502 is orbit-control's response and is not asserted here.",
        ],
        "expected": {
            "acceptedBeforeComplete": True,
            "slowerThanDeliveryTimeout": True,
            "turnStatus": "completed",
            "roomStatus": "running",
            "leftAwaitingApproval": True,
            "toolResults": 1,
        },
        "actual": {
            "acceptedBeforeComplete": _accepted_before_activity(history, "resolveApproval"),
            "slowerThanDeliveryTimeout": _activity_slower_than(
                history, "resolveApproval", DELIVERY_TIMEOUT_S
            ),
            "turnStatus": outcome.turn_status,
            "roomStatus": snapshot.status,
            "leftAwaitingApproval": snapshot.status != "awaiting_approval",
            "toolResults": len(_of(suite.recorder.room(room), "tool.result")),
        },
    }


def _mixed_rows(room: str, when: datetime) -> list[DecidedApproval]:
    half = MAX_DECIDED_APPROVALS // 2
    return _rows(half, prefix=f"{room}-main", when=when, state="done") + _rows(
        half, prefix=f"{room}-child", when=when, state="done"
    )


def _signal_before_activity_completed(
    history: object, signal_name: str, activity_name: str
) -> bool:
    signal_at: int | None = None
    scheduled: set[int] = set()
    completed: int | None = None
    for event in history.events:  # type: ignore[attr-defined]
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_SIGNALED:
            attrs = event.workflow_execution_signaled_event_attributes
            if attrs.signal_name == signal_name and signal_at is None:
                signal_at = event.event_id
        elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            scheduled_attrs = event.activity_task_scheduled_event_attributes
            if scheduled_attrs.activity_type.name == activity_name:
                scheduled.add(event.event_id)
        elif event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
            done = event.activity_task_completed_event_attributes
            if done.scheduled_event_id in scheduled and completed is None:
                completed = event.event_id
    return signal_at is not None and completed is not None and signal_at < completed


async def _spawn_echo(suite: Suite, handle: WorkflowHandle, room: str) -> tuple[str, str]:
    await handle.execute_update("runTurn", {"turnId": "tn-spawn", "message": "spawn echo"})
    approval_id = await _wait_approval(suite, room)
    return approval_id, f"room:{room}/agent/call-spawn-echo"


async def _wait_scheduled(handle: WorkflowHandle, name: str) -> None:
    for _ in range(200):
        history = await handle.fetch_history()
        if _activity_scheduled(history, name) > 0:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"{handle.id} did not schedule {name}")


async def _reject_until_limit(handle: WorkflowHandle, approval_id: str) -> dict[str, object]:
    """New update ids, so Temporal does not replay an earlier APPROVAL_UNKNOWN."""

    last = {"type": "", "nonRetryable": False}
    for attempt in range(40):
        try:
            await handle.execute_update(
                "decide",
                {"decision": "allow", "approvalRequestId": approval_id},
                id=f"{approval_id}:{attempt}",
                result_type=DecideOutcome,
            )
            return {"type": "", "nonRetryable": False}
        except WorkflowUpdateFailedError as exc:
            last = _error_info(exc)
            if last["type"] == "DECIDED_APPROVALS_LIMIT":
                return last
        await asyncio.sleep(0.05)
    return last


async def case_s_id_14(suite: Suite) -> dict:
    room = "s-id-14"
    now = datetime.now(UTC)
    carry = RoomCarryOver(
        room_id=room,
        decided_approvals=_rows(MAX_DECIDED_APPROVALS, prefix="pre", when=now, state="done"),
    )
    handle = await suite.open_room(room, "id", carry)
    approval_id, child_id = await _spawn_echo(suite, handle, room)
    usage_before = _usage_count(suite.recorder.room(room))
    gated_before = _gated_echo(suite.recorder.room(room))
    child = suite.client.get_workflow_handle(child_id)
    await child.signal("resolve", ResolveSignal(approval_request_id=approval_id))
    await _wait_closed(handle)
    await _wait_closed(child)
    await asyncio.sleep(1)
    events = suite.recorder.room(room)
    failed = _of(events, "room.failed")
    failure = failed[0].get("failure") if failed else None
    ids = await handle.query(RoomWorkflow.decided_approval_ids)
    desc = await handle.describe()
    child_status = (await child.describe()).status
    usage_after = _usage_count(events)
    return {
        "id": "S-ID-14",
        "title": "A child decision is the 1025th row on the room table",
        "steps": [
            f"Carry {MAX_DECIDED_APPROVALS} done rows on the room.",
            "Spawn one child that parks on gated_echo, then signal its resolve.",
            "The room counts that decision. The child's own set is only a dedupe set.",
        ],
        "expected": {
            "workflowStatus": "FAILED",
            "childStatus": "CANCELED",
            "roomFailed": 1,
            "failure": {
                "code": "DECIDED_APPROVALS_LIMIT",
                "message": DECIDED_APPROVALS_LIMIT_MESSAGE,
            },
            "idCount": MAX_DECIDED_APPROVALS,
            "newIdStored": False,
            "gatedEcho": gated_before,
            "usageUnchanged": True,
        },
        "actual": {
            "workflowStatus": desc.status.name if desc.status else "",
            "childStatus": child_status.name if child_status else "",
            "roomFailed": len(failed),
            "failure": failure,
            "idCount": len(ids),
            "newIdStored": approval_id in ids,
            "gatedEcho": _gated_echo(events),
            "usageUnchanged": usage_before == usage_after,
        },
    }


async def case_s_id_15(suite: Suite) -> dict:
    now = datetime.now(UTC)
    main_room = "s-id-15-main"
    main = await suite.open_room(
        main_room,
        "id",
        RoomCarryOver(room_id=main_room, decided_approvals=_mixed_rows(main_room, now)),
    )
    main_approval = await suite.park(main)
    try:
        await suite.decide(main, main_approval, main_approval)
        main_error = ""
    except WorkflowUpdateFailedError as exc:
        main_error = _error_type(exc)
    await _wait_closed(main)
    main_failed = _of(suite.recorder.room(main_room), "room.failed")
    main_failure = main_failed[0].get("failure") if main_failed else None

    child_room = "s-id-15-child"
    child_room_handle = await suite.open_room(
        child_room,
        "id",
        RoomCarryOver(room_id=child_room, decided_approvals=_mixed_rows(child_room, now)),
    )
    child_approval, child_id = await _spawn_echo(suite, child_room_handle, child_room)
    await suite.client.get_workflow_handle(child_id).signal(
        "resolve", ResolveSignal(approval_request_id=child_approval)
    )
    await _wait_closed(child_room_handle)
    child_failed = _of(suite.recorder.room(child_room), "room.failed")
    child_failure = child_failed[0].get("failure") if child_failed else None
    child_desc = await child_room_handle.describe()
    main_desc = await main.describe()
    return {
        "id": "S-ID-15",
        "title": "Main and child share the cap and fail with the same code",
        "steps": [
            "Two rooms. Each carries 512 main rows and 512 child rows.",
            "On the first room the main decide is the 1025th.",
            "On the second room the child resolve is the 1025th.",
        ],
        "expected": {
            "mainStatus": "FAILED",
            "childStatus": "FAILED",
            "mainError": "DECIDED_APPROVALS_LIMIT",
            "mainFailure": {
                "code": "DECIDED_APPROVALS_LIMIT",
                "message": DECIDED_APPROVALS_LIMIT_MESSAGE,
            },
            "childFailure": {
                "code": "DECIDED_APPROVALS_LIMIT",
                "message": DECIDED_APPROVALS_LIMIT_MESSAGE,
            },
            "sameCode": True,
        },
        "actual": {
            "mainStatus": main_desc.status.name if main_desc.status else "",
            "childStatus": child_desc.status.name if child_desc.status else "",
            "mainError": main_error,
            "mainFailure": main_failure,
            "childFailure": child_failure,
            "sameCode": (main_failure or {}).get("code") == (child_failure or {}).get("code"),
        },
    }


async def case_s_id_16(suite: Suite) -> dict:
    room = "s-id-16"
    now = datetime.now(UTC)
    kept = "kept-0000"
    handle = await suite.open_room(
        room,
        "slow",
        RoomCarryOver(
            room_id=room,
            decided_approvals=_rows(
                MAX_DECIDED_APPROVALS - 1, prefix="kept", when=now, state="done"
            ),
        ),
    )
    child_approval, child_id = await _spawn_echo(suite, handle, room)
    main_approval = await suite.park(handle)
    started = await handle.start_update(
        "decide",
        {"decision": "allow", "approvalRequestId": main_approval},
        id=main_approval,
        result_type=DecideOutcome,
        wait_for_stage=WorkflowUpdateStage.ACCEPTED,
    )
    await _wait_scheduled(handle, "resolveApproval")
    await suite.client.get_workflow_handle(child_id).signal(
        "resolve", ResolveSignal(approval_request_id=child_approval)
    )
    fresh = await _reject_until_limit(handle, "fresh-1025")
    usage_before = _usage_count(suite.recorder.room(room))
    try:
        await handle.execute_update("runTurn", {"turnId": "tn-late", "message": "echo:later"})
        run_turn = {"type": "", "nonRetryable": False}
    except WorkflowUpdateFailedError as exc:
        run_turn = _error_info(exc)
    usage_after = _usage_count(suite.recorder.room(room))
    stored = await suite.decide(handle, kept, kept + ":replay")
    gated_during = _gated_echo(suite.recorder.room(room))
    await started.result()
    await _wait_closed(handle)
    expected_stored = {
        "decision": "allow",
        "agentId": "main",
        "resumeTurnId": resume_turn_id(kept),
        "turnStatus": "completed",
        "errorCode": None,
    }
    return {
        "id": "S-ID-16",
        "title": "While _fatal is set and the resume is still sleeping, new work is rejected",
        "steps": [
            "Carry 1023 done rows, spawn a child, and park the main agent.",
            f"Start a decide whose resolve sleeps {RESOLVE_DELAY_S}s. That row fills the cap.",
            "The child resolve sets _fatal during the sleep.",
            "Decide a new id, start a runTurn, and redeliver a done id before the sleep ends.",
        ],
        "expected": {
            "fresh": {"type": "DECIDED_APPROVALS_LIMIT", "nonRetryable": True},
            "runTurn": {"type": "DECIDED_APPROVALS_LIMIT", "nonRetryable": True},
            "stored": expected_stored,
            "gatedDuringSleep": 0,
            "runTurnAddedUsage": False,
        },
        "actual": {
            "fresh": fresh,
            "runTurn": run_turn,
            "stored": _outcome(stored),
            "gatedDuringSleep": gated_during,
            "runTurnAddedUsage": usage_before != usage_after,
        },
    }


async def case_s_id_17(suite: Suite) -> dict:
    room = "s-id-17"
    now = datetime.now(UTC)
    handle = await suite.open_room(
        room,
        "slow",
        RoomCarryOver(
            room_id=room,
            decided_approvals=_rows(
                MAX_DECIDED_APPROVALS - 1, prefix="pre", when=now, state="done"
            ),
        ),
    )
    child_approval, child_id = await _spawn_echo(suite, handle, room)
    main_approval = await suite.park(handle)
    started = await handle.start_update(
        "decide",
        {"decision": "allow", "approvalRequestId": main_approval},
        id=main_approval,
        result_type=DecideOutcome,
        wait_for_stage=WorkflowUpdateStage.ACCEPTED,
    )
    await _wait_scheduled(handle, "resolveApproval")
    await suite.client.get_workflow_handle(child_id).signal(
        "resolve", ResolveSignal(approval_request_id=child_approval)
    )
    await started.result()
    await _wait_closed(handle)
    await asyncio.sleep(1)
    events = suite.recorder.room(room)
    failed_at = [index for index, body in enumerate(events) if body.get("type") == "room.failed"]
    tool_at = [
        index
        for index, body in enumerate(events)
        if body.get("type") == "tool.result" and body.get("toolName") == "gated_echo"
    ]
    history = await handle.fetch_history()
    desc = await handle.describe()
    usage_at_close = _usage_count(events)
    await asyncio.sleep(1)
    usage_later = _usage_count(suite.recorder.room(room))
    failure = events[failed_at[0]].get("failure") if len(failed_at) == 1 else None
    return {
        "id": "S-ID-17",
        "title": "room.failed is emitted once, after the in-flight turn",
        "steps": [
            "Same window as S-ID-16: _fatal is set while resolveApproval is still sleeping.",
            "After that activity completes, exactly one room.failed follows its tool.result.",
        ],
        "expected": {
            "workflowStatus": "FAILED",
            "roomFailed": 1,
            "afterToolResult": True,
            "claimDuringTurn": True,
            "failure": {
                "code": "DECIDED_APPROVALS_LIMIT",
                "message": DECIDED_APPROVALS_LIMIT_MESSAGE,
            },
            "usageStable": True,
        },
        "actual": {
            "workflowStatus": desc.status.name if desc.status else "",
            "roomFailed": len(failed_at),
            "afterToolResult": len(tool_at) == 1
            and len(failed_at) == 1
            and failed_at[0] > tool_at[0],
            "claimDuringTurn": _signal_before_activity_completed(
                history, "claimDecision", "resolveApproval"
            ),
            "failure": failure,
            "usageStable": usage_at_close == usage_later,
        },
    }


async def case_f11(suite: Suite) -> dict:
    room = "f11"
    handle = await suite.open_room(room, "slow")
    approval_id = await suite.park(handle)
    await handle.start_update(
        "decide",
        {"decision": "allow", "approvalRequestId": approval_id},
        id=approval_id,
        result_type=DecideOutcome,
        wait_for_stage=WorkflowUpdateStage.ACCEPTED,
    )
    await _wait_scheduled(handle, "resolveApproval")
    await handle.cancel()
    await _wait_closed(handle)
    outcome = await handle.query(RoomWorkflow.decide_outcome, approval_id)
    desc = await handle.describe()
    return {
        "id": "F11",
        "title": "Cancelling the resume still stores done in finally",
        "steps": [
            f"Park, start decide, and cancel the workflow during the {RESOLVE_DELAY_S}s sleep.",
            "Query decideOutcome on the closed workflow.",
        ],
        "expected": {
            "workflowStatus": "CANCELED",
            "stored": {
                "decision": "allow",
                "agentId": "main",
                "resumeTurnId": resume_turn_id(approval_id),
                "turnStatus": "failed",
                "errorCode": None,
            },
        },
        "actual": {
            "workflowStatus": desc.status.name if desc.status else "",
            "stored": _outcome(outcome) if outcome is not None else None,
        },
    }


async def case_f14(suite: Suite) -> dict:
    room = "f14"
    handle = await suite.open_room(room, "id")
    approval_id = await suite.park(handle)
    rejected = True
    try:
        await handle.execute_update(
            "decide",
            {
                "decision": "allow",
                "approvalRequestId": approval_id,
                "resumeTurnId": "tn-2",
            },
            id=approval_id,
            result_type=DecideOutcome,
        )
        rejected = False
    except Exception as exc:  # noqa: BLE001 — the converter's error type is what this case records
        rejected = True
        del exc
    ids = await handle.query(RoomWorkflow.decided_approval_ids)
    events = suite.recorder.room(room)
    desc = await handle.describe()
    return {
        "id": "F14",
        "title": "An extra DecideRequest field is rejected",
        "steps": [
            "Park, then decide with resumeTurnId still on the payload.",
            "extra=forbid rejects it. The id is not stored and gated_echo does not run.",
        ],
        "expected": {
            "rejected": True,
            "stored": False,
            "gatedEcho": 0,
            "workflowStatus": "RUNNING",
        },
        "actual": {
            "rejected": rejected,
            "stored": approval_id in ids,
            "gatedEcho": _gated_echo(events),
            "workflowStatus": desc.status.name if desc.status else "",
        },
    }


CASES = [
    case_s_id_2,
    case_s_id_7,
    case_s_id_8,
    case_s_id_8_order,
    case_s_id_10,
    case_s_id_11,
    case_s_id_13,
    case_s_id_14,
    case_s_id_15,
    case_s_id_16,
    case_s_id_17,
    case_f11,
    case_f14,
]


async def _wait_run(handle: WorkflowHandle, previous: str) -> bool:
    for _ in range(100):
        if (await handle.describe()).run_id != previous:
            return True
        await asyncio.sleep(0.1)
    return False


async def _wait_closed(handle: WorkflowHandle) -> None:
    for _ in range(200):
        status = (await handle.describe()).status
        if status not in (
            WorkflowExecutionStatus.RUNNING,
            WorkflowExecutionStatus.CONTINUED_AS_NEW,
        ):
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"{handle.id} did not close")


async def _wait_approval(suite: Suite, room: str) -> str:
    for _ in range(CASE_TIMEOUT_S * 10):
        asked = _of(suite.recorder.room(room), "approval.asked")
        if asked:
            return str(asked[-1].get("approvalRequestId") or "")
        await asyncio.sleep(0.1)
    raise TimeoutError(f"{room} did not park")


async def _reset_postgres(url: str) -> None:
    conn = await asyncpg.connect(url)
    try:
        await conn.execute("DROP TABLE IF EXISTS orbit_agent_state, orbit_agent_idempotency")
    finally:
        await conn.close()
    # Create the tables once. Three workers calling ensure_schema together can
    # lose the CREATE TABLE IF NOT EXISTS race and exit.
    store = PostgresStateStore(
        lambda: asyncpg.connect(url),
        StateCipher(fernet=None, allow_plaintext=True),
    )
    await store.ensure_schema()


def _modes() -> dict[str, dict[str, str]]:
    base = {
        "ORBIT_MODEL_MODE": "mock",
        "ORBIT_E2E": "1",
        # Main requires a Fernet key for Postgres. This suite does not test encryption.
        "ORBIT_ALLOW_PLAINTEXT_STATE": "1",
    }
    return {
        "id": base,
        "ttl": {
            **base,
            "ORBIT_CAN_TURN_THRESHOLD": "1",
            "ORBIT_DECIDED_APPROVAL_TTL_S": "5",
        },
        "slow": {**base, "ORBIT_E2E_RESOLVE_DELAY_S": str(RESOLVE_DELAY_S)},
    }


async def run(out: Path, logs: Path) -> int:
    store = os.environ.get("ORBIT_TEST_POSTGRES_URL") or os.environ.get("ORBIT_STATE_STORE_URL", "")
    if not store:
        print("ORBIT_TEST_POSTGRES_URL is required", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    await _reset_postgres(store)
    recorder = Recorder()
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/internal/events", recorder.ingest)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    rows: list[dict] = []
    async with await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter
    ) as temporal:
        info = await temporal.client.workflow_service.get_system_info(GetSystemInfoRequest())
        base = {key: value for key, value in os.environ.items() if not key.startswith("ORBIT_")}
        base.update(
            {
                "TEMPORAL_ADDRESS": temporal.client.service_client.config.target_host,
                "TEMPORAL_NAMESPACE": temporal.client.namespace,
                "ORBIT_EVENT_INGEST_URL": f"http://127.0.0.1:{port}/internal/events",
                "ORBIT_INTERNAL_TOKEN": INGEST_TOKEN,
                "ORBIT_WORKER_BIND": "127.0.0.1",
                "ORBIT_STATE_STORE_URL": store,
                "PYTHONUNBUFFERED": "1",
            }
        )
        processes: dict[str, list[subprocess.Popen]] = {}
        for mode, extra in _modes().items():
            env = {
                **base,
                **extra,
                "TEMPORAL_TASK_QUEUE": QUEUES[mode],
                "ORBIT_WORKER_PORT": str(_free_port()),
            }
            processes[mode] = [
                _start("orbit_orch.main", env, logs, f"orbit-orch-{mode}"),
                _start("orbit_worker.main", env, logs, f"orbit-worker-{mode}"),
            ]
        suite = Suite(temporal.client, recorder, logs)
        try:
            for case in CASES:
                try:
                    row = await asyncio.wait_for(case(suite), CASE_TIMEOUT_S)
                    row["pass"] = _passes(row)
                except Exception as exc:  # noqa: BLE001
                    chain: list[str] = []
                    current: BaseException | None = exc
                    while current is not None and len(chain) < 6:
                        chain.append(f"{type(current).__name__}: {current}")
                        current = current.__cause__
                    row = {
                        "id": case.__name__.replace("case_", "").replace("_", "-").upper(),
                        "steps": [],
                        "expected": "the case completes",
                        "actual": " | ".join(chain),
                        "pass": False,
                    }
                rows.append(row)
        finally:
            _stop([process for pair in processes.values() for process in pair])
    await runner.cleanup()
    failed = [row["id"] for row in rows if not row["pass"]]
    report = {
        "suite": "e2e-decide-idempotency",
        "commit": _commit(""),
        "components": _versions(info.server_version),
        "setup": [
            "Fresh local Temporal dev server (in-memory).",
            "orbit-orch and orbit-worker on orbit-e2e-id (mock model, real Postgres, plaintext state allowed).",
            "orbit-orch and orbit-worker on orbit-e2e-id-ttl (TTL 5s, continue-as-new threshold 1).",
            f"orbit-orch and orbit-worker on orbit-e2e-id-slow (resolve activity sleeps {RESOLVE_DELAY_S}s).",
            "Workers post events to a recording ingest stub.",
            "Rooms are driven with runTurn and decide Updates. S-ID-11 and the child cap cases signal resolve.",
            "S-ID-12 is the control delivery_state table and is not run in this suite.",
            "S-ID-13 asserts the workflow half only. The HTTP status is orbit-control #16.",
        ],
        "cases": rows,
        "passed": len(rows) - len(failed),
        "failed": failed,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['id']}")
    print(f"{report['passed']}/{len(rows)} passed; report: {out}")
    return 1 if failed or recorder.unauthorized else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--out", type=Path, default=Path("artifacts/e2e-decide-idempotency.json")
    )
    run_parser.add_argument("--logs", type=Path, default=Path("e2e-logs/decide"))
    args = parser.parse_args()
    return asyncio.run(run(args.out, args.logs))


if __name__ == "__main__":
    sys.exit(main())

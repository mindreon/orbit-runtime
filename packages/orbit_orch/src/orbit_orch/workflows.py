"""Durable workflows. Activities are named, never implemented here."""

from datetime import UTC, datetime, timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, TemporalError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.models import (
        DECIDED_APPROVALS_LIMIT_MESSAGE,
        MAX_DECIDED_APPROVALS,
        AgentRunInput,
        ApprovalAsk,
        CloneRepoInput,
        CloneRepoOutput,
        CloseSessionInput,
        CloseSessionOutput,
        CloudAgentJobInput,
        CloudAgentSnapshot,
        DecideConfig,
        DecidedApproval,
        DecideOutcome,
        DecideRequest,
        DeliverToolResultInput,
        ExternalCall,
        GatewayExecuteInput,
        GatewayExecuteOutput,
        OpenPrInput,
        OpenPrOutput,
        OpenSessionInput,
        OpenSessionOutput,
        OrbitEvent,
        PushBranchInput,
        PushBranchOutput,
        ResolveApprovalInput,
        ResolveSignal,
        RoomCarryOver,
        RoomCommand,
        RoomFailure,
        RoomSnapshot,
        RoomWorkflowInput,
        RunTurnInput,
        SteerInput,
        TurnResult,
        resume_turn_id,
    )

    from orbit_orch.versioning import (
        AGENT_RUN_SURFACE,
        CLOUD_JOB_SURFACE,
        ROOM_CONTROL_SURFACE,
        ROOM_STAY_OPEN,
    )

_RETRY = RetryPolicy(maximum_attempts=3)
_TIMEOUT = timedelta(minutes=10)
_HARD_FANOUT = 8
_HARD_DEPTH = 4
# Unexpired decided rows that make the workflow log one warning. Not configurable.
_HIGH_WATERMARK = 512
_DEFAULT_TTL_S = 86400
_MIN_TTL_S = 60
_DEFAULT_CAN_TURNS = 200


def _env(name: str) -> str:
    """Workflow config comes from the worker environment. Replay reads the same value."""

    with workflow.unsafe.sandbox_unrestricted():
        import os

        return os.environ.get(name, "")


def _decided_ttl_s() -> int:
    """TTL for decided rows. Below 60 is raised to 60 unless this process is an e2e run."""

    raw = _env("ORBIT_DECIDED_APPROVAL_TTL_S")
    e2e = _env("ORBIT_E2E") == "1"
    try:
        value = int(raw) if raw else _DEFAULT_TTL_S
    except ValueError:
        value = _DEFAULT_TTL_S
    if value < 1:
        return 1 if e2e else _MIN_TTL_S
    if value < _MIN_TTL_S and not e2e:
        return _MIN_TTL_S
    return value


def _can_turn_threshold() -> int:
    raw = _env("ORBIT_CAN_TURN_THRESHOLD")
    try:
        return int(raw) if raw else _DEFAULT_CAN_TURNS
    except ValueError:
        return _DEFAULT_CAN_TURNS


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _prune_decided(entries: list[DecidedApproval], ttl_s: int) -> list[DecidedApproval]:
    """Drop expired ``done`` rows. A ``running`` row stays until its handler finishes."""

    now = workflow.now()
    limit = timedelta(seconds=ttl_s)
    kept: list[DecidedApproval] = []
    for item in entries:
        if item.state == "running" or now - _aware(item.decided_at) <= limit:
            kept.append(item)
    return kept


def _activity_outcome(decision: str) -> str:
    if decision in ("reject", "rejected"):
        return "rejected"
    # allow and allow-always both confirm the parked call. The outcome keeps the original word.
    return "allowed-once"


def _cap(value: int, hard: int) -> int:
    return min(max(value, 1), hard)


def _turn_payload(result: TurnResult) -> dict[str, object]:
    approval = None
    if result.approval is not None:
        approval = {
            "approvalRequestId": result.approval.approval_request_id,
            "toolName": result.approval.tool_name,
            "callId": result.approval.call_id or "",
            "reason": result.approval.reason or "",
        }
    return {
        "status": result.status,
        "approval": approval,
        "texts": [result.text] if result.text else [],
        "error": result.error,
        "errorCode": result.error_code or "",
        "retryable": result.retryable,
        "modelMode": result.model_mode,
        "modelName": result.model_name,
    }


def _turn_failure(result: TurnResult) -> ApplicationError:
    return ApplicationError(result.error, type="ModelRequestFailed", non_retryable=True)


async def _activity(name: str, arg: object, result_type: type, task_queue: str | None = None):
    kwargs = {
        "result_type": result_type,
        "start_to_close_timeout": _TIMEOUT,
        "retry_policy": _RETRY,
    }
    if task_queue:
        kwargs["task_queue"] = task_queue
    return await workflow.execute_activity(name, arg, **kwargs)


@workflow.defn
class AgentRunWorkflow:
    """One worker agent. It has its own session and stops when dissolved."""

    def __init__(self) -> None:
        self._stop = False
        self._resolves: list[str] = []
        self._handled: list[DecidedApproval] = []
        self._ttl_s = _DEFAULT_TTL_S
        self._pending_id = ""
        self._room_id = ""
        self._session_id = ""

    @workflow.signal
    async def dissolve(self) -> None:
        self._stop = True

    @workflow.signal
    async def incoming(self, message: str) -> None:
        del message

    @workflow.signal(name="resolve")
    async def resolve(self, req: ResolveSignal) -> None:
        self._resolves.append(req.approval_request_id)

    @workflow.run
    async def run(self, inp: AgentRunInput) -> str:
        workflow.patched(AGENT_RUN_SURFACE)
        workflow_id = workflow.info().workflow_id
        self._room_id = inp.room_id
        self._ttl_s = _decided_ttl_s()
        opened = await _activity(
            "openSession",
            OpenSessionInput(
                room_id=inp.room_id,
                turn_id=f"{workflow_id}:open",
                permission_preset=inp.permission_preset,
            ),
            OpenSessionOutput,
        )
        self._session_id = opened.session_id
        if self._stop:
            await self._close(inp.room_id, opened.session_id, workflow_id)
            return ""
        result = await _activity(
            "runTurn",
            RunTurnInput(
                room_id=inp.room_id,
                session_id=opened.session_id,
                turn_id=f"{workflow_id}:turn",
                message=inp.prompt,
                state_version=opened.state_version,
            ),
            TurnResult,
        )
        if result.status == "needs_approval" and result.approval is not None and not self._stop:
            self._pending_id = result.approval.approval_request_id
            result = await self._wait_resolve(result)
        await self._close(inp.room_id, opened.session_id, workflow_id)
        if result.status == "failed":
            raise _turn_failure(result)
        return result.text

    async def _wait_resolve(self, result: TurnResult) -> TurnResult:
        """Run the parked tool once. A second signal for that id is a warning."""

        await workflow.wait_condition(lambda: bool(self._resolves) or self._stop)
        if self._stop:
            return result
        # One second for a duplicate signal to land before the tool runs.
        await workflow.sleep(1)
        ran: str | None = None
        while self._resolves:
            approval_id = self._resolves.pop(0)
            if not self._claim_resolve(approval_id):
                continue
            if ran is None:
                ran = approval_id
        if ran is None:
            return result
        try:
            result = await _activity(
                "resolveApproval",
                ResolveApprovalInput(
                    room_id=self._room_id,
                    session_id=self._session_id,
                    turn_id=resume_turn_id(ran),
                    approval_request_id=ran,
                    outcome="allowed-once",  # type: ignore[arg-type]
                ),
                TurnResult,
            )
        finally:
            self._finish_resolve(ran)
        return result

    def _claim_resolve(self, approval_id: str) -> bool:
        # Same order as the room decide handler: prune, cap, then running.
        self._handled = _prune_decided(self._handled, self._ttl_s)
        if any(item.approval_request_id == approval_id for item in self._handled):
            workflow.logger.warning("duplicate resolve ignored approvalRequestId=%s", approval_id)
            return False
        if approval_id != self._pending_id:
            workflow.logger.warning("duplicate resolve ignored approvalRequestId=%s", approval_id)
            return False
        if len(self._handled) >= MAX_DECIDED_APPROVALS:
            workflow.logger.warning("decided_approvals limit approvalRequestId=%s", approval_id)
            return False
        self._handled.append(
            DecidedApproval(
                approval_request_id=approval_id,
                decided_at=workflow.now(),
                state="running",
            )
        )
        return True

    def _finish_resolve(self, approval_id: str) -> None:
        for index, item in enumerate(self._handled):
            if item.approval_request_id == approval_id and item.state == "running":
                self._handled[index] = item.model_copy(update={"state": "done"})
                return

    async def _close(self, room_id: str, session_id: str, workflow_id: str) -> None:
        await _activity(
            "closeSession",
            CloseSessionInput(
                room_id=room_id,
                session_id=session_id,
                turn_id=f"{workflow_id}:close",
            ),
            CloseSessionOutput,
        )


@workflow.defn
class RoomWorkflow:
    """One room. Human approval and external tools wait here."""

    @workflow.init
    def __init__(self, inp: RoomWorkflowInput) -> None:
        # Init runs before any update in the same workflow task. A decide that
        # arrives with continue-as-new must already see the carried table.
        self._room_id = inp.room_id
        self._preset = inp.permission_preset
        self._kind = inp.kind
        self._max_fanout = _cap(inp.max_fanout, _HARD_FANOUT)
        self._max_depth = _cap(inp.max_depth, _HARD_DEPTH)
        self._depth = 0
        self._gateway_queue = inp.gateway_task_queue
        self._status = "idle"
        self._session_id: str | None = None
        self._state_version = 0
        self._approval = None
        self._last_text = ""
        self._error = ""
        self._queue: list[RoomCommand] = []
        self._stop = False
        self._children: list[workflow.ChildWorkflowHandle] = []
        self._decided: list[DecidedApproval] = []
        self._pending: list[ApprovalAsk] = []
        self._fatal: str | None = None
        self._ttl_s = _decided_ttl_s()
        self._can_threshold = _can_turn_threshold()
        self._turns = 0
        self._can_requested = False
        self._watermark_logged = False
        # Do not prune here. S-ID-10 fails if the handler checks the cap before it prunes.
        if inp.carry_over is not None:
            self._decided = list(inp.carry_over.decided_approvals)
            if inp.carry_over.session_id:
                self._restore(inp.carry_over)

    @workflow.run
    async def run(self, inp: RoomWorkflowInput) -> RoomSnapshot:
        workflow.patched(ROOM_CONTROL_SURFACE)
        if self._session_id is None:
            # Control starts the workflow and polls getRoomView until a session exists.
            await self._open_session(f"{inp.room_id}:bootstrap")
        while not self._stop:
            await workflow.wait_condition(
                lambda: (
                    bool(self._queue)
                    or self._stop
                    or self._fatal is not None
                    or self._can_requested
                )
            )
            if self._fatal:
                await self._fail_decided_limit()
            if self._queue:
                await self._handle(self._queue.pop(0))
            if self._can_requested and workflow.all_handlers_finished():
                self._can_requested = False
                self._maybe_continue()
            if self._stop and not self._queue:
                break
        return self._snapshot()

    @workflow.signal
    async def command(self, cmd: RoomCommand) -> None:
        self._queue.append(cmd)

    @workflow.query
    def snapshot(self) -> RoomSnapshot:
        return self._snapshot()

    @workflow.query(name="getRoomView")
    def get_room_view(self) -> dict[str, str]:
        approval_id = ""
        if self._approval is not None:
            approval_id = self._approval.approval_request_id
        return {
            "roomId": self._room_id,
            "state": self._status,
            "sessionId": self._session_id or "",
            "kind": self._kind,
            "pendingApprovalRequestId": approval_id,
        }

    @workflow.update(name="runTurn")
    async def update_run_turn(self, req: dict[str, str]) -> dict[str, object]:
        turn_id = req.get("turnId") or req.get("turn_id") or ""
        message = req.get("message") or ""
        if self._status != "running":
            raise ApplicationError(f"runTurn is illegal from {self._status}")
        result = await _activity(
            "runTurn",
            RunTurnInput(
                room_id=self._room_id,
                session_id=self._session_id or "",
                turn_id=turn_id,
                message=message,
                state_version=self._state_version,
            ),
            TurnResult,
        )
        await self._after_turn(result, turn_id)
        return _turn_payload(result)

    @workflow.update(name="decide")
    async def update_decide(self, req: DecideRequest) -> DecideOutcome:
        # F1 order is below, not "write running before anything else".
        approval_id = req.approval_request_id
        if self._find_decided(approval_id) is not None:
            return await self._await_decided(approval_id)

        # Prune, then the hard cap, then the running row. All of this is before the first await.
        self._decided = _prune_decided(self._decided, self._ttl_s)
        self._note_watermark()
        if len(self._decided) >= MAX_DECIDED_APPROVALS:
            self._fatal = "DECIDED_APPROVALS_LIMIT"
            raise ApplicationError(
                DECIDED_APPROVALS_LIMIT_MESSAGE,
                type="DECIDED_APPROVALS_LIMIT",
                non_retryable=True,
            )
        resume_id = resume_turn_id(approval_id)
        self._decided.append(
            DecidedApproval(
                approval_request_id=approval_id,
                decided_at=workflow.now(),
                state="running",
            )
        )
        self._drop_pending(approval_id)
        self._note_watermark()
        outcome: DecideOutcome | None = None
        try:
            outcome = await self._resume(req, resume_id)
            return outcome
        finally:
            # Also runs when the handler is cancelled, so a waiter is not stuck on running.
            self._mark_done(approval_id, outcome, req, resume_id)
            self._turns += 1
            self._can_requested = True

    @update_decide.validator
    def validate_decide(self, req: DecideRequest) -> None:
        approval_id = req.approval_request_id
        if self._find_decided(approval_id) is not None:
            return
        if any(item.approval_request_id == approval_id for item in self._pending):
            return
        if self._approval is not None and self._approval.approval_request_id == approval_id:
            return
        raise ApplicationError(
            "approval is not pending",
            type="APPROVAL_UNKNOWN",
            non_retryable=True,
        )

    @workflow.query(name="decideOutcome")
    def decide_outcome(self, approval_request_id: str) -> DecideOutcome | None:
        item = self._find_decided(approval_request_id)
        if item is None or item.state != "done":
            return None
        return item.outcome

    @workflow.query(name="decidedApprovalIds")
    def decided_approval_ids(self) -> list[str]:
        return [item.approval_request_id for item in self._decided]

    @workflow.query(name="decideConfig")
    def decide_config(self) -> DecideConfig:
        return DecideConfig(ttl_s=self._ttl_s, max_decided=MAX_DECIDED_APPROVALS)

    @workflow.signal(name="steer")
    async def on_steer(self, req: dict[str, str]) -> None:
        self._queue.append(
            RoomCommand(
                kind="steer",
                turn_id=req.get("turnId") or "steer",
                message=req.get("instruction") or "",
            )
        )

    @workflow.signal(name="abort")
    async def on_abort(self, req: dict[str, str]) -> None:
        self._queue.append(RoomCommand(kind="abort", turn_id=req.get("turnId") or "abort"))

    async def _handle(self, cmd: RoomCommand) -> None:
        if cmd.kind == "open":
            await self._open(cmd)
        elif cmd.kind == "message":
            await self._message(cmd)
        elif cmd.kind == "approve":
            await self._approve(cmd)
        elif cmd.kind == "abort":
            await self._abort(cmd)
        elif cmd.kind == "steer":
            await self._steer(cmd)

    async def _open(self, cmd: RoomCommand) -> None:
        if self._session_id:
            return
        await self._open_session(cmd.turn_id)

    async def _open_session(self, turn_id: str) -> None:
        opened = await _activity(
            "openSession",
            OpenSessionInput(
                room_id=self._room_id,
                turn_id=turn_id,
                permission_preset=self._preset,  # type: ignore[arg-type]
            ),
            OpenSessionOutput,
        )
        self._session_id = opened.session_id
        self._state_version = opened.state_version
        self._status = "running"

    async def _steer(self, cmd: RoomCommand) -> None:
        self._require("running", cmd.kind)
        result = await _activity(
            "steer",
            SteerInput(
                room_id=self._room_id,
                session_id=self._session_id or "",
                turn_id=cmd.turn_id,
                state_version=self._state_version,
                hint=cmd.message,
            ),
            TurnResult,
        )
        await self._after_turn(result, cmd.turn_id)

    async def _message(self, cmd: RoomCommand) -> None:
        self._require("running", cmd.kind)
        result = await _activity(
            "runTurn",
            RunTurnInput(
                room_id=self._room_id,
                session_id=self._session_id or "",
                turn_id=cmd.turn_id,
                message=cmd.message,
                state_version=self._state_version,
            ),
            TurnResult,
        )
        await self._after_turn(result, cmd.turn_id)

    async def _approve(self, cmd: RoomCommand) -> None:
        self._require("awaiting_approval", cmd.kind)
        if self._approval is None or self._session_id is None:
            raise ApplicationError("no parked approval")
        result = await _activity(
            "resolveApproval",
            ResolveApprovalInput(
                room_id=self._room_id,
                session_id=self._session_id,
                turn_id=cmd.turn_id,
                approval_request_id=self._approval.approval_request_id,
                outcome=cmd.outcome,
            ),
            TurnResult,
        )
        await self._after_turn(result, cmd.turn_id)

    async def _after_turn(self, result: TurnResult, turn_id: str) -> None:
        self._apply_turn(result)
        step = 0
        while result.status == "needs_external" and result.external is not None:
            step += 1
            result = await self._dispatch_external(
                result.external, f"{turn_id}:x{step}", result.state_version
            )
            self._apply_turn(result)
        await self._close_if_done(turn_id)

    async def _dispatch_external(
        self,
        call: ExternalCall,
        turn_id: str,
        state_version: int,
    ) -> TurnResult:
        output, metadata, state = await self._execute_external(call)
        return await _activity(
            "deliverToolResult",
            DeliverToolResultInput(
                room_id=self._room_id,
                session_id=self._session_id or "",
                turn_id=turn_id,
                state_version=state_version,
                tool_name=call.tool_name,
                call_id=call.call_id,
                output=output,
                metadata=metadata,
                result_state=state,  # type: ignore[arg-type]
            ),
            TurnResult,
        )

    async def _execute_external(self, call: ExternalCall) -> tuple[str, dict[str, str], str]:
        if call.tool_name == "agent_spawn":
            return await self._spawn(call)
        if call.tool_name == "agent_send":
            return await self._send(call)
        if call.tool_name == "agent_wait":
            return await self._wait_children()
        if call.tool_name == "team_dissolve":
            return await self._dissolve()
        gateway = await _activity(
            "gatewayExecute",
            GatewayExecuteInput(
                room_id=self._room_id,
                session_id=self._session_id or "",
                tool_name=call.tool_name,
                call_id=call.call_id,
                arguments=call.arguments,
            ),
            GatewayExecuteOutput,
            task_queue=self._gateway_queue,
        )
        return gateway.output, gateway.metadata, gateway.result_state

    async def _spawn(self, call: ExternalCall) -> tuple[str, dict[str, str], str]:
        if len(self._children) >= self._max_fanout:
            return "fan-out limit", {}, "error"
        if self._depth + 1 >= self._max_depth:
            return "depth limit", {}, "error"
        child_id = f"{workflow.info().workflow_id}/agent/{call.call_id}"
        handle = await workflow.start_child_workflow(
            AgentRunWorkflow.run,
            AgentRunInput(
                room_id=self._room_id,
                prompt=call.arguments.get("prompt", ""),
                permission_preset=self._preset,  # type: ignore[arg-type]
                depth=self._depth + 1,
                max_depth=self._max_depth,
            ),
            id=child_id,
        )
        self._children.append(handle)
        return "started", {"workflow_id": child_id, "status": "started"}, "success"

    async def _send(self, call: ExternalCall) -> tuple[str, dict[str, str], str]:
        child_id = call.arguments.get("workflow_id", "")
        handle = next((child for child in self._children if child.id == child_id), None)
        if handle is None:
            return "unknown child", {}, "error"
        await handle.signal("incoming", call.arguments.get("message", ""))
        return "sent", {"status": "sent"}, "success"

    async def _wait_children(self) -> tuple[str, dict[str, str], str]:
        texts: list[str] = []
        for handle in self._children:
            try:
                texts.append(str(await handle))
            except TemporalError as exc:
                workflow.logger.warning("child %s failed: %s", handle.id, exc)
                texts.append(exc.__class__.__name__)
        text = "\n".join(texts)
        return text, {"text": text}, "success"

    async def _dissolve(self) -> tuple[str, dict[str, str], str]:
        for handle in self._children:
            try:
                await handle.signal("dissolve")
            except TemporalError as exc:
                workflow.logger.warning("dissolve skipped for %s: %s", handle.id, exc)
            # Child handles are asyncio tasks. cancel() returns a bool.
            handle.cancel()
        return "dissolved", {"dissolved": "true"}, "success"

    async def _close_if_done(self, turn_id: str) -> None:
        if self._status != "closed" or self._session_id is None:
            return
        closed = await _activity(
            "closeSession",
            CloseSessionInput(
                room_id=self._room_id,
                session_id=self._session_id,
                turn_id=turn_id + ":close",
            ),
            CloseSessionOutput,
        )
        self._state_version = closed.state_version

    async def _abort(self, cmd: RoomCommand) -> None:
        if self._status not in ("running", "awaiting_approval", "awaiting_external"):
            raise ApplicationError(f"abort is illegal from {self._status}")
        await self._dissolve()
        if self._session_id is not None:
            closed = await _activity(
                "closeSession",
                CloseSessionInput(
                    room_id=self._room_id,
                    session_id=self._session_id,
                    turn_id=cmd.turn_id,
                ),
                CloseSessionOutput,
            )
            self._state_version = closed.state_version
        self._status = "closed"
        self._stop = True

    def _apply_turn(self, result: TurnResult) -> None:
        if result.status == "failed":
            # The worker kept the previous state, so the room keeps its status.
            self._error = result.error
            return
        self._error = ""
        self._state_version = result.state_version
        self._last_text = result.text
        if result.status == "needs_approval" and result.approval is not None:
            self._approval = result.approval
            self._pending = [result.approval]
            self._status = "awaiting_approval"
            return
        self._approval = None
        self._pending = []
        if result.status == "needs_external":
            self._status = "awaiting_external"
            return
        if result.status == "completed":
            # Control saves the user message, then waits on the runTurn Update.
            # Closing here returns the workflow before that Update finishes.
            if workflow.patched(ROOM_STAY_OPEN):
                self._status = "running"
                return
            self._status = "closed"
            self._stop = True
            return
        self._status = "running"

    def _restore(self, carry: RoomCarryOver) -> None:
        self._session_id = carry.session_id
        self._state_version = carry.state_version
        self._status = carry.status
        self._pending = list(carry.pending_approvals)
        self._approval = self._pending[0] if self._pending else None
        self._last_text = carry.last_text
        if carry.preset:
            self._preset = carry.preset

    def _find_decided(self, approval_id: str) -> DecidedApproval | None:
        for item in self._decided:
            if item.approval_request_id == approval_id:
                return item
        return None

    def _drop_pending(self, approval_id: str) -> None:
        self._pending = [item for item in self._pending if item.approval_request_id != approval_id]
        if self._approval is not None and self._approval.approval_request_id == approval_id:
            self._approval = self._pending[0] if self._pending else None

    def _note_watermark(self) -> None:
        count = len(self._decided)
        if count < _HIGH_WATERMARK or self._watermark_logged:
            return
        self._watermark_logged = True
        workflow.logger.warning(
            "decided_approvals_high_watermark{roomId=%s,count=%s}",
            self._room_id,
            count,
        )

    async def _await_decided(self, approval_id: str) -> DecideOutcome:
        if self._done_outcome(approval_id) is None:
            await workflow.wait_condition(
                lambda: self._done_outcome(approval_id) is not None or self._fatal is not None
            )
        outcome = self._done_outcome(approval_id)
        if outcome is not None:
            return outcome
        raise ApplicationError(
            DECIDED_APPROVALS_LIMIT_MESSAGE,
            type="DECIDED_APPROVALS_LIMIT",
            non_retryable=True,
        )

    def _done_outcome(self, approval_id: str) -> DecideOutcome | None:
        item = self._find_decided(approval_id)
        if item is None or item.state != "done" or item.outcome is None:
            return None
        return item.outcome

    async def _resume(self, req: DecideRequest, resume_id: str) -> DecideOutcome:
        if self._session_id is None:
            raise ApplicationError(
                "approval is not pending",
                type="APPROVAL_UNKNOWN",
                non_retryable=True,
            )
        # Already decided on the room: do not signal a child. The main agent resumes here.
        result = await _activity(
            "resolveApproval",
            ResolveApprovalInput(
                room_id=self._room_id,
                session_id=self._session_id,
                turn_id=resume_id,
                approval_request_id=req.approval_request_id,
                outcome=_activity_outcome(req.decision),  # type: ignore[arg-type]
            ),
            TurnResult,
        )
        await self._after_turn(result, resume_id)
        return DecideOutcome(
            decision=req.decision,
            agent_id="main",
            resume_turn_id=resume_id,
            turn_status=result.status,
            error_code=result.error_code,
        )

    def _mark_done(
        self,
        approval_id: str,
        outcome: DecideOutcome | None,
        req: DecideRequest,
        resume_id: str,
    ) -> None:
        if outcome is None:
            outcome = DecideOutcome(
                decision=req.decision,
                agent_id="main",
                resume_turn_id=resume_id,
                turn_status="failed",
            )
        for index, item in enumerate(self._decided):
            if item.approval_request_id == approval_id:
                self._decided[index] = item.model_copy(update={"state": "done", "outcome": outcome})
                return

    async def _fail_decided_limit(self) -> None:
        event = OrbitEvent(
            type="room.failed",
            session_id=self._session_id or self._room_id,
            room_id=self._room_id,
            text=DECIDED_APPROVALS_LIMIT_MESSAGE,
            failure=RoomFailure(),
        )
        try:
            await _activity("ingestRoomEvent", event, bool)
        except TemporalError:
            workflow.logger.warning("room.failed ingest failed roomId=%s", self._room_id)
        for child in self._children:
            child.cancel()
        raise ApplicationError(
            DECIDED_APPROVALS_LIMIT_MESSAGE,
            type="DECIDED_APPROVALS_LIMIT",
            non_retryable=True,
        )

    def _maybe_continue(self) -> None:
        if self._children or self._fatal:
            return
        suggested = workflow.info().is_continue_as_new_suggested()
        if self._turns < self._can_threshold and not suggested:
            return
        if self._can_threshold <= 0 and not suggested:
            return
        self._decided = _prune_decided(self._decided, self._ttl_s)
        workflow.continue_as_new(
            RoomWorkflowInput(
                room_id=self._room_id,
                permission_preset=self._preset,  # type: ignore[arg-type]
                kind=self._kind,
                max_fanout=self._max_fanout,
                max_depth=self._max_depth,
                gateway_task_queue=self._gateway_queue,
                carry_over=RoomCarryOver(
                    room_id=self._room_id,
                    session_id=self._session_id,
                    state_version=self._state_version,
                    preset=self._preset,  # type: ignore[arg-type]
                    status=self._status,  # type: ignore[arg-type]
                    pending_approvals=list(self._pending),
                    decided_approvals=list(self._decided),
                    child_workflow_ids=[child.id for child in self._children],
                    last_text=self._last_text,
                ),
            )
        )

    def _require(self, expected: str, kind: str) -> None:
        if self._status != expected:
            raise ApplicationError(f"{kind} is illegal from {self._status}")

    def _snapshot(self) -> RoomSnapshot:
        return RoomSnapshot(
            status=self._status,  # type: ignore[arg-type]
            room_id=self._room_id,
            session_id=self._session_id,
            state_version=self._state_version,
            approval=self._approval,
            last_text=self._last_text,
            error=self._error,
            child_workflow_ids=[child.id for child in self._children],
        )


@workflow.defn
class CloudAgentJob:
    """clone, one turn loop, push a branch, and open a pull request."""

    def __init__(self) -> None:
        self._outcome = ""

    @workflow.signal
    async def approve(self, outcome: str) -> None:
        self._outcome = outcome

    @workflow.run
    async def run(self, inp: CloudAgentJobInput) -> CloudAgentSnapshot:
        workflow.patched(CLOUD_JOB_SURFACE)
        cloned = await _activity(
            "cloneRepo",
            CloneRepoInput(session_id=inp.job_id, repo_url=inp.repo_url),
            CloneRepoOutput,
        )
        opened = await _activity(
            "openSession",
            OpenSessionInput(
                room_id=inp.job_id,
                turn_id=f"{inp.job_id}:open",
                permission_preset=inp.permission_preset,
            ),
            OpenSessionOutput,
        )
        result = await _activity(
            "runTurn",
            RunTurnInput(
                room_id=inp.job_id,
                session_id=opened.session_id,
                turn_id=f"{inp.job_id}:turn",
                message=inp.prompt,
                state_version=opened.state_version,
            ),
            TurnResult,
        )
        while result.status == "needs_approval" and result.approval is not None:
            await workflow.wait_condition(lambda: bool(self._outcome))
            outcome = self._outcome
            self._outcome = ""
            result = await _activity(
                "resolveApproval",
                ResolveApprovalInput(
                    room_id=inp.job_id,
                    session_id=opened.session_id,
                    turn_id=f"{inp.job_id}:approve:{result.state_version}",
                    approval_request_id=result.approval.approval_request_id,
                    outcome=outcome,  # type: ignore[arg-type]
                ),
                TurnResult,
            )
        if result.status == "failed":
            await _activity(
                "closeSession",
                CloseSessionInput(
                    room_id=inp.job_id,
                    session_id=opened.session_id,
                    turn_id=f"{inp.job_id}:close",
                ),
                CloseSessionOutput,
            )
            raise _turn_failure(result)
        pushed = await _activity(
            "pushBranch",
            PushBranchInput(
                session_id=inp.job_id,
                workdir=cloned.workdir,
                branch=inp.branch,
            ),
            PushBranchOutput,
        )
        pr = await _activity(
            "openPr",
            OpenPrInput(
                session_id=inp.job_id,
                repo_url=inp.repo_url,
                branch=inp.branch,
                title=inp.prompt[:80],
            ),
            OpenPrOutput,
        )
        await _activity(
            "closeSession",
            CloseSessionInput(
                room_id=inp.job_id,
                session_id=opened.session_id,
                turn_id=f"{inp.job_id}:close",
            ),
            CloseSessionOutput,
        )
        return CloudAgentSnapshot(
            status="closed",
            job_id=inp.job_id,
            session_id=opened.session_id,
            workdir=cloned.workdir,
            branch=pushed.branch,
            pr_url=pr.pr_url,
            last_text=result.text,
        )

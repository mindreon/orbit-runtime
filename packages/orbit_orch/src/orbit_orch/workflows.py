"""RoomWorkflow: durable FSM. Activities are named, never implemented here."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.models import (
        CloseSessionInput,
        CloseSessionOutput,
        OpenSessionInput,
        OpenSessionOutput,
        ResolveApprovalInput,
        RoomCommand,
        RoomSnapshot,
        RoomWorkflowInput,
        RunTurnInput,
        TurnResult,
    )

_RETRY = RetryPolicy(maximum_attempts=3)
_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class RoomWorkflow:
    """One room. Human approval waits here, not inside the agent process."""

    def __init__(self) -> None:
        self._room_id = ""
        self._preset = "workspace-write"
        self._status = "idle"
        self._session_id: str | None = None
        self._state_version = 0
        self._approval = None
        self._last_text = ""
        self._error = ""
        self._queue: list[RoomCommand] = []
        self._stop = False

    @workflow.run
    async def run(self, inp: RoomWorkflowInput) -> RoomSnapshot:
        self._room_id = inp.room_id
        self._preset = inp.permission_preset
        while not self._stop:
            await workflow.wait_condition(lambda: bool(self._queue) or self._stop)
            if self._stop and not self._queue:
                break
            await self._handle(self._queue.pop(0))
        return self._snapshot()

    @workflow.signal
    async def command(self, cmd: RoomCommand) -> None:
        self._queue.append(cmd)

    @workflow.query
    def snapshot(self) -> RoomSnapshot:
        return self._snapshot()

    async def _handle(self, cmd: RoomCommand) -> None:
        if cmd.kind == "open":
            await self._open(cmd)
        elif cmd.kind == "message":
            await self._message(cmd)
        elif cmd.kind == "approve":
            await self._approve(cmd)
        elif cmd.kind == "abort":
            await self._abort(cmd)

    async def _open(self, cmd: RoomCommand) -> None:
        self._require("idle", cmd.kind)
        opened = await workflow.execute_activity(
            "openSession",
            OpenSessionInput(
                room_id=self._room_id,
                turn_id=cmd.turn_id,
                permission_preset=self._preset,
            ),
            result_type=OpenSessionOutput,
            start_to_close_timeout=_TIMEOUT,
            retry_policy=_RETRY,
        )
        self._session_id = opened.session_id
        self._state_version = opened.state_version
        self._status = "running"

    async def _message(self, cmd: RoomCommand) -> None:
        self._require("running", cmd.kind)
        # The message Activity is registered by the worker under this name.
        # Importing the worker here would pull AgentScope into workflow code.
        result = await workflow.execute_activity(
            "runTurn",
            RunTurnInput(
                room_id=self._room_id,
                session_id=self._session_id or "",
                turn_id=cmd.turn_id,
                message=cmd.message,
                state_version=self._state_version,
            ),
            result_type=TurnResult,
            start_to_close_timeout=_TIMEOUT,
            retry_policy=_RETRY,
        )
        self._apply_turn(result)
        await self._close_if_done(cmd.turn_id)

    async def _approve(self, cmd: RoomCommand) -> None:
        self._require("awaiting_approval", cmd.kind)
        if self._approval is None or self._session_id is None:
            raise ApplicationError("no parked approval")
        result = await workflow.execute_activity(
            "resolveApproval",
            ResolveApprovalInput(
                room_id=self._room_id,
                session_id=self._session_id,
                turn_id=cmd.turn_id,
                approval_request_id=self._approval.approval_request_id,
                outcome=cmd.outcome,
            ),
            result_type=TurnResult,
            start_to_close_timeout=_TIMEOUT,
            retry_policy=_RETRY,
        )
        self._apply_turn(result)
        await self._close_if_done(cmd.turn_id)

    async def _close_if_done(self, turn_id: str) -> None:
        if self._status != "closed" or self._session_id is None:
            return
        closed = await workflow.execute_activity(
            "closeSession",
            CloseSessionInput(
                room_id=self._room_id,
                session_id=self._session_id,
                turn_id=turn_id + ":close",
            ),
            result_type=CloseSessionOutput,
            start_to_close_timeout=_TIMEOUT,
            retry_policy=_RETRY,
        )
        self._state_version = closed.state_version

    async def _abort(self, cmd: RoomCommand) -> None:
        if self._status not in ("running", "awaiting_approval"):
            raise ApplicationError(f"abort is illegal from {self._status}")
        if self._session_id is not None:
            closed = await workflow.execute_activity(
                "closeSession",
                CloseSessionInput(
                    room_id=self._room_id,
                    session_id=self._session_id,
                    turn_id=cmd.turn_id,
                ),
                result_type=CloseSessionOutput,
                start_to_close_timeout=_TIMEOUT,
                retry_policy=_RETRY,
            )
            self._state_version = closed.state_version
        self._status = "closed"
        self._stop = True

    def _apply_turn(self, result: TurnResult) -> None:
        self._state_version = result.state_version
        self._last_text = result.text
        if result.status == "needs_approval":
            self._approval = result.approval
            self._status = "awaiting_approval"
            return
        self._approval = None
        if result.status == "completed":
            self._status = "closed"
            self._stop = True
            return
        self._status = "running"

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
        )

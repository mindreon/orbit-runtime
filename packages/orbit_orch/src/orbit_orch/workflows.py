"""Durable workflows. Activities are named, never implemented here."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, TemporalError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.models import (
        AgentRunInput,
        CloneRepoInput,
        CloneRepoOutput,
        CloseSessionInput,
        CloseSessionOutput,
        CloudAgentJobInput,
        CloudAgentSnapshot,
        DeliverToolResultInput,
        ExternalCall,
        GatewayExecuteInput,
        GatewayExecuteOutput,
        OpenPrInput,
        OpenPrOutput,
        OpenSessionInput,
        OpenSessionOutput,
        PushBranchInput,
        PushBranchOutput,
        ResolveApprovalInput,
        RoomCommand,
        RoomSnapshot,
        RoomWorkflowInput,
        RunTurnInput,
        SteerInput,
        TurnResult,
    )

    from orbit_orch.versioning import (
        AGENT_RUN_SURFACE,
        CLOUD_JOB_SURFACE,
        ROOM_CONTROL_SURFACE,
    )

_RETRY = RetryPolicy(maximum_attempts=3)
_TIMEOUT = timedelta(minutes=10)
_HARD_FANOUT = 8
_HARD_DEPTH = 4


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
    }


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

    @workflow.signal
    async def dissolve(self) -> None:
        self._stop = True

    @workflow.signal
    async def incoming(self, message: str) -> None:
        del message

    @workflow.run
    async def run(self, inp: AgentRunInput) -> str:
        workflow.patched(AGENT_RUN_SURFACE)
        workflow_id = workflow.info().workflow_id
        opened = await _activity(
            "openSession",
            OpenSessionInput(
                room_id=inp.room_id,
                turn_id=f"{workflow_id}:open",
                permission_preset=inp.permission_preset,
            ),
            OpenSessionOutput,
        )
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
        await self._close(inp.room_id, opened.session_id, workflow_id)
        return result.text

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
        self._children: list[workflow.ChildWorkflowHandle] = []
        self._max_fanout = 4
        self._max_depth = 2
        self._depth = 0
        self._gateway_queue = "orbit-gateway"
        self._kind = "solo"

    @workflow.run
    async def run(self, inp: RoomWorkflowInput) -> RoomSnapshot:
        workflow.patched(ROOM_CONTROL_SURFACE)
        self._room_id = inp.room_id
        self._preset = inp.permission_preset
        self._kind = inp.kind
        self._max_fanout = _cap(inp.max_fanout, _HARD_FANOUT)
        self._max_depth = _cap(inp.max_depth, _HARD_DEPTH)
        self._gateway_queue = inp.gateway_task_queue
        # Control starts the workflow and polls getRoomView until a session exists.
        await self._open_session(f"{inp.room_id}:bootstrap")
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
    async def update_decide(self, req: dict[str, str]) -> dict[str, object]:
        if self._status != "awaiting_approval" or self._approval is None:
            raise ApplicationError(f"decide is illegal from {self._status}")
        decision = req.get("decision") or "allow"
        outcome = "rejected" if decision in ("reject", "rejected") else "allowed-once"
        turn_id = req.get("resumeTurnId") or req.get("turnId") or "decide"
        result = await _activity(
            "resolveApproval",
            ResolveApprovalInput(
                room_id=self._room_id,
                session_id=self._session_id or "",
                turn_id=turn_id,
                approval_request_id=req.get("approvalRequestId")
                or self._approval.approval_request_id,
                outcome=outcome,  # type: ignore[arg-type]
            ),
            TurnResult,
        )
        await self._after_turn(result, turn_id)
        return {"decision": decision, "turn": _turn_payload(result)}

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
            result = await self._dispatch_external(result.external, f"{turn_id}:x{step}", result.state_version)
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
        self._state_version = result.state_version
        self._last_text = result.text
        if result.status == "needs_approval":
            self._approval = result.approval
            self._status = "awaiting_approval"
            return
        self._approval = None
        if result.status == "needs_external":
            self._status = "awaiting_external"
            return
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

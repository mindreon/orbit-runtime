"""AgentRuntime: one Activity rebuilds the agent from a saved AgentState.

Nothing about AgentScope crosses this class's return values. A parked
approval or external tool call is a TurnResult. The live agent object is
discarded when the Activity returns, and the next Activity builds a new
one from the blob.
"""

import json
import logging
from uuid import uuid4

from agentscope.agent import Agent
from agentscope.event import (
    ConfirmResult,
    ExternalExecutionResultEvent,
    RequireExternalExecutionEvent,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
    UserInterruptEvent,
)
from agentscope.message import (
    HintBlock,
    Msg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)
from agentscope.middleware import TracingMiddleware
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from agentscope.types import ReplyFinishedReason
from orbit_contracts.models import (
    AbortSessionInput,
    ApprovalAsk,
    CloseSessionOutput,
    DeliverToolResultInput,
    ExternalCall,
    OpenSessionInput,
    OpenSessionOutput,
    OrbitEvent,
    ResolveApprovalInput,
    RunTurnInput,
    SteerInput,
    TurnFailure,
    TurnResult,
)

from orbit_worker.chat_model import ModelConfig, ModelRequestError, build_chat_model
from orbit_worker.events import MemoryEventIngest
from orbit_worker.isolation import IsolationSnapshot
from orbit_worker.store import MemoryStateStore, SessionBlob, StateStore
from orbit_worker.tools import orbit_tools

logger = logging.getLogger(__name__)

_PRESETS: dict[str, PermissionMode] = {
    "workspace-write": PermissionMode.ACCEPT_EDITS,
    "read-only": PermissionMode.EXPLORE,
    "danger-full-access": PermissionMode.BYPASS,
}

_BOOL_METADATA = {"ok", "dissolved"}


def _gated_echo(text: str) -> ToolChunk:
    """A custom tool. Custom tools ask for confirmation unless a rule allows them."""

    return ToolChunk(
        content=[TextBlock(text=f"echo:{text}")],
        state=ToolResultState.SUCCESS,
    )


class AgentRuntime:
    def __init__(
        self,
        store: StateStore | None = None,
        ingest: MemoryEventIngest | None = None,
        isolation: IsolationSnapshot | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self._model_config = model_config or ModelConfig()
        self._store: StateStore = store if store is not None else MemoryStateStore()
        self._ingest = ingest if ingest is not None else MemoryEventIngest()
        self._isolation = isolation or IsolationSnapshot(
            mode="local",
            share_net=False,
            backend="local",
            image_digest="",
            cpu_max="",
            memory_max="",
            cgroup_applied=False,
        )

    async def open_session(self, inp: OpenSessionInput) -> OpenSessionOutput:
        # Same turn id reopens the same session. A retry must not fork state.
        existing = await self._store.find_by_idempotency(
            inp.room_id, _key(inp.turn_id, "openSession")
        )
        if existing is not None:
            return OpenSessionOutput(
                session_id=existing.session_id,
                state_version=existing.state_version,
            )
        if inp.permission_preset not in _PRESETS:
            raise ValueError(f"unknown permission preset: {inp.permission_preset}")
        state = AgentState()
        state.permission_context = PermissionContext(mode=_PRESETS[inp.permission_preset])
        blob = SessionBlob(
            session_id=uuid4().hex,
            room_id=inp.room_id,
            state_version=1,
            agent_state=state.model_dump(mode="json"),
            permission_preset=inp.permission_preset,
            isolation_mode=self._isolation.mode,
            share_net=self._isolation.share_net,
            backend=self._isolation.backend,
        )
        blob.idempotency[_key(inp.turn_id, "openSession")] = {
            "session_id": blob.session_id,
        }
        await self._store.put(blob)
        await self._emit(
            blob,
            "session.status",
            f"open isolation={blob.isolation_mode} share_net={blob.share_net}",
        )
        await self._emit(blob, "agent.started", blob.session_id)
        return OpenSessionOutput(session_id=blob.session_id, state_version=1)

    async def run_turn(self, inp: RunTurnInput) -> TurnResult:
        blob = await self._require(inp.session_id)
        cached = _cached_turn(blob, inp.turn_id, "runTurn")
        if cached is not None:
            return cached
        if blob.state_version != inp.state_version:
            raise ValueError(
                f"state version {inp.state_version} does not match {blob.state_version}"
            )
        agent = self._agent(blob)
        result = await self._drive(
            agent, UserMsg(name="user", content=inp.message), blob, inp.turn_id
        )
        _remember(blob, inp.turn_id, "runTurn", result)
        await self._store.put(blob)
        await self._emit_turn(blob, result)
        return result

    async def resolve_approval(self, inp: ResolveApprovalInput) -> TurnResult:
        blob = await self._require(inp.session_id)
        cached = _cached_turn(blob, inp.turn_id, "resolveApproval")
        if cached is not None:
            return cached
        if blob.state_version < 1:
            raise ValueError("session has no persisted state")
        agent = self._agent(blob)
        event = _confirm_event(agent, inp)
        result = await self._drive(agent, event, blob, inp.turn_id)
        _remember(blob, inp.turn_id, "resolveApproval", result)
        await self._store.put(blob)
        await self._emit_turn(blob, result)
        return result

    async def deliver_tool_result(self, inp: DeliverToolResultInput) -> TurnResult:
        blob = await self._require(inp.session_id)
        cached = _cached_turn(blob, inp.turn_id, "deliverToolResult")
        if cached is not None:
            return cached
        if blob.state_version != inp.state_version:
            raise ValueError(
                f"state version {inp.state_version} does not match {blob.state_version}"
            )
        agent = self._agent(blob)
        event = _external_result(agent, inp)
        result = await self._drive(agent, event, blob, inp.turn_id)
        _remember(blob, inp.turn_id, "deliverToolResult", result)
        await self._store.put(blob)
        await self._emit(blob, "tool.result", inp.output)
        await self._emit_turn(blob, result)
        return result

    async def steer(self, inp: SteerInput) -> TurnResult:
        blob = await self._require(inp.session_id)
        cached = _cached_turn(blob, inp.turn_id, "steer")
        if cached is not None:
            return cached
        if blob.state_version != inp.state_version:
            raise ValueError(
                f"state version {inp.state_version} does not match {blob.state_version}"
            )
        agent = self._agent(blob)
        await agent.observe(
            Msg(name="user", role="user", content=[HintBlock(hint=inp.hint, source="system")])
        )
        result = await self._drive(agent, None, blob, inp.turn_id)
        _remember(blob, inp.turn_id, "steer", result)
        await self._store.put(blob)
        return result

    async def abort_session(self, inp: AbortSessionInput) -> CloseSessionOutput:
        blob = await self._require(inp.session_id)
        cached = blob.idempotency.get(_key(inp.turn_id, "abort"))
        if cached is not None:
            return CloseSessionOutput(closed=True, state_version=int(cached["state_version"]))
        agent = self._agent(blob)
        pending = agent.state.get_awaiting_tool_calls(agent.name)
        if pending:
            await self._drive(
                agent,
                UserInterruptEvent(reply_id=agent.state.reply_id),
                blob,
                inp.turn_id,
            )
        version = await self.close_session(inp.session_id, inp.turn_id)
        blob = await self._require(inp.session_id)
        blob.idempotency[_key(inp.turn_id, "abort")] = {"state_version": version}
        await self._store.put(blob)
        return CloseSessionOutput(closed=True, state_version=version)

    async def close_session(self, session_id: str, turn_id: str) -> int:
        blob = await self._require(session_id)
        cached = blob.idempotency.get(_key(turn_id, "closeSession"))
        if cached is not None:
            return int(cached["state_version"])
        blob.closed = True
        blob.state_version += 1
        blob.idempotency[_key(turn_id, "closeSession")] = {
            "state_version": blob.state_version,
        }
        await self._store.put(blob)
        await self._emit(blob, "agent.finished", session_id)
        await self._emit(blob, "session.status", "closed")
        return blob.state_version

    def _agent(self, blob: SessionBlob) -> Agent:
        state = AgentState.model_validate(blob.agent_state)
        return Agent(
            name="orbit",
            system_prompt="You are an Orbit business agent.",
            model=build_chat_model(self._model_config),
            toolkit=Toolkit(),
            state=state,
            middlewares=[TracingMiddleware()],
        )

    async def _drive(
        self,
        agent: Agent,
        inputs: Msg | UserConfirmResultEvent | ExternalExecutionResultEvent | UserInterruptEvent | None,
        blob: SessionBlob,
        turn_id: str,
    ) -> TurnResult:
        # Tools are code, not part of the saved blob, so each rebuild registers them.
        if await agent.toolkit.get_tool("gated_echo") is None:
            await agent.toolkit.add_tool(
                FunctionTool(
                    _gated_echo,
                    name="gated_echo",
                    description="Echo text back. Requires a human to allow it.",
                )
            )
        for tool in orbit_tools():
            if await agent.toolkit.get_tool(tool.name) is None:
                await agent.toolkit.add_tool(tool)
        approval: ApprovalAsk | None = None
        external: ExternalCall | None = None
        text = ""
        finished: str | None = None
        try:
            async for event in agent.reply_stream(inputs, yield_final_msg=True):
                if isinstance(event, RequireUserConfirmEvent) and event.tool_calls:
                    call = event.tool_calls[0]
                    approval = ApprovalAsk(
                        approval_request_id=f"apr-{call.id}",
                        tool_name=call.name,
                        call_id=call.id,
                        reason="tool requires confirmation",
                    )
                elif isinstance(event, RequireExternalExecutionEvent) and event.tool_calls:
                    call = event.tool_calls[0]
                    external = ExternalCall(
                        tool_name=call.name,
                        call_id=call.id,
                        arguments=_arguments(call),
                    )
                elif isinstance(event, Msg):
                    reason = event.finished_reason
                    finished = None if reason is None else getattr(reason, "value", reason)
                    text = event.get_text_content() or ""
        except ModelRequestError as exc:
            # The half-finished agent state is dropped, so the blob stays at
            # the version the caller sent and the turn can be retried.
            logger.warning(
                "session %s turn failed [%s]: %s", blob.session_id, exc.code, exc.log_detail
            )
            failure = TurnFailure(
                turnId=turn_id,
                agentId=blob.session_id,
                errorCode=exc.code,
                retryable=exc.retryable,
                message=str(exc),
            )
            await self._emit(blob, "turn.failed", failure.message, failure=failure)
            await self._emit(blob, "session.status", f"turn failed: {exc}")
            return self._turn(
                status="failed",
                session_id=blob.session_id,
                state_version=blob.state_version,
                error=str(exc),
                error_code=exc.code,
                retryable=exc.retryable,
            )
        blob.agent_state = agent.state.model_dump(mode="json")
        blob.state_version += 1
        status = "continue"
        if approval is not None and finished != ReplyFinishedReason.COMPLETED.value:
            status = "needs_approval"
            external = None
        elif external is not None and finished != ReplyFinishedReason.COMPLETED.value:
            status = "needs_external"
        elif finished == ReplyFinishedReason.COMPLETED.value:
            status = "completed"
            approval = None
            external = None
        return self._turn(
            status=status,
            session_id=blob.session_id,
            state_version=blob.state_version,
            approval=approval,
            external=external,
            text=text,
        )

    def _turn(self, **fields: object) -> TurnResult:
        return TurnResult(
            model_mode=self._model_config.mode,
            model_name=self._model_config.name,
            **fields,  # type: ignore[arg-type]
        )

    async def _require(self, session_id: str) -> SessionBlob:
        blob = await self._store.get(session_id)
        if blob is None:
            raise KeyError(f"unknown session {session_id}")
        return blob

    async def _emit(
        self,
        blob: SessionBlob,
        kind: str,
        text: str,
        failure: TurnFailure | None = None,
    ) -> None:
        await self._ingest.emit(
            OrbitEvent(
                type=kind,  # type: ignore[arg-type]
                session_id=blob.session_id,
                room_id=blob.room_id,
                text=text,
                runtime_version=blob.runtime_version,
                permission_preset=blob.permission_preset,
                model_mode=self._model_config.mode,
                model_name=self._model_config.name,
                failure=failure,
            )
        )

    async def _emit_turn(self, blob: SessionBlob, result: TurnResult) -> None:
        if result.approval is not None:
            await self._emit(blob, "approval.asked", result.approval.tool_name)
        if result.external is not None:
            await self._emit(blob, "tool.call", result.external.tool_name)
        if result.text:
            await self._emit(blob, "assistant.message", result.text)


def _confirm_event(agent: Agent, inp: ResolveApprovalInput) -> UserConfirmResultEvent:
    pending = [
        call
        for call in agent.state.get_awaiting_tool_calls(agent.name)
        if getattr(call.state, "value", call.state) in ("asking", "ASKING")
    ]
    if not pending:
        raise ValueError("session is not parked on a confirmation")
    allowed = inp.outcome == "allowed-once"
    return UserConfirmResultEvent(
        reply_id=agent.state.reply_id,
        confirm_results=[
            ConfirmResult(confirmed=allowed, tool_call=call) for call in pending
        ],
    )


def _external_result(agent: Agent, inp: DeliverToolResultInput) -> ExternalExecutionResultEvent:
    pending = [
        call
        for call in agent.state.get_awaiting_tool_calls(agent.name)
        if call.id == inp.call_id
    ]
    if not pending:
        raise ValueError("session is not parked on that external call")
    state = ToolResultState.SUCCESS if inp.result_state == "success" else ToolResultState.ERROR
    return ExternalExecutionResultEvent(
        reply_id=agent.state.reply_id,
        execution_results=[
            ToolResultBlock(
                id=inp.call_id,
                name=inp.tool_name,
                output=inp.output,
                state=state,
                metadata=_metadata(inp.metadata),
            )
        ],
    )


def _arguments(call: ToolCallBlock) -> dict[str, str]:
    try:
        parsed = json.loads(call.input or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): "" if value is None else str(value) for key, value in parsed.items()}


def _metadata(raw: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in raw.items():
        if key in _BOOL_METADATA:
            parsed[key] = value == "true"
        else:
            parsed[key] = value
    return parsed


def _key(turn_id: str, activity: str) -> str:
    return f"{turn_id}:{activity}"


def _remember(blob: SessionBlob, turn_id: str, activity: str, result: TurnResult) -> None:
    blob.idempotency[_key(turn_id, activity)] = result.model_dump(mode="json")


def _cached_turn(blob: SessionBlob, turn_id: str, activity: str) -> TurnResult | None:
    raw = blob.idempotency.get(_key(turn_id, activity))
    if raw is None:
        return None
    return TurnResult.model_validate(raw)

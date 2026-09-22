"""AgentRuntime: one Activity rebuilds the agent from a saved AgentState.

Nothing about AgentScope crosses this class's return values. A parked
approval is just a TurnResult; the live agent object is discarded when the
Activity returns, and the next Activity builds a new one from the blob.
"""

from uuid import uuid4

from agentscope.agent import Agent
from agentscope.event import (
    ConfirmResult,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
)
from agentscope.message import Msg, TextBlock, ToolResultState, UserMsg
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from agentscope.types import ReplyFinishedReason
from orbit_contracts.models import (
    ApprovalAsk,
    OpenSessionInput,
    OpenSessionOutput,
    ResolveApprovalInput,
    RunTurnInput,
    TurnResult,
)

from orbit_worker.mock_model import MockChatModel
from orbit_worker.store import MemoryStateStore, SessionBlob

_PRESETS: dict[str, PermissionMode] = {
    "workspace-write": PermissionMode.ACCEPT_EDITS,
    "read-only": PermissionMode.EXPLORE,
    "danger-full-access": PermissionMode.BYPASS,
}


def _gated_echo(text: str) -> ToolChunk:
    """A custom tool. Custom tools ask for confirmation unless a rule allows them."""

    return ToolChunk(
        content=[TextBlock(text=f"echo:{text}")],
        state=ToolResultState.SUCCESS,
    )


class AgentRuntime:
    def __init__(self, store: MemoryStateStore) -> None:
        self._store = store

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
        )
        blob.idempotency[_key(inp.turn_id, "openSession")] = {
            "session_id": blob.session_id,
        }
        await self._store.put(blob)
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
        result = await self._drive(agent, UserMsg(name="user", content=inp.message), blob)
        _remember(blob, inp.turn_id, "runTurn", result)
        await self._store.put(blob)
        return result

    async def resolve_approval(self, inp: ResolveApprovalInput) -> TurnResult:
        blob = await self._require(inp.session_id)
        cached = _cached_turn(blob, inp.turn_id, "resolveApproval")
        if cached is not None:
            return cached
        agent = self._agent(blob)
        event = _confirm_event(agent, inp)
        result = await self._drive(agent, event, blob)
        _remember(blob, inp.turn_id, "resolveApproval", result)
        await self._store.put(blob)
        return result

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
        return blob.state_version

    def _agent(self, blob: SessionBlob) -> Agent:
        state = AgentState.model_validate(blob.agent_state)
        toolkit = Toolkit()
        return Agent(
            name="orbit",
            system_prompt="You are an Orbit business agent.",
            model=MockChatModel(),
            toolkit=toolkit,
            state=state,
        )

    async def _drive(
        self,
        agent: Agent,
        inputs: Msg | UserConfirmResultEvent,
        blob: SessionBlob,
    ) -> TurnResult:
        # The toolkit is attached before the reply so the parked tool still resolves.
        # Tools are code, not part of the saved blob, so each rebuild registers them.
        if await agent.toolkit.get_tool("gated_echo") is None:
            await agent.toolkit.add_tool(
                FunctionTool(
                    _gated_echo,
                    name="gated_echo",
                    description="Echo text back. Requires a human to allow it.",
                )
            )
        approval: ApprovalAsk | None = None
        text = ""
        finished: str | None = None
        async for event in agent.reply_stream(inputs, yield_final_msg=True):
            if isinstance(event, RequireUserConfirmEvent) and event.tool_calls:
                call = event.tool_calls[0]
                approval = ApprovalAsk(
                    approval_request_id=f"apr-{call.id}",
                    tool_name=call.name,
                    call_id=call.id,
                    reason="custom tool requires confirmation",
                )
            elif isinstance(event, Msg):
                reason = event.finished_reason
                finished = None if reason is None else getattr(reason, "value", reason)
                text = event.get_text_content() or ""
        blob.agent_state = agent.state.model_dump(mode="json")
        blob.state_version += 1
        status = "continue"
        if approval is not None and finished != ReplyFinishedReason.COMPLETED.value:
            status = "needs_approval"
        elif finished == ReplyFinishedReason.COMPLETED.value:
            status = "completed"
            approval = None
        return TurnResult(
            status=status,  # type: ignore[arg-type]
            session_id=blob.session_id,
            state_version=blob.state_version,
            approval=approval,
            text=text,
        )

    async def _require(self, session_id: str, state_version: int | None = None) -> SessionBlob:
        blob = await self._store.get(session_id)
        if blob is None:
            raise KeyError(f"unknown session {session_id}")
        if state_version is not None and blob.state_version != state_version:
            raise ValueError(
                f"state version {state_version} does not match {blob.state_version}"
            )
        return blob


def _confirm_event(agent: Agent, inp: ResolveApprovalInput) -> UserConfirmResultEvent:
    pending = agent.state.get_awaiting_tool_calls(agent.name)
    if not pending:
        raise ValueError("session is not parked on a confirmation")
    allowed = inp.outcome == "allowed-once"
    return UserConfirmResultEvent(
        reply_id=agent.state.reply_id,
        confirm_results=[
            ConfirmResult(confirmed=allowed, tool_call=call) for call in pending
        ],
    )


def _key(turn_id: str, activity: str) -> str:
    return f"{turn_id}:{activity}"


def _remember(blob: SessionBlob, turn_id: str, activity: str, result: TurnResult) -> None:
    blob.idempotency[_key(turn_id, activity)] = result.model_dump(mode="json")


def _cached_turn(blob: SessionBlob, turn_id: str, activity: str) -> TurnResult | None:
    raw = blob.idempotency.get(_key(turn_id, activity))
    if raw is None:
        return None
    return TurnResult.model_validate(raw)

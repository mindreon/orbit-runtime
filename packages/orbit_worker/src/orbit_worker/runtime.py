"""AgentRuntime: one Activity rebuilds the agent from a saved AgentState.

Nothing about AgentScope crosses this class's return values. A parked
approval or external tool call is a TurnResult. The live agent object is
discarded when the Activity returns, and the next Activity builds a new
one from the blob.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import re
from typing import Any

from agentscope.agent import Agent, ReActConfig
from agentscope.event import (
    ConfirmResult,
    ExternalExecutionResultEvent,
    RequireExternalExecutionEvent,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
    UserInterruptEvent,
)
from agentscope.message import (
    Msg,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)
from agentscope.middleware import TracingMiddleware
from agentscope.permission import (
    AdditionalWorkingDirectory,
    PermissionBehavior,
    PermissionContext,
    PermissionMode,
    PermissionRule,
)
from agentscope.skill import Skill
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from agentscope.types import ReplyFinishedReason
from orbit_contracts.models import (
    ApprovalAsk,
    DeliverToolResultInput,
    DeliverToolResultsInput,
    ExternalCall,
    OpenSessionInput,
    OpenSessionOutput,
    OrbitEvent,
    ResolveApprovalInput,
    RunTurnInput,
    TurnFailure,
    TurnResult,
)
from temporalio import activity

from orbit_worker.agent_config import AgentConfig, overriding_agent_config
from orbit_worker.budget_middleware import BudgetExceeded, OrbitBudgetMiddleware
from orbit_worker.chat_model import ModelConfig, ModelRequestError, build_chat_model
from orbit_worker.events import MemoryEventIngest
from orbit_worker.isolation import IsolationSnapshot
from orbit_worker.ledger_middleware import OrbitLedgerMiddleware, ToolLedger
from orbit_worker.mcp_connectors import McpRegistry, attach_mcp_clients, specs_for_storage
from orbit_worker.mock_tools import mock_tools
from orbit_worker.planning_tools import TemporalPlanPort, planning_tools
from orbit_worker.policy_middleware import OrbitPolicyMiddleware
from orbit_worker.sandbox import current_sandbox
from orbit_worker.secrets import redact_text
from orbit_worker.settings import MockSettings
from orbit_worker.store import (
    STATE_UNREADABLE_CODE,
    MemoryStateStore,
    SessionBlob,
    StateStore,
    StateUnreadableError,
)
from orbit_worker.task_stream import current_task_context
from orbit_worker.team_tools import notes_of, stage_prompt, team_tools
from orbit_worker.tools import orbit_tools
from orbit_worker.turn_events import TurnEvents
from orbit_worker.workspace import WORKSPACE_DIR

logger = logging.getLogger(__name__)

_PRESETS: dict[str, PermissionMode] = {
    "workspace-write": PermissionMode.ACCEPT_EDITS,
    "read-only": PermissionMode.EXPLORE,
    "danger-full-access": PermissionMode.BYPASS,
}

_BOOL_METADATA = {"ok", "dissolved"}
_AGENT_NAME = "orbit"
# What AgentScope itself writes as the result of a call it closes on an interruption.
_INTERRUPTED = "<system-reminder>The tool call has been interrupted by the user.</system-reminder>"
_BASE_PROMPT = "You are an Orbit business agent."
_WORKSPACE_PROMPT = (
    f"You have a workspace at {WORKSPACE_DIR}: Bash, Read, Write and Edit work on it, and it is kept for this task, so "
    "files you leave there are still there when the conversation goes on. Files you want the user to have, such as a "
    "report or a script, go in it; what you say in a reply is not a file. Use absolute paths."
)


def _attach_skills(toolkit: Toolkit, skills: tuple[Skill, ...]) -> None:
    """Hand the staged skills to AgentScope, which names them in the system prompt and reads their files on request."""
    if not skills:
        return
    group = toolkit.tool_groups[0]
    present = {item.name for item in group.skills_or_loaders if isinstance(item, Skill)}
    group.skills_or_loaders.extend(skill for skill in skills if skill.name not in present)


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
        tool_ledger: ToolLedger | None = None,
    ) -> None:
        self._tool_ledger = tool_ledger
        self._planning_tools = planning_tools(
            TemporalPlanPort(lambda context: f"task/{context.tenant_id}/{context.task_id}")
        )
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
        self._mcp = McpRegistry()

    async def open_session(self, inp: OpenSessionInput) -> OpenSessionOutput:
        # An attempt's session id is its attempt id, so the checkpoint store finds it from any worker. Any other
        # session id follows from the turn that opens it. Either way, opening again after a retry must not fork state.
        context = current_task_context()
        session_id = (
            context.attempt_id
            if context is not None
            else hashlib.sha256(f"{inp.room_id}:{inp.turn_id}:open".encode()).hexdigest()[:32]
        )
        existing = await self._store.get(session_id)
        if existing is not None:
            return OpenSessionOutput(
                session_id=existing.session_id,
                state_version=existing.state_version,
                carried=bool(existing.idempotency.get(_key(inp.turn_id, "openSession"), {}).get("carried")),
            )
        if inp.permission_preset not in _PRESETS:
            raise ValueError(f"unknown permission preset: {inp.permission_preset}")
        carried = await self._carried_state(inp.continue_from, session_id)
        state = AgentState.model_validate(carried) if carried is not None else AgentState()
        if carried is not None:
            # The attempt this one carries on may have been cut short with a call still running, parked on an approval or
            # waiting for an answer. Under this attempt's id the call would run again, and the side-effect ledger keys a
            # call by (attempt, call id), so a repeat would not be recognised: the call ends here, as interrupted.
            closed = close_unfinished_tool_calls(state, _AGENT_NAME)
            if closed:
                logger.info("session %s carries on %s: closed %d unfinished tool calls", session_id, inp.continue_from, closed)
        # The preset is this attempt's own: the mode may have changed since the session it carries on.
        state.permission_context = PermissionContext(
            mode=_PRESETS[inp.permission_preset],
            allow_rules=_allow_rules_by_tool(inp.allow_rules),
            # Files in the workspace are the agent's to edit; the tools decide on their own paths against this.
            working_directories={WORKSPACE_DIR: AdditionalWorkingDirectory(path=WORKSPACE_DIR, source="orbit")},
        )
        blob = SessionBlob(
            session_id=session_id,
            task_id=inp.room_id,
            state_version=1,
            agent_state=state.model_dump(mode="json"),
            permission_preset=inp.permission_preset,
            isolation_mode=self._isolation.mode,
            share_net=self._isolation.share_net,
            backend=self._isolation.backend,
            mcp_connectors=specs_for_storage(list(inp.mcp_connectors)),
        )
        blob.idempotency[_key(inp.turn_id, "openSession")] = {
            "session_id": blob.session_id,
            "carried": carried is not None,
        }
        await self._store.put(blob)
        await self._emit(
            blob,
            "session.status",
            f"open isolation={blob.isolation_mode} share_net={blob.share_net}",
            turn_id=inp.turn_id,
        )
        await self._emit(blob, "agent.started", blob.session_id, turn_id=inp.turn_id)
        return OpenSessionOutput(session_id=blob.session_id, state_version=1, carried=carried is not None)

    async def _carried_state(self, previous_id: str, session_id: str) -> dict | None:
        """The agent state of the session a follow-up carries on, as its own copy. A session that is gone or cannot be
        read is not a reason to fail the attempt: it starts fresh and says so."""
        if not previous_id or previous_id == session_id:
            return None
        try:
            previous = await self._store.get(previous_id)
        except StateUnreadableError as exc:
            logger.warning("session %s cannot be carried on (%s); starting fresh", previous_id, exc.reason)
            return None
        if previous is None:
            logger.warning("session %s to carry on is gone; starting fresh", previous_id)
            return None
        return previous.agent_state

    async def run_turn(self, inp: RunTurnInput) -> TurnResult:
        try:
            blob = await self._require(inp.session_id)
        except StateUnreadableError as exc:
            return await self._unreadable(inp, inp.state_version, exc)
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
        await self._emit_turn(blob, result, inp.turn_id)
        return result

    async def emit_event(self, event: OrbitEvent) -> None:
        """Ingest an event the workflow built. The worker stamps the model identity."""

        stamped = event.model_copy(
            update={
                "model_mode": self._model_config.mode,
                "model_name": self._model_config.name,
            }
        )
        await self._ingest.emit(stamped)

    async def resolve_approval(self, inp: ResolveApprovalInput) -> TurnResult:
        try:
            blob = await self._require(inp.session_id)
        except StateUnreadableError as exc:
            return await self._unreadable(inp, 0, exc)
        cached = _cached_turn(blob, inp.turn_id, "resolveApproval")
        if cached is not None:
            return cached
        # E2E only. Unset in production, so a resume is not delayed.
        delay = MockSettings().e2e_resolve_delay_s
        if delay:
            await asyncio.sleep(delay)
        await self._emit(blob, "turn.started", turn_id=inp.turn_id)
        if blob.state_version < 1:
            raise ValueError("session has no persisted state")
        agent = self._agent(blob)
        event = _confirm_event(agent, inp)
        result = await self._drive(agent, event, blob, inp.turn_id)
        _remember(blob, inp.turn_id, "resolveApproval", result)
        await self._store.put(blob)
        await self._emit_turn(blob, result, inp.turn_id)
        return result

    async def deliver_tool_result(self, inp: DeliverToolResultInput) -> TurnResult:
        try:
            blob = await self._require(inp.session_id)
        except StateUnreadableError as exc:
            return await self._unreadable(inp, inp.state_version, exc)
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
        await self._emit_turn(blob, result, inp.turn_id)
        return result

    async def deliver_tool_results(self, inp: DeliverToolResultsInput) -> TurnResult:
        """`deliver_tool_result` for several calls at once: the answers to every external call the session is parked on."""
        try:
            blob = await self._require(inp.session_id)
        except StateUnreadableError as exc:
            return await self._unreadable(inp, inp.state_version, exc)
        cached = _cached_turn(blob, inp.turn_id, "deliverToolResults")
        if cached is not None:
            return cached
        if blob.state_version != inp.state_version:
            raise ValueError(
                f"state version {inp.state_version} does not match {blob.state_version}"
            )
        agent = self._agent(blob)
        event = _external_results(agent, inp)
        result = await self._drive(agent, event, blob, inp.turn_id)
        _remember(blob, inp.turn_id, "deliverToolResults", result)
        await self._store.put(blob)
        await self._emit_turn(blob, result, inp.turn_id)
        return result

    def model_config_for(self, config: AgentConfig) -> ModelConfig:
        """A profile may pick another model of the same provider, and say how big its context window is (it replaces the
        worker's `ORBIT_MODEL_CONTEXT_SIZE`). A mock model has neither to pick."""
        if self._model_config.mode != "real":
            return self._model_config
        changes: dict[str, object] = {}
        if config.model:
            changes["name"] = config.model
        if config.context_size:
            changes["context_size"] = config.context_size
        return dataclasses.replace(self._model_config, **changes) if changes else self._model_config

    def _staged_skills(self) -> tuple[Skill, ...]:
        context = current_task_context()
        return context.skills if context is not None else ()

    def _agent_config(self) -> AgentConfig:
        override = overriding_agent_config()
        if override is not None:
            return override
        context = current_task_context()
        return context.agent if context is not None else AgentConfig()

    def _agent(self, blob: SessionBlob) -> Agent:
        state = AgentState.model_validate(blob.agent_state)
        config = self._agent_config()
        prompt = _BASE_PROMPT + (f"\n\n{_WORKSPACE_PROMPT}" if current_sandbox() is not None else "")
        # Persona first, then the work instructions (ADR-0013); a team's prompt is appended to the instructions.
        prompt += f"\n\n{config.soul}" if config.soul else ""
        prompt += f"\n\n{config.instructions}" if config.instructions else ""
        context = current_task_context()
        if context is not None and context.team is not None:
            prompt += f"\n\n{stage_prompt(context.team)}"
        return Agent(
            name=_AGENT_NAME,
            system_prompt=prompt,
            model=build_chat_model(self.model_config_for(config)),
            toolkit=Toolkit(),
            state=state,
            # A cancelled activity must end as cancelled, not as a reply that "finished" after an interrupt (06 §3 S4).
            react_config=ReActConfig(interruption_raise_cancelled_error=True),
            middlewares=[
                TracingMiddleware(),
                # Outside the policy and the ledger: a call it refuses is not checkpointed nor recorded as started.
                OrbitBudgetMiddleware(),
                *(
                    [
                        OrbitPolicyMiddleware(self._tool_ledger),
                        OrbitLedgerMiddleware(self._tool_ledger),
                    ]
                    if self._tool_ledger
                    else []
                ),
            ],
        )

    async def _drive(
        self,
        agent: Agent,
        inputs: Msg
        | UserConfirmResultEvent
        | ExternalExecutionResultEvent
        | UserInterruptEvent
        | None,
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
        task = current_task_context()
        team = task.team if task is not None else None
        for tool in [
            *orbit_tools(),
            # A team stage is its own protocol (07): its agents give work out and take it in, and change no plan.
            *([] if team is not None else self._planning_tools),
            *(team_tools(team) if team is not None else []),
            *mock_tools(self._model_config.mode == "mock"),
            *(sandbox.tools() if (sandbox := current_sandbox()) is not None else []),
        ]:
            if await agent.toolkit.get_tool(tool.name) is None:
                await agent.toolkit.add_tool(tool)
        await attach_mcp_clients(
            agent.toolkit, [*blob.mcp_connectors, *self._agent_config().mcp_connectors], self._mcp
        )
        _attach_skills(agent.toolkit, self._staged_skills())
        approval: ApprovalAsk | None = None
        approvals: list[ApprovalAsk] = []
        externals: list[ExternalCall] = []
        text = ""
        context_start = len(agent.state.context)
        finished: str | None = None
        events = TurnEvents(
            {call.id: call.name for call in agent.state.get_awaiting_tool_calls(agent.name)},
            activity_attempt=_activity_attempt(),
        )
        # A reply that must end in an object of the node's schema. Passed for every new message; a reply that resumes (an
        # approval, an answer) keeps the schema of the one it parked, which is in the saved state (06 §3 S7).
        context = current_task_context()
        schema = context.output_schema if context is not None and isinstance(inputs, Msg) else None
        output: dict[str, Any] | None = None
        stream = agent.reply_stream(inputs, structured_schema=schema, yield_final_msg=True)  # type: ignore[arg-type]
        try:
            while True:
                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    break
                except (ModelRequestError, BudgetExceeded):
                    raise
                except Exception as exc:
                    if not events.in_model_call:
                        raise
                    # A malformed response is a provider failure, not an
                    # Activity failure: Temporal must not retry the turn.
                    raise ModelRequestError(
                        "provider_error",
                        f"{type(exc).__name__} while reading the model response; "
                        f"model={self._model_config.name}",
                    ) from None
                for kind, fields in events.observe(event):
                    await self._emit(blob, kind, turn_id=turn_id, **fields)
                if isinstance(event, RequireUserConfirmEvent) and event.tool_calls:
                    # AgentScope may announce the calls of one step in several events; keep every one.
                    approvals.extend(
                        ApprovalAsk(
                            approval_request_id=f"apr-{call.id}",
                            tool_name=call.name,
                            call_id=call.id,
                            reason="tool requires confirmation",
                            detail=redact_text(_call_detail(call)),
                            allow_rule=_not_yet_allowed(_offered_rule(call), agent.state.permission_context),
                        )
                        for call in event.tool_calls
                        if all(call.id != known.call_id for known in approvals)
                    )
                    approval = approvals[0]
                elif isinstance(event, RequireExternalExecutionEvent) and event.tool_calls:
                    # Each call of a concurrent step announces itself; a team's leader may open several in one step.
                    externals.extend(
                        ExternalCall(tool_name=call.name, call_id=call.id, arguments=_arguments(call))
                        for call in event.tool_calls
                        if all(call.id != known.call_id for known in externals)
                    )
                elif isinstance(event, Msg):
                    reason = event.finished_reason
                    finished = None if reason is None else getattr(reason, "value", reason)
                    text = event.get_text_content() or ""
                    if event.structured_output is not None:
                        output = dict(event.structured_output)
            for kind, fields in events.flush():
                await self._emit(blob, kind, turn_id=turn_id, **fields)
        except asyncio.CancelledError:
            # An interrupt or a stop cancels the activity. What the agent had done so far is kept, so the attempt that
            # replaces this one carries on from it instead of starting from nothing.
            if _cancelled_by_workflow():
                await self._keep_interrupted(agent, blob, turn_id)
            raise
        except BudgetExceeded as exc:
            # The turn stopped between two steps (05 §4): every call of the last batch has its result, so the state is whole
            # and is kept, and the attempt that goes on after budget is granted starts from it.
            logger.info("session %s stopped: %s", blob.session_id, exc)
            close_unfinished_tool_calls(agent.state, agent.name)
            blob.agent_state = agent.state.model_dump(mode="json")
            blob.state_version += 1
            await self._store.put(blob)
            failure = TurnFailure(
                turn_id=turn_id, agent_id=blob.agent.agent_id, error_code="budget", retryable=False, message=str(exc)
            )
            await self._emit(blob, "turn.failed", failure.message, turn_id=turn_id, failure=failure)
            await self._emit(blob, "session.status", f"turn stopped: {exc}", turn_id=turn_id)
            return self._turn(
                status="failed",
                session_id=blob.session_id,
                state_version=blob.state_version,
                error=str(exc),
                error_code="budget",
                retryable=False,
            )
        except ModelRequestError as exc:
            # The half-finished agent state is dropped, so the blob stays at
            # the version the caller sent and the turn can be retried.
            logger.warning(
                "session %s turn failed [%s]: %s", blob.session_id, exc.code, exc.log_detail
            )
            failure = TurnFailure(
                turn_id=turn_id,
                agent_id=blob.agent.agent_id,
                error_code=exc.code,
                retryable=exc.retryable,
                message=str(exc),
            )
            await self._emit(blob, "turn.failed", failure.message, turn_id=turn_id, failure=failure)
            await self._emit(blob, "session.status", f"turn failed: {exc}", turn_id=turn_id)
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
        turn_notes = notes_of(agent.state, agent.name, context_start)
        status = "continue"
        if approval is not None and finished != ReplyFinishedReason.COMPLETED.value:
            status = "needs_approval"
            externals = []
        elif externals and finished != ReplyFinishedReason.COMPLETED.value:
            status = "needs_external"
        elif finished == ReplyFinishedReason.COMPLETED.value:
            status = "completed"
            approval = None
            externals = []
        return self._turn(
            status=status,
            session_id=blob.session_id,
            state_version=blob.state_version,
            approval=approval,
            approvals=approvals,
            external=externals[0] if externals else None,
            externals=externals,
            text=text,
            output=output,
            notes=[text for text, _ in turn_notes],
            note_mentions=[mentions for _, mentions in turn_notes],
        )

    async def _keep_interrupted(self, agent: Agent, blob: SessionBlob, turn_id: str) -> None:
        """Save the state of a turn that was cancelled, with every call it left open closed as interrupted. Nothing here
        may fail the cancellation: if the state cannot be saved the session stays at its last saved version."""
        try:
            close_unfinished_tool_calls(agent.state, agent.name)
            blob.agent_state = agent.state.model_dump(mode="json")
            blob.state_version += 1
            await self._store.put(blob)
            await self._emit(blob, "session.status", "turn interrupted", turn_id=turn_id)
        except Exception:
            logger.warning("session %s: the interrupted turn could not be saved", blob.session_id, exc_info=True)

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

    async def _unreadable(
        self,
        inp: RunTurnInput | ResolveApprovalInput | DeliverToolResultInput,
        state_version: int,
        exc: StateUnreadableError,
    ) -> TurnResult:
        # Returned, not raised: a retry reads the same blob, so Temporal must
        # not retry. Nothing is written; the stored blob stays as it was.
        logger.warning(
            "session %s turn %s failed [%s]: %s",
            inp.session_id,
            inp.turn_id,
            STATE_UNREADABLE_CODE,
            exc.reason,
        )
        # Events need the room identity the unreadable blob would have given.
        stand_in = SessionBlob(
            session_id=inp.session_id,
            task_id=inp.room_id,
            state_version=state_version,
            agent_state={},
            permission_preset="",
        )
        failure = TurnFailure(
            turn_id=inp.turn_id,
            agent_id=stand_in.agent.agent_id,
            error_code=STATE_UNREADABLE_CODE,
            retryable=False,
            message=str(exc),
        )
        await self._emit(
            stand_in, "turn.failed", failure.message, turn_id=inp.turn_id, failure=failure
        )
        await self._emit(stand_in, "session.status", f"turn failed: {exc}", turn_id=inp.turn_id)
        return self._turn(
            status="failed",
            session_id=inp.session_id,
            state_version=state_version,
            error=str(exc),
            error_code=STATE_UNREADABLE_CODE,
            retryable=False,
        )

    async def _emit(
        self,
        blob: SessionBlob,
        kind: str,
        text: str = "",
        *,
        turn_id: str = "",
        **fields: object,
    ) -> None:
        agent = blob.agent
        event = OrbitEvent(
            type=kind,  # type: ignore[arg-type]
            session_id=blob.session_id,
            room_id=blob.task_id,
            text=text,
            runtime_version=blob.runtime_version,
            permission_preset=blob.permission_preset,
            turn_id=turn_id,
            agent_id=agent.agent_id,
            parent_agent_id=agent.parent_agent_id,
            parent_session_id=agent.parent_session_id,
            depth=agent.depth,
            persona=agent.persona,
            agent_path=agent.agent_path,
            model_mode=self._model_config.mode,
            model_name=self._model_config.name,
            **fields,  # type: ignore[arg-type]
        )
        # Suspected secrets are replaced, never raised: the blob is already saved.
        await self._ingest.emit(
            event.model_copy(
                update={
                    "text": redact_text(event.text),
                    "delta": redact_text(event.delta),
                    "args_preview": redact_text(event.args_preview),
                }
            )
        )

    async def _emit_turn(self, blob: SessionBlob, result: TurnResult, turn_id: str) -> None:
        # tool.call and tool.result were emitted while the turn streamed.
        if result.approval is not None:
            await self._emit(
                blob,
                "approval.asked",
                turn_id=turn_id,
                tool_name=result.approval.tool_name,
                call_id=result.approval.call_id or "",
                approval_request_id=result.approval.approval_request_id,
            )
        if result.text:
            await self._emit(blob, "assistant.message", result.text, turn_id=turn_id)


def close_unfinished_tool_calls(state: AgentState, agent_name: str) -> int:
    """End every tool call of the agent that has no result yet, as AgentScope does when a reply is interrupted
    (`Agent._close_unfinished_tool_calls`): the call is FINISHED and a result in state INTERRUPTED follows it. That covers
    calls that were running, parked on a confirmation (ASKING) or submitted for an outside answer (SUBMITTED). Returns
    how many were closed.

    A call counts as answered when a result with its id is anywhere in the context: a reply that resumes after a
    confirmation writes its results into a new assistant message, not the one that holds the call."""
    answered = {
        block.id
        for message in state.context
        if not isinstance(message.content, str)
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }
    closed = 0
    for message in state.context:
        if message.role != "assistant" or message.name != agent_name or isinstance(message.content, str):
            continue
        for block in list(message.content):
            if isinstance(block, ToolCallBlock) and block.id not in answered:
                block.state = ToolCallState.FINISHED
                message.content.append(
                    ToolResultBlock(
                        id=block.id, name=block.name, output=_INTERRUPTED, state=ToolResultState.INTERRUPTED
                    )
                )
                answered.add(block.id)
                closed += 1
    return closed


def _cancelled_by_workflow() -> bool:
    """Whether the cancel that is ending this turn was asked for by the workflow (an interrupt, a stop, a task cancel).
    A heartbeat timeout, a worker shutdown or a pause also cancel the activity, but Temporal then retries the same
    activity with the state version it was given, so nothing may be saved for those. Outside an activity (tests, SOP
    steps) there is no one else to retry, and the state is kept as it always was."""
    try:
        details = activity.cancellation_details()
    except RuntimeError:
        return True
    if details is None:
        return False  # inside an activity, a cancel that says nothing about why is not taken for the workflow's
    return bool(details.cancel_requested) and not (
        details.worker_shutdown or details.timed_out or details.paused or details.reset or details.not_found
    )


def _activity_attempt() -> int:
    try:
        return activity.info().attempt
    except RuntimeError:
        return 1


def _allow_rules_by_tool(specs: list[dict[str, str | None]]) -> dict[str, list[PermissionRule]]:
    rules: dict[str, list[PermissionRule]] = {}
    for spec in specs:
        rule = _allow_rule(spec)
        rules.setdefault(rule.tool_name, []).append(rule)
    return rules


def _confirm_event(agent: Agent, inp: ResolveApprovalInput) -> UserConfirmResultEvent:
    pending = [
        call
        for call in agent.state.get_awaiting_tool_calls(agent.name)
        if getattr(call.state, "value", call.state) in ("asking", "ASKING")
    ]
    if not pending:
        raise ValueError("session is not parked on a confirmation")
    allowed = inp.outcome == "allowed-once"
    results: list[ConfirmResult] = []
    for call in pending:
        confirmed = inp.decisions.get(call.id, allowed)
        rule = inp.rules.get(call.id)
        # A rule a person allowed for the rest of the task joins the ones the agent checks its next calls against.
        rules = [_allow_rule(rule)] if confirmed and rule else None
        results.append(ConfirmResult(confirmed=confirmed, tool_call=call, rules=rules))
    return UserConfirmResultEvent(reply_id=agent.state.reply_id, confirm_results=results)


def _allow_rule(spec: dict[str, str | None]) -> PermissionRule:
    return PermissionRule(
        tool_name=str(spec["tool_name"]), rule_content=spec.get("rule_content"), behavior=PermissionBehavior.ALLOW, source="task"
    )


def _call_detail(call: ToolCallBlock) -> str:
    """What a call is made with, for a person to read: the command, the path, else the arguments."""
    arguments = _arguments(call)
    for key in ("command", "file_path", "path", "url"):
        if arguments.get(key):
            return arguments[key][:500]
    return json.dumps(arguments, ensure_ascii=False)[:500]


# One program and its arguments: nothing that runs a second command, substitutes one, or reads a script from the input.
_SIMPLE_COMMAND = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*(?:\s[^;&|`$<(){}\n]*)?$")


_WORDS = re.compile(r"[A-Za-z0-9_./-]+")
# Programs that are asked about every time: deleting, privilege, permissions, disks, and running other programs.
_ASK_EVERY_TIME = frozenset(
    ["rm", "rmdir", "sudo", "su", "doas", "chmod", "chown", "chgrp", "dd", "mkfs", "mount", "umount", "kill", "killall", "pkill", "shutdown", "reboot", "sh", "bash", "zsh", "dash", "eval", "exec", "xargs", "env", "nohup", "ssh", "scp"]
)


def _not_yet_allowed(rule: dict[str, str | None] | None, context: PermissionContext) -> dict[str, str | None] | None:
    """The rule to offer, unless the task already allows it: a call that still asks, with the rule in force, asks because
    of how it is written (a heredoc, a substitution), and offering the same "always" again would promise what it cannot."""
    if rule is None:
        return None
    held = context.allow_rules.get(str(rule["tool_name"]), [])
    return None if any(item.rule_content in (None, rule["rule_content"]) for item in held) else rule


def _offered_rule(call: ToolCallBlock) -> dict[str, str | None] | None:
    """What "always allow" would allow for the rest of the task, if anything.

    A file tool: the path pattern AgentScope suggests. Bash: the program, when the command is one simple command; for the
    commands an agent really writes (`a || b`, `cd x && python3 y`, a heredoc) no program says what is allowed, so it is
    every Bash command of the task. Either way nothing is offered for a command that names a program in
    `_ASK_EVERY_TIME`, and what AgentScope's safety checks always ask about is still asked."""
    if call.name != "Bash":
        suggested = [rule for rule in call.suggested_rules if rule.behavior == PermissionBehavior.ALLOW]
        return {"tool_name": suggested[0].tool_name, "rule_content": suggested[0].rule_content} if suggested else None
    command = _arguments(call).get("command", "").strip()
    if any(word.rsplit("/", 1)[-1] in _ASK_EVERY_TIME for word in _WORDS.findall(command)):
        return None
    if _SIMPLE_COMMAND.match(command):
        return {"tool_name": "Bash", "rule_content": f"{command.split()[0]}:*"}
    return {"tool_name": "Bash", "rule_content": None}


def _external_result(agent: Agent, inp: DeliverToolResultInput) -> ExternalExecutionResultEvent:
    pending = [
        call for call in agent.state.get_awaiting_tool_calls(agent.name) if call.id == inp.call_id
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


def _external_results(agent: Agent, inp: DeliverToolResultsInput) -> ExternalExecutionResultEvent:
    """The answers to every external call the session is parked on, as one event: AgentScope resumes the reply only when
    each open call has its result, so what is given has to be exactly those. They are put in the order the calls were made,
    which is the order the agent reads them in."""
    pending = [
        call.id
        for call in agent.state.get_awaiting_tool_calls(agent.name)
        if getattr(call.state, "value", call.state) in ("submitted", "SUBMITTED")
    ]
    given = {item.call_id: item for item in inp.results}
    if not pending or set(given) != set(pending) or len(given) != len(inp.results):
        raise ValueError(f"session is parked on external calls {sorted(pending)}, not {sorted(given)}")
    return ExternalExecutionResultEvent(
        reply_id=agent.state.reply_id,
        execution_results=[
            ToolResultBlock(
                id=call_id,
                name=given[call_id].tool_name,
                output=given[call_id].output,
                state=ToolResultState.SUCCESS if given[call_id].result_state == "success" else ToolResultState.ERROR,
            )
            for call_id in pending
        ],
    )


def _arguments(call: ToolCallBlock) -> dict[str, str]:
    try:
        parsed = json.loads(call.input or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    # Structured values (ask_user's `questions`) stay JSON; everything else is its string form, as before.
    return {
        str(key): json.dumps(value, ensure_ascii=False)
        if isinstance(value, dict | list)
        else "" if value is None else str(value)
        for key, value in parsed.items()
    }


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

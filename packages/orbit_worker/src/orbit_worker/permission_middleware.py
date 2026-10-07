"""The part of the permission presets AgentScope's engine cannot express (see `permissions`).

It runs inside `OrbitPolicyMiddleware`: the engine decides first, this middleware then turns the decision into a DENY (a
catastrophic command, a write that is not allowed), an ASK (a write to a protected file in `full`) or, for the presets that
trust the classifier, an ALLOW in place of an ASK. The policy middleware outside it can still refuse whatever it returns.
"""

from __future__ import annotations

from typing import Any

from agentscope.message import ToolCallState
from agentscope.middleware import MiddlewareBase
from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
    PermissionMode,
)

from orbit_worker.command_risk import assess
from orbit_worker.permissions import (
    CATASTROPHIC_DENIED,
    FILE_TOOLS,
    PROTECTED_ASK,
    WRITES_DENIED,
    PermissionPlan,
)

_ASKING = (PermissionBehavior.ASK, PermissionBehavior.PASSTHROUGH)


class OrbitPermissionMiddleware(MiddlewareBase):
    def __init__(self, plan: PermissionPlan) -> None:
        self._plan = plan

    async def on_check_permission(self, agent: Any, input_kwargs: dict, next_handler: Any) -> PermissionDecision:
        decision = await next_handler(**input_kwargs)
        tool = input_kwargs["tool"]
        tool_input = input_kwargs["tool_input"]
        if tool.name == "Bash":
            return self._bash(decision, str(tool_input.get("command") or ""), _confirmed(input_kwargs["tool_call"]))
        if tool.name in FILE_TOOLS:
            return await self._file(decision, tool, tool_input, _confirmed(input_kwargs["tool_call"]))
        return decision

    def _bash(self, decision: PermissionDecision, command: str, confirmed: bool) -> PermissionDecision:
        plan = self._plan
        found = assess(command)
        # Catastrophic is refused in every preset and under every earlier decision, an approval or an allow rule included.
        if found.level == "catastrophic":
            return _deny(CATASTROPHIC_DENIED.format(reason=found.reason), "Safety check: catastrophic command")
        if decision.behavior == PermissionBehavior.DENY or plan.read_only:
            return decision
        if plan.write_scope == "none" and found.writes:
            return _deny(WRITES_DENIED.format(reason=found.reason or "Bash"), "This task may not write files")
        if plan.full:
            # BYPASS skips the safety asks of the tool; a write to a dotfile, .ssh, .git or .env is put back as an ask,
            # unless a person already said yes to this very call (asking again would never end).
            if found.protected_write and not confirmed:
                return _ask(PROTECTED_ASK.format(reason=found.reason))
            return decision
        if decision.behavior not in _ASKING:
            return decision
        if plan.auto_builtin and found.skill_script:
            return _allow("a skill script of the workspace")
        # Without the right to write, a medium command (python x.py) may still write: only a low one is trusted.
        trusted = ("low", "medium") if plan.write_scope == "workspace" else ("low",)
        if plan.auto_commands and found.level in trusted:
            return _allow(f"a {found.level} risk command")
        return decision

    async def _file(
        self, decision: PermissionDecision, tool: Any, tool_input: dict[str, Any], confirmed: bool
    ) -> PermissionDecision:
        plan = self._plan
        if decision.behavior == PermissionBehavior.DENY or plan.read_only:
            return decision
        if plan.write_scope == "none":
            return _deny(WRITES_DENIED.format(reason=tool.name), "This task may not write files")
        if plan.full and not confirmed:
            # BYPASS skips the tool's safety ask for a dangerous file or directory; it is put back here.
            own = await tool.check_permissions(tool_input, PermissionContext(mode=PermissionMode.BYPASS))
            if own.behavior == PermissionBehavior.ASK and own.bypass_immune:
                return own
        return decision


def _confirmed(call: Any) -> bool:
    """A person already allowed this very call: it comes back through the engine as allowed, and must not be asked again."""
    return getattr(call, "state", None) == ToolCallState.ALLOWED


def _deny(message: str, reason: str) -> PermissionDecision:
    return PermissionDecision(behavior=PermissionBehavior.DENY, message=message, decision_reason=reason)


def _ask(message: str) -> PermissionDecision:
    return PermissionDecision(
        behavior=PermissionBehavior.ASK,
        message=message,
        decision_reason="Safety check: protected file or directory",
        bypass_immune=True,
    )


def _allow(why: str) -> PermissionDecision:
    return PermissionDecision(
        behavior=PermissionBehavior.ALLOW,
        message=f"Permission granted: {why}",
        decision_reason="Allowed by the permission preset of the task",
    )
